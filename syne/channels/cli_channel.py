"""Syne CLI — Interactive terminal chat, Pi-style fixed prompt at bottom.

Uses ANSI scroll regions: content scrolls in the upper area, prompt is
fixed at the bottom. Modeled after Pi's TUI (../pi) component ordering
where the editor always renders last and stays at the bottom.
"""

import asyncio
import getpass
import logging
import os
import re
import select
import shutil
import sys
import time
import termios
import tty

from ..agent import SyneAgent
from ..config import load_settings
from ..db.models import get_or_create_user, get_identity
from ..llm.provider import StreamCallbacks

logger = logging.getLogger("syne.cli_channel")

# ── ANSI helpers ──
_DIM = "\033[2m"
_DIM_ITALIC = "\033[2;3m"
_ITALIC = "\033[3m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM_GREEN = "\033[2;32m"
_DIM_RED = "\033[2;31m"
_DIM_YELLOW = "\033[2;33m"
_DIM_CYAN = "\033[2;36m"

# ── Terminal state ──
_term_rows = 38
_term_cols = 125
_sr_end = 36      # last row of scroll region
_sep_row = 37     # separator ────
_input_row = 38   # > prompt

# ── Spinner ──
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_task: asyncio.Task | None = None


def _write(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _term_height() -> int:
    try:
        import fcntl, struct
        result = fcntl.ioctl(sys.stdin.fileno(), termios.TIOCGWINSZ, b'\x00' * 8)
        rows = struct.unpack('HHHH', result)[0]
        if rows > 0:
            return rows
    except Exception:
        pass
    return shutil.get_terminal_size((80, 24)).lines


def _setup_screen():
    """Clear screen and set up scroll region with fixed prompt at bottom."""
    global _term_rows, _term_cols, _sr_end, _sep_row, _input_row
    _term_rows = _term_height()
    _term_cols = _term_width()
    _sr_end = _term_rows - 2     # scroll region: rows 1.._sr_end
    _sep_row = _term_rows - 1    # separator line
    _input_row = _term_rows      # prompt "> " line
    _write("\033[2J\033[H")                  # clear screen + home
    _write(f"\033[1;{_sr_end}r")             # set scroll region
    _write(f"\033[{_sr_end};1H")             # cursor at bottom of scroll region


def _draw_prompt(buf="", cursor_col=0, status_left="", status_right=""):
    """Draw the fixed prompt area (separator with status + input line).

    Like Pi's footer: status info embedded in the separator line.
    """
    w = _term_cols
    _write("\0337")  # save cursor
    # Separator line with optional status (like Pi footer)
    if status_left or status_right:
        # ── status_left ──────────── status_right ──
        mid = w - len(status_left) - len(status_right) - 4
        if mid < 4:
            mid = 4
        sep = f"{_DIM}──{_RESET} {_DIM}{status_left}{_RESET} {_DIM}{'─' * mid}{_RESET} {_DIM}{status_right}{_RESET}"
        _write(f"\033[{_sep_row};1H\033[2K{sep}")
    else:
        _write(f"\033[{_sep_row};1H\033[2K{_DIM}{'─' * w}{_RESET}")
    _write(f"\033[{_input_row};1H\033[2K> {buf}")
    _write(f"\033[{_input_row};{cursor_col + 3}H")
    _write("\0338")  # restore cursor


def _move_to_input(buf="", cursor_col=0):
    """Move cursor to the input line for typing."""
    _write(f"\033[{_input_row};{cursor_col + 3}H")


def _scroll_print(text: str):
    """Print text in the scroll region (auto-scrolls when full)."""
    _write("\0337")                          # save cursor
    _write(f"\033[{_sr_end};1H")             # move to bottom of scroll region
    _write("\n")                             # scroll up one line
    _write(f"\033[{_sr_end};1H")             # back to bottom
    _write(text)
    _write("\0338")                          # restore cursor


# ── Spinner (in scroll region) ──

async def _start_spinner(message: str = "Working..."):
    global _spinner_task
    _stop_spinner()

    async def _spin():
        i = 0
        try:
            while True:
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                _write(f"\033[{_sr_end};1H\033[2K  {_DIM_CYAN}{frame}{_RESET} {_DIM}{message}{_RESET} {_DIM}(Ctrl+C to interrupt){_RESET}")
                i += 1
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass

    _spinner_task = asyncio.create_task(_spin())


def _stop_spinner():
    global _spinner_task
    if _spinner_task and not _spinner_task.done():
        _spinner_task.cancel()
        _write(f"\033[{_sr_end};1H\033[2K")  # clear spinner line
    _spinner_task = None


# ── Markdown stream ──
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
_MD_CODE = re.compile(r'`([^`]+)`')
_MD_HEADING = re.compile(r'^(#{1,3})\s+(.+)$')


class _MarkdownStream:
    def __init__(self):
        self._buf = ""
        self._in_code_block = False

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line)
            _write("\n")

    def flush(self) -> None:
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""

    def reset(self) -> None:
        self._buf = ""
        self._in_code_block = False

    def _emit_line(self, line: str) -> None:
        if line.strip().startswith("```"):
            self._in_code_block = not self._in_code_block
            if self._in_code_block:
                lang = line.strip()[3:].strip()
                if lang:
                    _write(f"  {_DIM}[{lang}]{_RESET}")
            return
        if self._in_code_block:
            _write(f"  {_DIM_CYAN}{line}{_RESET}")
            return
        m = _MD_HEADING.match(line)
        if m:
            _write(f"  {_BOLD}{m.group(2)}{_RESET}")
            return
        if line.strip() in ("---", "***", "___"):
            return
        line = _MD_BOLD.sub(f'{_BOLD}\\1{_RESET}', line)
        line = _MD_CODE.sub(f'{_DIM_CYAN}\\1{_RESET}', line)
        line = _MD_ITALIC.sub(f'{_ITALIC}\\1{_RESET}', line)
        _write(f"  {line}")


# ── Slash commands ──
_SLASH_COMMANDS = [
    "/help", "/models", "/status", "/memory", "/compact",
    "/clear", "/cost", "/context", "/exit",
]

# ── Token usage ──
class _SessionUsage:
    __slots__ = ("total_in", "total_out")
    def __init__(self):
        self.total_in = 0
        self.total_out = 0

_session_usage = _SessionUsage()


def _short_model_name(model: str) -> str:
    name = model
    name = re.sub(r'-preview(-\d{2}-\d{2})?$', '', name)
    name = re.sub(r'-\d{6,}$', '', name)
    name = re.sub(r'-\d{4}$', '', name)
    return name


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


# ── Tool activity labels ──
_TOOL_LABELS = {
    "read_source": "Read", "file_read": "Read", "file_write": "Write",
    "exec": "Run", "db_query": "Query", "web_fetch": "Fetch",
    "web_search": "Search", "send_message": "Send", "send_file": "SendFile",
    "send_voice": "Voice", "send_reaction": "React",
    "manage_schedule": "Schedule", "update_config": "Config",
    "update_ability": "Ability", "memory_search": "MemSearch",
    "memory_add": "MemAdd",
}

_last_tool_key = ""


def _format_tool_activity(name: str, args: dict, result_preview: str) -> None:
    """Display tool call in Pi style: bold title, accent path, muted output."""
    global _last_tool_key
    _stop_spinner()

    dedup_key = f"{name}:{list(args.values())[:1]}"
    if dedup_key == _last_tool_key:
        return
    _last_tool_key = dedup_key

    # Build Pi-style tool header: "bold_label accent_arg"
    label = _TOOL_LABELS.get(name, name)
    arg_hint = ""
    if args:
        for key in ("path", "file_path", "file", "command", "query", "action", "url", "name", "key"):
            if key in args:
                v = str(args[key])
                if len(v) > 80:
                    v = v[:77] + "..."
                arg_hint = v
                break
        if not arg_hint:
            v = str(list(args.values())[0])
            if len(v) > 80:
                v = v[:77] + "..."
            arg_hint = v

    # Pi style: exec → "$ command", read → "read path", write → "write path"
    if name == "exec" and arg_hint:
        header = f"  {_BOLD}$ {arg_hint}{_RESET}"
    elif arg_hint:
        header = f"  {_BOLD}{label.lower()}{_RESET} {_DIM_CYAN}{arg_hint}{_RESET}"
    else:
        header = f"  {_BOLD}{label.lower()}{_RESET}"

    w = _term_width()
    _write(f"\n  {_DIM}{'─' * (w - 4)}{_RESET}\n")  # top border
    _write(f"{header}\n")

    preview = result_preview.strip().replace("\n", " ")
    if len(preview) > 100:
        preview = preview[:97] + "..."
    if preview:
        is_error = preview.lower().startswith("error")
        color = _DIM_RED if is_error else _DIM
        _write(f"  {color}{preview}{_RESET}\n")
    _write(f"  {_DIM}{'─' * (w - 4)}{_RESET}\n")  # bottom border


# ══════════════════════════════════════════════════════════════════════════════
# Raw terminal input — reads keys in the fixed prompt area
# ══════════════════════════════════════════════════════════════════════════════

def _read_key(fd: int) -> tuple[str, str]:
    """Read one key press from raw fd. Returns (key_name, char_data)."""
    ch = os.read(fd, 1)
    if not ch:
        return ("eof", "")
    b = ch[0]

    if b == 0x1b:  # ESC — start of escape sequence
        # Check for more bytes (timeout 50ms)
        if select.select([fd], [], [], 0.05)[0]:
            seq = ch
            while select.select([fd], [], [], 0.01)[0]:
                seq += os.read(fd, 1)
            s = seq.decode("utf-8", errors="ignore")
            if s == "\x1b[A": return ("up", "")
            if s == "\x1b[B": return ("down", "")
            if s == "\x1b[C": return ("right", "")
            if s == "\x1b[D": return ("left", "")
            if s == "\x1b[H": return ("home", "")
            if s == "\x1b[F": return ("end", "")
            if s == "\x1b[3~": return ("delete", "")
            if s == "\x1b[13;2u": return ("shift-enter", "")
            if s == "\x1b[27;2;13~": return ("shift-enter", "")
            return ("esc-seq", s)
        return ("escape", "")

    if b == 0x0d or b == 0x0a:  # Enter
        return ("enter", "")
    if b == 0x7f or b == 0x08:  # Backspace
        return ("backspace", "")
    if b == 0x03:  # Ctrl+C
        return ("ctrl-c", "")
    if b == 0x04:  # Ctrl+D
        return ("ctrl-d", "")
    if b == 0x09:  # Tab
        return ("tab", "")
    if b == 0x01:  # Ctrl+A (home)
        return ("home", "")
    if b == 0x05:  # Ctrl+E (end)
        return ("end", "")
    if b == 0x0b:  # Ctrl+K (kill to end)
        return ("ctrl-k", "")
    if b == 0x15:  # Ctrl+U (kill line)
        return ("ctrl-u", "")

    # Multi-byte UTF-8
    if b >= 0x80:
        needed = 1 if b < 0xe0 else (2 if b < 0xf0 else 3)
        more = os.read(fd, needed)
        return ("char", (ch + more).decode("utf-8", errors="ignore"))

    if b >= 0x20:
        return ("char", chr(b))

    return ("unknown", "")


def _redraw_input(buf: str, cursor: int):
    """Redraw the input line in the fixed prompt area."""
    # For multiline, show line count + current line
    lines = buf.split("\n")
    if len(lines) > 1:
        # Find which line the cursor is on
        pos = 0
        cur_line_idx = 0
        for i, ln in enumerate(lines):
            if pos + len(ln) >= cursor:
                cur_line_idx = i
                break
            pos += len(ln) + 1  # +1 for \n
        cur_line = lines[cur_line_idx]
        col_in_line = cursor - sum(len(lines[j]) + 1 for j in range(cur_line_idx))
        prefix = f"({len(lines)}L) > "
        _write(f"\033[{_input_row};1H\033[2K{prefix}{cur_line}")
        _write(f"\033[{_input_row};{len(prefix) + col_in_line + 1}H")
    else:
        _write(f"\033[{_input_row};1H\033[2K> {buf}")
        _write(f"\033[{_input_row};{cursor + 3}H")


def _blocking_read_line(history: list[str]) -> str | None:
    """Read a line with editing in the fixed prompt area. Blocking."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf = ""
    cursor = 0
    hist_idx = -1
    hist_save = ""

    # Draw clean prompt
    _write(f"\033[{_input_row};1H\033[2K> ")

    try:
        tty.setraw(fd)
        while True:
            key, data = _read_key(fd)

            if key == "enter":
                if buf.endswith("\\"):
                    buf = buf[:-1] + "\n"
                    cursor = len(buf)
                    # Temporarily exit raw to redraw
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)
                    continue
                return buf

            elif key == "shift-enter":
                buf = buf[:cursor] + "\n" + buf[cursor:]
                cursor += 1
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

            elif key == "ctrl-c":
                raise KeyboardInterrupt

            elif key == "ctrl-d":
                if not buf:
                    return None  # EOF

            elif key == "backspace":
                if cursor > 0:
                    buf = buf[:cursor - 1] + buf[cursor:]
                    cursor -= 1
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "delete":
                if cursor < len(buf):
                    buf = buf[:cursor] + buf[cursor + 1:]
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "left":
                if cursor > 0:
                    cursor -= 1
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "right":
                if cursor < len(buf):
                    cursor += 1
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "home":
                cursor = 0
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

            elif key == "end":
                cursor = len(buf)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

            elif key == "ctrl-u":
                buf = buf[cursor:]
                cursor = 0
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

            elif key == "ctrl-k":
                buf = buf[:cursor]
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

            elif key == "up":
                if history:
                    if hist_idx == -1:
                        hist_save = buf
                        hist_idx = len(history) - 1
                    elif hist_idx > 0:
                        hist_idx -= 1
                    buf = history[hist_idx]
                    cursor = len(buf)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "down":
                if hist_idx >= 0:
                    hist_idx += 1
                    if hist_idx >= len(history):
                        hist_idx = -1
                        buf = hist_save
                    else:
                        buf = history[hist_idx]
                    cursor = len(buf)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _redraw_input(buf, cursor)
                    tty.setraw(fd)

            elif key == "tab":
                # Simple slash command completion
                if buf.startswith("/") and "\n" not in buf:
                    matches = [c for c in _SLASH_COMMANDS if c.startswith(buf)]
                    if len(matches) == 1:
                        buf = matches[0]
                        cursor = len(buf)
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        _redraw_input(buf, cursor)
                        tty.setraw(fd)

            elif key == "char":
                buf = buf[:cursor] + data + buf[cursor:]
                cursor += len(data)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                _redraw_input(buf, cursor)
                tty.setraw(fd)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ══════════════════════════════════════════════════════════════════════════════
# Main CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

async def run_cli(debug: bool = False, yolo: bool = False, fresh: bool = False, node_client=None):
    remote_mode = node_client is not None
    if debug:
        logging.getLogger("syne").setLevel(logging.DEBUG)
    else:
        logging.getLogger("syne").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.WARNING)

    agent = None
    if not remote_mode:
        settings = load_settings()
        agent = SyneAgent(settings)

    listen_task = None

    try:
        # ── Remote mode setup ──
        if remote_mode:
            await node_client.connect()
            listen_task = asyncio.create_task(node_client.listen())
        # ── Local mode setup ──
        else:
            await agent.start()
            agent._cli_cwd = os.getcwd()
            from ..tools.file_ops import set_workspace
            set_workspace(os.getcwd())
            from ..main import _auto_migrate_google_oauth
            await _auto_migrate_google_oauth()

            username = getpass.getuser()
            user = await get_or_create_user(
                name=username, platform="cli",
                platform_id=f"cli:{username}",
                display_name=username, is_dm=True,
            )
            if user.get("access_level") != "owner":
                from ..db.models import update_user
                await update_user("cli", f"cli:{username}", access_level="owner")
                user = dict(user)
                user["access_level"] = "owner"

        # ── Callbacks & approval (local mode) ──
        if not remote_mode:
            if agent.conversations:
                agent.conversations.add_status_callback(_cli_status_callback)

            if not yolo:
                _always_allowed: set[str] = set()

                async def _approval_callback(tool_name: str, args: dict) -> tuple[bool, str]:
                    if tool_name != "file_write":
                        return True, ""
                    file_path = args.get("path", "?")
                    resolved = os.path.abspath(file_path)
                    if resolved in _always_allowed:
                        return True, ""

                    content = args.get("content", "")
                    lines_count = content.count("\n") + 1
                    size = len(content.encode("utf-8"))

                    if os.path.isfile(resolved):
                        try:
                            import difflib
                            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                                old_lines = f.readlines()
                            new_lines = content.splitlines(keepends=True)
                            diff = list(difflib.unified_diff(old_lines, new_lines, fromfile=file_path, tofile=file_path, lineterm=""))
                            if diff:
                                _write(f"\n  {_BOLD}Edit file{_RESET} {file_path}\n")
                                shown = 0
                                for dline in diff:
                                    if dline.startswith("@@"):
                                        _write(f"  {_DIM_CYAN}{dline.rstrip()}{_RESET}\n")
                                    elif dline.startswith("-") and not dline.startswith("---"):
                                        _write(f"  {_DIM_RED}{dline.rstrip()}{_RESET}\n")
                                    elif dline.startswith("+") and not dline.startswith("+++"):
                                        _write(f"  {_DIM_GREEN}{dline.rstrip()}{_RESET}\n")
                                    else:
                                        _write(f"  {_DIM}{dline.rstrip()}{_RESET}\n")
                                    shown += 1
                                    if shown >= 30:
                                        remaining = len(diff) - shown
                                        if remaining > 0:
                                            _write(f"  {_DIM}  ... {remaining} more lines{_RESET}\n")
                                        break
                            else:
                                _write(f"\n  {_BOLD}Write to{_RESET} {file_path} {_DIM}(no changes){_RESET}\n")
                        except Exception:
                            _write(f"\n  {_BOLD}Write to{_RESET} {file_path} ({lines_count} lines, {size:,} bytes)\n")
                    else:
                        _write(f"\n  {_BOLD}New file{_RESET} {file_path} ({lines_count} lines, {size:,} bytes)\n")
                        preview_lines = content.split("\n")[:10]
                        for pline in preview_lines:
                            _write(f"  {_DIM_GREEN}+ {pline}{_RESET}\n")
                        if lines_count > 10:
                            _write(f"  {_DIM}  ... {lines_count - 10} more lines{_RESET}\n")

                    _write(f"\n  {_BOLD}Allow edit to {os.path.basename(file_path)}?{_RESET} (y/a/n): ")
                    loop = asyncio.get_running_loop()
                    resp = await loop.run_in_executor(None, lambda: input().strip().lower())
                    if resp in ("y", "yes"):
                        return True, ""
                    elif resp in ("a", "always"):
                        _always_allowed.add(resolved)
                        return True, ""
                    else:
                        return False, "User declined file write."

                agent.tools.set_approval_callback(_approval_callback)

        # ── Set up screen with scroll region ──
        _setup_screen()

        # ── Print header in scroll region ──
        if remote_mode:
            meta = node_client.server_meta
            agent_name = meta.get("agent_name", "Syne")
            motto = meta.get("motto", "")
            model_short = _short_model_name(meta.get("model", "unknown"))
            tool_count = meta.get("tool_count", 0)
            _write(f"\n  {_BOLD}{agent_name}{_RESET}\n")
            if motto:
                _write(f"  {_DIM_ITALIC}{motto}{_RESET}\n")
            _write(f"  {_DIM}Remote | Model: {model_short} | Tools: {tool_count} | /help{_RESET}\n")
        else:
            identity = await get_identity()
            agent_name = identity.get("name", "Syne")
            motto = identity.get("motto", "")
            model_short = _short_model_name(getattr(agent.provider, 'chat_model', 'unknown'))
            tool_count = len(agent.tools.list_tools('owner'))
            _write(f"\n  {_BOLD}{agent_name}{_RESET}\n")
            if motto:
                _write(f"  {_DIM_ITALIC}{motto}{_RESET}\n")
            _write(f"  {_DIM}Model: {model_short} | Tools: {tool_count} | /help{_RESET}\n")
            try:
                from ..update_checker import get_pending_update_notice
                update_notice = await get_pending_update_notice()
                if update_notice:
                    _write(f"  {_BOLD}{_DIM_YELLOW}{update_notice}{_RESET}\n")
            except Exception:
                pass

        # ── REPL setup ──
        cwd = os.getcwd()
        chat_id = ""

        if remote_mode:
            chat_id = f"node:{node_client.node_id}:{cwd}"
            from ..node.executor import execute_tool
            _r_streamed_text = False
            _r_streamed_thinking = False
            _r_in_thinking = False
            _r_thinking_done = False
            _r_reasoning_visible = node_client.server_meta.get("reasoning_visible", False)
            _r_md = _MarkdownStream()

            def _remote_on_response(text: str, done: bool):
                nonlocal _r_streamed_text, _r_in_thinking, _r_thinking_done
                _stop_spinner()
                if not text and done:
                    return
                if not _r_streamed_text:
                    _r_streamed_text = True
                    if _r_streamed_thinking and not _r_thinking_done:
                        _r_thinking_done = True
                        _write(f"{_RESET}\n\n")
                    _r_in_thinking = False
                _r_md.feed(text)

            def _remote_on_thinking(text: str):
                nonlocal _r_streamed_thinking, _r_in_thinking
                _stop_spinner()
                if not _r_reasoning_visible:
                    if not _r_streamed_thinking:
                        _r_streamed_thinking = True
                        _write(f"  {_DIM_ITALIC}Thinking...{_RESET}")
                    return
                if not _r_streamed_thinking:
                    _r_streamed_thinking = True
                    _r_in_thinking = True
                    _write(f"  {_DIM_ITALIC}")
                if not _r_in_thinking:
                    _r_in_thinking = True
                    _write(f"  {_DIM_ITALIC}")
                _write(text)

            def _remote_on_tool_activity(name: str, args: dict, result_preview: str):
                nonlocal _r_streamed_text
                if _r_streamed_text:
                    _write("\n")
                    _r_streamed_text = False
                _format_tool_activity(name, args, result_preview)

            def _remote_on_status(message: str):
                nonlocal _r_streamed_text, _r_in_thinking, _r_thinking_done
                _stop_spinner()
                if _r_streamed_text:
                    _write("\n")
                    _r_streamed_text = False
                if _r_in_thinking and not _r_thinking_done:
                    _r_thinking_done = True
                    _write(f"{_RESET}\n")
                    _r_in_thinking = False

            node_client._on_response = _remote_on_response
            node_client._on_thinking = _remote_on_thinking
            node_client._on_tool_activity = _remote_on_tool_activity
            node_client._on_status = _remote_on_status
            node_client._on_tool_request = execute_tool
        else:
            cwd = agent._cli_cwd or os.getcwd()
            username = user.get("name", getpass.getuser())
            import hashlib
            cwd_hash = hashlib.sha256(cwd.encode()).hexdigest()[:8]
            chat_id = f"{username}:{cwd_hash}"

        # Handle fresh start
        if fresh and remote_mode:
            # Remote mode: send /new to gateway to clear session in DB
            _write(f"  {_DIM}Starting fresh conversation...{_RESET}\n")
            await node_client.send_message("/new", cwd=os.getcwd())
        elif fresh:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                result = await conn.fetch("""
                    UPDATE sessions SET status = 'closed'
                    WHERE platform = 'cli' AND platform_chat_id = $1 AND status = 'active'
                    RETURNING id
                """, chat_id)
                if result:
                    session_ids = [r["id"] for r in result]
                    await conn.execute("DELETE FROM messages WHERE session_id = ANY($1::int[])", session_ids)
            key = f"cli:{chat_id}"
            if key in agent.conversations._active:
                del agent.conversations._active[key]
            _write(f"  {_DIM}Starting fresh conversation...{_RESET}\n")

        # Load conversation history
        _history: list[str] = []
        try:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT m.role, m.content FROM messages m
                    JOIN sessions s ON m.session_id = s.id
                    WHERE s.platform = 'cli' AND s.platform_chat_id = $1 AND s.status = 'active'
                      AND m.role IN ('user', 'assistant')
                    ORDER BY m.created_at ASC
                """, chat_id)
            if rows:
                _write(f"\n  {_DIM}── conversation resumed ──{_RESET}\n")
                display_rows = rows[-20:]
                if len(rows) > 20:
                    _write(f"  {_DIM}... {len(rows) - 20} earlier messages ...{_RESET}\n")
                for row in display_rows:
                    content = row["content"].strip()
                    if not content:
                        continue
                    if row["role"] == "user":
                        _history.append(content)
                        _write(f"\n  \033[48;2;52;53;65m {content} \033[49m\n")
                    else:
                        _write(f"  {content}\n")
                _write(f"\n  {_DIM}── end of history ──{_RESET}\n")
            else:
                async with get_connection() as conn:
                    user_rows = await conn.fetch("""
                        SELECT m.content FROM messages m
                        JOIN sessions s ON m.session_id = s.id
                        WHERE s.platform = 'cli' AND s.platform_chat_id = $1 AND m.role = 'user'
                        ORDER BY m.created_at ASC
                    """, chat_id)
                for row in user_rows:
                    content = row["content"].strip()
                    if content and not content.startswith("/"):
                        _history.append(content)
        except Exception:
            pass

        # ── Draw initial prompt area ──
        _draw_prompt()

        # ── REPL loop ──
        global _last_tool_key
        _last_ctrl_c = 0.0
        loop = asyncio.get_running_loop()

        while True:
            try:
                # Move cursor to input area and read
                _move_to_input()
                user_input = await loop.run_in_executor(None, _blocking_read_line, _history)
            except KeyboardInterrupt:
                now = time.time()
                if now - _last_ctrl_c < 0.5:
                    break
                _last_ctrl_c = now
                # Show warning in scroll region
                _write(f"\033[{_sr_end};1H\033[2K")
                _write(f"  {_DIM}Press Ctrl+C again to exit.{_RESET}")
                _draw_prompt()
                continue

            if user_input is None:  # EOF
                break

            user_input = user_input.strip()
            if not user_input:
                _draw_prompt()
                continue

            _last_ctrl_c = 0.0
            _last_tool_key = ""
            _history.append(user_input)

            # Echo user message in scroll region (Pi style: subtle bg #343541)
            _write(f"\033[{_sr_end};1H\033[2K")
            _write(f"\n  \033[48;2;52;53;65m {user_input} \033[49m\n")

            # Clear input line
            _write(f"\033[{_input_row};1H\033[2K> ")

            # ── Slash commands ──
            if user_input.startswith("/"):
                cmd_word = user_input.split()[0].lower()
                if cmd_word in ("/exit", "/quit", "/q"):
                    break

                if remote_mode:
                    if cmd_word == "/help":
                        _write(f"\n  {_BOLD}CLI Commands{_RESET}\n\n"
                               f"  /help   — Show this help\n"
                               f"  /cost   — Show session token usage\n"
                               f"  /exit   — Exit CLI\n\n"
                               f"  {_DIM}All other /commands and messages are sent to the server.{_RESET}\n"
                               f"  {_DIM}Shift+Enter or \\+Enter for new line.{_RESET}\n\n")
                        _draw_prompt()
                        continue
                    elif cmd_word == "/cost":
                        _write(f"\n  {_BOLD}Session token usage{_RESET}\n"
                               f"  Input:  {_format_tokens(_session_usage.total_in)}\n"
                               f"  Output: {_format_tokens(_session_usage.total_out)}\n"
                               f"  Total:  {_format_tokens(_session_usage.total_in + _session_usage.total_out)}\n\n")
                        _draw_prompt()
                        continue
                else:
                    handled = await _handle_cli_command(user_input, agent, user, chat_id)
                    if handled == "exit":
                        break
                    if handled:
                        _draw_prompt()
                        continue

            # ── Process message ──
            await _start_spinner()

            if remote_mode:
                _r_streamed_text = False
                _r_streamed_thinking = False
                _r_in_thinking = False
                _r_thinking_done = False

                try:
                    if not node_client._connected.is_set():
                        _stop_spinner()
                        _write(f"  {_DIM_YELLOW}Reconnecting...{_RESET}\n")
                        await node_client.connect()
                        listen_task = asyncio.create_task(node_client.listen())
                        _write(f"  {_DIM_GREEN}Reconnected.{_RESET}\n")
                        await _start_spinner()

                    await node_client.send_message(user_input, cwd=cwd)
                    _stop_spinner()
                    _r_md.flush()
                    _r_md.reset()
                    _write(_RESET)
                    if _r_streamed_text:
                        _write("\n")
                except (KeyboardInterrupt, asyncio.CancelledError):
                    _stop_spinner()
                    _write(f"{_RESET}\n  {_DIM_YELLOW}Cancelled{_RESET}\n")

            else:
                # ── Local mode message handling ──
                _streamed_any_text = False
                _streamed_any_thinking = False
                _in_thinking = False
                _thinking_done = False
                _md = _MarkdownStream()

                _reasoning_visible = False
                _active_conv = agent.conversations._active.get(f"cli:{chat_id}")
                if _active_conv:
                    _reasoning_visible = _active_conv.reasoning_visible

                def _on_text(chunk: str):
                    nonlocal _streamed_any_text, _in_thinking, _thinking_done
                    _stop_spinner()
                    if not _streamed_any_text:
                        _streamed_any_text = True
                        if _streamed_any_thinking and not _thinking_done:
                            _thinking_done = True
                            _write(f"{_RESET}\n\n")
                        _in_thinking = False
                    _md.feed(chunk)

                def _on_thinking(chunk: str):
                    nonlocal _streamed_any_thinking, _in_thinking
                    _stop_spinner()
                    if not _reasoning_visible:
                        if not _streamed_any_thinking:
                            _streamed_any_thinking = True
                            _write(f"  {_DIM_ITALIC}Thinking...{_RESET}")
                        return
                    if not _streamed_any_thinking:
                        _streamed_any_thinking = True
                        _in_thinking = True
                        _write(f"  {_DIM_ITALIC}")
                    if not _in_thinking:
                        _in_thinking = True
                        _write(f"  {_DIM_ITALIC}")
                    _write(chunk)

                try:
                    stream_cbs = StreamCallbacks(on_text=_on_text, on_thinking=_on_thinking)
                    agent.conversations.set_stream_callbacks(stream_cbs)

                    async def _on_tool_detail(name: str, args: dict, result_preview: str):
                        nonlocal _streamed_any_text
                        if _streamed_any_text:
                            _write("\n")
                            _streamed_any_text = False
                        _format_tool_activity(name, args, result_preview)

                    agent.conversations.set_tool_detail_callback(_on_tool_detail)

                    async def _on_tool(name: str):
                        pass

                    agent.conversations.set_tool_callback(_on_tool)

                    from ..communication.inbound import InboundContext
                    cli_inbound = InboundContext(
                        channel="cli", platform="cli", chat_type="direct",
                        conversation_label=user.get("display_name") or user.get("name", "user"),
                        chat_id=chat_id,
                    )
                    response = await agent.conversations.handle_message(
                        platform="cli", chat_id=chat_id, user=user,
                        message=user_input,
                        message_metadata={"cwd": agent._cli_cwd, "inbound": cli_inbound},
                    )

                    _stop_spinner()
                    _md.flush()
                    _md.reset()
                    _write(_RESET)

                    if _streamed_any_text:
                        _write("\n")
                    elif response:
                        _write(f"  {response}\n")

                    _active_conv_final = agent.conversations._active.get(f"cli:{chat_id}")
                    in_tok = 0
                    out_tok = 0
                    if _active_conv_final:
                        last_resp = getattr(_active_conv_final, '_last_chat_response', None)
                        if last_resp:
                            in_tok = last_resp.input_tokens
                            out_tok = last_resp.output_tokens

                    if in_tok or out_tok:
                        _session_usage.total_in += in_tok
                        _session_usage.total_out += out_tok

                except (KeyboardInterrupt, asyncio.CancelledError):
                    _stop_spinner()
                    _write(f"{_RESET}\n  {_DIM_YELLOW}Cancelled{_RESET}\n")

            # Redraw prompt with Pi-style footer stats
            stat_left = f"↑{_format_tokens(_session_usage.total_in)} ↓{_format_tokens(_session_usage.total_out)}"
            stat_right = ""
            if not remote_mode:
                stat_right = _short_model_name(getattr(agent.provider, 'chat_model', ''))
            _draw_prompt(status_left=stat_left, status_right=stat_right)

    except Exception as e:
        _write(f"\n  {_DIM_RED}Error: {e}{_RESET}\n")
        if debug:
            import traceback
            _write(traceback.format_exc() + "\n")

    finally:
        _stop_spinner()
        # Reset scroll region and clean up
        _write("\033[r")  # reset scroll region to full screen
        _write(f"\033[{_term_rows};1H\n")  # move to bottom
        _write(f"  {_DIM}Goodbye!{_RESET}\n")
        if remote_mode:
            if listen_task:
                listen_task.cancel()
            await node_client.disconnect()
        elif agent:
            await agent.stop()


async def _cli_status_callback(session_id: int, message: str):
    _write(f"\n  {_DIM_ITALIC}{message}{_RESET}\n")


async def _handle_cli_command(command: str, agent: SyneAgent, user: dict, chat_id: str) -> str | bool:
    cmd = command.split()[0].lower()

    if cmd in ("/exit", "/quit", "/q"):
        return "exit"

    elif cmd == "/help":
        _write(f"\n  {_BOLD}CLI Commands{_RESET}\n\n"
               f"  /help     — Show this help\n"
               f"  /status   — Show agent status\n"
               f"  /models   — Switch model\n"
               f"  /memory   — Search memories\n"
               f"  /compact  — Compact conversation now\n"
               f"  /clear    — Clear conversation history\n"
               f"  /cost     — Show session token usage\n"
               f"  /context  — Show context usage\n"
               f"  /exit     — Exit CLI\n\n"
               f"  {_DIM}Shift+Enter or \\+Enter for new line.{_RESET}\n\n")
        return True

    elif cmd == "/cost":
        _write(f"\n  {_BOLD}Session token usage{_RESET}\n"
               f"  Input:  {_format_tokens(_session_usage.total_in)}\n"
               f"  Output: {_format_tokens(_session_usage.total_out)}\n"
               f"  Total:  {_format_tokens(_session_usage.total_in + _session_usage.total_out)}\n\n")
        return True

    elif cmd == "/context":
        conv = agent.conversations._active.get(f"cli:{chat_id}")
        if conv:
            msg_count = len(conv._message_cache)
            from ..context import estimate_messages_tokens
            est_tokens = estimate_messages_tokens(conv._message_cache)
            ctx_window = conv.provider.context_window
            pct = min(100, int(est_tokens / ctx_window * 100)) if ctx_window else 0
            _write(f"\n  {_BOLD}Context usage{_RESET}\n"
                   f"  Messages: {msg_count}\n"
                   f"  Estimated tokens: ~{_format_tokens(est_tokens)}\n"
                   f"  Context window: {_format_tokens(ctx_window)}\n"
                   f"  Usage: ~{pct}%\n\n")
        else:
            _write(f"  {_DIM}No active conversation.{_RESET}\n\n")
        return True

    elif cmd == "/status":
        from ..db.models import get_config
        identity = await get_identity()
        model_name = getattr(agent.provider, 'chat_model', 'unknown')
        provider_name = agent.provider.name
        from ..db.connection import get_connection
        async with get_connection() as conn:
            mem_count = await conn.fetchval("SELECT COUNT(*) FROM memory")
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE active = true")
            group_count = await conn.fetchval("SELECT COUNT(*) FROM groups WHERE enabled = true")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
            total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        tool_count = len(agent.tools.list_tools("owner"))
        ability_count = len(agent.abilities.list_all()) if agent.abilities else 0
        auto_capture = await get_config("memory.auto_capture", False)
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        _active_entry = next((m for m in models if m.get("key") == active_key), {})
        _model_params = _active_entry.get("params") or {}
        thinking = _model_params.get("thinking_budget", "default")
        from .. import __version__ as syne_version
        _write(f"\n  {_BOLD}{identity.get('name', 'Syne')}{_RESET} · Syne v{syne_version}\n"
               f"  Model: {model_name} ({provider_name})\n"
               f"  Memories: {mem_count}\n"
               f"  Users: {user_count} | Groups: {group_count}\n"
               f"  Sessions: {session_count} active · {total_messages} messages\n"
               f"  Tools: {tool_count} | Abilities: {ability_count}\n"
               f"  Thinking: {thinking}\n"
               f"  Auto-capture: {'ON' if auto_capture else 'OFF'}\n\n")
        return True

    elif cmd == "/clear":
        key = f"cli:{chat_id}"
        if key in agent.conversations._active:
            conv = agent.conversations._active[key]
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute("DELETE FROM messages WHERE session_id = $1", conv.session_id)
            conv._message_cache.clear()
            _write(f"  {_DIM_GREEN}Conversation cleared.{_RESET}\n\n")
        else:
            _write(f"  {_DIM}No active conversation.{_RESET}\n\n")
        return True

    elif cmd == "/memory":
        parts = command.split(maxsplit=1)
        query = parts[1].strip() if len(parts) > 1 else ""
        if not query:
            _write(f"  {_DIM}Usage: /memory <search query>{_RESET}\n\n")
            return True
        from ..db.connection import get_connection
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT content, metadata, created_at FROM memory
                ORDER BY embedding <=> (
                    SELECT embedding FROM memory
                    WHERE content ILIKE '%' || $1 || '%'
                    LIMIT 1
                )
                LIMIT 5
            """, query)
        if not rows:
            async with get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT content, metadata, created_at FROM memory
                    WHERE content ILIKE '%' || $1 || '%'
                    ORDER BY created_at DESC LIMIT 5
                """, query)
        if rows:
            _write(f"\n  {_BOLD}Memory search: {query}{_RESET}\n\n")
            for row in rows:
                content = row["content"]
                date = row["created_at"].strftime("%Y-%m-%d") if row["created_at"] else ""
                _write(f"  {_DIM}{date}{_RESET} {content}\n")
            _write("\n")
        else:
            _write(f"  {_DIM}No memories found for '{query}'.{_RESET}\n\n")
        return True

    elif cmd == "/compact":
        conv = agent.conversations._active.get(f"cli:{chat_id}")
        if not conv:
            _write(f"  {_DIM}No active conversation.{_RESET}\n\n")
            return True
        msg_count = len(conv._message_cache)
        if msg_count < 4:
            _write(f"  {_DIM}Conversation too short to compact.{_RESET}\n\n")
            return True
        _write(f"  {_DIM}Compacting conversation...{_RESET}\n")
        from ..compaction import compact_session
        result = await compact_session(session_id=conv.session_id, provider=conv.provider)
        if result:
            await conv.load_history()
            _write(f"  {_DIM_GREEN}Compacted: {result['messages_before']} -> {result['messages_after']} messages{_RESET}\n\n")
        else:
            _write(f"  {_DIM}Nothing to compact.{_RESET}\n\n")
        return True

    elif cmd == "/models":
        from ..db.models import get_config, set_config
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "")
        if not models:
            _write(f"  {_DIM}No models registered.{_RESET}\n\n")
            return True
        _write(f"\n  {_BOLD}Select model:{_RESET}\n")
        for i, m in enumerate(models):
            key = m.get("key", "")
            label = m.get("label", key)
            model_id = m.get("model_id", "")
            active = " (active)" if key == active_key else ""
            _write(f"  {i+1}. {label} [{model_id}]{active}\n")
        _write(f"\n  {_DIM}Type number to select, or press Enter to cancel:{_RESET} ")
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, lambda: input().strip())
        if resp and resp.isdigit():
            idx = int(resp) - 1
            if 0 <= idx < len(models):
                new_key = models[idx].get("key", "")
                if new_key != active_key:
                    await set_config("provider.active_model", new_key)
                    new_provider = await agent.create_provider_for_model(new_key)
                    if new_provider:
                        agent.provider = new_provider
                        agent.conversations.provider = new_provider
                        conv_key = f"cli:{chat_id}"
                        if conv_key in agent.conversations._active:
                            del agent.conversations._active[conv_key]
                        _write(f"  {_DIM_GREEN}Switched to {models[idx].get('label', new_key)}{_RESET}\n\n")
                    else:
                        _write(f"  {_DIM_RED}Failed to create provider for {new_key}{_RESET}\n\n")
                else:
                    _write(f"  {_DIM}Already using {models[idx].get('label', new_key)}.{_RESET}\n\n")
        return True

    return False


# ── Arrow-key selector ──
async def cli_select(options: list[str], default: int = 0) -> int:
    selected = default
    total = len(options)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _draw(first: bool = False):
        if not first:
            sys.stdout.write(f"\033[{total}A")
        for i, label in enumerate(options):
            sys.stdout.write(f"\033[2K\r")
            if i == selected:
                sys.stdout.write(f"  \033[1;36m> {label}\033[0m")
            else:
                sys.stdout.write(f"    {label}")
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _read_selection() -> int:
        nonlocal selected
        _draw(first=True)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\r" or ch == "\n":
                    return selected
                elif ch == "\x03":
                    return default
                elif ch == "\x1b":
                    seq = sys.stdin.read(2)
                    if seq == "[A":
                        selected = (selected - 1) % total
                    elif seq == "[B":
                        selected = (selected + 1) % total
                    else:
                        continue
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _draw(first=False)
                    tty.setraw(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _read_selection)
    return result
