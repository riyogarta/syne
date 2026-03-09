"""Syne CLI — Interactive terminal chat with full tool access.

Inspired by Pi's TUI approach: output goes directly to stdout (terminal handles
scrollback), prompt rendered with ─ borders at bottom, no full-screen mode.
"""

import asyncio
import getpass
import json
import logging
import os
import re
import shutil
import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

# Register Shift+Enter as a distinct key (prompt_toolkit maps it to Enter by default)
ANSI_SEQUENCES["\x1b[27;2;13~"] = "<shift-enter>"  # xterm modifyOtherKeys
ANSI_SEQUENCES["\x1b[13;2u"] = "<shift-enter>"      # kitty keyboard protocol

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
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_BOLD_CYAN = "\033[1;36m"
_DIM_GREEN = "\033[2;32m"
_DIM_RED = "\033[2;31m"
_DIM_YELLOW = "\033[2;33m"
_DIM_CYAN = "\033[2;36m"
_BG_DIM = "\033[48;5;236m"  # subtle dark background for user messages

# ── Spinner ──
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_task: asyncio.Task | None = None


def _write(text: str):
    """Write directly to stdout."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _separator():
    """Print a full-width ─ separator line."""
    w = _term_width()
    _write(f"{_DIM}{'─' * w}{_RESET}\n")


async def _start_spinner(message: str = "Working..."):
    """Start an animated spinner on the current line."""
    global _spinner_task
    _stop_spinner()  # cancel any existing

    async def _spin():
        i = 0
        try:
            while True:
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                _write(f"\r  {_DIM_CYAN}{frame}{_RESET} {_DIM}{message}{_RESET}\033[K")
                i += 1
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            _write(f"\r\033[2K")  # clear spinner line

    _spinner_task = asyncio.create_task(_spin())


def _stop_spinner():
    """Stop the spinner if running."""
    global _spinner_task
    if _spinner_task and not _spinner_task.done():
        _spinner_task.cancel()
    _spinner_task = None


# ── Markdown stream ──
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
_MD_CODE = re.compile(r'`([^`]+)`')
_MD_HEADING = re.compile(r'^(#{1,3})\s+(.+)$')


class _MarkdownStream:
    """Line-buffered markdown-to-ANSI converter for streaming output."""

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


# ── Slash command completer ──
_SLASH_COMMANDS = [
    ("/help", "Show available commands"),
    ("/models", "Switch model"),
    ("/status", "Show agent status"),
    ("/memory", "Search memories"),
    ("/compact", "Compact conversation now"),
    ("/clear", "Clear conversation history"),
    ("/cost", "Show session token usage"),
    ("/context", "Show context usage"),
    ("/exit", "Exit CLI"),
]

from prompt_toolkit.completion import Completer, Completion


class _SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if "\n" in text:
            return
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        if " " in stripped:
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(stripped):
                yield Completion(cmd, start_position=-len(stripped), display_meta=desc)


# ── Session token usage tracker ──
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
    "read_source": "Read",
    "file_read": "Read",
    "file_write": "Write",
    "exec": "Run",
    "db_query": "Query",
    "web_fetch": "Fetch",
    "web_search": "Search",
    "send_message": "Send",
    "send_file": "SendFile",
    "send_voice": "Voice",
    "send_reaction": "React",
    "manage_schedule": "Schedule",
    "update_config": "Config",
    "update_ability": "Ability",
    "memory_search": "MemSearch",
    "memory_add": "MemAdd",
}

_last_tool_key = ""


def _format_tool_activity(name: str, args: dict, result_preview: str) -> None:
    """Display a tool call as a bullet point line."""
    global _last_tool_key
    _stop_spinner()

    dedup_key = f"{name}:{list(args.values())[:1]}"
    if dedup_key == _last_tool_key:
        return
    _last_tool_key = dedup_key

    label = _TOOL_LABELS.get(name, name)

    arg_hint = ""
    if args:
        for key in ("path", "file_path", "file", "command", "query", "action", "url", "name", "key"):
            if key in args:
                v = str(args[key])
                if len(v) > 60:
                    v = v[:57] + "..."
                arg_hint = v
                break
        if not arg_hint:
            v = str(list(args.values())[0])
            if len(v) > 60:
                v = v[:57] + "..."
            arg_hint = v

    if arg_hint:
        _write(f"\n  {_DIM_CYAN}●{_RESET} {label}({_DIM_YELLOW}{arg_hint}{_RESET})\n")
    else:
        _write(f"\n  {_DIM_CYAN}●{_RESET} {label}\n")

    preview = result_preview.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    if preview:
        is_error = preview.lower().startswith("error")
        color = _DIM_RED if is_error else _DIM
        _write(f"    {_DIM}⎿{_RESET}  {color}{preview}{_RESET}\n")


# ── Build prompt session with keybindings ──

def _build_prompt_session(history: InMemoryHistory | None = None) -> PromptSession:
    """Create a PromptSession with Pi-style keybindings."""
    kb = KeyBindings()

    @kb.add(Keys.Enter, eager=True)
    def _submit(event):
        buf = event.current_buffer
        # Backslash at end = line continuation
        if buf.text.endswith("\\"):
            buf.text = buf.text[:-1]
            buf.insert_text("\n")
            return
        buf.validate_and_handle()

    @kb.add("<shift-enter>")
    def _newline_shift(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        message="> ",
        multiline=True,
        key_bindings=kb,
        completer=_SlashCompleter(),
        history=history,
        bottom_toolbar=lambda: HTML(f'<style fg="ansibrightblack">{"─" * _term_width()}</style>'),
        enable_open_in_editor=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

async def run_cli(debug: bool = False, yolo: bool = False, fresh: bool = False, node_client=None):
    """Run Syne in interactive CLI mode."""
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
                    # Simple blocking input for approval during tool execution
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

        # ── Display header ──
        _write("\n")
        if remote_mode:
            meta = node_client.server_meta
            agent_name = meta.get("agent_name", "Syne")
            motto = meta.get("motto", "")
            model_short = _short_model_name(meta.get("model", "unknown"))
            tool_count = meta.get("tool_count", 0)
            _write(f"  {_BOLD}{agent_name}{_RESET}\n")
            if motto:
                _write(f"  {_DIM_ITALIC}{motto}{_RESET}\n")
            _write(f"  {_DIM}Remote | Model: {model_short} | Tools: {tool_count} | /help{_RESET}\n\n")
        else:
            identity = await get_identity()
            agent_name = identity.get("name", "Syne")
            motto = identity.get("motto", "")
            model_short = _short_model_name(getattr(agent.provider, 'chat_model', 'unknown'))
            tool_count = len(agent.tools.list_tools('owner'))
            _write(f"  {_BOLD}{agent_name}{_RESET}\n")
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
            _write("\n")

        # ── REPL setup ──
        cwd = os.getcwd()
        chat_id = ""

        if remote_mode:
            chat_id = f"node:{node_client.node_id}:{cwd}"

            # Wire remote mode callbacks
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
        if fresh and not remote_mode:
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

        # Load conversation history from DB
        _history = InMemoryHistory()
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
                _write(f"  {_DIM}── conversation resumed ──{_RESET}\n")
                display_rows = rows[-20:]
                if len(rows) > 20:
                    _write(f"  {_DIM}... {len(rows) - 20} earlier messages ...{_RESET}\n")
                for row in display_rows:
                    content = row["content"].strip()
                    if not content:
                        continue
                    if row["role"] == "user":
                        _history.append_string(content)
                        _write(f"\n  {_DIM}> {content}{_RESET}\n")
                    else:
                        _write(f"  {content}\n")
                _write(f"\n  {_DIM}── end of history ──{_RESET}\n\n")
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
                        _history.append_string(content)
        except Exception:
            pass

        # Build prompt session
        session = _build_prompt_session(_history)

        # ── REPL loop ──
        _last_ctrl_c = 0.0

        while True:
            try:
                _separator()
                user_input = await session.prompt_async()
            except KeyboardInterrupt:
                now = time.time()
                if now - _last_ctrl_c < 0.5:
                    _write(f"\n  {_DIM}Goodbye!{_RESET}\n")
                    break
                _last_ctrl_c = now
                _write(f"\n  {_DIM}Press Ctrl+C again to exit.{_RESET}\n")
                continue
            except EOFError:
                _write(f"\n  {_DIM}Goodbye!{_RESET}\n")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            _last_ctrl_c = 0.0
            global _last_tool_key
            _last_tool_key = ""

            # ── Slash commands ──
            if user_input.startswith("/"):
                cmd_word = user_input.split()[0].lower()
                if cmd_word in ("/exit", "/quit", "/q"):
                    _write(f"  {_DIM}Goodbye!{_RESET}\n")
                    break

                if remote_mode:
                    if cmd_word == "/help":
                        _write(f"\n  {_BOLD}CLI Commands{_RESET}\n\n"
                               f"  /help   — Show this help\n"
                               f"  /cost   — Show session token usage\n"
                               f"  /exit   — Exit CLI\n\n"
                               f"  {_DIM}All other /commands and messages are sent to the server.{_RESET}\n"
                               f"  {_DIM}Shift+Enter or \\+Enter for new line.{_RESET}\n\n")
                        continue
                    elif cmd_word == "/cost":
                        _write(f"\n  {_BOLD}Session token usage{_RESET}\n"
                               f"  Input:  {_format_tokens(_session_usage.total_in)}\n"
                               f"  Output: {_format_tokens(_session_usage.total_out)}\n"
                               f"  Total:  {_format_tokens(_session_usage.total_in + _session_usage.total_out)}\n\n")
                        continue
                else:
                    handled = await _handle_cli_command(user_input, agent, user, chat_id)
                    if handled == "exit":
                        break
                    if handled:
                        continue

            # ── Process message ──
            _write("\n")
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
                    continue

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
                        _write(f"\n  {_DIM}↑{_format_tokens(in_tok)} ↓{_format_tokens(out_tok)}{_RESET}\n")

                except (KeyboardInterrupt, asyncio.CancelledError):
                    _stop_spinner()
                    _write(f"{_RESET}\n  {_DIM_YELLOW}Cancelled{_RESET}\n")
                    continue

            _write("\n")

    except Exception as e:
        _write(f"\n  {_DIM_RED}Error: {e}{_RESET}\n")
        if debug:
            import traceback
            _write(traceback.format_exc() + "\n")

    finally:
        if remote_mode:
            listen_task.cancel()
            await node_client.disconnect()
        elif agent:
            await agent.stop()


async def _cli_status_callback(session_id: int, message: str):
    """Display status messages (compaction, etc.)."""
    _write(f"\n  {_DIM_ITALIC}{message}{_RESET}\n")


async def _handle_cli_command(command: str, agent: SyneAgent, user: dict, chat_id: str) -> str | bool:
    """Handle CLI-specific slash commands. Returns 'exit', True (handled), or False."""
    cmd = command.split()[0].lower()

    if cmd in ("/exit", "/quit", "/q"):
        _write(f"  {_DIM}Goodbye!{_RESET}\n")
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
        from ..db.models import get_config, list_users
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
        _write("\n")
        return True

    return False


# ── Arrow-key selector (kept for compatibility) ──
async def cli_select(options: list[str], default: int = 0) -> int:
    """Arrow-key selector for CLI choices."""
    import termios
    import tty

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
