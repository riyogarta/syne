"""Syne CLI — Interactive terminal chat with full tool access."""

import asyncio
import getpass
import json
import logging
import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..agent import SyneAgent
from ..config import load_settings
from ..db.models import get_or_create_user, get_identity
from ..llm.provider import StreamCallbacks

logger = logging.getLogger("syne.cli_channel")
console = Console()

# ── ANSI helpers for raw streaming output ──
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


import re

# Regex patterns for inline markdown → ANSI conversion
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)')
_MD_CODE = re.compile(r'`([^`]+)`')
_MD_HEADING = re.compile(r'^(#{1,3})\s+(.+)$')


class _MarkdownStream:
    """Line-buffered markdown-to-ANSI converter for streaming output.

    Accumulates text chunks, converts complete lines from markdown to ANSI,
    and writes to stdout. Handles code blocks and inline formatting.
    """

    def __init__(self):
        self._buf = ""
        self._in_code_block = False

    def feed(self, chunk: str) -> None:
        """Feed a text chunk — process and output complete lines."""
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line)
            sys.stdout.write("\n")
        # Partial line — hold in buffer
        sys.stdout.flush()

    def flush(self) -> None:
        """Flush remaining buffer (partial last line)."""
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""
        sys.stdout.flush()

    def reset(self) -> None:
        """Reset state for next message."""
        self._buf = ""
        self._in_code_block = False

    def _emit_line(self, line: str) -> None:
        """Convert a single line and write to stdout."""
        # Code block toggle
        if line.strip().startswith("```"):
            self._in_code_block = not self._in_code_block
            if self._in_code_block:
                # Opening — show language hint dimly if present
                lang = line.strip()[3:].strip()
                if lang:
                    sys.stdout.write(f"{_DIM}  [{lang}]{_RESET}")
            else:
                # Closing — just skip
                pass
            return

        # Inside code block — dim, no markdown processing
        if self._in_code_block:
            sys.stdout.write(f"{_DIM_CYAN}{line}{_RESET}")
            return

        # Heading
        m = _MD_HEADING.match(line)
        if m:
            sys.stdout.write(f"{_BOLD}{m.group(2)}{_RESET}")
            return

        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            return

        # Inline: bold, italic, code
        line = _MD_BOLD.sub(f'{_BOLD}\\1{_RESET}', line)
        line = _MD_CODE.sub(f'{_DIM_CYAN}\\1{_RESET}', line)
        line = _MD_ITALIC.sub(f'{_ITALIC}\\1{_RESET}', line)

        sys.stdout.write(line)


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


class _SlashCompleter(Completer):
    """Autocomplete slash commands when input starts with /."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # Only complete if the entire input so far is a slash command (no prior text)
        if "\n" in text:
            return
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return
        # Only complete the first word
        if " " in stripped:
            return
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(stripped):
                yield Completion(
                    cmd,
                    start_position=-len(stripped),
                    display_meta=desc,
                )


async def cli_select(options: list[str], default: int = 0) -> int:
    """Arrow-key selector for CLI choices.

    Displays options vertically. User navigates with up/down and confirms with Enter.
    Selected option is marked with '>'.

    Args:
        options: List of option labels (e.g. ["Yes", "No", "Always allow"])
        default: Index of initially selected option (0-based)

    Returns:
        Index of chosen option
    """
    import termios
    import tty

    selected = default
    total = len(options)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _draw(first: bool = False):
        """Draw all options. If not first, move cursor up first."""
        if not first:
            # Move cursor up to start of option list and clear
            sys.stdout.write(f"\033[{total}A")
        for i, label in enumerate(options):
            # Clear entire line, carriage return, then write
            sys.stdout.write(f"\033[2K\r")
            if i == selected:
                sys.stdout.write(f"  \033[1;36m> {label}\033[0m")
            else:
                sys.stdout.write(f"    {label}")
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _read_selection() -> int:
        nonlocal selected
        # Initial render in normal mode
        _draw(first=True)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\r" or ch == "\n":
                    return selected
                elif ch == "\x03":  # Ctrl+C
                    return default
                elif ch == "\x1b":  # Escape sequence
                    seq = sys.stdin.read(2)
                    if seq == "[A":  # Up arrow
                        selected = (selected - 1) % total
                    elif seq == "[B":  # Down arrow
                        selected = (selected + 1) % total
                    else:
                        continue
                    # Re-render — temporarily restore cooked mode for cursor control
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    _draw(first=False)
                    tty.setraw(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _read_selection)
    return result
_prompt_session: PromptSession | None = None


# ── Session token usage tracker ──
class _SessionUsage:
    """Accumulate token counts across the entire CLI session."""
    __slots__ = ("total_in", "total_out")

    def __init__(self):
        self.total_in = 0
        self.total_out = 0

_session_usage = _SessionUsage()


def _short_model_name(model: str) -> str:
    """Extract a short display name from a full model ID.

    Examples:
        claude-sonnet-4-20250514 → sonnet-4
        gemini-2.5-pro-preview-05-06 → gemini-2.5-pro
        gpt-5-0806 → gpt-5
    """
    name = model
    # Strip date suffixes like -20250514, -0806, -preview-05-06
    import re
    name = re.sub(r'-preview(-\d{2}-\d{2})?$', '', name)
    name = re.sub(r'-\d{6,}$', '', name)
    name = re.sub(r'-\d{4}$', '', name)
    return name


def _format_tokens(n: int) -> str:
    """Format token count: 1234 → '1,234'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


# Friendly display names for tools
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


def _format_tool_activity(name: str, args: dict, result_preview: str) -> None:
    """Display a tool call as a bullet point line (Claude Code style).

    Examples:
        ● Read(syne/boot.py)
          ⎿  1,245 chars
        ● Run(pip install -e .)
          ⎿  Installed successfully
    """
    label = _TOOL_LABELS.get(name, name)

    # Build compact arg hint
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

    # Tool line
    if arg_hint:
        sys.stdout.write(f"\n  {_DIM_CYAN}●{_RESET} {label}({_DIM_YELLOW}{arg_hint}{_RESET})\n")
    else:
        sys.stdout.write(f"\n  {_DIM_CYAN}●{_RESET} {label}\n")

    # Result preview
    preview = result_preview.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:77] + "..."
    if preview:
        is_error = preview.lower().startswith("error")
        color = _DIM_RED if is_error else _DIM
        sys.stdout.write(f"    {_DIM}⎿{_RESET}  {color}{preview}{_RESET}\n")
    sys.stdout.flush()


async def run_cli(debug: bool = False, yolo: bool = False, fresh: bool = False, node_client=None):
    """Run Syne in interactive CLI mode (resumes previous session by default).

    Args:
        node_client: Optional NodeClient for remote mode. When provided, messages
            are sent via WebSocket to the gateway instead of local SyneAgent.
    """
    remote_mode = node_client is not None
    if debug:
        logging.getLogger("syne").setLevel(logging.DEBUG)
    else:
        # Suppress noisy loggers in CLI mode — only show errors
        logging.getLogger("syne").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.WARNING)

    agent = None
    if not remote_mode:
        settings = load_settings()
        agent = SyneAgent(settings)

    global _prompt_session

    # Build prompt_toolkit keybindings: Enter=submit, Shift+Enter / Esc+Enter=newline
    _kb = KeyBindings()

    @_kb.add(Keys.BracketedPaste)
    def _paste(event):
        """Paste inserts text as-is (newlines preserved, no submit)."""
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        event.current_buffer.insert_text(data)

    @_kb.add(Keys.Enter, eager=True)
    def _submit(event):
        """Enter submits the input."""
        event.current_buffer.validate_and_handle()

    @_kb.add(Keys.Escape, Keys.Enter)
    def _newline_esc(event):
        """Esc + Enter inserts a newline (fallback for terminals without Shift+Enter)."""
        event.current_buffer.insert_text("\n")

    # Some terminals send Shift+Enter as Escape [13;2u or just \x1b\r
    # prompt_toolkit maps this to s-enter on supported terminals
    try:
        @_kb.add("s-enter")
        def _newline_shift(event):
            """Shift+Enter inserts a newline."""
            event.current_buffer.insert_text("\n")
    except Exception:
        pass  # Terminal doesn't support — Esc+Enter still works

    # Ctrl+L to clear screen
    @_kb.add("c-l")
    def _clear_screen(event):
        """Ctrl+L clears the terminal."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        event.app.renderer.clear()

    _prompt_session = PromptSession(
        key_bindings=_kb,
        multiline=True,
        completer=_SlashCompleter(),
        complete_while_typing=True,
    )

    try:
        # ── Remote mode setup ──
        if remote_mode:
            await node_client.connect()
            listen_task = asyncio.create_task(node_client.listen())

        # ── Local mode setup ──
        else:
            await agent.start()

            # Set CLI working directory to where user launched `syne cli`
            agent._cli_cwd = os.getcwd()

            # Override workspace for file_write — CLI uses caller's directory
            from ..tools.file_ops import set_workspace
            set_workspace(os.getcwd())

            # Auto-migrate Google OAuth if needed
            from ..main import _auto_migrate_google_oauth
            await _auto_migrate_google_oauth()

            # Get or create CLI user (always owner)
            username = getpass.getuser()
            user = await get_or_create_user(
                name=username,
                platform="cli",
                platform_id=f"cli:{username}",
                display_name=username,
                is_dm=True,  # CLI is always direct interaction
            )

            # Ensure CLI user is owner
            if user.get("access_level") != "owner":
                from ..db.models import update_user
                await update_user("cli", f"cli:{username}", access_level="owner")
                user = dict(user)
                user["access_level"] = "owner"

        # ── Local mode: callbacks & approval ──
        if not remote_mode:
            if agent.conversations:
                agent.conversations.add_status_callback(_cli_status_callback)
                agent.conversations._tool_status = None

            if not yolo:
                _always_allowed: set[str] = set()

                async def _approval_callback(tool_name: str, args: dict) -> tuple[bool, str]:
                    """Ask user for approval before file writes."""
                    if tool_name != "file_write":
                        return True, ""

                    file_path = args.get("path", "?")
                    resolved = os.path.abspath(file_path)

                    if resolved in _always_allowed:
                        return True, ""

                    _status_ref = getattr(agent.conversations, '_active_status', None)
                    if _status_ref:
                        _status_ref.stop()

                    content = args.get("content", "")
                    lines = content.count("\n") + 1
                    size = len(content.encode("utf-8"))

                    console.print()
                    if os.path.isfile(resolved):
                        try:
                            import difflib
                            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                                old_lines = f.readlines()
                            new_lines = content.splitlines(keepends=True)
                            diff = list(difflib.unified_diff(
                                old_lines, new_lines,
                                fromfile=file_path, tofile=file_path,
                                lineterm=""
                            ))
                            if diff:
                                console.print(f"[bold]Edit file[/bold] {file_path}")
                                shown = 0
                                for line in diff:
                                    if line.startswith("@@"):
                                        console.print(f"[cyan]{line.rstrip()}[/cyan]")
                                    elif line.startswith("-") and not line.startswith("---"):
                                        console.print(f"[red]{line.rstrip()}[/red]")
                                    elif line.startswith("+") and not line.startswith("+++"):
                                        console.print(f"[green]{line.rstrip()}[/green]")
                                    else:
                                        console.print(f"[dim]{line.rstrip()}[/dim]")
                                    shown += 1
                                    if shown >= 30:
                                        remaining = len(diff) - shown
                                        if remaining > 0:
                                            console.print(f"[dim]  ... {remaining} more lines[/dim]")
                                        break
                            else:
                                console.print(f"[bold]Write to[/bold] {file_path} [dim](no changes)[/dim]")
                        except Exception:
                            console.print(f"[bold]Write to[/bold] {file_path} ({lines} lines, {size:,} bytes)")
                    else:
                        console.print(f"[bold]New file[/bold] {file_path} ({lines} lines, {size:,} bytes)")
                        preview = content.split("\n")[:10]
                        for pline in preview:
                            console.print(f"[green]+ {pline}[/green]")
                        if lines > 10:
                            console.print(f"[dim]  ... {lines - 10} more lines[/dim]")

                    console.print()
                    console.print(f"[bold]Do you want to make this edit to {os.path.basename(file_path)}?[/bold]")

                    choices = ["Yes", "Yes, always allow this file", "No"]
                    try:
                        chosen = await cli_select(choices, default=0)
                    except (EOFError, KeyboardInterrupt):
                        chosen = 1

                    if _status_ref:
                        _status_ref.start()

                    if chosen == 0:
                        return True, ""
                    elif chosen == 1:
                        _always_allowed.add(resolved)
                        return True, ""
                    else:
                        return False, "User declined file write."

                agent.tools.set_approval_callback(_approval_callback)

        # ── Display header ──
        if remote_mode:
            meta = node_client.server_meta
            agent_name = meta.get("agent_name", "Syne")
            motto = meta.get("motto", "")
            model_short = _short_model_name(meta.get("model", "unknown"))
            tool_count = meta.get("tool_count", 0)

            console.print()
            console.print(Panel(
                f"[bold]{agent_name}[/bold]" + (f"\n[dim italic]{motto}[/dim italic]" if motto else ""),
                style="blue",
                subtitle=f"Remote | Model: {model_short} | Tools: {tool_count} | Type /help for commands",
            ))
            console.print()
        else:
            identity = await get_identity()
            agent_name = identity.get("name", "Syne")
            motto = identity.get("motto", "")
            model_short = _short_model_name(getattr(agent.provider, 'chat_model', 'unknown'))

            console.print()
            console.print(Panel(
                f"[bold]{agent_name}[/bold]" + (f"\n[dim italic]{motto}[/dim italic]" if motto else ""),
                style="blue",
                subtitle=f"Model: {model_short} | Tools: {len(agent.tools.list_tools('owner'))} | Type /help for commands",
            ))

            try:
                from ..update_checker import get_pending_update_notice
                update_notice = await get_pending_update_notice()
                if update_notice:
                    console.print(f"[bold yellow]{update_notice}[/bold yellow]")
            except Exception:
                pass

            console.print()

        # ── REPL setup ──
        cwd = os.getcwd()
        chat_id = ""
        username = ""

        if remote_mode:
            chat_id = f"node:{node_client.node_id}:{cwd}"

            # Wire NodeClient callbacks for streaming display
            from ..node.executor import execute_tool
            _r_streamed_text = False
            _r_streamed_thinking = False
            _r_in_thinking = False
            _r_thinking_done = False
            _r_status = [None]  # mutable container for closure access
            _r_reasoning_visible = node_client.server_meta.get("reasoning_visible", False)
            _r_md = _MarkdownStream()

            def _remote_on_response(text: str, done: bool):
                nonlocal _r_streamed_text, _r_in_thinking, _r_thinking_done
                if not text and done:
                    return
                if not _r_streamed_text:
                    _r_streamed_text = True
                    if _r_status[0]:
                        _r_status[0].stop()
                        _r_status[0] = None
                    if _r_streamed_thinking and not _r_thinking_done:
                        _r_thinking_done = True
                        sys.stdout.write(f"{_RESET}\n")
                        sys.stdout.flush()
                    _r_in_thinking = False
                _r_md.feed(text)

            def _remote_on_thinking(text: str):
                nonlocal _r_streamed_thinking, _r_in_thinking
                if not _r_reasoning_visible:
                    return
                if not _r_streamed_thinking:
                    _r_streamed_thinking = True
                    if _r_status[0]:
                        _r_status[0].stop()
                        _r_status[0] = None
                    _r_in_thinking = True
                    sys.stdout.write(_DIM_ITALIC)
                if not _r_in_thinking:
                    _r_in_thinking = True
                    sys.stdout.write(_DIM_ITALIC)
                sys.stdout.write(text)
                sys.stdout.flush()

            def _remote_on_tool_activity(name: str, args: dict, result_preview: str):
                nonlocal _r_streamed_text
                if _r_streamed_text:
                    sys.stdout.write("\n")
                    _r_streamed_text = False
                if _r_status[0]:
                    _r_status[0].stop()
                    _r_status[0] = None
                _format_tool_activity(name, args, result_preview)
                _r_status[0] = console.status("[bold blue]Thinking...", spinner="dots")
                _r_status[0].start()

            def _remote_on_status(message: str):
                nonlocal _r_streamed_text, _r_in_thinking, _r_thinking_done
                if _r_streamed_text:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    _r_streamed_text = False
                if _r_in_thinking and not _r_thinking_done:
                    _r_thinking_done = True
                    sys.stdout.write(f"{_RESET}\n")
                    sys.stdout.flush()
                    _r_in_thinking = False
                # Show as spinner
                if _r_status[0]:
                    _r_status[0].stop()
                _r_status[0] = console.status(f"[bold blue]{message}", spinner="dots")
                _r_status[0].start()

            node_client._on_response = _remote_on_response
            node_client._on_thinking = _remote_on_thinking
            node_client._on_tool_activity = _remote_on_tool_activity
            node_client._on_status = _remote_on_status
            node_client._on_tool_request = execute_tool

        else:
            cwd = agent._cli_cwd or os.getcwd()
            username = user.get("name", getpass.getuser())
            chat_id = f"cli:{username}:{cwd}"

            # Fresh start: close old session and clear history
            if fresh:
                from ..db.connection import get_connection
                async with get_connection() as conn:
                    result = await conn.fetch("""
                        UPDATE sessions SET status = 'closed'
                        WHERE platform = 'cli' AND platform_chat_id = $1 AND status = 'active'
                        RETURNING id
                    """, chat_id)
                    if result:
                        session_ids = [r["id"] for r in result]
                        await conn.execute(
                            "DELETE FROM messages WHERE session_id = ANY($1::int[])",
                            session_ids,
                        )
                key = f"cli:{chat_id}"
                if key in agent.conversations._active:
                    del agent.conversations._active[key]
                console.print("[dim]Starting fresh conversation...[/dim]")

            # Load input history from DB
            try:
                from prompt_toolkit.history import InMemoryHistory
                _history = InMemoryHistory()
                from ..db.connection import get_connection
                async with get_connection() as conn:
                    rows = await conn.fetch("""
                        SELECT m.content FROM messages m
                        JOIN sessions s ON m.session_id = s.id
                        WHERE s.platform = 'cli' AND s.platform_chat_id = $1
                          AND m.role = 'user'
                        ORDER BY m.created_at ASC
                    """, chat_id)
                for row in rows:
                    content = row["content"].strip()
                    if content and not content.startswith("/"):
                        _history.append_string(content)
                _prompt_session.history = _history
            except Exception:
                pass

        # ── REPL loop ──
        import time as _time
        _last_ctrl_c = 0.0
        while True:
            try:
                user_input = await _get_input()
            except (KeyboardInterrupt, asyncio.CancelledError):
                now = _time.time()
                if now - _last_ctrl_c < 2.0:
                    console.print("\n[dim]Goodbye![/dim]")
                    break
                _last_ctrl_c = now
                console.print("\n[dim](Press Ctrl+C again to exit)[/dim]")
                continue

            if user_input is None:
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Built-in CLI commands
            if user_input.startswith("/"):
                cmd_word = user_input.split()[0].lower()
                if cmd_word in ("/exit", "/quit", "/q"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                if remote_mode:
                    # Remote mode: only /exit, /help, /cost handled locally
                    if cmd_word == "/help":
                        console.print(Panel(
                            "[bold]CLI Commands[/bold]\n\n"
                            "/help          \u2014 Show this help\n"
                            "/cost          \u2014 Show session token usage\n"
                            "/exit          \u2014 Exit CLI\n"
                            "\n[dim]All other /commands and messages are sent to the server.[/dim]\n"
                            "[dim]Shift+Enter or Esc+Enter for new line. Ctrl+L to clear screen.[/dim]",
                            style="blue",
                        ))
                        continue
                    elif cmd_word == "/cost":
                        console.print(
                            f"[bold]Session token usage[/bold]\n"
                            f"  Input:  {_format_tokens(_session_usage.total_in)}\n"
                            f"  Output: {_format_tokens(_session_usage.total_out)}\n"
                            f"  Total:  {_format_tokens(_session_usage.total_in + _session_usage.total_out)}"
                        )
                        continue
                    # All other slash commands: fall through and send to server
                else:
                    handled = await _handle_cli_command(user_input, agent, user, chat_id)
                    if handled == "exit":
                        break
                    if handled:
                        continue

            # Phase 2: Process message
            console.print()

            if remote_mode:
                # Reset streaming state for this message
                _r_streamed_text = False
                _r_streamed_thinking = False
                _r_in_thinking = False
                _r_thinking_done = False
                _r_status[0] = console.status("[bold blue]Thinking...", spinner="dots")
                _r_status[0].start()

                try:
                    # Auto-reconnect if connection lost
                    if not node_client._connected.is_set():
                        _r_status[0].stop()
                        console.print("[yellow]Connection lost. Reconnecting...[/yellow]")
                        await node_client.connect()
                        listen_task = asyncio.create_task(node_client.listen())
                        console.print("[green]Reconnected.[/green]")
                        _r_status[0] = console.status("[bold blue]Thinking...", spinner="dots")
                        _r_status[0].start()

                    await node_client.send_message(user_input, cwd=cwd)

                    # Cleanup
                    _r_md.flush()
                    _r_md.reset()
                    if _r_status[0]:
                        _r_status[0].stop()
                        _r_status[0] = None
                    sys.stdout.write(_RESET)
                    sys.stdout.flush()

                    if _r_streamed_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()

                    console.print()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    sys.stdout.write(_RESET)
                    sys.stdout.flush()
                    if _r_status[0]:
                        _r_status[0].stop()
                        _r_status[0] = None
                    console.print("\n[yellow]Cancelled[/yellow]\n")
                    continue

            else:
                # ── Local mode message handling ──
                _streamed_any_text = False
                _streamed_any_thinking = False
                _in_thinking = False
                _thinking_done = False
                status = None
                _md = _MarkdownStream()

                _reasoning_visible = False
                _active_conv = agent.conversations._active.get(f"cli:{chat_id}")
                if _active_conv:
                    _reasoning_visible = _active_conv.reasoning_visible

                def _on_text(chunk: str):
                    nonlocal _streamed_any_text, _in_thinking, _thinking_done, status
                    if not _streamed_any_text:
                        _streamed_any_text = True
                        if status:
                            status.stop()
                            status = None
                        if _streamed_any_thinking and not _thinking_done:
                            _thinking_done = True
                            sys.stdout.write(f"{_RESET}\n")
                            sys.stdout.flush()
                        _in_thinking = False
                    _md.feed(chunk)

                def _on_thinking(chunk: str):
                    nonlocal _streamed_any_thinking, _in_thinking, status
                    if not _reasoning_visible:
                        return
                    if not _streamed_any_thinking:
                        _streamed_any_thinking = True
                        if status:
                            status.stop()
                            status = None
                        _in_thinking = True
                        sys.stdout.write(_DIM_ITALIC)
                    if not _in_thinking:
                        _in_thinking = True
                        sys.stdout.write(_DIM_ITALIC)
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

                try:
                    msg = user_input
                    status = console.status("[bold blue]Thinking...", spinner="dots")
                    status.start()
                    agent.conversations._active_status = status
                    _cli_status_callback._active_status = status

                    stream_cbs = StreamCallbacks(on_text=_on_text, on_thinking=_on_thinking)
                    agent.conversations.set_stream_callbacks(stream_cbs)

                    async def _on_tool_detail(name: str, args: dict, result_preview: str):
                        nonlocal status, _streamed_any_text
                        if _streamed_any_text:
                            sys.stdout.write("\n")
                            _streamed_any_text = False
                        if status:
                            status.stop()
                            status = None
                        _format_tool_activity(name, args, result_preview)
                        status = console.status("[bold blue]Thinking...", spinner="dots")
                        status.start()
                        agent.conversations._active_status = status

                    agent.conversations.set_tool_detail_callback(_on_tool_detail)

                    async def _on_tool(name: str):
                        nonlocal status
                        if status:
                            status.update(f"[bold blue]Running {name}...")

                    agent.conversations.set_tool_callback(_on_tool)

                    from ..communication.inbound import InboundContext
                    cli_inbound = InboundContext(
                        channel="cli",
                        platform="cli",
                        chat_type="direct",
                        conversation_label=user.get("display_name") or user.get("name", "user"),
                        chat_id=chat_id,
                    )
                    response = await agent.conversations.handle_message(
                        platform="cli",
                        chat_id=chat_id,
                        user=user,
                        message=msg,
                        message_metadata={"cwd": agent._cli_cwd, "inbound": cli_inbound},
                    )

                    _md.flush()
                    _md.reset()
                    _stop_status(status, agent)
                    status = None

                    sys.stdout.write(_RESET)
                    sys.stdout.flush()

                    if _streamed_any_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    elif response:
                        _display_response(response)

                    _active_conv_final = agent.conversations._active.get(f"cli:{chat_id}")
                    if _active_conv_final and hasattr(_active_conv_final, '_last_response_usage'):
                        in_tok = _active_conv_final._last_response_usage.get("input_tokens", 0)
                        out_tok = _active_conv_final._last_response_usage.get("output_tokens", 0)
                    elif response:
                        in_tok = 0
                        out_tok = 0
                    else:
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
                        console.print(f"[dim][{_format_tokens(in_tok)} in | {_format_tokens(out_tok)} out][/dim]")

                    console.print()
                except (KeyboardInterrupt, asyncio.CancelledError):
                    sys.stdout.write(_RESET)
                    sys.stdout.flush()
                    _stop_status(status, agent)
                    status = None
                    console.print("\n[yellow]Cancelled[/yellow]\n")
                    continue

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()

    # Cleanup
    if remote_mode:
        listen_task.cancel()
        await node_client.disconnect()
    else:
        await agent.stop()
    try:
        os.system("stty sane 2>/dev/null")
    except Exception:
        pass
    os._exit(0)


def _stop_status(status, agent):
    """Safely stop spinner and clean up callbacks."""
    try:
        if status:
            status.stop()
    except Exception:
        pass
    try:
        agent.conversations._active_status = None
        _cli_status_callback._active_status = None
        agent.conversations.set_tool_callback(None)
        agent.conversations.set_stream_callbacks(None)
        agent.conversations.set_tool_detail_callback(None)
    except Exception:
        pass


def _make_separator() -> str:
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return "─" * cols


async def _get_input(model_name: str = "") -> str | None:
    """Get user input using prompt_toolkit (supports Shift+Enter for newlines)."""
    from prompt_toolkit.formatted_text import HTML

    sep = _make_separator()

    # Top border
    sys.stdout.write(f"{_DIM}{sep}{_RESET}\n")
    sys.stdout.flush()

    prompt_str = "> "
    try:
        result = await _prompt_session.prompt_async(
            prompt_str,
            multiline=True,
            prompt_continuation="  ",
            bottom_toolbar=HTML(f'<style bg="" fg="ansibrightblack">{sep}</style>'),
        )
        return result
    except EOFError:
        return None


def _display_response(response: str):
    """Display agent response with markdown rendering."""
    try:
        # Check if response contains code blocks or markdown
        if "```" in response or "**" in response or "# " in response:
            md = Markdown(response)
            console.print(md)
        else:
            console.print(response)
    except Exception:
        # Fallback to plain text
        console.print(response)


async def _cli_status_callback(session_id: int, message: str):
    """Display status messages (compaction, etc.) in terminal."""
    # Pause spinner if active so message is visible
    _status = getattr(_cli_status_callback, '_active_status', None)
    if _status:
        _status.stop()
    console.print(f"\n[dim italic]{message}[/dim italic]")
    if _status:
        _status.start()


async def _handle_cli_command(
    command: str, agent: SyneAgent, user: dict, chat_id: str
) -> str | bool:
    """Handle CLI-specific slash commands.

    Returns:
        - "exit" to quit
        - True if handled
        - False if not handled (pass to agent)
    """
    cmd = command.split()[0].lower()
    args = command[len(cmd):].strip()

    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        return "exit"

    elif cmd == "/help":
        console.print(Panel(
            "[bold]CLI Commands[/bold]\n\n"
            "/help          \u2014 Show this help\n"
            "/status        \u2014 Show agent status\n"
            "/models        \u2014 Switch model\n"
            "/memory        \u2014 Search memories\n"
            "/compact       \u2014 Compact conversation now\n"
            "/clear         \u2014 Clear conversation history\n"
            "/cost          \u2014 Show session token usage\n"
            "/context       \u2014 Show context usage\n"
            "/exit          \u2014 Exit CLI\n"
            "\n[dim]All other messages are sent to the agent.[/dim]\n"
            "[dim]Shift+Enter or Esc+Enter for new line. Ctrl+L to clear screen.[/dim]",
            style="blue",
        ))
        return True

    elif cmd == "/cost":
        console.print(
            f"[bold]Session token usage[/bold]\n"
            f"  Input:  {_format_tokens(_session_usage.total_in)}\n"
            f"  Output: {_format_tokens(_session_usage.total_out)}\n"
            f"  Total:  {_format_tokens(_session_usage.total_in + _session_usage.total_out)}"
        )
        return True

    elif cmd == "/context":
        conv = agent.conversations._active.get(f"cli:{chat_id}")
        if conv:
            msg_count = len(conv._message_cache)
            # Estimate token usage
            from ..context import estimate_messages_tokens
            est_tokens = estimate_messages_tokens(conv._message_cache)
            ctx_window = conv.provider.context_window
            pct = min(100, int(est_tokens / ctx_window * 100)) if ctx_window else 0
            console.print(
                f"[bold]Context usage[/bold]\n"
                f"  Messages: {msg_count}\n"
                f"  Estimated tokens: ~{_format_tokens(est_tokens)}\n"
                f"  Context window: {_format_tokens(ctx_window)}\n"
                f"  Usage: ~{pct}%"
            )
        else:
            console.print("[dim]No active conversation.[/dim]")
        return True

    elif cmd == "/status":
        # Reuse the status logic
        from ..db.models import get_config, list_users

        identity = await get_identity()
        model_name = getattr(agent.provider, 'chat_model', 'unknown')
        provider_name = agent.provider.name

        # Count memories
        from ..db.connection import get_connection
        async with get_connection() as conn:
            mem_count = await conn.fetchval("SELECT COUNT(*) FROM memory")
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users WHERE active = true")
            group_count = await conn.fetchval("SELECT COUNT(*) FROM groups WHERE enabled = true")
            session_count = await conn.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
            )
            total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")

        tool_count = len(agent.tools.list_tools("owner"))
        ability_count = len(agent.abilities.list_all()) if agent.abilities else 0

        auto_capture = await get_config("memory.auto_capture", False)
        # Read thinking from active model params
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        _active_entry = next((m for m in models if m.get("key") == active_key), {})
        _model_params = _active_entry.get("params") or {}
        thinking = _model_params.get("thinking_budget", "default")

        from .. import __version__ as syne_version

        console.print(Panel(
            f"[bold]{identity.get('name', 'Syne')}[/bold] \u00b7 Syne v{syne_version}\n"
            f"Model: {model_name} ({provider_name})\n"
            f"Memories: {mem_count}\n"
            f"Users: {user_count} | Groups: {group_count}\n"
            f"Sessions: {session_count} active \u2022 {total_messages} messages\n"
            f"Tools: {tool_count} | Abilities: {ability_count}\n"
            f"Thinking: {thinking}\n"
            f"Auto-capture: {'ON' if auto_capture else 'OFF'}",
            title="Status",
            style="blue",
        ))
        return True

    elif cmd == "/clear":
        # Clear current conversation
        key = f"cli:{chat_id}"
        if key in agent.conversations._active:
            conv = agent.conversations._active[key]
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "DELETE FROM messages WHERE session_id = $1", conv.session_id
                )
            conv._message_cache.clear()
            console.print("[green]Conversation cleared.[/green]")
        else:
            console.print("[dim]No active conversation.[/dim]")
        return True

    elif cmd == "/compact":
        conv = agent.conversations._active.get(f"cli:{chat_id}")
        if not conv:
            console.print("[dim]No active conversation.[/dim]")
            return True
        msg_count = len(conv._message_cache)
        if msg_count < 4:
            console.print("[dim]Conversation too short to compact.[/dim]")
            return True
        console.print("[dim]Compacting conversation...[/dim]")
        from ..compaction import compact_session
        result = await compact_session(
            session_id=conv.session_id,
            provider=conv.provider,
        )
        if result:
            await conv.load_history()
            console.print(
                f"[green]Compacted: {result['messages_before']} -> {result['messages_after']} messages[/green]"
            )
        else:
            console.print("[dim]Nothing to compact.[/dim]")
        return True

    elif cmd == "/models":
        from ..db.models import get_config, set_config

        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "")

        if not models:
            console.print("[dim]No models registered.[/dim]")
            return True

        # Build option list — mark current with (active)
        options = []
        keys = []
        for m in models:
            key = m.get("key", "")
            label = m.get("label", key)
            model_id = m.get("model_id", "")
            suffix = " (active)" if key == active_key else ""
            options.append(f"{label} [{model_id}]{suffix}")
            keys.append(key)

        # Find current index
        current_idx = 0
        for i, k in enumerate(keys):
            if k == active_key:
                current_idx = i
                break

        console.print("[bold]Select model:[/bold]")
        try:
            chosen = await cli_select(options, default=current_idx)
        except (EOFError, KeyboardInterrupt):
            return True

        new_key = keys[chosen]
        if new_key == active_key:
            console.print(f"[dim]Already using {models[chosen].get('label', new_key)}.[/dim]")
            return True

        # Switch: update DB, create new provider, replace everywhere
        await set_config("provider.active_model", new_key)
        new_provider = await agent.create_provider_for_model(new_key)
        if new_provider:
            agent.provider = new_provider
            agent.conversations.provider = new_provider  # ConversationManager uses this
            # Invalidate cached conversation so next message uses new provider
            conv_key = f"cli:{chat_id}"
            if conv_key in agent.conversations._active:
                del agent.conversations._active[conv_key]
            new_label = models[chosen].get("label", new_key)
            console.print(f"[green]Switched to {new_label}[/green]")
        else:
            console.print(f"[red]Failed to create provider for {new_key}[/red]")
        return True

    # Unknown command — pass to agent
    return False
