"""Syne CLI — Interactive terminal chat with full tool access."""

import asyncio
import getpass
import logging
import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..agent import SyneAgent
from ..config import load_settings
from ..db.models import get_or_create_user, get_identity

logger = logging.getLogger("syne.cli_channel")
console = Console()
_prompt_session: PromptSession | None = None


async def run_cli(debug: bool = False, yolo: bool = False, fresh: bool = False):
    """Run Syne in interactive CLI mode (resumes previous session by default)."""
    if debug:
        logging.getLogger("syne").setLevel(logging.DEBUG)
    else:
        # Suppress noisy loggers in CLI mode — only show errors
        logging.getLogger("syne").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.WARNING)

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

    _prompt_session = PromptSession(key_bindings=_kb, multiline=True)

    try:
        await agent.start()

        # Set CLI working directory to where user launched `syne cli`
        agent._cli_cwd = os.getcwd()

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
        )

        # Ensure CLI user is owner
        if user.get("access_level") != "owner":
            from ..db.models import update_user
            await update_user("cli", f"cli:{username}", access_level="owner")
            user = dict(user)
            user["access_level"] = "owner"

        # Set up callbacks
        if agent.conversations:
            agent.conversations.set_status_callback(_cli_status_callback)
            # Tool activity indicator — will be wired to status spinner in REPL
            agent.conversations._tool_status = None  # Rich Status object, set during processing

        # File write approval (unless --yolo)
        if not yolo:
            _always_allowed: set[str] = set()  # paths auto-approved for this session

            async def _approval_callback(tool_name: str, args: dict) -> tuple[bool, str]:
                """Ask user for approval before file writes. Returns (approved, reason)."""
                if tool_name != "file_write":
                    return True, ""

                file_path = args.get("path", "?")
                resolved = os.path.abspath(file_path)

                if resolved in _always_allowed:
                    return True, ""

                # Pause spinner if active
                _status_ref = getattr(agent.conversations, '_active_status', None)
                if _status_ref:
                    _status_ref.stop()

                content = args.get("content", "")
                lines = content.count("\n") + 1
                size = len(content.encode("utf-8"))

                console.print(f"\n[bold yellow]📝 Write to {file_path}?[/bold yellow] ({lines} lines, {size:,} bytes)")
                console.print("[dim]  [y] Yes  [n] No  [a] Always allow this file[/dim]")

                try:
                    choice = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("  > ").strip().lower()
                    )
                except (EOFError, KeyboardInterrupt):
                    choice = "n"

                # Resume spinner
                if _status_ref:
                    _status_ref.start()

                if choice == "a":
                    _always_allowed.add(resolved)
                    return True, ""
                elif choice in ("y", "yes"):
                    return True, ""
                else:
                    return False, "User declined file write."

            agent.tools.set_approval_callback(_approval_callback)

        # Display header
        identity = await get_identity()
        agent_name = identity.get("name", "Syne")
        motto = identity.get("motto", "")

        console.print()
        console.print(Panel(
            f"[bold]{agent_name}[/bold]" + (f"\n[dim italic]{motto}[/dim italic]" if motto else ""),
            style="blue",
            subtitle=f"Model: {agent.provider.name} | Tools: {len(agent.tools.list_tools('owner'))} | Type /help for commands",
        ))

        # Check for pending update notice
        try:
            from ..update_checker import get_pending_update_notice
            update_notice = await get_pending_update_notice()
            if update_notice:
                console.print(f"[bold yellow]{update_notice}[/bold yellow]")
        except Exception:
            pass

        console.print()

        # REPL loop — session is per-directory
        cwd = agent._cli_cwd or os.getcwd()
        chat_id = f"cli:{username}:{cwd}"

        # Fresh start: close old session and clear history
        if fresh:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                # Close old sessions and delete messages for this chat_id
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
            # Also clear in-memory cache
            key = f"cli:{chat_id}"
            if key in agent.conversations._active:
                del agent.conversations._active[key]
            console.print("[dim]Starting fresh conversation...[/dim]")

        # Load input history from DB (user messages for this directory)
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
                    LIMIT 500
                """, chat_id)
            for row in rows:
                content = row["content"].strip()
                if content and not content.startswith("/"):
                    _history.append_string(content)
            _prompt_session.history = _history
        except Exception:
            pass
        import time as _time
        _last_ctrl_c = 0.0
        while True:
            # Phase 1: Get input (idle)
            try:
                user_input = await _get_input()
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl+C at prompt → double-tap to exit
                now = _time.time()
                if now - _last_ctrl_c < 2.0:
                    console.print("\n[dim]Goodbye![/dim]")
                    break
                _last_ctrl_c = now
                console.print("\n[dim](Press Ctrl+C again to exit)[/dim]")
                continue

            if user_input is None:
                # EOF (Ctrl+D)
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Built-in CLI commands
            if user_input.startswith("/"):
                handled = await _handle_cli_command(user_input, agent, user, chat_id)
                if handled == "exit":
                    break
                if handled:
                    continue

            # Phase 2: Process (Ctrl+C or Esc = cancel this request)
            console.print()
            status = None
            try:
                msg = user_input
                status = console.status("[bold blue]Thinking...", spinner="dots")
                status.start()
                agent.conversations._active_status = status
                _cli_status_callback._active_status = status

                # Wire tool callback to update spinner
                _TOOL_LABELS = {
                    "exec": "🔧 Running command...",
                    "memory_search": "🔍 Searching memory...",
                    "memory_store": "💾 Storing memory...",
                    "web_search": "🌐 Searching web...",
                    "web_fetch": "🌐 Fetching page...",
                    "spawn_subagent": "🤖 Spawning sub-agent...",
                    "file_read": "📖 Reading file...",
                    "file_write": "📝 Writing file...",
                    "read_source": "📖 Reading source...",
                    "update_config": "⚙️ Updating config...",
                    "update_soul": "✨ Updating soul...",
                    "update_ability": "🧩 Updating ability...",
                    "manage_schedule": "⏰ Managing schedule...",
                    "manage_user": "👤 Managing user...",
                    "manage_group": "👥 Managing group...",
                }

                async def _on_tool(name: str):
                    label = _TOOL_LABELS.get(name, f"🔧 {name}...")
                    status.update(f"[bold blue]{label}")

                agent.conversations.set_tool_callback(_on_tool)

                response = await agent.conversations.handle_message(
                    platform="cli",
                    chat_id=chat_id,
                    user=user,
                    message=msg,
                    message_metadata={"cwd": agent._cli_cwd},
                )

                # Cleanup spinner
                _stop_status(status, agent)
                status = None

                # Display response
                if response:
                    _display_response(response)
                console.print()
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl+C during processing → cancel, back to prompt
                _stop_status(status, agent)
                status = None
                console.print("\n[yellow]⚡ Cancelled[/yellow]\n")
                continue

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()

    # Cleanup
    await agent.stop()
    # Restore terminal state — prompt_toolkit/executor can leave it corrupted
    try:
        os.system("stty sane 2>/dev/null")
    except Exception:
        pass
    # Force exit to avoid threading cleanup error from executor thread
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
    except Exception:
        pass


async def _get_input() -> str | None:
    """Get user input using prompt_toolkit (supports Shift+Enter for newlines)."""
    try:
        result = await _prompt_session.prompt_async(
            "> ",
            multiline=True,
            prompt_continuation="… ",
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
            "/help          — Show this help\n"
            "/status        — Show agent status\n"
            "/model         — Show/switch model\n"
            "/memory        — Search memories\n"
            "/compact       — Compact conversation\n"
            "/clear         — Clear conversation history\n"
            "/think [level] — Set thinking budget\n"
            "/exit          — Exit CLI\n"
            "\n[dim]All other messages are sent to the agent.[/dim]\n"
            "[dim]Shift+Enter or Esc+Enter for new line.[/dim]",
            style="blue",
        ))
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

        tool_count = len(agent.tools.list_tools("owner"))
        ability_count = len(agent.abilities.list_all()) if agent.abilities else 0

        auto_capture = await get_config("memory.auto_capture", False)
        thinking = await get_config("session.thinking_budget", "default")

        from .. import __version__ as syne_version

        console.print(Panel(
            f"[bold]{identity.get('name', 'Syne')}[/bold] · Syne v{syne_version}\n"
            f"🤖 Model: {model_name} ({provider_name})\n"
            f"📚 Memories: {mem_count}\n"
            f"👥 Users: {user_count} | Groups: {group_count}\n"
            f"💬 Active sessions: {session_count}\n"
            f"🔧 Tools: {tool_count} | Abilities: {ability_count}\n"
            f"💭 Thinking: {thinking}\n"
            f"📝 Auto-capture: {'ON' if auto_capture else 'OFF'}",
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
            console.print("[green]✓ Conversation cleared.[/green]")
        else:
            console.print("[dim]No active conversation.[/dim]")
        return True

    elif cmd == "/compact":
        # Forward to agent as regular message
        return False

    elif cmd == "/model":
        if args:
            # Forward model switch to agent
            return False
        console.print(f"[bold]Current model:[/bold] {agent.provider.chat_model} ({agent.provider.name})")
        return True

    elif cmd == "/think":
        if args:
            return False  # Let agent handle
        thinking = await agent.conversations._active.get(
            f"cli:{chat_id}", None
        )
        if thinking:
            console.print(f"[bold]Thinking budget:[/bold] {thinking.thinking_budget or 'default'}")
        return True

    # Unknown command — pass to agent
    return False
