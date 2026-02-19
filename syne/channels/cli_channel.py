"""Syne CLI — Interactive terminal chat with full tool access."""

import asyncio
import getpass
import logging
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ..agent import SyneAgent
from ..config import load_settings
from ..db.models import get_or_create_user, get_identity

logger = logging.getLogger("syne.cli_channel")
console = Console()


async def run_cli(debug: bool = False):
    """Run Syne in interactive CLI mode."""
    if debug:
        logging.getLogger("syne").setLevel(logging.DEBUG)

    settings = load_settings()
    agent = SyneAgent(settings)

    try:
        await agent.start()

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

        # Set up status callback for compaction notifications
        if agent.conversations:
            agent.conversations.set_status_callback(_cli_status_callback)

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
        console.print()

        # REPL loop
        chat_id = f"cli:{username}"
        while True:
            try:
                # Prompt
                user_input = _get_input()
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

                # Send to agent
                console.print()
                with console.status("[bold blue]Thinking...", spinner="dots"):
                    response = await agent.conversations.handle_message(
                        platform="cli",
                        chat_id=chat_id,
                        user=user,
                        message=user_input,
                    )

                # Display response
                if response:
                    _display_response(response)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim](Ctrl+C — type /exit to quit)[/dim]")
                continue

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if debug:
            import traceback
            traceback.print_exc()
        raise
    finally:
        await agent.stop()


def _get_input() -> str | None:
    """Get user input, supporting multiline with \\ continuation."""
    try:
        lines = []
        prompt = "[bold green]>[/bold green] "
        
        while True:
            if not lines:
                console.print(prompt, end="")
                line = input()
            else:
                console.print("[dim]...[/dim] ", end="")
                line = input()

            if line.endswith("\\"):
                lines.append(line[:-1])
                continue
            else:
                lines.append(line)
                break

        return "\n".join(lines)
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
    console.print(f"\n[dim italic]{message}[/dim italic]")


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
            "[dim]Use \\\\ at end of line for multiline input.[/dim]",
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
            group_count = await conn.fetchval("SELECT COUNT(*) FROM groups WHERE active = true")
            session_count = await conn.fetchval(
                "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
            )

        tool_count = len(agent.tools.list_tools("owner"))
        ability_count = len(agent.abilities.list_abilities()) if agent.abilities else 0

        auto_capture = await get_config("memory.auto_capture", False)
        thinking = await get_config("session.thinking_budget", "default")

        console.print(Panel(
            f"[bold]{identity.get('name', 'Syne')}[/bold]\n"
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
