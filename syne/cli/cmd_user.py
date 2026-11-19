"""User commands.

Usage:
    syne user delete <platform_id> [--platform telegram] [--yes]
"""

import asyncio

import click

from . import cli
from .shared import console


@cli.group()
def user():
    """Manage users."""


@user.command("delete")
@click.argument("platform_id")
@click.option("--platform", default="telegram", show_default=True,
              type=click.Choice(["telegram", "whatsapp", "discord", "cli"]))
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def user_delete(platform_id, platform, yes):
    """Delete a user: DM history, schedules, group entries. Memories stay."""
    async def _run():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import delete_user_preview, delete_user

        await init_db(load_settings().database_url)
        try:
            p = await delete_user_preview(platform, platform_id)
            if not p:
                console.print(f"[red]User {platform} {platform_id} not found.[/red]")
                return 1
            if p["access_level"] == "owner":
                console.print("[red]Refused: an owner cannot be deleted.[/red]")
                return 1
            console.print(f"[bold]Delete {p['name']}[/bold] ({platform} {platform_id}, {p['access_level']})")
            console.print(f"  DM deleted:              {len(p['dm_session_ids'])} session(s), {p['dm_messages']} message(s)")
            console.print(f"  Group messages anonymised: {len(p['group_message_ids'])}")
            console.print(f"  Shared sessions detached:  {p['shared_sessions']} (history kept)")
            console.print(f"  Schedules deleted:         {p['schedules']}")
            console.print(f"  Group member entries:      {p['groups']}")
            console.print(f"  Memories kept, owner cleared: {p['memories']}")
            if not yes and not click.confirm("This cannot be undone. Continue?", default=False):
                console.print("Cancelled.")
                return 1
            res = await delete_user(platform, platform_id)
            console.print("[green]Deleted.[/green]" if res else "[red]User not found.[/red]")
            if res:
                console.print("[dim]A running Syne still caches this user's open chat until it "
                              "restarts or the chat goes idle.[/dim]")
            return 0 if res else 1
        finally:
            await close_db()

    code = asyncio.run(_run())
    if code:
        raise SystemExit(code)
