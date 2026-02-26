"""Identity and prompt commands."""

import asyncio
import click

from . import cli
from .shared import console

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
def identity(key, value):
    """View or set identity values. Usage: syne identity [key] [value]"""
    async def _identity():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import get_identity, set_identity

        settings = load_settings()
        await init_db(settings.database_url)

        if key and value:
            await set_identity(key, value)
            console.print(f"[green]✓ {key} = {value}[/green]")
        elif key:
            data = await get_identity()
            val = data.get(key)
            if val is not None:
                console.print(f"[bold]{key}[/bold] = {val}")
            else:
                console.print(f"[yellow]Key '{key}' not found.[/yellow]")
        else:
            data = await get_identity()
            table = Table(title="Identity", show_header=True)
            table.add_column("Key", style="bold")
            table.add_column("Value")
            for k, v in data.items():
                table.add_row(k, v)
            console.print(table)

        await close_db()

    asyncio.run(_identity())


@cli.command()
def prompt():
    """Show the current system prompt."""
    async def _prompt():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.boot import build_system_prompt

        settings = load_settings()
        await init_db(settings.database_url)

        p = await build_system_prompt()
        console.print(Panel(Syntax(p, "markdown"), title="System Prompt", style="blue"))

        await close_db()

    asyncio.run(_prompt())
