"""Start and CLI mode commands."""

import asyncio
import click

from . import cli
from .shared import console


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def start(debug):
    """Start Syne agent (auto-configures autostart on first run)."""
    from ..node.client import is_node_mode
    if is_node_mode():
        console.print("[yellow]This machine is configured as a remote node.[/yellow]")
        console.print("Use [bold]syne node start[/bold] / [bold]syne node stop[/bold] instead.")
        return

    from pathlib import Path

    # Auto-setup systemd service if not installed yet
    service_file = Path.home() / ".config" / "systemd" / "user" / "syne.service"
    if not service_file.exists():
        console.print("[dim]Setting up autostart service...[/dim]")
        from .helpers import _setup_service
        _setup_service()
        return

    if debug:
        import logging
        logging.getLogger("syne").setLevel(logging.DEBUG)

    from syne.main import run
    console.print("[bold blue]Starting Syne...[/bold blue]")
    asyncio.run(run())


@cli.command(name="cli")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--yolo", is_flag=True, help="Skip file write approvals (auto-yes)")
@click.option("--new", "-n", is_flag=True, help="Start fresh conversation (clear history in this directory)")
@click.option("--local", is_flag=True, help="Force local mode even if node is configured")
def cli_mode(debug, yolo, new, local):
    """Interactive CLI chat (resumes previous conversation by default).

    Auto-detects remote node mode if ~/.syne/node.json exists.
    """
    from syne.channels.cli_channel import run_cli

    node_client = None
    if not local:
        from syne.node.client import is_node_mode, NodeClient
        if is_node_mode():
            node_client = NodeClient()
            if debug:
                console.print("[dim]Remote mode: connecting via gateway...[/dim]")

    asyncio.run(run_cli(debug=debug, yolo=yolo, fresh=new, node_client=node_client))
