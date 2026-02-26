"""Start and CLI mode commands."""

import asyncio
import click

from . import cli
from .shared import console


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def start(debug):
    """Start Syne agent."""
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
def cli_mode(debug, yolo, new):
    """Interactive CLI chat (resumes previous conversation by default)."""
    from syne.channels.cli_channel import run_cli
    asyncio.run(run_cli(debug=debug, yolo=yolo, fresh=new))
