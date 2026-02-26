"""Syne CLI — command line interface."""

import click
from syne import __version__
from .shared import console


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="syne")
@click.pass_context
def cli(ctx):
    """Syne — AI Agent Framework with Unlimited Memory"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Import all command modules (registers commands onto cli group)
from . import cmd_init  # noqa: E402, F401
from . import cmd_start  # noqa: E402, F401
from . import cmd_status  # noqa: E402, F401
from . import cmd_db  # noqa: E402, F401
from . import cmd_identity  # noqa: E402, F401
from . import cmd_memory  # noqa: E402, F401
from . import cmd_repair  # noqa: E402, F401
from . import cmd_service  # noqa: E402, F401
from . import cmd_update  # noqa: E402, F401
from . import cmd_uninstall  # noqa: E402, F401
from . import cmd_backup  # noqa: E402, F401


def main():
    """CLI entry point."""
    cli()
