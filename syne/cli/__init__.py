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
        _show_help()


def _show_help():
    """Show all available commands grouped by category."""
    console.print(f"[bold]Syne v{__version__}[/bold] — AI Agent Framework with Unlimited Memory\n")

    groups = {
        "Setup": [
            ("init", "Initialize Syne: authenticate, setup database, configure"),
            ("repair", "Diagnose and repair Syne installation (--fix to auto-repair)"),
            ("update", "Update to latest release"),
            ("updatedev", "Update from git (dev, always reinstall)"),
            ("reauth", "Re-authenticate OAuth provider"),
            ("uninstall", "Completely remove Syne"),
        ],
        "Usage": [
            ("start", "Start Syne agent (Telegram bot)"),
            ("cli", "Interactive CLI chat"),
            ("status", "Show agent status"),
        ],
        "Data": [
            ("identity", "View or set agent identity"),
            ("prompt", "Show current system prompt"),
            ("memory stats", "Show memory statistics"),
            ("memory search", "Search memories by similarity"),
            ("memory add", "Manually add a memory"),
            ("db init", "Initialize database schema"),
            ("db reset", "Reset database (DROP ALL + re-init)"),
            ("backup", "Backup database to .sql.gz file"),
            ("restore", "Restore database from backup file"),
        ],
        "Service": [
            ("autostart", "Configure systemd autostart (--enable/--disable)"),
            ("stop", "Stop running Syne process"),
            ("restart", "Restart Syne (stop + start)"),
        ],
    }

    for category, commands in groups.items():
        console.print(f"  [bold cyan]{category}[/bold cyan]")
        for name, desc in commands:
            console.print(f"    [bold]syne {name:16s}[/bold] {desc}")
        console.print()

    console.print("[dim]Run 'syne <command> --help' for details on a specific command.[/dim]")


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


@cli.command(name="help", hidden=True)
def help_cmd():
    """Show all available commands."""
    _show_help()


def main():
    """CLI entry point."""
    import sys
    try:
        cli(standalone_mode=False)
    except click.UsageError as e:
        if e.ctx:
            click.echo(e.ctx.command.get_usage(e.ctx), err=True)
        click.echo(f"Try 'syne help' for help.\n", err=True)
        click.echo(f"Error: {e.format_message()}", err=True)
        sys.exit(2)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        sys.exit(e.code)
    except click.Abort:
        click.echo("Aborted!", err=True)
        sys.exit(1)
