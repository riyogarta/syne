"""Service management commands — autostart, stop, restart."""

import os
import subprocess
import click

from . import cli
from .shared import console
from .helpers import _setup_service


@cli.command()
@click.option("--enable/--disable", default=True, help="Enable or disable autostart")
def autostart(enable):
    """Configure systemd autostart for Syne."""
    from pathlib import Path

    if not enable:
        service_dir = Path.home() / ".config" / "systemd" / "user"
        service_file = service_dir / "syne.service"
        subprocess.run(["systemctl", "--user", "stop", "syne"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "syne"], capture_output=True)
        if service_file.exists():
            service_file.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("[yellow]Autostart disabled.[/yellow]")
        return

    _setup_service()
    console.print()
    console.print("[bold]Commands:[/bold]")
    console.print("  systemctl --user start syne    # Start now")
    console.print("  systemctl --user stop syne     # Stop")
    console.print("  systemctl --user restart syne  # Restart")
    console.print("  systemctl --user status syne   # Status")
    console.print("  journalctl --user -u syne -f   # Logs")


@cli.command()
def stop():
    """Stop running Syne process."""
    # Try systemd first
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "syne"],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == "active":
        subprocess.run(["systemctl", "--user", "stop", "syne"])
        console.print("[green]✓ Syne stopped (systemd)[/green]")
        return

    # Fall back to pkill
    result = subprocess.run(["pkill", "-f", "syne.main"], capture_output=True)
    if result.returncode == 0:
        console.print("[green]✓ Syne process killed[/green]")
    else:
        console.print("[yellow]Syne is not running.[/yellow]")


@cli.command()
def restart():
    """Restart Syne (stop + start)."""
    # Try systemd
    result = subprocess.run(
        ["systemctl", "--user", "is-enabled", "syne"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("Restarting service... Please wait a moment.")
        subprocess.run(["systemctl", "--user", "restart", "syne"])
        console.print("[green]✓ Service restarted (systemd)[/green]")
        return

    # Manual: kill + start
    console.print("Restarting service... Please wait a moment.")
    subprocess.run(["pkill", "-f", "syne.main"], capture_output=True)
    import time
    time.sleep(2)

    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    venv_python = os.path.join(syne_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_python):
        console.print(f"[red]Venv not found at {venv_python}. Run 'syne init' first.[/red]")
        return
    log_file = open("/tmp/syne.log", "a")
    try:
        subprocess.Popen(
            [venv_python, "-m", "syne.main"],
            cwd=syne_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()
    console.print("[green]✓ Service restarted (PID in background)[/green]")
    console.print("[dim]Logs: tail -f /tmp/syne.log[/dim]")
