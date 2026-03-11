"""Service management commands — stop, restart."""

import os
import subprocess
import click

from . import cli
from .shared import console


def _is_server() -> bool:
    """Check if this machine is a Syne server (has .env from syne init)."""
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.exists(os.path.join(syne_dir, ".env"))


@cli.command()
def stop():
    """Stop running Syne process."""
    if not _is_server():
        console.print("[yellow]This machine is not a Syne server.[/yellow]")
        return

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
    if not _is_server():
        console.print("[yellow]This machine is not a Syne server.[/yellow]")
        return

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
    log_dir = os.path.expanduser("~/.log-syne")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "syne.log")
    log_file = open(log_path, "a")
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
    console.print(f"[dim]Logs: tail -f {log_path}[/dim]")
