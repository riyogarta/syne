"""Uninstall command."""

import os
import subprocess
import click

from . import cli
from .shared import console


@cli.command()
def uninstall():
    """Completely remove Syne — service, containers, data, source code."""
    import shutil
    from pathlib import Path

    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    compose_file = os.path.join(syne_dir, "docker-compose.yml")
    service_file = Path.home() / ".config" / "systemd" / "user" / "syne.service"
    symlink_path = os.path.expanduser("~/.local/bin/syne")

    console.print()
    console.print("[bold red]Syne Uninstall[/bold red]")
    console.print()
    console.print("This will permanently remove:")
    console.print(f"  - Systemd service (stop, disable, delete)")
    console.print(f"  - Docker containers, volume, and image (ALL DATA LOST)")
    console.print(f"  - CLI symlink ({symlink_path})")
    console.print(f"  - Shell PATH entry (bashrc/zshrc)")
    console.print(f"  - Ollama embedding model (if installed)")
    console.print(f"  - Source code ({syne_dir})")
    console.print()

    confirm = input("Type 'yes' to confirm uninstall: ").strip().lower()
    if confirm != "yes":
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print()

    # 1. Stop and remove systemd service
    console.print("[bold]1/6[/bold] Removing systemd service...")
    try:
        subprocess.run(["systemctl", "--user", "stop", "syne"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "syne"], capture_output=True)
        if service_file.exists():
            service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        console.print("  [green]✓ Service removed[/green]")
    except FileNotFoundError:
        console.print("  [dim]systemctl not found — skipping[/dim]")

    # 2. Stop and remove Docker containers + volume + image
    console.print("[bold]2/6[/bold] Removing Docker containers and data...")
    try:
        if os.path.isfile(compose_file):
            subprocess.run(
                ["docker", "compose", "-f", compose_file, "down", "-v", "--remove-orphans"],
                capture_output=True, cwd=syne_dir,
            )
        # Also clean up any leftover containers
        for name in ["syne-db", "syne-agent"]:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        # Remove Docker image
        subprocess.run(["docker", "rmi", "pgvector/pgvector:pg16"], capture_output=True)
        console.print("  [green]✓ Containers, data, and images removed[/green]")
    except FileNotFoundError:
        console.print("  [dim]docker not found — skipping[/dim]")

    # 3. Remove CLI symlink
    console.print("[bold]3/6[/bold] Removing CLI symlink...")
    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
        os.remove(symlink_path)
        console.print(f"  [green]✓ Removed {symlink_path}[/green]")
    else:
        console.print(f"  [dim]Not found: {symlink_path}[/dim]")

    # 4. Clean up shell rc (remove Syne CLI PATH line)
    console.print("[bold]4/6[/bold] Cleaning shell config...")
    for rc_file in [os.path.expanduser("~/.bashrc"), os.path.expanduser("~/.zshrc")]:
        if os.path.exists(rc_file):
            try:
                with open(rc_file, "r") as f:
                    lines = f.readlines()
                new_lines = []
                skip_next = False
                for line in lines:
                    if skip_next:
                        skip_next = False
                        continue
                    if "# Syne CLI" in line:
                        skip_next = True  # Skip the PATH export line after comment
                        continue
                    if ".local/bin" in line and "syne" in line.lower():
                        continue
                    new_lines.append(line)
                if len(new_lines) != len(lines):
                    with open(rc_file, "w") as f:
                        f.writelines(new_lines)
                    console.print(f"  [green]✓ Cleaned {os.path.basename(rc_file)}[/green]")
            except Exception:
                pass

    # 5. Remove Ollama models if present
    console.print("[bold]5/6[/bold] Cleaning up Ollama models...")
    try:
        removed = []
        for model in ["qwen3-embedding:0.6b", "qwen3:0.6b"]:
            result = subprocess.run(
                ["ollama", "rm", model],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                removed.append(model)
        if removed:
            console.print(f"  [green]✓ Removed {', '.join(removed)}[/green]")
        else:
            console.print("  [dim]No Ollama models to remove[/dim]")
    except FileNotFoundError:
        console.print("  [dim]ollama not found — skipping[/dim]")

    # 6. Remove source code (self-destruct — must be last)
    console.print("[bold]6/6[/bold] Removing source code...")
    try:
        shutil.rmtree(syne_dir)
        console.print(f"  [green]✓ Removed {syne_dir}[/green]")
    except Exception as e:
        console.print(f"  [yellow]Could not fully remove {syne_dir}: {e}[/yellow]")
        console.print(f"  [dim]Remove manually: rm -rf {syne_dir}[/dim]")

    console.print()
    console.print("[bold green]Syne has been completely uninstalled.[/bold green]")
