"""Update and re-auth commands."""

import asyncio
import os
import subprocess
import sys
import click

from . import cli
from .shared import console, _strip_env_quotes, _get_db_url
from .helpers import (
    _ensure_evaluator_if_enabled, _run_schema_migration,
    _restart_service, _create_symlink,
)


def _do_update(syne_dir: str):
    """Shared update logic: venv + pip install + symlink + PATH + restart service."""
    venv_dir = os.path.join(syne_dir, ".venv")
    venv_pip = os.path.join(venv_dir, "bin", "pip")
    venv_syne = os.path.join(venv_dir, "bin", "syne")

    console.print("Setting up virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], cwd=syne_dir)

    console.print("Installing...")
    result = subprocess.run([venv_pip, "install", "-e", ".", "-q"], cwd=syne_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Install failed: {result.stderr}[/red]")
        return False

    # Ensure syne is callable from anywhere
    local_bin = os.path.expanduser("~/.local/bin")
    os.makedirs(local_bin, exist_ok=True)
    target = os.path.join(local_bin, "syne")
    if os.path.exists(venv_syne):
        if os.path.exists(target) or os.path.islink(target):
            os.remove(target)
        os.symlink(venv_syne, target)

    # Ensure ~/.local/bin is in PATH
    path_line = 'export PATH="$HOME/.local/bin:$PATH"'
    shell_rc = os.path.expanduser("~/.bashrc")
    if os.path.exists(os.path.expanduser("~/.zshrc")):
        shell_rc = os.path.expanduser("~/.zshrc")
    try:
        with open(shell_rc, "r") as f:
            content = f.read()
        if path_line not in content and ".local/bin" not in content:
            with open(shell_rc, "a") as f:
                f.write(f"\n# Syne CLI\n{path_line}\n")
    except Exception:
        pass

    # Run schema migrations (safe — uses IF NOT EXISTS / DO $$ checks)
    _run_schema_migration(syne_dir)

    # Ensure Ollama + evaluator model if auto_capture is enabled (non-fatal)
    _ensure_evaluator_if_enabled(syne_dir)

    # Restart systemd service if active
    _restart_service()

    return True


@cli.command()
def reauth():
    """Re-authenticate OAuth provider (refresh expired tokens)."""
    async def _reauth():
        syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dsn = _get_db_url(syne_dir)
        if not dsn:
            dsn = os.environ.get("SYNE_DATABASE_URL", "")
        if not dsn:
            console.print("[red]SYNE_DATABASE_URL not set. Is .env configured?[/red]")
            return

        from syne.db.connection import init_db, close_db
        from syne.db.models import get_config, set_config

        await init_db(dsn)

        # Determine current provider
        primary = await get_config("provider.primary", None)
        if not primary:
            console.print("[red]No provider configured. Run `syne init` first.[/red]")
            return

        driver = primary.get("driver", "") if isinstance(primary, dict) else ""

        if driver == "codex":
            console.print("[bold]Re-authenticating Codex (GPT) OAuth...[/bold]")
            from syne.auth.codex_oauth import login_codex
            try:
                creds = await login_codex()
                await set_config("credential.codex_access_token", creds["access_token"])
                await set_config("credential.codex_refresh_token", creds["refresh_token"])
                await set_config("credential.codex_expires_at", creds["expires_at"])
                console.print("[green]Codex OAuth tokens refreshed.[/green]")
            except Exception as e:
                console.print(f"[red]Codex re-auth failed: {e}[/red]")
                return

        elif driver in ("google", "google_cca"):
            console.print("[bold]Re-authenticating Google Gemini OAuth...[/bold]")
            from syne.auth.google_oauth import login_google
            try:
                creds = await login_google()
                await set_config("credential.google_oauth_access_token", creds.access_token)
                await set_config("credential.google_oauth_refresh_token", creds.refresh_token)
                await set_config("credential.google_oauth_expires_at", creds.expires_at)
                await set_config("credential.google_oauth_project_id", creds.project_id)
                if creds.email:
                    await set_config("credential.google_oauth_email", creds.email)
                console.print("[green]Google OAuth tokens refreshed.[/green]")
            except Exception as e:
                console.print(f"[red]Google re-auth failed: {e}[/red]")
                return

        else:
            console.print(f"[yellow]Provider '{driver}' doesn't use OAuth. No re-auth needed.[/yellow]")
            return

        await close_db()

        # Restart service to pick up new tokens
        _restart_service()

    asyncio.run(_reauth())


@cli.command()
def update():
    """Update to latest release (skip if version unchanged)."""
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from syne import __version__ as current_version

    # Fetch remote
    console.print("Checking for updates...")
    subprocess.run(["git", "fetch", "origin"], cwd=syne_dir, capture_output=True, text=True)

    # Read remote __version__ to compare
    result = subprocess.run(
        ["git", "show", "origin/main:syne/__init__.py"],
        cwd=syne_dir, capture_output=True, text=True,
    )
    remote_version = current_version  # fallback
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("__version__"):
                remote_version = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    if remote_version == current_version:
        console.print(f"[green]Already up to date (v{current_version})[/green]")
        return

    console.print(f"[yellow]New version available: v{remote_version} (current: v{current_version})[/yellow]")

    console.print("Pulling latest code...")
    result = subprocess.run(["git", "pull", "origin", "main"], cwd=syne_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Git pull failed: {result.stderr}[/red]")
        return
    console.print(f"[dim]{result.stdout.strip()}[/dim]")

    if _do_update(syne_dir):
        # Re-read version after update
        try:
            r = subprocess.run(
                [os.path.join(syne_dir, ".venv", "bin", "python"), "-c",
                 "from syne import __version__; print(__version__)"],
                cwd=syne_dir, capture_output=True, text=True,
            )
            new_ver = r.stdout.strip() if r.returncode == 0 else "?"
        except Exception:
            new_ver = "?"
        console.print(f"[green]Syne updated to v{new_ver}[/green]")


@cli.command(name="updatedev", hidden=True)
def update_dev():
    """Pull latest code and reinstall (always, ignoring version)."""
    syne_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    console.print("Pulling latest code...")
    result = subprocess.run(["git", "pull", "origin", "main"], cwd=syne_dir, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Git pull failed: {result.stderr}[/red]")
        return
    console.print(f"[dim]{result.stdout.strip()}[/dim]")

    if _do_update(syne_dir):
        console.print("[green]Syne updated (dev)[/green]")
