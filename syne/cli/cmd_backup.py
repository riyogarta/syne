"""Backup and restore commands."""

import os
import subprocess
import sys
from datetime import datetime
import click

from . import cli
from .shared import console, _read_env_value, _get_syne_dir


def _get_backup_dir():
    """Return the backup directory path."""
    return os.path.join(_get_syne_dir(), "backup")


def _list_backups():
    """List backup files sorted newest-first. Returns list of (path, size, mtime)."""
    backup_dir = _get_backup_dir()
    if not os.path.isdir(backup_dir):
        return []
    files = []
    for f in os.listdir(backup_dir):
        if f.endswith(".sql.gz") or f.endswith(".sql"):
            path = os.path.join(backup_dir, f)
            stat = os.stat(path)
            files.append((path, stat.st_size, stat.st_mtime))
    files.sort(key=lambda x: x[2], reverse=True)
    return files


def _format_size(size):
    """Format byte size to human-readable string."""
    if size > 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    elif size > 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} bytes"


def run_backup(output=None):
    """Core backup logic. Returns (success, message, filename).

    Used by both CLI `syne backup` and Telegram `/backup`.
    """
    syne_dir = _get_syne_dir()
    db_user = _read_env_value("SYNE_DB_USER", syne_dir)
    db_name = _read_env_value("SYNE_DB_NAME", syne_dir)

    if not db_user or not db_name:
        return False, "Cannot read DB credentials from .env.", None

    # Default output path
    if output is None:
        backup_dir = os.path.join(syne_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output = os.path.join(backup_dir, f"syne-backup-{timestamp}.sql.gz")
    output = os.path.expanduser(output)

    # Check container
    check = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or check.stdout.strip() != "running":
        return False, "syne-db container is not running.", None

    try:
        dump_proc = subprocess.Popen(
            ["docker", "exec", "syne-db", "pg_dump", "-U", db_user, "-d", db_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        with open(output, "wb") as f:
            gzip_proc = subprocess.Popen(
                ["gzip"], stdin=dump_proc.stdout, stdout=f, stderr=subprocess.PIPE,
            )
            dump_proc.stdout.close()
            gzip_proc.wait()
            dump_proc.wait()

        if dump_proc.returncode != 0:
            stderr = dump_proc.stderr.read().decode() if dump_proc.stderr else ""
            if os.path.exists(output):
                os.remove(output)
            return False, f"pg_dump failed: {stderr.strip()[:300]}", None

        size_str = _format_size(os.path.getsize(output))
        return True, f"{os.path.basename(output)} ({size_str})", output

    except FileNotFoundError as e:
        return False, f"Command not found: {e}", None
    except Exception as e:
        return False, f"Backup failed: {e}", None


def run_restore(filepath):
    """Core restore logic. Returns (success, message).

    Used by both CLI `syne restore` and Telegram `/restore`.
    Caller is responsible for confirmation prompt.
    """
    from .helpers import _run_schema_migration

    syne_dir = _get_syne_dir()
    db_user = _read_env_value("SYNE_DB_USER", syne_dir)
    db_name = _read_env_value("SYNE_DB_NAME", syne_dir)

    if not db_user or not db_name:
        return False, "Cannot read DB credentials from .env."

    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    # Check container
    check = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or check.stdout.strip() != "running":
        return False, "syne-db container is not running."

    # Drop and recreate database
    subprocess.run(
        ["docker", "exec", "syne-db", "psql", "-U", db_user, "-d", "postgres",
         "-c", f"DROP DATABASE IF EXISTS {db_name} WITH (FORCE);"],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "exec", "syne-db", "psql", "-U", db_user, "-d", "postgres",
         "-c", f"CREATE DATABASE {db_name} OWNER {db_user};"],
        capture_output=True,
    )

    try:
        is_gzip = filepath.endswith(".gz")

        if is_gzip:
            gunzip_proc = subprocess.Popen(
                ["gunzip", "-c", filepath],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            psql_proc = subprocess.Popen(
                ["docker", "exec", "-i", "syne-db", "psql", "-U", db_user, "-d", db_name, "-q"],
                stdin=gunzip_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            gunzip_proc.stdout.close()
            psql_proc.wait()
            gunzip_proc.wait()
            returncode = psql_proc.returncode
            stderr = psql_proc.stderr.read().decode() if psql_proc.stderr else ""
        else:
            with open(filepath, "rb") as f:
                result = subprocess.run(
                    ["docker", "exec", "-i", "syne-db", "psql", "-U", db_user, "-d", db_name, "-q"],
                    stdin=f, capture_output=True,
                )
                returncode = result.returncode
                stderr = result.stderr.decode() if result.stderr else ""

        # Schema migration
        _run_schema_migration(syne_dir)

        if returncode != 0:
            return True, f"Restored (with warnings): {stderr.strip()[:200]}"
        return True, "Database restored successfully."

    except FileNotFoundError as e:
        return False, f"Command not found: {e}"
    except Exception as e:
        return False, f"Restore failed: {e}"


@cli.command()
@click.option("--output", "-o", default=None, help="Output file path (default: ~/syne/backup/syne-backup-TIMESTAMP.sql.gz)")
def backup(output):
    """Backup Syne database to a compressed SQL file."""
    console.print("[bold]Backing up database...[/bold]")
    success, message, path = run_backup(output)
    if success:
        console.print(f"[green]Backup saved: {message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")


@cli.command()
@click.argument("file", required=False, default=None)
@click.option("--list", "-l", "list_backups", is_flag=True, help="List available backups")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def restore(file, list_backups, yes):
    """Restore Syne database from a backup file.

    FILE is required — specify a filename or full path.
    Use --list to see available backups.
    """
    # No args and no --list: show available backups
    if file is None and not list_backups:
        list_backups = True

    # --list: show available backups and exit
    if list_backups:
        backups = _list_backups()
        if not backups:
            console.print("[yellow]No backups found in ~/syne/backup/[/yellow]")
            console.print("[dim]Create one with: syne backup[/dim]")
            return
        console.print(f"[bold]Available backups ({len(backups)}):[/bold]")
        for i, (path, size, mtime) in enumerate(backups, 1):
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            label = " [green](latest)[/green]" if i == 1 else ""
            console.print(f"  {i}. {os.path.basename(path)}  {_format_size(size)}  {ts}{label}")
        console.print(f"\n[dim]Restore with: syne restore <filename>[/dim]")
        return

    # Resolve file: explicit path or filename in backup dir
    if not os.path.isabs(file) and not os.path.exists(file):
        candidate = os.path.join(_get_backup_dir(), file)
        if os.path.exists(candidate):
            file = candidate
    file = os.path.expanduser(file)

    if not os.path.exists(file):
        console.print(f"[red]File not found: {file}[/red]")
        console.print("[dim]Use 'syne restore --list' to see available backups.[/dim]")
        return

    # Confirmation
    if not yes:
        if not click.confirm("This will REPLACE all current data. Continue?"):
            return

    console.print(f"[bold]Restoring database from {os.path.basename(file)}...[/bold]")
    success, message = run_restore(file)
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
