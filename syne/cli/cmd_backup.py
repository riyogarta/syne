"""Backup and restore commands."""

import os
import subprocess
import sys
from datetime import datetime
import click

from . import cli
from .shared import console, _read_env_value, _get_syne_dir


@cli.command()
@click.option("--output", "-o", default=None, help="Output file path (default: ~/syne-backup-TIMESTAMP.sql.gz)")
def backup(output):
    """Backup Syne database to a compressed SQL file."""
    syne_dir = _get_syne_dir()
    db_user = _read_env_value("SYNE_DB_USER", syne_dir)
    db_name = _read_env_value("SYNE_DB_NAME", syne_dir)

    if not db_user or not db_name:
        console.print("[red]Cannot read DB credentials from .env. Run 'syne init' first.[/red]")
        return

    # Default output path
    if output is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        output = os.path.expanduser(f"~/syne-backup-{timestamp}.sql.gz")

    output = os.path.expanduser(output)

    # Check that syne-db container is running
    check = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or check.stdout.strip() != "running":
        console.print("[red]syne-db container is not running.[/red]")
        console.print("[dim]Start it with: docker start syne-db[/dim]")
        return

    console.print(f"[bold]Backing up database '{db_name}'...[/bold]")

    # pg_dump → gzip → file (streaming via Popen pipes)
    try:
        dump_proc = subprocess.Popen(
            ["docker", "exec", "syne-db", "pg_dump", "-U", db_user, "-d", db_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with open(output, "wb") as f:
            gzip_proc = subprocess.Popen(
                ["gzip"],
                stdin=dump_proc.stdout,
                stdout=f,
                stderr=subprocess.PIPE,
            )
            # Allow dump_proc to receive SIGPIPE if gzip exits
            dump_proc.stdout.close()
            gzip_proc.wait()
            dump_proc.wait()

        if dump_proc.returncode != 0:
            stderr = dump_proc.stderr.read().decode() if dump_proc.stderr else ""
            console.print(f"[red]pg_dump failed (exit {dump_proc.returncode}): {stderr.strip()}[/red]")
            # Clean up partial file
            if os.path.exists(output):
                os.remove(output)
            return

        # Show result
        size = os.path.getsize(output)
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"

        console.print(f"[green]✓ Backup saved: {output} ({size_str})[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Command not found: {e}[/red]")
        console.print("[dim]Ensure 'docker' and 'gzip' are installed.[/dim]")
    except Exception as e:
        console.print(f"[red]Backup failed: {e}[/red]")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.confirmation_option(prompt="This will REPLACE all current data. Continue?")
def restore(file):
    """Restore Syne database from a backup file (.sql or .sql.gz)."""
    from .helpers import _run_schema_migration

    syne_dir = _get_syne_dir()
    db_user = _read_env_value("SYNE_DB_USER", syne_dir)
    db_name = _read_env_value("SYNE_DB_NAME", syne_dir)

    if not db_user or not db_name:
        console.print("[red]Cannot read DB credentials from .env. Run 'syne init' first.[/red]")
        return

    file = os.path.expanduser(file)

    # Check that syne-db container is running
    check = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", "syne-db"],
        capture_output=True, text=True,
    )
    if check.returncode != 0 or check.stdout.strip() != "running":
        console.print("[red]syne-db container is not running.[/red]")
        console.print("[dim]Start it with: docker start syne-db[/dim]")
        return

    console.print(f"[bold]Restoring database from {os.path.basename(file)}...[/bold]")

    # Drop and recreate database to ensure clean restore
    console.print("[dim]Dropping existing data...[/dim]")
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
        is_gzip = file.endswith(".gz")

        if is_gzip:
            # gunzip -c file | docker exec -i syne-db psql
            gunzip_proc = subprocess.Popen(
                ["gunzip", "-c", file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            psql_proc = subprocess.Popen(
                ["docker", "exec", "-i", "syne-db", "psql", "-U", db_user, "-d", db_name, "-q"],
                stdin=gunzip_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            gunzip_proc.stdout.close()
            psql_proc.wait()
            gunzip_proc.wait()
            returncode = psql_proc.returncode
            stderr = psql_proc.stderr.read().decode() if psql_proc.stderr else ""
        else:
            # Plain SQL: pipe directly
            with open(file, "rb") as f:
                result = subprocess.run(
                    ["docker", "exec", "-i", "syne-db", "psql", "-U", db_user, "-d", db_name, "-q"],
                    stdin=f,
                    capture_output=True,
                )
                returncode = result.returncode
                stderr = result.stderr.decode() if result.stderr else ""

        if returncode != 0:
            console.print(f"[yellow]Restore completed with warnings: {stderr.strip()[:200]}[/yellow]")
        else:
            console.print("[green]✓ Database restored successfully[/green]")

        # Run schema migration to ensure compatibility with current version
        console.print("[dim]Running schema migration for compatibility...[/dim]")
        _run_schema_migration(syne_dir)

    except FileNotFoundError as e:
        console.print(f"[red]Command not found: {e}[/red]")
        console.print("[dim]Ensure 'docker' and 'gunzip' are installed.[/dim]")
    except Exception as e:
        console.print(f"[red]Restore failed: {e}[/red]")
