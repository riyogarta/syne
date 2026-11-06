"""Config commands — read, write, list, delete config entries from CLI.

Usage:
    syne config                          # list ALL keys
    syne config <prefix>                 # list keys starting with <prefix>
    syne config get <key>                # print single value
    syne config set <key> <value>        # upsert (value is JSON-parsed if possible)
    syne config delete <key>             # remove a key
    syne config unset <key>              # alias for delete
    syne config backup                   # snapshot config only (not the whole DB)
    syne config restore [file]           # restore a config snapshot (no file → list)

Value parsing for `set`:
    "true" / "false"     → bool
    integer text          → int
    float text            → float
    valid JSON            → parsed structure (list/dict/etc.)
    anything else         → string
"""

import asyncio
import json

import click
from rich.table import Table

from . import cli
from .shared import console


def _parse_value(raw: str):
    """Best-effort typed parse of a CLI value argument."""
    if raw is None:
        return None
    s = raw.strip()
    lower = s.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none"):
        return None
    # int?
    try:
        return int(s)
    except ValueError:
        pass
    # float?
    try:
        return float(s)
    except ValueError:
        pass
    # JSON structure?
    if s and s[0] in "[{\"":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    # Fallback: bare string
    return s


def _render_value(v) -> str:
    """Compact rendering for table display."""
    if isinstance(v, (dict, list)):
        text = json.dumps(v, ensure_ascii=False)
        if len(text) > 80:
            text = text[:77] + "..."
        return text
    if isinstance(v, str):
        return v if len(v) <= 80 else v[:77] + "..."
    return str(v)


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """View or modify configuration stored in the DB.

    Usage:
      syne config                    list ALL config keys
      syne config list [prefix]      list keys, optionally filtered by prefix
      syne config get <key>          print single value
      syne config set <key> <value>  upsert (value auto-parsed: bool/int/JSON/str)
      syne config delete <key>       remove a key
      syne config backup             snapshot config only (not the whole DB)
      syne config restore [file]     restore a config snapshot (no file → list)
    """
    if ctx.invoked_subcommand is not None:
        return

    # No subcommand → list all
    _run_list(None)


def _run_list(prefix):
    async def _list():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import list_config

        settings = load_settings()
        await init_db(settings.database_url)
        try:
            entries = await list_config(prefix or None)
        finally:
            await close_db()

        if not entries:
            if prefix:
                console.print(f"[yellow]No config keys match prefix '{prefix}'.[/yellow]")
            else:
                console.print("[yellow]Config table is empty.[/yellow]")
            return

        title = f"Config ({len(entries)} keys)" + (f" — prefix '{prefix}'" if prefix else "")
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Description", style="dim")
        for entry in entries:
            table.add_row(
                entry["key"],
                _render_value(entry["value"]),
                (entry.get("description") or "")[:60],
            )
        console.print(table)

    asyncio.run(_list())


@config.command("list")
@click.argument("prefix", required=False)
def config_list(prefix):
    """List config keys, optionally filtered by prefix."""
    _run_list(prefix)


@config.command("get")
@click.argument("key")
def config_get(key):
    """Print the value of a config key."""
    async def _get():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import get_config

        settings = load_settings()
        await init_db(settings.database_url)
        try:
            value = await get_config(key, default=_MISSING)
        finally:
            await close_db()

        if value is _MISSING:
            console.print(f"[yellow]Key '{key}' not found.[/yellow]")
            raise click.exceptions.Exit(1)
        # Pretty-print structured values, raw print scalars.
        if isinstance(value, (dict, list)):
            console.print_json(json.dumps(value, ensure_ascii=False))
        else:
            console.print(f"[bold]{key}[/bold] = {value!r}")

    asyncio.run(_get())


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--description", "-d", default=None,
              help="Optional description stored alongside the value.")
def config_set(key, value, description):
    """Set a config key. Value is auto-parsed (bool/int/float/JSON/string)."""
    parsed = _parse_value(value)

    async def _set():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import set_config

        settings = load_settings()
        await init_db(settings.database_url)
        try:
            await set_config(key, parsed, description=description)
        finally:
            await close_db()
        console.print(
            f"[green]✓[/green] {key} = "
            f"[bold]{_render_value(parsed)}[/bold] "
            f"[dim]({type(parsed).__name__})[/dim]"
        )

    asyncio.run(_set())


@config.command("delete")
@click.argument("key")
def config_delete(key):
    """Remove a config key."""
    async def _delete():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.models import delete_config

        settings = load_settings()
        await init_db(settings.database_url)
        try:
            removed = await delete_config(key)
        finally:
            await close_db()

        if removed:
            console.print(f"[green]✓[/green] Deleted '{key}'.")
        else:
            console.print(f"[yellow]Key '{key}' was not present.[/yellow]")
            raise click.exceptions.Exit(1)

    asyncio.run(_delete())


# Alias: `syne config unset foo` == `syne config delete foo`
@config.command("unset", hidden=True)
@click.argument("key")
@click.pass_context
def config_unset(ctx, key):
    """Alias for `delete`."""
    ctx.invoke(config_delete, key=key)


@config.command("backup")
def config_backup():
    """Snapshot the config table and per-user model overrides.

    Unlike `syne backup`, this saves only config, so restoring it does not
    roll back memories or chat history.
    """
    async def _backup():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.config_snapshot import create_snapshot

        await init_db(load_settings().database_url)
        try:
            path = await create_snapshot("manual")
        finally:
            await close_db()
        console.print(f"[green]✓[/green] Config snapshot saved: {path}")
        console.print("[dim]Contains credentials — keep it private.[/dim]")

    asyncio.run(_backup())


@config.command("restore")
@click.argument("file", required=False, default=None)
@click.option("--prefix", "-p", default=None,
              help="Restore only keys starting with this prefix (e.g. 'provider.'). "
                   "User model overrides are then left alone.")
@click.option("--no-users", is_flag=True,
              help="Leave per-user model overrides as they are.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_restore(file, prefix, no_users, yes):
    """Restore config from a snapshot. Without FILE, list snapshots.

    Keys that exist now but not in the snapshot are left alone. Restart
    Syne afterwards (`syne restart`) so the running service picks it up.
    """
    import os
    from datetime import datetime
    from syne.db.config_snapshot import list_snapshots, load_snapshot, snapshot_dir

    if file is None:
        snaps = list_snapshots()
        if not snaps:
            console.print(f"[yellow]No config snapshots in {snapshot_dir()}[/yellow]")
            console.print("[dim]Create one with: syne config backup[/dim]")
            return
        console.print(f"[bold]Config snapshots ({len(snaps)}), newest first:[/bold]")
        for i, (path, mtime) in enumerate(snaps, 1):
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"  {i}. {os.path.basename(path)}  {ts}")
        console.print("\n[dim]Restore with: syne config restore <filename>[/dim]")
        return

    if not os.path.exists(file):
        candidate = os.path.join(snapshot_dir(), file)
        if os.path.exists(candidate):
            file = candidate
    if not os.path.exists(file):
        console.print(f"[red]File not found: {file}[/red]")
        raise click.exceptions.Exit(1)

    try:
        data = load_snapshot(file)
    except (ValueError, json.JSONDecodeError) as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(1)

    if prefix:
        no_users = True

    async def _restore():
        from syne.config import load_settings
        from syne.db.connection import init_db, close_db
        from syne.db.config_snapshot import create_snapshot, diff_snapshot, restore_snapshot

        await init_db(load_settings().database_url)
        try:
            diff = await diff_snapshot(data, prefix)
            if no_users:
                diff["users"] = []
            if not (diff["changed"] or diff["added"] or diff["users"]):
                console.print("[green]Config already matches the snapshot. Nothing to do.[/green]")
                return

            console.print(f"[bold]Snapshot:[/bold] {os.path.basename(file)} ({data.get('created_at', '?')})")
            for k in diff["changed"]:
                console.print(f"  [yellow]~[/yellow] {k}")
            for k in diff["added"]:
                console.print(f"  [green]+[/green] {k}")
            for platform, pid, now, then in diff["users"]:
                console.print(f"  [cyan]user[/cyan] {platform}:{pid}  model {now or '-'} → {then or '-'}")
            if diff["extra"]:
                console.print(f"[dim]  {len(diff['extra'])} key(s) not in the snapshot will be left as they are.[/dim]")

            if not yes and not click.confirm("Apply these changes?"):
                return

            safety = await create_snapshot("pre-restore")
            result = await restore_snapshot(data, prefix, users=not no_users)
        finally:
            await close_db()

        console.print(f"[green]✓[/green] Restored {result['keys']} key(s)"
                      + ("" if no_users else f" and {result['users']} user model override(s)") + ".")
        console.print(f"[dim]Previous config saved as {os.path.basename(safety)}[/dim]")
        console.print("Run [bold]syne restart[/bold] to apply.")

    asyncio.run(_restore())


# Sentinel — distinguishes "missing" from a legitimate stored None value.
_MISSING = object()
