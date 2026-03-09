"""CLI commands for Syne remote node and gateway management."""

import asyncio
import click

from . import cli
from .shared import console


# ── Node commands (run on remote machine) ──

@cli.group()
def node():
    """Remote node commands (for connecting to a Syne server)."""
    pass


@node.command(name="init")
def node_init():
    """Pair this machine with a Syne server as a remote node."""
    asyncio.run(_node_init_async())


async def _node_init_async():
    from ..node.client import load_node_config, pair_with_server

    existing = load_node_config()
    if existing:
        console.print(f"[yellow]Already paired as '{existing.get('display_name', '?')}'[/yellow]")
        console.print(f"  Gateway: {existing.get('gateway', '?')}")
        console.print(f"  Node ID: {existing.get('node_id', '?')}")
        if not click.confirm("Re-pair with a different server?", default=False):
            return

    console.print("[bold]Remote Node Setup[/bold]")
    console.print("[dim]Connect this machine to your Syne server.[/dim]")
    console.print("[dim]You need a pairing token from the server (run syne gateway token there).[/dim]\n")
    gateway_url = click.prompt("Gateway URL (e.g. wss://syne.example.com:8443)")
    pairing_token = click.prompt("Pairing token")

    # Normalize URL
    if not gateway_url.startswith("ws"):
        gateway_url = f"wss://{gateway_url}"

    if gateway_url.startswith("ws://"):
        console.print("[yellow]WARNING: Using unencrypted ws://. Tokens will be sent in plain text.[/yellow]")
        console.print("[yellow]Use wss:// in production for security.[/yellow]")
        if not click.confirm("Continue anyway?", default=False):
            return

    console.print(f"\n[dim]Connecting to {gateway_url}...[/dim]")

    try:
        config = await pair_with_server(gateway_url, pairing_token)
        console.print(f"\n[green]Paired successfully![/green]")
        console.print(f"  Node ID: {config['node_id']}")
        console.print(f"  Name: {config['display_name']}")
        console.print(f"\nRun [bold]syne node cli[/bold] to start chatting.")
    except ConnectionError as e:
        console.print(f"\n[red]Pairing failed: {e}[/red]")
        console.print("[dim]Check that the gateway is running and the token hasn't expired.[/dim]")
    except Exception as e:
        console.print(f"\n[red]Pairing failed: {e}[/red]")


@node.command(name="cli")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def node_cli(debug):
    """Start interactive CLI connected to the Syne server.

    This is an alias for 'syne cli' in remote mode.
    """
    from ..node.client import NodeClient, load_node_config

    config = load_node_config()
    if not config:
        console.print("[red]Node not configured. Run 'syne node init' first.[/red]")
        return

    from ..channels.cli_channel import run_cli
    client = NodeClient(config)
    asyncio.run(run_cli(debug=debug, node_client=client))


@node.command(name="status")
def node_status():
    """Show node connection status."""
    from ..node.client import load_node_config

    config = load_node_config()
    if not config:
        console.print("[dim]Not configured as a remote node.[/dim]")
        return

    console.print(f"[bold]Node Status[/bold]")
    console.print(f"  Node ID: {config.get('node_id', '?')}")
    console.print(f"  Display name: {config.get('display_name', '?')}")
    console.print(f"  Gateway: {config.get('gateway', '?')}")


# ── Gateway commands (run on server) ──

@cli.group()
def gateway():
    """Gateway management (for the Syne server)."""
    pass


@gateway.command(name="token")
@click.argument("name")
def gateway_token(name):
    """Generate a one-time pairing token for a new node.

    NAME is the alias for the node (e.g. 'mypc', 'laptop').
    This name is used to target the node from Telegram or other channels.
    """
    asyncio.run(_gateway_token_async(name))


async def _gateway_token_async(name: str):
    from ..db.connection import init_db
    from ..gateway.auth import generate_pairing_token, ensure_paired_nodes_table

    from .shared import _get_db_url
    db_url = _get_db_url()
    await init_db(db_url)
    await ensure_paired_nodes_table()

    try:
        token = await generate_pairing_token(node_name=name)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        from ..db.connection import close_db
        await close_db()
        return

    console.print(f"\n[bold]Pairing Token for '{name}'[/bold] (expires in 10 minutes):\n")
    console.print(f"  [bold green]{token}[/bold green]\n")
    console.print(f"[dim]Use this token with 'syne node init' on the remote machine.[/dim]")
    console.print(f"[dim]The node will be accessible as '{name}' from Telegram.[/dim]")

    from ..db.connection import close_db
    await close_db()


@gateway.command(name="list")
def gateway_list():
    """List all paired nodes."""
    asyncio.run(_gateway_list_async())


async def _gateway_list_async():
    from ..db.connection import init_db, close_db
    from ..gateway.auth import list_nodes, ensure_paired_nodes_table

    from .shared import _get_db_url
    db_url = _get_db_url()
    await init_db(db_url)
    await ensure_paired_nodes_table()

    nodes = await list_nodes()
    if not nodes:
        console.print("[dim]No paired nodes.[/dim]")
    else:
        console.print(f"[bold]Paired Nodes ({len(nodes)})[/bold]\n")
        for n in nodes:
            status = "[green]active[/green]" if n["active"] else "[red]revoked[/red]"
            last_seen = n["last_seen"].strftime("%Y-%m-%d %H:%M") if n["last_seen"] else "never"
            console.print(
                f"  {n['display_name']:20s}  {status}  "
                f"  last seen: {last_seen}  "
                f"  id: {n['node_id']}"
            )

    await close_db()


@gateway.command(name="revoke")
@click.argument("node_id")
def gateway_revoke(node_id):
    """Revoke a node's access."""
    asyncio.run(_gateway_revoke_async(node_id))


async def _gateway_revoke_async(node_id: str):
    from ..db.connection import init_db, close_db
    from ..gateway.auth import revoke_node, ensure_paired_nodes_table

    from .shared import _get_db_url
    db_url = _get_db_url()
    await init_db(db_url)
    await ensure_paired_nodes_table()

    if await revoke_node(node_id):
        console.print(f"[green]Node '{node_id}' revoked.[/green]")
    else:
        console.print(f"[red]Node '{node_id}' not found.[/red]")

    await close_db()


@gateway.command(name="enable")
def gateway_enable():
    """Enable the gateway WebSocket server."""
    asyncio.run(_gateway_toggle_async(True))


@gateway.command(name="disable")
def gateway_disable():
    """Disable the gateway WebSocket server."""
    asyncio.run(_gateway_toggle_async(False))


async def _gateway_toggle_async(enable: bool):
    from ..db.connection import init_db, close_db
    from ..db.models import set_config

    from .shared import _get_db_url
    db_url = _get_db_url()
    await init_db(db_url)

    await set_config("gateway.enabled", enable)
    state = "enabled" if enable else "disabled"
    console.print(f"[green]Gateway {state}.[/green]")

    await close_db()

    from .helpers import _restart_service
    console.print("[dim]Restarting Syne...[/dim]")
    _restart_service()
