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

    console.print("[bold]Pair with Syne Server[/bold]\n")
    gateway_url = click.prompt("Gateway URL (e.g. wss://syne.example.com:8765)")
    pairing_token = click.prompt("Pairing token (from 'syne gateway token' on server)")
    display_name = click.prompt("Display name for this node", default="")

    # Normalize URL
    if not gateway_url.startswith("ws"):
        gateway_url = f"wss://{gateway_url}"

    console.print(f"\n[dim]Connecting to {gateway_url}...[/dim]")

    try:
        config = await pair_with_server(gateway_url, pairing_token, display_name)
        console.print(f"\n[green]Paired successfully![/green]")
        console.print(f"  Node ID: {config['node_id']}")
        console.print(f"  Display name: {config['display_name']}")
        console.print(f"\nRun [bold]syne node cli[/bold] to start chatting.")
    except Exception as e:
        console.print(f"\n[red]Pairing failed: {e}[/red]")


@node.command(name="cli")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def node_cli(debug):
    """Start interactive CLI connected to the Syne server."""
    asyncio.run(_node_cli_async(debug))


async def _node_cli_async(debug: bool):
    import logging
    import os
    import sys

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    from ..node.client import NodeClient, load_node_config
    from ..node.executor import execute_tool

    config = load_node_config()
    if not config:
        console.print("[red]Node not configured. Run 'syne node init' first.[/red]")
        return

    client = NodeClient(config)

    # Set up response display
    def on_response(text: str, done: bool):
        sys.stdout.write(text)
        sys.stdout.flush()
        if done:
            sys.stdout.write("\n")
            sys.stdout.flush()

    client._on_response = on_response
    client._on_tool_request = execute_tool

    try:
        console.print(f"[dim]Connecting to {config['gateway']}...[/dim]")
        await client.connect()
        console.print(f"[green]Connected as {client.display_name}[/green]\n")

        # Start listener in background
        listen_task = asyncio.create_task(client.listen())

        # REPL loop — input() must run in executor to not block asyncio
        loop = asyncio.get_event_loop()
        cwd = os.getcwd()
        prompt = f"{os.path.basename(cwd)} > "

        while True:
            try:
                user_input = await loop.run_in_executor(None, lambda: input(prompt))
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            try:
                await client.send_message(user_input, cwd=cwd)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        listen_task.cancel()

    except ConnectionError as e:
        console.print(f"[red]Connection failed: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.disconnect()


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
def gateway_token():
    """Generate a one-time pairing token for a new node."""
    asyncio.run(_gateway_token_async())


async def _gateway_token_async():
    from ..db.connection import init_db
    from ..gateway.auth import generate_pairing_token, ensure_paired_nodes_table

    from .shared import _get_db_url
    db_url = _get_db_url()
    await init_db(db_url)
    await ensure_paired_nodes_table()

    token = await generate_pairing_token()
    console.print(f"\n[bold]Pairing Token[/bold] (expires in 10 minutes):\n")
    console.print(f"  [bold green]{token}[/bold green]\n")
    console.print("[dim]Use this token with 'syne node init' on the remote machine.[/dim]")

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
    if enable:
        console.print("[dim]Restart Syne for the gateway to start: syne restart[/dim]")

    await close_db()
