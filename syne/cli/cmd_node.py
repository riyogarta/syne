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

        # Set up and start the background daemon service
        console.print("\n[bold]Setting up background service...[/bold]")
        from .helpers import _setup_node_service
        _setup_node_service()

        console.print(f"\nThe node daemon is running in the background.")
        console.print(f"You can also use [bold]syne node cli[/bold] for interactive chat.")
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


@node.command(name="run")
def node_run():
    """Run the node daemon (maintains persistent gateway connection).

    This is the long-running process that keeps the node connected to the
    gateway so it can receive remote commands from Telegram. Normally managed
    by the syne-node systemd service — you don't need to run this manually.
    """
    asyncio.run(_node_run_async())


async def _node_run_async():
    import signal

    from ..node.client import NodeClient, load_node_config
    from ..node.executor import execute_tool

    config = load_node_config()
    if not config:
        console.print("[red]Node not configured. Run 'syne node init' first.[/red]")
        return

    client = NodeClient(config)

    # Wire up tool executor
    client._on_tool_request = execute_tool

    # Graceful shutdown
    shutdown = asyncio.Event()

    def _signal_handler():
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    console.print(f"[bold]Syne Node Daemon[/bold]")
    console.print(f"  Node: {config.get('display_name', '?')}")
    console.print(f"  Gateway: {config.get('gateway', '?')}")

    retry_delay = 5
    max_retry_delay = 60

    while not shutdown.is_set():
        try:
            console.print(f"[dim]Connecting to gateway...[/dim]")
            await client.connect()
            console.print(f"[green]Connected to gateway[/green]")
            retry_delay = 5  # Reset on successful connect

            # Listen until disconnected or shutdown
            listen_task = asyncio.create_task(client.listen())
            shutdown_task = asyncio.create_task(shutdown.wait())

            done, pending = await asyncio.wait(
                [listen_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            if shutdown.is_set():
                break

            console.print("[yellow]Disconnected from gateway[/yellow]")

        except ConnectionError as e:
            console.print(f"[yellow]Connection failed: {e}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        if not shutdown.is_set():
            console.print(f"[dim]Reconnecting in {retry_delay}s...[/dim]")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=retry_delay)
            except asyncio.TimeoutError:
                pass
            retry_delay = min(retry_delay * 2, max_retry_delay)

    # Clean disconnect
    try:
        await client.disconnect()
    except Exception:
        pass
    console.print("[dim]Node daemon stopped.[/dim]")


@node.command(name="status")
def node_status():
    """Show node connection status."""
    import subprocess as sp
    from ..node.client import load_node_config

    config = load_node_config()
    if not config:
        console.print("[dim]Not configured as a remote node.[/dim]")
        return

    console.print(f"[bold]Node Status[/bold]")
    console.print(f"  Node ID: {config.get('node_id', '?')}")
    console.print(f"  Display name: {config.get('display_name', '?')}")
    console.print(f"  Gateway: {config.get('gateway', '?')}")

    # Check systemd service status
    result = sp.run(
        ["systemctl", "--user", "is-active", "syne-node"],
        capture_output=True, text=True,
    )
    service_status = result.stdout.strip()
    if service_status == "active":
        console.print(f"  Service: [green]running[/green]")
    elif service_status == "inactive":
        console.print(f"  Service: [yellow]stopped[/yellow]")
    else:
        console.print(f"  Service: [dim]{service_status or 'not installed'}[/dim]")


@node.command(name="stop")
def node_stop():
    """Stop the node daemon service."""
    import subprocess as sp

    result = sp.run(
        ["systemctl", "--user", "stop", "syne-node"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]Node daemon stopped.[/green]")
    else:
        console.print(f"[red]Failed to stop: {result.stderr.strip()}[/red]")


@node.command(name="start")
def node_start():
    """Start the node daemon service (auto-setup if not installed yet)."""
    import subprocess as sp
    from pathlib import Path

    from ..node.client import load_node_config
    config = load_node_config()
    if not config:
        console.print("[red]Node not configured. Run 'syne node init' first.[/red]")
        return

    # Auto-setup service if not installed yet
    service_file = Path.home() / ".config" / "systemd" / "user" / "syne-node.service"
    if not service_file.exists():
        console.print("[dim]Service not installed yet, setting up...[/dim]")
        from .helpers import _setup_node_service
        _setup_node_service()
        return

    result = sp.run(
        ["systemctl", "--user", "start", "syne-node"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]Node daemon started.[/green]")
    else:
        console.print(f"[red]Failed to start: {result.stderr.strip()}[/red]")
        console.print("[dim]Try: systemctl --user status syne-node[/dim]")


@node.command(name="restart")
def node_restart():
    """Restart the node daemon service."""
    import subprocess as sp
    from pathlib import Path

    from ..node.client import load_node_config
    config = load_node_config()
    if not config:
        console.print("[red]Node not configured. Run 'syne node init' first.[/red]")
        return

    # Auto-setup service if not installed yet
    service_file = Path.home() / ".config" / "systemd" / "user" / "syne-node.service"
    if not service_file.exists():
        console.print("[dim]Service not installed yet, setting up...[/dim]")
        from .helpers import _setup_node_service
        _setup_node_service()
        return

    result = sp.run(
        ["systemctl", "--user", "restart", "syne-node"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]Node daemon restarted.[/green]")
    else:
        console.print(f"[red]Failed to restart: {result.stderr.strip()}[/red]")
        console.print("[dim]Try: systemctl --user status syne-node[/dim]")


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
