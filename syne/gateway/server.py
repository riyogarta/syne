"""Syne Gateway — WebSocket server for remote node connections.

Embeds into the main Syne process as an asyncio task.
Handles node authentication, message routing, and tool execution bridging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import websockets
from websockets.asyncio.server import ServerConnection

from . import auth
from .protocol import (
    NODE_TOOLS,
    PROTOCOL_VERSION,
    ConnectedMsg,
    ErrorMsg,
    MetaMsg,
    ResponseChunkMsg,
    ToolRequestMsg,
    decode,
    encode,
)

if TYPE_CHECKING:
    from ..agent import SyneAgent

logger = logging.getLogger("syne.gateway")

# Default port — 8443 is Cloudflare-compatible for proxied WSS
DEFAULT_PORT = 8443


@dataclass
class NodeConnection:
    """Represents a connected remote node."""

    ws: ServerConnection
    node_id: str
    display_name: str
    platform: str = "linux"
    cwd: str = ""
    # Pending tool requests: request_id → Future
    _pending_tools: dict[str, asyncio.Future] = field(default_factory=dict)

    async def send(self, msg) -> None:
        """Send a protocol message to the node."""
        await self.ws.send(encode(msg))

    async def request_tool(self, tool_name: str, args: dict, timeout: float = 120) -> dict:
        """Send a tool execution request to node and wait for result.

        Args:
            tool_name: Name of the tool to execute on node.
            args: Tool arguments.
            timeout: Max seconds to wait for result.

        Returns:
            dict with 'result' (str) and 'success' (bool).

        Raises:
            asyncio.TimeoutError: If node doesn't respond in time.
            ConnectionError: If node disconnects.
        """
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_tools[request_id] = future

        try:
            await self.send(ToolRequestMsg(
                request_id=request_id,
                tool=tool_name,
                args=args,
            ))
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool request timeout: {tool_name} on node {self.node_id}")
            raise
        finally:
            self._pending_tools.pop(request_id, None)

    def resolve_tool(self, request_id: str, result: str, success: bool) -> None:
        """Resolve a pending tool request future."""
        future = self._pending_tools.get(request_id)
        if future and not future.done():
            future.set_result({"result": result, "success": success})


class Gateway:
    """WebSocket gateway server for remote node connections."""

    def __init__(self, agent: SyneAgent, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.agent = agent
        self.host = host
        self.port = port
        self._server = None
        self._nodes: dict[str, NodeConnection] = {}  # node_id → NodeConnection

    async def start(self) -> None:
        """Start the WebSocket server (with TLS if certs are available)."""
        await auth.ensure_paired_nodes_table()

        ssl_context = self._load_ssl_context()
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=10,
        )
        proto = "wss" if ssl_context else "ws"
        logger.info(f"Gateway WebSocket server started on {proto}://{self.host}:{self.port}")

    @staticmethod
    def _load_ssl_context() -> Optional[ssl.SSLContext]:
        """Load TLS certificates if available. Returns None for plain WS."""
        # Check common cert locations
        cert_paths = [
            ("/etc/ssl/cf-origin.pem", "/etc/ssl/cf-origin-key.pem"),
            (os.path.expanduser("~/.syne/gateway-cert.pem"), os.path.expanduser("~/.syne/gateway-key.pem")),
        ]
        for cert_file, key_file in cert_paths:
            if os.path.exists(cert_file) and os.path.exists(key_file):
                try:
                    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    ctx.load_cert_chain(cert_file, key_file)
                    logger.info(f"TLS enabled: {cert_file}")
                    return ctx
                except Exception as e:
                    logger.warning(f"Failed to load TLS cert {cert_file}: {e}")
        return None

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Gateway WebSocket server stopped")

    def get_node(self, name_or_id: str) -> Optional[NodeConnection]:
        """Get a connected node by ID or display_name alias."""
        # Try exact node_id first
        node = self._nodes.get(name_or_id)
        if node:
            return node
        # Try alias (display_name)
        for conn in self._nodes.values():
            if conn.display_name == name_or_id:
                return conn
        return None

    async def _send_meta(self, node: NodeConnection) -> None:
        """Send server metadata to a node (for CLI header display)."""
        try:
            from ..db.models import get_identity, get_config
            identity = await get_identity()
            models = await get_config("provider.models", [])
            active_key = await get_config("provider.active_model", "")
            tool_count = len(self.agent.tools.list_tools("owner")) if self.agent.tools else 0

            # Check for per-node model override
            node_record = await auth.get_node(node.node_id)
            node_model_key = (node_record or {}).get("model", "")
            effective_key = node_model_key or active_key
            active_entry = next((m for m in models if m.get("key") == effective_key), {})
            model_id = active_entry.get("model_id", effective_key)
            reasoning_visible = active_entry.get("reasoning_visible", False)

            await node.send(MetaMsg(
                agent_name=identity.get("name", "Syne"),
                motto=identity.get("motto", ""),
                model=model_id,
                tool_count=tool_count,
                reasoning_visible=reasoning_visible,
            ))
        except Exception as e:
            logger.warning(f"Failed to send meta to {node.node_id}: {e}")

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a new WebSocket connection."""
        node: Optional[NodeConnection] = None

        try:
            # First message must be 'connect' with auth
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            msg = decode(raw)

            if msg.get("type") == "pair":
                # Pairing flow
                await self._handle_pairing(ws, msg)
                return

            if msg.get("type") != "connect":
                await ws.send(encode(ErrorMsg(message="Expected 'connect' message", code="auth_required")))
                return

            # Verify permanent token
            node_id = msg.get("node_id", "")
            token = msg.get("token", "")
            if not node_id or not token:
                await ws.send(encode(ErrorMsg(message="Missing node_id or token", code="auth_invalid")))
                return

            if not await auth.verify_node_token(node_id, token):
                await ws.send(encode(ErrorMsg(message="Invalid credentials", code="auth_failed")))
                return

            # Check protocol version
            client_version = msg.get("protocol_version", 0)
            if client_version != PROTOCOL_VERSION:
                logger.warning(
                    f"Protocol version mismatch: server={PROTOCOL_VERSION}, "
                    f"node={client_version} ({node_id}). Node may need updating."
                )

            # Authenticated — register connection
            node = NodeConnection(
                ws=ws,
                node_id=node_id,
                display_name=msg.get("display_name", node_id),
                platform=msg.get("platform", "linux"),
                cwd=msg.get("cwd", ""),
            )
            self._nodes[node_id] = node
            logger.info(f"Node connected: {node_id} ({node.display_name})")

            # Send connected acknowledgement
            await node.send(ConnectedMsg(session_id=0, display_name=node.display_name))

            # Send server metadata for CLI header
            await self._send_meta(node)

            # Message loop
            async for raw_msg in ws:
                try:
                    msg = decode(raw_msg)
                    await self._handle_node_message(node, msg)
                except json.JSONDecodeError:
                    await node.send(ErrorMsg(message="Invalid JSON", code="parse_error"))
                except Exception as e:
                    logger.error(f"Error handling message from {node_id}: {e}", exc_info=True)
                    await node.send(ErrorMsg(message=str(e), code="internal_error"))

        except asyncio.TimeoutError:
            try:
                await ws.send(encode(ErrorMsg(message="Auth timeout", code="timeout")))
            except Exception:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            if node:
                self._nodes.pop(node.node_id, None)
                # Cancel pending tool requests
                for fut in list(node._pending_tools.values()):
                    if not fut.done():
                        fut.set_exception(ConnectionError("Node disconnected"))
                logger.info(f"Node disconnected: {node.node_id}")

    async def _handle_pairing(self, ws: ServerConnection, msg: dict) -> None:
        """Handle pairing request from a new node."""
        pairing_token = msg.get("token", "")
        if not pairing_token:
            await ws.send(encode(ErrorMsg(message="Missing pairing token", code="pair_invalid")))
            return

        valid, node_name = await auth.verify_pairing_token(pairing_token)
        if not valid:
            await ws.send(encode(ErrorMsg(message="Invalid or expired pairing token", code="pair_failed")))
            return

        # Token valid — register the node
        # node_name from token takes priority (set by admin via `syne gateway token <name>`)
        # Falls back to client-provided display_name or "node"
        display_name = node_name or msg.get("display_name", "node")
        node_id = f"{display_name}-{uuid.uuid4().hex[:8]}"
        platform = msg.get("platform", "linux")

        permanent_token = await auth.register_node(node_id, display_name, platform)

        # Send back the permanent token
        await ws.send(json.dumps({
            "type": "paired",
            "node_id": node_id,
            "token": permanent_token,
            "display_name": display_name,
        }))
        logger.info(f"Node paired: {node_id} ({display_name})")

    async def _handle_node_message(self, node: NodeConnection, msg: dict) -> None:
        """Route an incoming message from a connected node.

        IMPORTANT: 'message' (chat) is dispatched as a background task so the
        message loop stays free to receive 'tool_result' messages.  Without
        this, chat processing that invokes node-side tools would deadlock:
        the server awaits the tool result, but the result can't be received
        because the message loop is blocked waiting for chat to finish.
        """
        msg_type = msg.get("type", "")

        if msg_type == "disconnect":
            logger.info(f"Node {node.node_id} sent graceful disconnect")
            await node.ws.close()
        elif msg_type == "message":
            # Run in background so message loop can still receive tool_result
            asyncio.create_task(self._handle_chat(node, msg))
        elif msg_type == "tool_result":
            node.resolve_tool(
                msg.get("request_id", ""),
                msg.get("result", ""),
                msg.get("success", True),
            )
        else:
            await node.send(ErrorMsg(message=f"Unknown message type: {msg_type}", code="unknown_type"))

    async def _handle_chat(self, node: NodeConnection, msg: dict) -> None:
        """Handle a chat message from the node — forward to conversation system."""
        text = msg.get("text", "").strip()
        if not text:
            return

        cwd = msg.get("cwd", node.cwd) or ""
        node.cwd = cwd

        # Handle slash commands server-side (same as local CLI)
        if text.startswith("/"):
            handled = await self._handle_slash_command(node, text, cwd)
            if handled:
                return

        try:
            from .conversation_remote import handle_remote_message
            await handle_remote_message(
                agent=self.agent,
                node=node,
                message=text,
                cwd=cwd,
            )

            # Content was already streamed via callbacks; send completion signal
            await node.send(ResponseChunkMsg(text="", done=True))
        except Exception as e:
            logger.error(f"Chat error for node {node.node_id}: {e}", exc_info=True)
            from ..communication.errors import classify_error
            await node.send(ErrorMsg(message=classify_error(e), code="chat_error"))

    async def _handle_slash_command(self, node: NodeConnection, cmd: str, cwd: str) -> bool:
        """Handle slash commands from the node. Returns True if handled."""
        from .conversation_remote import _make_chat_id

        cmd_name = cmd.split()[0].lower()
        chat_id = _make_chat_id(node.node_id, cwd)
        session_key = f"node:{chat_id}"

        if cmd_name == "/compact":
            conv = self.agent.conversations._active.get(session_key)
            if not conv:
                await node.send(ResponseChunkMsg(text="No active conversation.", done=False))
                await node.send(ResponseChunkMsg(text="", done=True))
                return True
            msg_count = len(conv._message_cache) if conv._message_cache else 0
            if msg_count < 4:
                await node.send(ResponseChunkMsg(text="Conversation too short to compact.", done=False))
                await node.send(ResponseChunkMsg(text="", done=True))
                return True
            await node.send(ResponseChunkMsg(text="Compacting conversation...", done=False))
            try:
                from ..compaction import compact_session
                result = await compact_session(session_id=conv.session_id, provider=conv.provider)
                if result:
                    await conv.load_history()
                    await node.send(ResponseChunkMsg(
                        text=f"Compacted: {result['messages_before']} → {result['messages_after']} messages",
                        done=False,
                    ))
                else:
                    await node.send(ResponseChunkMsg(text="Nothing to compact.", done=False))
            except Exception as e:
                logger.error(f"Compact error for node {node.node_id}: {e}", exc_info=True)
                await node.send(ResponseChunkMsg(text=f"Compact failed: {e}", done=False))
            await node.send(ResponseChunkMsg(text="", done=True))
            return True

        if cmd_name == "/new":
            # Close session in DB + clear cache (same as local CLI fresh start)
            from ..db.connection import get_connection
            conv = self.agent.conversations._active.pop(session_key, None)
            async with get_connection() as conn:
                result = await conn.fetch("""
                    UPDATE sessions SET status = 'closed'
                    WHERE platform = 'node' AND platform_chat_id = $1 AND status = 'active'
                    RETURNING id
                """, chat_id)
                if result:
                    session_ids = [r["id"] for r in result]
                    await conn.execute("DELETE FROM messages WHERE session_id = ANY($1::int[])", session_ids)
            await node.send(ResponseChunkMsg(text="Session cleared. Starting fresh.", done=False))
            await node.send(ResponseChunkMsg(text="", done=True))
            return True

        return False
