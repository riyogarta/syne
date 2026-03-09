"""Remote conversation handler — bridges node messages to the Syne conversation system.

This module does NOT modify conversation.py. Instead, it:
1. Creates a user/session for the node (platform='node')
2. Injects a tool execution interceptor that routes node-side tools over WebSocket
3. Wires streaming callbacks so LLM output is sent to the node in real time
4. Uses the existing ConversationManager.handle_message() for everything else
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import TYPE_CHECKING

from ..db.models import get_or_create_user
from ..communication.inbound import InboundContext
from ..llm.provider import StreamCallbacks
from .protocol import NODE_TOOLS, ResponseChunkMsg, ThinkingChunkMsg, ToolActivityMsg

if TYPE_CHECKING:
    from ..agent import SyneAgent
    from .server import NodeConnection

logger = logging.getLogger("syne.gateway.remote")


def _make_chat_id(node_id: str, cwd: str) -> str:
    """Build a per-directory chat ID for node sessions.

    Format: node:<node_id>:<cwd_hash_8chars>
    Different directories = different sessions (like syne cli).
    """
    cwd_hash = hashlib.sha256(cwd.encode()).hexdigest()[:8]
    return f"node:{node_id}:{cwd_hash}"


async def _get_node_user(node_id: str, display_name: str) -> dict:
    """Get or create the user record for a remote node. Always owner access."""
    user = await get_or_create_user(
        name=display_name,
        platform="node",
        platform_id=f"node:{node_id}",
        display_name=display_name,
        is_dm=True,
    )

    # Ensure node user has owner access
    if user.get("access_level") != "owner":
        from ..db.models import update_user
        await update_user("node", f"node:{node_id}", access_level="owner")
        user = dict(user)
        user["access_level"] = "owner"

    return user


async def handle_remote_message(
    agent: SyneAgent,
    node: "NodeConnection",
    message: str,
    cwd: str,
) -> str:
    """Handle a chat message from a remote node.

    This is the main entry point from gateway/server.py.
    Routes the message through the standard conversation system,
    with node-side tool execution intercepted via the node's WebSocket.

    Args:
        agent: The SyneAgent instance.
        node: The connected NodeConnection.
        message: User's message text.
        cwd: Current working directory on the node.

    Returns:
        Agent response string.
    """
    if not agent.conversations:
        return "Agent not ready — conversation manager not initialized."

    user = await _get_node_user(node.node_id, node.display_name)
    chat_id = _make_chat_id(node.node_id, cwd)

    # Build inbound context
    inbound = InboundContext(
        channel="node",
        platform="node",
        chat_type="direct",
        conversation_label=node.display_name,
        chat_id=chat_id,
    )

    # Set up the node tool interceptor on the conversation manager.
    # This tells _execute_single_tool to route NODE_TOOLS to the node.
    # We store it on the manager keyed by chat_id so concurrent nodes don't clash.
    agent.conversations._node_connections[chat_id] = node

    # --- Wire streaming callbacks so LLM output reaches the node in real time ---
    loop = asyncio.get_running_loop()

    def _on_text(delta: str) -> None:
        """Sync callback — schedule async send on the running loop."""
        loop.call_soon_threadsafe(
            loop.create_task,
            node.send(ResponseChunkMsg(text=delta, done=False)),
        )

    def _on_thinking(delta: str) -> None:
        loop.call_soon_threadsafe(
            loop.create_task,
            node.send(ThinkingChunkMsg(text=delta)),
        )

    async def _on_tool_detail(name: str, args: dict, result_preview: str) -> None:
        await node.send(ToolActivityMsg(name=name, args=args, result_preview=result_preview))

    stream_cbs = StreamCallbacks(on_text=_on_text, on_thinking=_on_thinking)

    # Save previous callbacks so we can restore them after (manager is shared)
    prev_stream = agent.conversations._stream_callbacks
    prev_tool_detail = agent.conversations._tool_detail_callback

    agent.conversations.set_stream_callbacks(stream_cbs)
    agent.conversations.set_tool_detail_callback(_on_tool_detail)

    try:
        await agent.conversations.handle_message(
            platform="node",
            chat_id=chat_id,
            user=user,
            message=message,
            message_metadata={"cwd": cwd, "inbound": inbound},
        )
        # Response was already streamed via callbacks — return None
        return None
    finally:
        # Restore previous callbacks and clean up node connection
        agent.conversations.set_stream_callbacks(prev_stream)
        agent.conversations.set_tool_detail_callback(prev_tool_detail)
        agent.conversations._node_connections.pop(chat_id, None)
