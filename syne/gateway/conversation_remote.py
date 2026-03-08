"""Remote conversation handler — bridges node messages to the Syne conversation system.

This module does NOT modify conversation.py. Instead, it:
1. Creates a user/session for the node (platform='node')
2. Injects a tool execution interceptor that routes node-side tools over WebSocket
3. Uses the existing ConversationManager.handle_message() for everything else
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from ..db.models import get_or_create_user
from ..communication.inbound import InboundContext
from .protocol import NODE_TOOLS

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

    try:
        response = await agent.conversations.handle_message(
            platform="node",
            chat_id=chat_id,
            user=user,
            message=message,
            message_metadata={"cwd": cwd, "inbound": inbound},
        )
        return response or ""
    finally:
        # Clean up node connection reference (don't leak)
        agent.conversations._node_connections.pop(chat_id, None)
