"""WebSocket protocol message definitions for Syne Gateway ↔ Node communication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# --- Node → Gateway messages ---

@dataclass
class ConnectMsg:
    """Node sends this to authenticate on connection."""
    token: str
    node_id: str
    platform: str = "linux"
    cwd: str = ""
    type: str = field(default="connect", init=False)


@dataclass
class ChatMsg:
    """User typed a message in the node CLI."""
    text: str
    cwd: str = ""
    type: str = field(default="message", init=False)


@dataclass
class ToolResultMsg:
    """Node finished executing a tool locally."""
    request_id: str
    result: str
    success: bool = True
    type: str = field(default="tool_result", init=False)


# --- Gateway → Node messages ---

@dataclass
class ConnectedMsg:
    """Acknowledge successful connection."""
    session_id: int
    display_name: str = ""
    type: str = field(default="connected", init=False)


@dataclass
class ResponseChunkMsg:
    """Streaming text chunk from LLM."""
    text: str
    done: bool = False
    type: str = field(default="response_chunk", init=False)


@dataclass
class ToolRequestMsg:
    """Gateway asks node to execute a tool locally."""
    request_id: str
    tool: str
    args: dict = field(default_factory=dict)
    type: str = field(default="tool_request", init=False)


@dataclass
class ErrorMsg:
    """Error message."""
    message: str
    code: str = ""
    type: str = field(default="error", init=False)


# --- Serialization helpers ---

def encode(msg) -> str:
    """Serialize a protocol message to JSON string."""
    return json.dumps(asdict(msg))


def decode(raw: str | bytes) -> dict[str, Any]:
    """Parse a JSON message from WebSocket. Returns raw dict."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


# Node-side tool names — these execute on the node, not the server
NODE_TOOLS = frozenset({"exec", "file_read", "file_write", "read_source"})
