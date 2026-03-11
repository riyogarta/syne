"""Tests for syne.gateway.server — NodeConnection, Gateway routing."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from syne.gateway.server import NodeConnection, Gateway
from syne.gateway.protocol import (
    ConnectedMsg,
    ErrorMsg,
    ResponseChunkMsg,
    ToolRequestMsg,
    encode,
    decode,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_ws():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


def _make_node(node_id="node-1", display_name="laptop", ws=None):
    """Create a NodeConnection with a mock WS."""
    return NodeConnection(
        ws=ws or _make_mock_ws(),
        node_id=node_id,
        display_name=display_name,
        platform="linux",
        cwd="/home/user",
    )


# ── NodeConnection ───────────────────────────────────────────────────


class TestNodeConnection:
    def test_fields(self):
        node = _make_node()
        assert node.node_id == "node-1"
        assert node.display_name == "laptop"
        assert node.platform == "linux"
        assert node._pending_tools == {}

    @pytest.mark.asyncio
    async def test_send(self):
        ws = _make_mock_ws()
        node = _make_node(ws=ws)
        msg = ResponseChunkMsg(text="hello", done=False)
        await node.send(msg)
        ws.send.assert_called_once()
        sent_data = json.loads(ws.send.call_args[0][0])
        assert sent_data["type"] == "response_chunk"
        assert sent_data["text"] == "hello"

    def test_resolve_tool_success(self):
        node = _make_node()
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        node._pending_tools["r1"] = future

        node.resolve_tool("r1", "output data", True)
        assert future.done()
        assert future.result() == {"result": "output data", "success": True}
        loop.close()

    def test_resolve_tool_nonexistent_request(self):
        """Resolving a non-existent request should not raise."""
        node = _make_node()
        node.resolve_tool("nonexistent", "data", True)  # Should not raise

    def test_resolve_tool_already_done(self):
        """Resolving an already-done future should not raise."""
        node = _make_node()
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result({"result": "first", "success": True})
        node._pending_tools["r1"] = future

        node.resolve_tool("r1", "second", True)  # Should not raise
        assert future.result()["result"] == "first"  # Unchanged
        loop.close()

    @pytest.mark.asyncio
    async def test_request_tool_sends_and_resolves(self):
        ws = _make_mock_ws()
        node = _make_node(ws=ws)

        async def _simulate_response():
            await asyncio.sleep(0.01)
            # Find the pending request_id and resolve it
            for req_id in list(node._pending_tools.keys()):
                node.resolve_tool(req_id, "ls output", True)

        task = asyncio.create_task(_simulate_response())
        result = await node.request_tool("exec", {"command": "ls"}, timeout=5)

        assert result["result"] == "ls output"
        assert result["success"] is True
        ws.send.assert_called_once()
        await task

    @pytest.mark.asyncio
    async def test_request_tool_timeout(self):
        ws = _make_mock_ws()
        node = _make_node(ws=ws)

        with pytest.raises(asyncio.TimeoutError):
            await node.request_tool("exec", {"command": "sleep 999"}, timeout=0.01)

        # Pending tools should be cleaned up
        assert len(node._pending_tools) == 0


# ── Gateway ──────────────────────────────────────────────────────────


class TestGateway:
    def test_init(self):
        agent = MagicMock()
        gw = Gateway(agent, host="0.0.0.0", port=9999)
        assert gw.host == "0.0.0.0"
        assert gw.port == 9999
        assert gw._nodes == {}

    def test_get_node_by_id(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node(node_id="abc-123")
        gw._nodes["abc-123"] = node
        assert gw.get_node("abc-123") is node

    def test_get_node_by_display_name(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node(node_id="abc-123", display_name="mypc")
        gw._nodes["abc-123"] = node
        assert gw.get_node("mypc") is node

    def test_get_node_not_found(self):
        agent = MagicMock()
        gw = Gateway(agent)
        assert gw.get_node("nonexistent") is None

    def test_get_node_prefers_exact_id(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node1 = _make_node(node_id="laptop", display_name="other")
        node2 = _make_node(node_id="xyz", display_name="laptop")
        gw._nodes["laptop"] = node1
        gw._nodes["xyz"] = node2
        # Exact ID match should win
        assert gw.get_node("laptop") is node1


class TestHandleNodeMessage:
    @pytest.mark.asyncio
    async def test_disconnect_message(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node()

        await gw._handle_node_message(node, {"type": "disconnect"})
        node.ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_result_message(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        node._pending_tools["r1"] = future

        await gw._handle_node_message(node, {
            "type": "tool_result",
            "request_id": "r1",
            "result": "ok",
            "success": True,
        })

        assert future.done()
        assert future.result() == {"result": "ok", "success": True}

    @pytest.mark.asyncio
    async def test_unknown_message_sends_error(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node()

        await gw._handle_node_message(node, {"type": "foobar"})
        node.ws.send.assert_called_once()
        sent = json.loads(node.ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "unknown" in sent["message"].lower()

    @pytest.mark.asyncio
    async def test_chat_message_dispatches_background_task(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node()

        with patch.object(gw, "_handle_chat", new_callable=AsyncMock) as mock_chat:
            await gw._handle_node_message(node, {"type": "message", "text": "hello"})
            # Give the background task time to start
            await asyncio.sleep(0.05)
            mock_chat.assert_called_once()


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_unknown_slash_not_handled(self):
        agent = MagicMock()
        gw = Gateway(agent)
        node = _make_node()

        result = await gw._handle_slash_command(node, "/unknown", "/home")
        assert result is False

    @pytest.mark.asyncio
    async def test_compact_no_active_conversation(self):
        agent = MagicMock()
        agent.conversations = MagicMock()
        agent.conversations._active = {}
        gw = Gateway(agent)
        node = _make_node()

        result = await gw._handle_slash_command(node, "/compact", "/home")
        assert result is True
        # Should send "No active conversation" + done
        assert node.ws.send.call_count >= 1

    @pytest.mark.asyncio
    @patch("syne.db.connection.get_connection")
    async def test_new_closes_session(self, mock_get_conn):
        conn = AsyncMock()
        conn.fetch.return_value = [{"id": 1}]
        conn.execute.return_value = None
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        agent = MagicMock()
        agent.conversations = MagicMock()
        agent.conversations._active = {}
        gw = Gateway(agent)
        node = _make_node()

        result = await gw._handle_slash_command(node, "/new", "/home")
        assert result is True
