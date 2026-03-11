"""Tests for syne.node.client — NodeClient, config loading, pairing."""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from syne.node.client import (
    load_node_config,
    save_node_config,
    is_node_mode,
    NodeClient,
)


# ── Config file operations ───────────────────────────────────────────


class TestNodeConfig:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "node.json")
            with patch("syne.node.client.NODE_CONFIG_PATH", path):
                config = {"node_id": "n1", "token": "t1", "gateway": "ws://host:8443"}
                save_node_config(config)

                # Check file permissions
                mode = os.stat(path).st_mode & 0o777
                assert mode == 0o600

                loaded = load_node_config()
                assert loaded == config

    def test_load_nonexistent(self):
        with patch("syne.node.client.NODE_CONFIG_PATH", "/tmp/nonexistent_config_xyz.json"):
            result = load_node_config()
            assert result is None

    def test_is_node_mode_true(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            with patch("syne.node.client.NODE_CONFIG_PATH", f.name):
                assert is_node_mode() is True

    def test_is_node_mode_false(self):
        with patch("syne.node.client.NODE_CONFIG_PATH", "/tmp/nonexistent_xyz.json"):
            assert is_node_mode() is False


# ── NodeClient init ──────────────────────────────────────────────────


class TestNodeClientInit:
    def test_init_with_config(self):
        config = {
            "node_id": "n1",
            "token": "secret",
            "gateway": "ws://host:8443",
            "display_name": "laptop",
        }
        client = NodeClient(config=config)
        assert client.node_id == "n1"
        assert client.token == "secret"
        assert client.gateway_url == "ws://host:8443"
        assert client.display_name == "laptop"

    def test_init_no_config_raises(self):
        with patch("syne.node.client.load_node_config", return_value=None):
            with pytest.raises(RuntimeError, match="not configured"):
                NodeClient()

    def test_default_display_name(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443"}
        client = NodeClient(config=config)
        assert client.display_name == "n1"  # Falls back to node_id


# ── NodeClient dispatch ──────────────────────────────────────────────


class TestNodeClientDispatch:
    def _make_client(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443", "display_name": "test"}
        client = NodeClient(config=config)
        client._ws = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_response_chunk_accumulates(self):
        client = self._make_client()
        await client._dispatch({"type": "response_chunk", "text": "hello ", "done": False})
        await client._dispatch({"type": "response_chunk", "text": "world", "done": False})
        assert client._last_response == "hello world"

    @pytest.mark.asyncio
    async def test_response_done_sets_event(self):
        client = self._make_client()
        assert not client._response_done.is_set()
        await client._dispatch({"type": "response_chunk", "text": "", "done": True})
        assert client._response_done.is_set()

    @pytest.mark.asyncio
    async def test_tool_request_without_handler(self):
        client = self._make_client()
        await client._dispatch({
            "type": "tool_request",
            "request_id": "r1",
            "tool": "exec",
            "args": {},
        })
        # Should send back error result
        client._ws.send.assert_called_once()
        sent = json.loads(client._ws.send.call_args[0][0])
        assert sent["type"] == "tool_result"
        assert sent["success"] is False

    @pytest.mark.asyncio
    async def test_tool_request_with_handler(self):
        client = self._make_client()
        client._on_tool_request = AsyncMock(return_value=("output", True))

        await client._dispatch({
            "type": "tool_request",
            "request_id": "r1",
            "tool": "exec",
            "args": {"command": "ls"},
        })

        client._on_tool_request.assert_called_once()
        sent = json.loads(client._ws.send.call_args[0][0])
        assert sent["success"] is True
        assert sent["result"] == "output"

    @pytest.mark.asyncio
    async def test_thinking_chunk_callback(self):
        client = self._make_client()
        received = []
        client._on_thinking = lambda text: received.append(text)

        await client._dispatch({"type": "thinking_chunk", "text": "hmm"})
        assert received == ["hmm"]

    @pytest.mark.asyncio
    async def test_tool_activity_callback(self):
        client = self._make_client()
        received = []
        client._on_tool_activity = lambda n, a, r: received.append(n)

        await client._dispatch({
            "type": "tool_activity",
            "name": "exec",
            "args": {},
            "result_preview": "ok",
        })
        assert received == ["exec"]

    @pytest.mark.asyncio
    async def test_status_callback(self):
        client = self._make_client()
        received = []
        client._on_status = lambda msg: received.append(msg)

        await client._dispatch({"type": "status", "message": "Running..."})
        assert received == ["Running..."]

    @pytest.mark.asyncio
    async def test_meta_stored(self):
        client = self._make_client()
        await client._dispatch({
            "type": "meta",
            "agent_name": "Syne",
            "model": "gemini",
        })
        assert client.server_meta["agent_name"] == "Syne"

    @pytest.mark.asyncio
    async def test_error_sets_done(self):
        client = self._make_client()
        await client._dispatch({"type": "error", "message": "bad"})
        assert client._response_done.is_set()
        assert "Error: bad" in client._last_response

    @pytest.mark.asyncio
    async def test_response_callback_fired(self):
        client = self._make_client()
        received = []
        client._on_response = lambda text, done: received.append((text, done))

        await client._dispatch({"type": "response_chunk", "text": "hi", "done": False})
        assert received == [("hi", False)]


class TestNodeClientSendMessage:
    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443"}
        client = NodeClient(config=config)
        with pytest.raises(ConnectionError):
            await client.send_message("hello")

    @pytest.mark.asyncio
    async def test_timeout_returns_error_message(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443"}
        client = NodeClient(config=config)
        client._ws = AsyncMock()

        # Don't set _response_done → will timeout
        with patch.object(client._response_done, "wait", side_effect=asyncio.TimeoutError):
            result = await client.send_message("test")
        assert "timed out" in result.lower()


class TestNodeClientDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_sends_and_closes(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443"}
        client = NodeClient(config=config)
        client._ws = AsyncMock()
        client._connected.set()

        await client.disconnect()
        assert client._ws is None
        assert not client._connected.is_set()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        config = {"node_id": "n1", "token": "t", "gateway": "ws://h:8443"}
        client = NodeClient(config=config)
        await client.disconnect()  # Should not raise
