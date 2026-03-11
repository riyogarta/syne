"""Tests for syne.gateway.protocol — message types, encoding, decoding."""

import json
import pytest

from syne.gateway.protocol import (
    PROTOCOL_VERSION,
    NODE_TOOLS,
    ConnectMsg,
    ChatMsg,
    ToolResultMsg,
    ConnectedMsg,
    ResponseChunkMsg,
    ToolRequestMsg,
    ErrorMsg,
    ThinkingChunkMsg,
    ToolActivityMsg,
    StatusMsg,
    MetaMsg,
    encode,
    decode,
)


# ── Protocol constants ───────────────────────────────────────────────


class TestProtocolConstants:
    def test_version_is_int(self):
        assert isinstance(PROTOCOL_VERSION, int)
        assert PROTOCOL_VERSION >= 1

    def test_node_tools_is_frozenset(self):
        assert isinstance(NODE_TOOLS, frozenset)

    def test_node_tools_contains_expected(self):
        assert "exec" in NODE_TOOLS
        assert "file_read" in NODE_TOOLS
        assert "file_write" in NODE_TOOLS
        assert "read_source" in NODE_TOOLS

    def test_node_tools_does_not_contain_server_tools(self):
        assert "web_search" not in NODE_TOOLS
        assert "memory_search" not in NODE_TOOLS
        assert "send_message" not in NODE_TOOLS


# ── Node → Gateway messages ──────────────────────────────────────────


class TestConnectMsg:
    def test_fields(self):
        msg = ConnectMsg(token="abc", node_id="node-1", display_name="laptop")
        assert msg.type == "connect"
        assert msg.token == "abc"
        assert msg.node_id == "node-1"
        assert msg.display_name == "laptop"
        assert msg.protocol_version == PROTOCOL_VERSION

    def test_defaults(self):
        msg = ConnectMsg(token="x", node_id="n")
        assert msg.platform == "linux"
        assert msg.cwd == ""


class TestChatMsg:
    def test_type(self):
        msg = ChatMsg(text="hello")
        assert msg.type == "message"
        assert msg.text == "hello"

    def test_with_cwd(self):
        msg = ChatMsg(text="ls", cwd="/home/user")
        assert msg.cwd == "/home/user"


class TestToolResultMsg:
    def test_success(self):
        msg = ToolResultMsg(request_id="r1", result="output", success=True)
        assert msg.type == "tool_result"
        assert msg.success is True

    def test_failure(self):
        msg = ToolResultMsg(request_id="r2", result="error", success=False)
        assert msg.success is False


# ── Gateway → Node messages ──────────────────────────────────────────


class TestConnectedMsg:
    def test_fields(self):
        msg = ConnectedMsg(session_id=42, display_name="laptop")
        assert msg.type == "connected"
        assert msg.session_id == 42
        assert msg.protocol_version == PROTOCOL_VERSION


class TestResponseChunkMsg:
    def test_streaming(self):
        msg = ResponseChunkMsg(text="hello ", done=False)
        assert msg.type == "response_chunk"
        assert msg.done is False

    def test_done(self):
        msg = ResponseChunkMsg(text="", done=True)
        assert msg.done is True


class TestToolRequestMsg:
    def test_fields(self):
        msg = ToolRequestMsg(request_id="r1", tool="exec", args={"command": "ls"})
        assert msg.type == "tool_request"
        assert msg.tool == "exec"
        assert msg.args == {"command": "ls"}

    def test_empty_args_default(self):
        msg = ToolRequestMsg(request_id="r1", tool="file_read")
        assert msg.args == {}


class TestErrorMsg:
    def test_fields(self):
        msg = ErrorMsg(message="bad request", code="parse_error")
        assert msg.type == "error"


class TestThinkingChunkMsg:
    def test_fields(self):
        msg = ThinkingChunkMsg(text="reasoning...")
        assert msg.type == "thinking_chunk"


class TestToolActivityMsg:
    def test_fields(self):
        msg = ToolActivityMsg(name="exec", args={"command": "ls"}, result_preview="file1\nfile2")
        assert msg.type == "tool_activity"


class TestStatusMsg:
    def test_fields(self):
        msg = StatusMsg(message="Running exec...")
        assert msg.type == "status"


class TestMetaMsg:
    def test_defaults(self):
        msg = MetaMsg()
        assert msg.type == "meta"
        assert msg.agent_name == ""
        assert msg.tool_count == 0
        assert msg.reasoning_visible is False

    def test_with_values(self):
        msg = MetaMsg(agent_name="Syne", model="gemini-2.0-flash", tool_count=23)
        assert msg.agent_name == "Syne"
        assert msg.tool_count == 23


# ── Encoding / decoding ─────────────────────────────────────────────


class TestEncodeDecode:
    def test_encode_produces_valid_json(self):
        msg = ChatMsg(text="hello")
        encoded = encode(msg)
        parsed = json.loads(encoded)
        assert parsed["type"] == "message"
        assert parsed["text"] == "hello"

    def test_decode_string(self):
        raw = '{"type": "connect", "token": "abc"}'
        result = decode(raw)
        assert result["type"] == "connect"
        assert result["token"] == "abc"

    def test_decode_bytes(self):
        raw = b'{"type": "error", "message": "test"}'
        result = decode(raw)
        assert result["type"] == "error"

    def test_roundtrip(self):
        msg = ToolRequestMsg(request_id="r1", tool="exec", args={"command": "pwd"})
        encoded = encode(msg)
        decoded = decode(encoded)
        assert decoded["type"] == "tool_request"
        assert decoded["request_id"] == "r1"
        assert decoded["args"]["command"] == "pwd"

    def test_decode_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            decode("not json")
