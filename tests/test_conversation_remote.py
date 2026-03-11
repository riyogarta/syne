"""Tests for syne.gateway.conversation_remote — chat ID generation, session routing."""

import pytest
from syne.gateway.conversation_remote import _make_chat_id, _make_tg_chat_id


# ── Chat ID generation ───────────────────────────────────────────────


class TestMakeChatId:
    def test_format(self):
        result = _make_chat_id("node-abc", "/home/user")
        assert result.startswith("node:node-abc:")
        # cwd hash is 8 hex chars
        parts = result.split(":")
        assert len(parts) == 3
        assert len(parts[2]) == 8

    def test_different_cwd_different_id(self):
        id1 = _make_chat_id("node-1", "/home/user/project-a")
        id2 = _make_chat_id("node-1", "/home/user/project-b")
        assert id1 != id2

    def test_same_cwd_same_id(self):
        id1 = _make_chat_id("node-1", "/home/user")
        id2 = _make_chat_id("node-1", "/home/user")
        assert id1 == id2

    def test_different_node_different_id(self):
        id1 = _make_chat_id("node-1", "/home")
        id2 = _make_chat_id("node-2", "/home")
        assert id1 != id2

    def test_empty_cwd(self):
        result = _make_chat_id("node-1", "")
        assert result.startswith("node:node-1:")


class TestMakeTgChatId:
    def test_format(self):
        result = _make_tg_chat_id("node-abc")
        assert result == "tgremote:node-abc"

    def test_no_cwd_component(self):
        """Telegram chat IDs don't include cwd — always same session per node."""
        result = _make_tg_chat_id("node-1")
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
