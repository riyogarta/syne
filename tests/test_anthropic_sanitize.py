"""Tests for Anthropic conversation sanitization."""

import pytest
from syne.llm.anthropic import AnthropicProvider


class TestSanitizeConversation:
    """Test _sanitize_conversation ensures valid tool_use/tool_result pairing."""

    def test_normal_conversation_unchanged(self):
        """Plain user/assistant messages pass through unchanged."""
        conv = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        assert result == conv

    def test_matched_tool_use_and_result(self):
        """Properly paired tool_use + tool_result pass through."""
        conv = [
            {"role": "user", "content": "run a command"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Running..."},
                {"type": "tool_use", "id": "t1", "name": "exec", "input": {"cmd": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file1\nfile2"},
            ]},
            {"role": "assistant", "content": "Done! Found file1 and file2."},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        assert len(result) == 4
        # Assistant with tool_use preserved
        assert result[1]["role"] == "assistant"
        assert any(b.get("type") == "tool_use" for b in result[1]["content"])
        # Tool result preserved
        assert result[2]["role"] == "user"
        assert any(b.get("type") == "tool_result" for b in result[2]["content"])

    def test_orphaned_tool_result_dropped(self):
        """tool_result without preceding tool_use is dropped."""
        conv = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t_orphan", "content": "some result"},
            ]},
            {"role": "assistant", "content": "response"},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        # The orphaned tool_result should be dropped
        # Two consecutive assistant messages get merged → user + assistant = 2
        assert len(result) == 2
        # No tool_results remain
        assert all(
            not (isinstance(m.get("content"), list) and 
                 any(b.get("type") == "tool_result" for b in m["content"] if isinstance(b, dict)))
            for m in result
        )

    def test_orphaned_tool_use_converted_to_text(self):
        """tool_use without following tool_result is converted to plain text."""
        conv = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll try"},
                {"type": "tool_use", "id": "t2", "name": "exec", "input": {}},
            ]},
            # No tool_result follows — next message is a plain user message
            {"role": "user", "content": "what happened?"},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        # Assistant message should be converted to plain text
        assert result[1]["role"] == "assistant"
        assert isinstance(result[1]["content"], str)
        assert "I'll try" in result[1]["content"]

    def test_mismatched_ids_filtered(self):
        """tool_results with wrong IDs are filtered out."""
        conv = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t_real", "name": "exec", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t_real", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "t_wrong", "content": "orphan"},
            ]},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        # Only matched result should remain
        tool_results = [
            b for m in result if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "t_real"

    def test_consecutive_same_role_merged(self):
        """Consecutive same-role messages are merged (Anthropic requires alternating)."""
        conv = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response"},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "first" in result[0]["content"]
        assert "second" in result[0]["content"]

    def test_empty_conversation(self):
        """Empty conversation returns empty."""
        assert AnthropicProvider._sanitize_conversation([]) == []

    def test_multi_round_tool_calls(self):
        """Multiple rounds of tool_use → tool_result work correctly."""
        conv = [
            {"role": "user", "content": "search and fetch"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Searching..."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "result1"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Now fetching..."},
                {"type": "tool_use", "id": "t2", "name": "fetch", "input": {"url": "http://x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "page content"},
            ]},
            {"role": "assistant", "content": "Here's what I found."},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        assert len(result) == 6
        # All tool_use and tool_result pairs preserved
        tool_uses = [
            b for m in result if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        tool_results = [
            b for m in result if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(tool_uses) == 2
        assert len(tool_results) == 2

    def test_trimmed_context_recovery(self):
        """Simulates context trimming that removes assistant tool_use but leaves tool_result."""
        # After trimming, the first message visible is a tool_result without its tool_use
        conv = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t_trimmed", "content": "leftover"},
            ]},
            {"role": "assistant", "content": "Based on the results..."},
            {"role": "user", "content": "thanks"},
        ]
        result = AnthropicProvider._sanitize_conversation(conv)
        # Orphaned tool_result at start should be dropped
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        assert "thanks" in result[1]["content"]
