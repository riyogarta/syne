"""Tests for context window management."""

import pytest
from syne.context import ContextManager, estimate_tokens, ChatMessage


class TestTokenEstimation:

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        text = "Hello, how are you today?"
        tokens = estimate_tokens(text)
        assert 5 < tokens < 15  # Rough estimate

    def test_longer_text(self):
        text = "a" * 3500  # ~1000 tokens
        tokens = estimate_tokens(text)
        assert 900 < tokens < 1100


class TestContextManager:

    def test_no_trim_when_fits(self):
        mgr = ContextManager(max_context_tokens=10000)
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
        ]
        result = mgr.trim_context(messages)
        assert len(result) == 2

    def test_trim_old_history(self):
        mgr = ContextManager(max_context_tokens=500)  # Very small
        messages = [
            ChatMessage(role="system", content="You are helpful."),
        ]
        # Add many history messages
        for i in range(50):
            messages.append(ChatMessage(role="user", content=f"Message number {i} with some padding text"))
            messages.append(ChatMessage(role="assistant", content=f"Response to message {i}"))
        messages.append(ChatMessage(role="user", content="Current question"))

        result = mgr.trim_context(messages)
        # Should be shorter
        assert len(result) < len(messages)
        # System prompt should be kept
        assert result[0].role == "system"
        # Current message should be kept
        assert result[-1].content == "Current question"

    def test_should_compact(self):
        mgr = ContextManager(max_context_tokens=100000)
        # Small context — no compact
        small = [ChatMessage(role="user", content="Hi")]
        assert mgr.should_compact(small) is False

        # Large context — should compact
        large = [ChatMessage(role="user", content="x" * 300000)]
        assert mgr.should_compact(large) is True

    def test_usage_stats(self):
        mgr = ContextManager(max_context_tokens=10000)
        messages = [
            ChatMessage(role="system", content="System prompt here"),
            ChatMessage(role="user", content="Hello"),
        ]
        usage = mgr.get_usage(messages)
        assert "used_tokens" in usage
        assert "max_tokens" in usage
        assert "usage_percent" in usage
        assert usage["message_count"] == 2
