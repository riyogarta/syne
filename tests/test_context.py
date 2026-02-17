"""Tests for context window management and session config."""

import pytest
from syne.context import ContextManager, estimate_tokens, ChatMessage
from syne.channels.telegram import TelegramChannel


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


class TestThinkingBudget:
    """Tests for /think command levels and formatting."""

    LEVELS = {
        "off": 0,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "max": 24576,
    }

    def test_all_levels_defined(self):
        assert self.LEVELS["off"] == 0
        assert self.LEVELS["low"] == 1024
        assert self.LEVELS["medium"] == 4096
        assert self.LEVELS["high"] == 8192
        assert self.LEVELS["max"] == 24576

    def test_format_thinking_off(self):
        assert TelegramChannel._format_thinking_level(0) == "off"

    def test_format_thinking_low(self):
        assert TelegramChannel._format_thinking_level(1024) == "low"

    def test_format_thinking_medium(self):
        assert TelegramChannel._format_thinking_level(4096) == "medium"

    def test_format_thinking_high(self):
        assert TelegramChannel._format_thinking_level(8192) == "high"

    def test_format_thinking_max(self):
        assert TelegramChannel._format_thinking_level(24576) == "max"

    def test_format_thinking_default(self):
        assert TelegramChannel._format_thinking_level(None) == "default"

    def test_format_thinking_custom(self):
        assert TelegramChannel._format_thinking_level(2048) == "2048 tokens"

    def test_format_thinking_string_budget(self):
        """DB may return string — should handle int conversion."""
        assert TelegramChannel._format_thinking_level("4096") == "medium"

    def test_conversation_default_thinking(self):
        """New conversation should have thinking_budget = None (model default)."""
        from syne.conversation import Conversation
        from unittest.mock import MagicMock
        conv = Conversation(
            provider=MagicMock(),
            memory=MagicMock(),
            tools=MagicMock(),
            context_mgr=MagicMock(),
            session_id=1,
            user={"id": 1, "access_level": "owner"},
            system_prompt="test",
        )
        assert conv.thinking_budget is None


class TestReasoningVisibility:
    """Tests for /reasoning command."""

    def test_chat_response_has_thinking_field(self):
        from syne.llm.provider import ChatResponse
        resp = ChatResponse(content="hello", model="test", thinking="I think...")
        assert resp.thinking == "I think..."

    def test_chat_response_thinking_default_none(self):
        from syne.llm.provider import ChatResponse
        resp = ChatResponse(content="hello", model="test")
        assert resp.thinking is None

    def test_parse_gemini_thinking_parts(self):
        """Gemini returns thinking in parts with thought=True."""
        from syne.llm.google import GoogleProvider
        # Simulate a Gemini response with thinking parts
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me analyze...", "thought": True},
                        {"text": "Here is my answer."},
                    ]
                }
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
        }
        # Use the static-ish _parse_cca_response
        provider = GoogleProvider.__new__(GoogleProvider)
        result = provider._parse_cca_response(data, "gemini-2.5-pro")
        assert result.content == "Here is my answer."
        assert result.thinking == "Let me analyze..."

    def test_parse_gemini_no_thinking(self):
        """Regular response without thinking parts."""
        from syne.llm.google import GoogleProvider
        data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Simple answer."}]
                }
            }],
            "usageMetadata": {},
        }
        provider = GoogleProvider.__new__(GoogleProvider)
        result = provider._parse_cca_response(data, "gemini-2.5-pro")
        assert result.content == "Simple answer."
        assert result.thinking is None
