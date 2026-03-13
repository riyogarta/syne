"""Tests for syne/context.py — token estimation and context window management."""

import pytest
from syne.context import (
    estimate_tokens, estimate_messages_tokens, ContextManager,
    DEFAULT_CHARS_PER_TOKEN, SAFETY_MARGIN,
)
from syne.llm.provider import ChatMessage


class TestEstimateTokens:
    """Tests for the estimate_tokens() helper."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        text = "hello"  # 5 chars / 4.0 = 1.25 -> 1
        assert estimate_tokens(text) == 1

    def test_long_text(self):
        text = "a" * 400  # 400 / 4.0 = 100
        assert estimate_tokens(text) == 100

    def test_returns_int(self):
        result = estimate_tokens("some text")
        assert isinstance(result, int)

    def test_known_value(self):
        text = "sixteen chars!!!"  # 16 chars / 4.0 = 4
        assert estimate_tokens(text) == int(len(text) / DEFAULT_CHARS_PER_TOKEN)

    def test_custom_chars_per_token(self):
        text = "a" * 100
        assert estimate_tokens(text, chars_per_token=2.0) == 50
        assert estimate_tokens(text, chars_per_token=5.0) == 20

    def test_default_is_4(self):
        assert DEFAULT_CHARS_PER_TOKEN == 4.0


class TestEstimateMessagesTokens:
    """Tests for the estimate_messages_tokens() helper."""

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msg = ChatMessage(role="user", content="hello")
        result = estimate_messages_tokens([msg])
        expected = estimate_tokens("hello") + 4  # content tokens + overhead
        assert result == expected

    def test_multiple_messages(self):
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]
        result = estimate_messages_tokens(msgs)
        expected = sum(estimate_tokens(m.content) + 4 for m in msgs)
        assert result == expected

    def test_overhead_per_message(self):
        # Each message adds 4 tokens of overhead regardless of content
        msg_empty = ChatMessage(role="user", content="")
        assert estimate_messages_tokens([msg_empty]) == 4

    def test_custom_chars_per_token(self):
        msg = ChatMessage(role="user", content="a" * 100)
        result_default = estimate_messages_tokens([msg])
        result_custom = estimate_messages_tokens([msg], chars_per_token=2.0)
        assert result_custom > result_default  # 2 chars/token = more tokens


class TestContextManagerInit:
    """Tests for ContextManager.__init__() — defaults and custom values."""

    def test_default_values(self):
        cm = ContextManager()
        assert cm.max_context_tokens == 128000
        assert cm.reserved_output == 4096
        # available includes safety margin
        expected = int((128000 - 4096) / SAFETY_MARGIN)
        assert cm.available == expected

    def test_custom_values(self):
        cm = ContextManager(
            max_context_tokens=200000,
            reserved_output_tokens=8192,
        )
        assert cm.max_context_tokens == 200000
        assert cm.reserved_output == 8192
        expected = int((200000 - 8192) / SAFETY_MARGIN)
        assert cm.available == expected

    def test_safety_margin_applied(self):
        cm = ContextManager(max_context_tokens=120000, reserved_output_tokens=0)
        # With 1.2 safety margin, available should be less than raw
        assert cm.available == int(120000 / SAFETY_MARGIN)
        assert cm.available < 120000

    def test_budget_calculations(self):
        cm = ContextManager()
        assert cm.system_budget == int(cm.available * 0.15)
        assert cm.memory_budget_tokens == int(cm.available * 0.10)
        assert cm.history_budget == int(cm.available * 0.65)

    def test_chars_per_token_stored(self):
        cm = ContextManager(chars_per_token=3.0)
        assert cm.chars_per_token == 3.0

    def test_chars_per_token_default(self):
        cm = ContextManager()
        assert cm.chars_per_token == DEFAULT_CHARS_PER_TOKEN


class TestShouldCompact:
    """Tests for ContextManager.should_compact()."""

    def test_below_threshold(self):
        cm = ContextManager(max_context_tokens=1000, reserved_output_tokens=0)
        # Small messages well below 80%
        msgs = [ChatMessage(role="user", content="hi")]
        assert cm.should_compact(msgs) is False

    def test_above_threshold(self):
        cm = ContextManager(max_context_tokens=100, reserved_output_tokens=0)
        # available = int(100 / 1.2) = 83, threshold 80% = 66 tokens needed
        # Big content to exceed that
        big_content = "x" * 400  # 100 tokens + 4 overhead = 104 > 66
        msgs = [ChatMessage(role="user", content=big_content)]
        assert cm.should_compact(msgs) is True

    def test_custom_threshold(self):
        cm = ContextManager(max_context_tokens=100, reserved_output_tokens=0)
        # available = 83, threshold=0.5 means compact at 41 tokens
        content = "x" * 200  # 50 tokens + 4 overhead = 54 > 41
        msgs = [ChatMessage(role="user", content=content)]
        assert cm.should_compact(msgs, threshold=0.5) is True


class TestGetUsage:
    """Tests for ContextManager.get_usage()."""

    def test_correct_stats(self):
        cm = ContextManager(max_context_tokens=10000, reserved_output_tokens=0)
        msgs = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="world"),
        ]
        usage = cm.get_usage(msgs)
        total = estimate_messages_tokens(msgs)

        assert usage["used_tokens"] == total
        assert usage["max_tokens"] == cm.available
        assert usage["usage_percent"] == round(total / cm.available * 100, 1)
        assert usage["remaining_tokens"] == cm.available - total
        assert usage["message_count"] == 2

    def test_empty_messages(self):
        cm = ContextManager(max_context_tokens=5000, reserved_output_tokens=1000)
        usage = cm.get_usage([])
        assert usage["used_tokens"] == 0
        assert usage["max_tokens"] == cm.available
        assert usage["usage_percent"] == 0.0
        assert usage["remaining_tokens"] == cm.available
        assert usage["message_count"] == 0


class TestTrimContext:
    """Tests for ContextManager.trim_context()."""

    def test_empty_messages(self):
        cm = ContextManager()
        assert cm.trim_context([]) == []

    def test_messages_fitting(self):
        cm = ContextManager(max_context_tokens=100000, reserved_output_tokens=0)
        msgs = [
            ChatMessage(role="system", content="Be helpful."),
            ChatMessage(role="user", content="Hi"),
        ]
        result = cm.trim_context(msgs)
        assert result == msgs

    def test_messages_needing_trim(self):
        # Small context window forces trimming
        cm = ContextManager(max_context_tokens=50, reserved_output_tokens=0)
        msgs = [
            ChatMessage(role="system", content="System prompt."),
            ChatMessage(role="user", content="old message " * 20),
            ChatMessage(role="assistant", content="old reply " * 20),
            ChatMessage(role="user", content="another old " * 20),
            ChatMessage(role="assistant", content="another reply " * 20),
            ChatMessage(role="user", content="current question"),
        ]
        result = cm.trim_context(msgs)
        # Result should be shorter than original
        assert len(result) < len(msgs)
        # Last message (current user) must be preserved
        assert result[-1].content == "current question"
        assert result[-1].role == "user"

    def test_system_messages_preserved(self):
        cm = ContextManager(max_context_tokens=80, reserved_output_tokens=0)
        system_msg = ChatMessage(role="system", content="Important system prompt.")
        msgs = [
            system_msg,
            ChatMessage(role="user", content="old " * 20),
            ChatMessage(role="assistant", content="reply " * 20),
            ChatMessage(role="user", content="newest"),
        ]
        result = cm.trim_context(msgs)
        # System message must be first
        assert result[0].role == "system"
        # The original system prompt content should still be there
        assert "Important system prompt" in result[0].content

    def test_trim_injects_notice_when_messages_dropped(self):
        cm = ContextManager(max_context_tokens=60, reserved_output_tokens=0)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="old " * 30),
            ChatMessage(role="assistant", content="old reply " * 30),
            ChatMessage(role="user", content="new"),
        ]
        result = cm.trim_context(msgs)
        # When messages are dropped, a notice system message is injected
        system_contents = [m.content for m in result if m.role == "system"]
        has_notice = any("CONTEXT NOTICE" in c for c in system_contents)
        assert has_notice


class TestSafetyMargin:
    """Tests for SAFETY_MARGIN behavior."""

    def test_safety_margin_value(self):
        assert SAFETY_MARGIN == 1.2

    def test_safety_margin_reduces_available(self):
        # Without safety margin, available = 200000 - 4096 = 195904
        # With safety margin, available = int(195904 / 1.2) = 163253
        cm = ContextManager(max_context_tokens=200000, reserved_output_tokens=4096)
        raw = 200000 - 4096
        assert cm.available < raw
        assert cm.available == int(raw / SAFETY_MARGIN)
