"""Tests for syne/context.py — token estimation and context window management."""

import pytest
from syne.context import estimate_tokens, estimate_messages_tokens, ContextManager
from syne.llm.provider import ChatMessage


class TestEstimateTokens:
    """Tests for the estimate_tokens() helper."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text(self):
        text = "hello"  # 5 chars / 3.5 = 1.42 -> 1
        assert estimate_tokens(text) == 1

    def test_long_text(self):
        text = "a" * 350  # 350 / 3.5 = 100
        assert estimate_tokens(text) == 100

    def test_returns_int(self):
        result = estimate_tokens("some text")
        assert isinstance(result, int)

    def test_known_value(self):
        # 14 chars / 3.5 = 4.0
        assert estimate_tokens("fourteen chars!") == int(len("fourteen chars!") / 3.5)


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


class TestContextManagerInit:
    """Tests for ContextManager.__init__() — defaults and custom values."""

    def test_default_values(self):
        cm = ContextManager()
        assert cm.max_context_tokens == 128000
        assert cm.reserved_output == 4096
        assert cm.available == 128000 - 4096

    def test_custom_values(self):
        cm = ContextManager(
            max_context_tokens=200000,
            reserved_output_tokens=8192,
        )
        assert cm.max_context_tokens == 200000
        assert cm.reserved_output == 8192
        assert cm.available == 200000 - 8192

    def test_budget_calculations_defaults(self):
        cm = ContextManager()
        available = 128000 - 4096
        assert cm.system_budget == int(available * 0.15)
        assert cm.memory_budget_tokens == int(available * 0.10)
        assert cm.history_budget == int(available * 0.65)

    def test_budget_calculations_custom(self):
        cm = ContextManager(
            max_context_tokens=100000,
            reserved_output_tokens=2000,
            system_prompt_budget=0.20,
            memory_budget=0.15,
            history_budget=0.55,
        )
        available = 100000 - 2000
        assert cm.system_budget == int(available * 0.20)
        assert cm.memory_budget_tokens == int(available * 0.15)
        assert cm.history_budget == int(available * 0.55)


class TestShouldCompact:
    """Tests for ContextManager.should_compact()."""

    def test_below_threshold(self):
        cm = ContextManager(max_context_tokens=1000, reserved_output_tokens=0)
        # Small messages well below 80%
        msgs = [ChatMessage(role="user", content="hi")]
        assert cm.should_compact(msgs) is False

    def test_above_threshold(self):
        cm = ContextManager(max_context_tokens=100, reserved_output_tokens=0)
        # Create messages that exceed 80% of 100 tokens
        # 80 tokens needed; each char ~0.286 tokens, plus 4 overhead per msg
        big_content = "x" * 400  # ~114 tokens + 4 overhead = ~118 > 80
        msgs = [ChatMessage(role="user", content=big_content)]
        assert cm.should_compact(msgs) is True

    def test_custom_threshold(self):
        cm = ContextManager(max_context_tokens=100, reserved_output_tokens=0)
        # available = 100, threshold=0.5 means compact at 50 tokens
        content = "x" * 200  # ~57 tokens + 4 overhead = ~61 > 50
        msgs = [ChatMessage(role="user", content=content)]
        assert cm.should_compact(msgs, threshold=0.5) is True

    def test_exactly_at_threshold(self):
        cm = ContextManager(max_context_tokens=100, reserved_output_tokens=0)
        # Edge: exactly at threshold should return True (>=)
        # Need exactly 80 tokens: threshold 0.8 * 100 = 80
        # estimate_messages_tokens = len(content)/3.5 + 4
        # So len(content)/3.5 + 4 = 80 => len(content) = 266
        msgs = [ChatMessage(role="user", content="x" * 266)]
        assert cm.should_compact(msgs) is True


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
        assert usage["max_tokens"] == 10000
        assert usage["usage_percent"] == round(total / 10000 * 100, 1)
        assert usage["remaining_tokens"] == 10000 - total
        assert usage["message_count"] == 2

    def test_empty_messages(self):
        cm = ContextManager(max_context_tokens=5000, reserved_output_tokens=1000)
        usage = cm.get_usage([])
        assert usage["used_tokens"] == 0
        assert usage["max_tokens"] == 4000
        assert usage["usage_percent"] == 0.0
        assert usage["remaining_tokens"] == 4000
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
