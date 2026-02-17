"""Tests for conversation compaction."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from syne.compaction import (
    compact_session,
    should_compact,
    should_compact_by_chars,
    auto_compact_check,
    get_session_stats,
    COMPACTION_PROMPT,
)
from syne.llm.provider import ChatMessage, ChatResponse


# ── Helpers ──────────────────────────────────────────────────

def make_mock_provider(summary_text: str = "## Summary\n- User discussed topic X") -> MagicMock:
    """Create a mock LLM provider that returns a summary."""
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=ChatResponse(
        content=summary_text,
        model="test-model",
        tool_calls=None,
    ))
    return provider


def make_message_rows(count: int, role: str = "user") -> list:
    """Create mock message rows."""
    rows = []
    for i in range(count):
        rows.append({
            "id": i + 1,
            "role": role if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "x" * 100,
        })
    return rows


# ── Tests ────────────────────────────────────────────────────

class TestCompactionPrompt:
    """Ensure the compaction prompt has the right properties."""

    def test_prompt_includes_factual_rule(self):
        assert "factual" in COMPACTION_PROMPT.lower()

    def test_prompt_blocks_assistant_suggestions(self):
        assert "assistant suggestions" in COMPACTION_PROMPT.lower()

    def test_prompt_blocks_assumptions(self):
        assert "assumptions" in COMPACTION_PROMPT.lower()

    def test_prompt_preserves_corrections(self):
        assert "corrected" in COMPACTION_PROMPT.lower()


class TestShouldCompact:

    @pytest.mark.asyncio
    async def test_below_threshold(self):
        """Session below threshold should not compact."""
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            conn.fetchrow = AsyncMock(return_value={"message_count": 50})

            result = await should_compact(1, threshold=100)
            assert result is False

    @pytest.mark.asyncio
    async def test_at_threshold(self):
        """Session at threshold should compact."""
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            conn.fetchrow = AsyncMock(return_value={"message_count": 100})

            result = await should_compact(1, threshold=100)
            assert result is True

    @pytest.mark.asyncio
    async def test_above_threshold(self):
        """Session above threshold should compact."""
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            conn.fetchrow = AsyncMock(return_value={"message_count": 150})

            result = await should_compact(1, threshold=100)
            assert result is True

    @pytest.mark.asyncio
    async def test_no_session(self):
        """Non-existent session should not compact."""
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            conn.fetchrow = AsyncMock(return_value=None)

            result = await should_compact(999, threshold=100)
            assert result is False


class TestShouldCompactByChars:

    @pytest.mark.asyncio
    async def test_below_char_threshold(self):
        with patch("syne.compaction.get_session_stats") as mock_stats:
            mock_stats.return_value = {"total_chars": 50000, "message_count": 40, "oldest_message": None, "newest_message": None}
            result = await should_compact_by_chars(1, char_threshold=80000)
            assert result is False

    @pytest.mark.asyncio
    async def test_above_char_threshold(self):
        with patch("syne.compaction.get_session_stats") as mock_stats:
            mock_stats.return_value = {"total_chars": 90000, "message_count": 60, "oldest_message": None, "newest_message": None}
            result = await should_compact_by_chars(1, char_threshold=80000)
            assert result is True


class TestGetSessionStats:

    @pytest.mark.asyncio
    async def test_returns_stats(self):
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            conn.fetchrow = AsyncMock(return_value={
                "message_count": 42,
                "total_chars": 12345,
                "oldest_message": "2026-01-01",
                "newest_message": "2026-02-18",
            })

            stats = await get_session_stats(1)
            assert stats["message_count"] == 42
            assert stats["total_chars"] == 12345


class TestCompactSession:

    @pytest.mark.asyncio
    async def test_too_few_messages(self):
        """Should return None when not enough messages to compact."""
        with patch("syne.compaction.get_connection") as mock_conn:
            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
            # get_session_stats call
            conn.fetchrow = AsyncMock(return_value={
                "message_count": 10,
                "total_chars": 500,
                "oldest_message": None,
                "newest_message": None,
                "count": 10,
            })

            provider = make_mock_provider()
            result = await compact_session(1, provider, keep_recent=20)
            assert result is None
            # LLM should NOT have been called
            provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_compaction(self):
        """Full compaction flow: summarize + delete + insert summary."""
        summary_text = "## Summary\n- Discussed project X\n- Decided on approach Y"
        provider = make_mock_provider(summary_text)

        with patch("syne.compaction.get_connection") as mock_conn, \
             patch("syne.compaction.get_session_stats") as mock_stats:

            conn = AsyncMock()
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=conn)
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)

            # First call: count messages
            conn.fetchrow = AsyncMock(return_value={"count": 50})
            # Fetch old messages to summarize
            old_messages = make_message_rows(30)
            conn.fetch = AsyncMock(return_value=old_messages)
            conn.execute = AsyncMock()

            # Stats calls (pre and post)
            mock_stats.side_effect = [
                {"message_count": 50, "total_chars": 15000, "oldest_message": None, "newest_message": None},
                {"message_count": 21, "total_chars": 3000, "oldest_message": None, "newest_message": None},
            ]

            result = await compact_session(1, provider, keep_recent=20)

            assert result is not None
            assert result["messages_before"] == 50
            assert result["messages_after"] == 21  # 20 kept + 1 summary
            assert result["messages_summarized"] == 30
            assert result["summary"] == summary_text

            # Verify LLM was called with compaction prompt
            provider.chat.assert_called_once()
            call_args = provider.chat.call_args
            messages = call_args.kwargs.get("messages") or call_args[1].get("messages") if len(call_args) > 1 else call_args.kwargs["messages"]
            assert messages[0].role == "system"
            assert "Summarize" in messages[0].content


class TestAutoCompactCheck:

    @pytest.mark.asyncio
    async def test_no_compact_needed(self):
        """Should return None when thresholds not reached."""
        with patch("syne.compaction.get_config") as mock_cfg, \
             patch("syne.compaction.should_compact") as mock_sc, \
             patch("syne.compaction.should_compact_by_chars") as mock_scc:
            mock_cfg.side_effect = [100, 80000]  # msg threshold, char threshold
            mock_sc.return_value = False
            mock_scc.return_value = False

            provider = make_mock_provider()
            result = await auto_compact_check(1, provider)
            assert result is None

    @pytest.mark.asyncio
    async def test_compact_on_message_threshold(self):
        """Should compact when message threshold reached."""
        with patch("syne.compaction.get_config") as mock_cfg, \
             patch("syne.compaction.should_compact") as mock_sc, \
             patch("syne.compaction.should_compact_by_chars") as mock_scc, \
             patch("syne.compaction.compact_session") as mock_compact:
            mock_cfg.side_effect = [100, 80000]
            mock_sc.return_value = True
            mock_scc.return_value = False
            mock_compact.return_value = {"summary": "test", "messages_before": 120, "messages_after": 21}

            provider = make_mock_provider()
            result = await auto_compact_check(1, provider)
            assert result is not None
            mock_compact.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_on_char_threshold(self):
        """Should compact when char threshold reached (even if message count is low)."""
        with patch("syne.compaction.get_config") as mock_cfg, \
             patch("syne.compaction.should_compact") as mock_sc, \
             patch("syne.compaction.should_compact_by_chars") as mock_scc, \
             patch("syne.compaction.compact_session") as mock_compact:
            mock_cfg.side_effect = [100, 80000]
            mock_sc.return_value = False
            mock_scc.return_value = True  # Chars over threshold
            mock_compact.return_value = {"summary": "test", "messages_before": 60, "messages_after": 21}

            provider = make_mock_provider()
            result = await auto_compact_check(1, provider)
            assert result is not None
            mock_compact.assert_called_once()
