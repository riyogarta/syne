"""Tests for syne/memory/evaluator.py — memory evaluation and filtering."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from syne.memory.evaluator import (
    _is_explicit_remember,
    evaluate_message,
    evaluate_and_store,
)


# ── _is_explicit_remember (pure function) ────────────────────


class TestIsExplicitRemember:
    def test_ingat_ini(self):
        assert _is_explicit_remember("ingat ini") is True

    def test_remember_this(self):
        assert _is_explicit_remember("remember this please") is True

    def test_jangan_lupa(self):
        assert _is_explicit_remember("jangan lupa beli susu") is True

    def test_dont_forget(self):
        assert _is_explicit_remember("don't forget to buy milk") is True

    def test_catat_prefix(self):
        assert _is_explicit_remember("catat: berat badan 70kg") is True

    def test_simpan_ini(self):
        assert _is_explicit_remember("simpan ini ya") is True

    def test_inget_ini(self):
        assert _is_explicit_remember("inget ini") is True

    def test_ingat_prefix(self):
        assert _is_explicit_remember("ingat: alamat rumah di jalan merdeka") is True

    def test_remember_prefix(self):
        assert _is_explicit_remember("remember: my birthday is April 18") is True

    def test_normal_message(self):
        assert _is_explicit_remember("I like pizza") is False

    def test_question_about_remember(self):
        assert _is_explicit_remember("do you remember what I said?") is False

    def test_short_message(self):
        assert _is_explicit_remember("ok") is False

    def test_empty_message(self):
        assert _is_explicit_remember("") is False

    def test_case_insensitive(self):
        assert _is_explicit_remember("INGAT INI") is True
        assert _is_explicit_remember("Remember This") is True


# ── evaluate_message quick filters ───────────────────────────


class TestEvaluateMessageQuickFilters:
    async def test_too_short(self, mock_provider):
        result = await evaluate_message(mock_provider, "hi")
        assert result is None
        mock_provider.chat.assert_not_called()

    async def test_skip_pattern_ok(self, mock_provider):
        result = await evaluate_message(mock_provider, "ok")
        assert result is None
        mock_provider.chat.assert_not_called()

    async def test_skip_pattern_thanks(self, mock_provider):
        result = await evaluate_message(mock_provider, "thanks")
        assert result is None

    async def test_skip_pattern_terima_kasih(self, mock_provider):
        result = await evaluate_message(mock_provider, "terima kasih")
        assert result is None

    async def test_question_short(self, mock_provider):
        result = await evaluate_message(mock_provider, "what time is it?")
        assert result is None
        mock_provider.chat.assert_not_called()

    async def test_question_mark_short(self, mock_provider):
        result = await evaluate_message(mock_provider, "ready?")
        assert result is None

    async def test_technical_git(self, mock_provider):
        result = await evaluate_message(mock_provider, "git push origin main")
        assert result is None
        mock_provider.chat.assert_not_called()

    async def test_technical_docker(self, mock_provider):
        result = await evaluate_message(mock_provider, "docker restart syne-db")
        assert result is None

    async def test_technical_with_explicit_remember(self, mock_provider):
        """Technical keyword BUT explicit remember — should NOT be filtered."""
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(
            content="STORE|fact|0.7|Git repo at /home/riyo"
        )
        result = await evaluate_message(mock_provider, "ingat ini: git repo di /home/riyo")
        # Should pass quick filter and reach LLM
        mock_provider.chat.assert_called_once()


# ── evaluate_message LLM path ────────────────────────────────


class TestEvaluateMessageLLM:
    async def test_store_response(self, mock_provider):
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(
            content="STORE|fact|0.7|User lives in Bandung"
        )
        result = await evaluate_message(mock_provider, "Saya tinggal di Bandung sejak 2020")
        assert result is not None
        assert result["category"] == "fact"
        assert result["importance"] == 0.7
        assert result["content"] == "User lives in Bandung"

    async def test_skip_response(self, mock_provider):
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(content="SKIP")
        result = await evaluate_message(mock_provider, "Saya tinggal di Bandung sejak 2020")
        assert result is None

    async def test_malformed_store(self, mock_provider):
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(content="STORE|fact|0.7")
        result = await evaluate_message(mock_provider, "some long enough message for testing")
        assert result is None

    async def test_importance_clamped(self, mock_provider):
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(
            content="STORE|fact|1.5|some content"
        )
        result = await evaluate_message(mock_provider, "some long enough message for testing")
        assert result is not None
        assert result["importance"] == 1.0

    async def test_importance_invalid(self, mock_provider):
        from tests.conftest import MockChatResponse
        mock_provider.chat.return_value = MockChatResponse(
            content="STORE|fact|abc|some content"
        )
        result = await evaluate_message(mock_provider, "some long enough message for testing")
        assert result is not None
        assert result["importance"] == 0.5

    async def test_provider_exception(self, mock_provider):
        mock_provider.chat.side_effect = Exception("API error")
        result = await evaluate_message(mock_provider, "some long enough message for testing")
        assert result is None


# ── evaluate_and_store ───────────────────────────────────────


class TestEvaluateAndStore:
    async def test_evaluator_returns_none(self, mock_provider):
        memory_engine = AsyncMock()
        with patch("syne.memory.evaluator.evaluate_message", new_callable=AsyncMock, return_value=None):
            result = await evaluate_and_store(mock_provider, memory_engine, "hi")
        assert result is None
        memory_engine.store_if_new.assert_not_called()

    async def test_store_succeeds(self, mock_provider):
        memory_engine = AsyncMock()
        memory_engine.store_if_new.return_value = 42
        eval_result = {"category": "fact", "importance": 0.7, "content": "test"}

        with patch("syne.memory.evaluator.evaluate_message", new_callable=AsyncMock, return_value=eval_result):
            result = await evaluate_and_store(mock_provider, memory_engine, "some message", user_id=1)

        assert result == 42
        memory_engine.store_if_new.assert_called_once()

    async def test_permanent_memory(self, mock_provider):
        memory_engine = AsyncMock()
        memory_engine.store_if_new.return_value = 42
        eval_result = {"category": "fact", "importance": 0.9, "content": "birthday info"}

        with patch("syne.memory.evaluator.evaluate_message", new_callable=AsyncMock, return_value=eval_result), \
             patch("asyncio.create_task") as mock_task:
            result = await evaluate_and_store(
                mock_provider, memory_engine,
                "ingat ini: ulang tahun saya 18 April",
                user_id=1,
            )

        # permanent=True should be passed to store_if_new
        call_kwargs = memory_engine.store_if_new.call_args[1]
        assert call_kwargs["permanent"] is True
        # graph extraction should be triggered
        mock_task.assert_called_once()

    async def test_non_permanent_no_graph(self, mock_provider):
        memory_engine = AsyncMock()
        memory_engine.store_if_new.return_value = 42
        eval_result = {"category": "fact", "importance": 0.5, "content": "casual info"}

        with patch("syne.memory.evaluator.evaluate_message", new_callable=AsyncMock, return_value=eval_result), \
             patch("asyncio.create_task") as mock_task:
            await evaluate_and_store(mock_provider, memory_engine, "I like coffee", user_id=1)

        call_kwargs = memory_engine.store_if_new.call_args[1]
        assert call_kwargs["permanent"] is False
        mock_task.assert_not_called()

    async def test_duplicate_memory_skipped(self, mock_provider):
        memory_engine = AsyncMock()
        memory_engine.store_if_new.return_value = None  # duplicate
        eval_result = {"category": "fact", "importance": 0.7, "content": "test"}

        with patch("syne.memory.evaluator.evaluate_message", new_callable=AsyncMock, return_value=eval_result):
            result = await evaluate_and_store(mock_provider, memory_engine, "some message", user_id=1)

        assert result is None
