"""Tests for memory evaluator."""

import pytest
from unittest.mock import AsyncMock
from syne.memory.evaluator import evaluate_message
from syne.llm.provider import ChatResponse


@pytest.fixture
def mock_provider():
    return AsyncMock()


class TestEvaluatorQuickFilters:
    """Test quick filter rules (no LLM needed)."""

    @pytest.mark.asyncio
    async def test_skip_short_messages(self, mock_provider):
        result = await evaluate_message(mock_provider, "ok")
        assert result is None
        mock_provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_greetings(self, mock_provider):
        for msg in ["hi", "halo", "hello", "thanks", "makasih", "ok", "sip"]:
            result = await evaluate_message(mock_provider, msg)
            assert result is None
        mock_provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_short_questions(self, mock_provider):
        result = await evaluate_message(mock_provider, "apa itu?")
        assert result is None
        mock_provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_meaningful_to_llm(self, mock_provider):
        mock_provider.chat = AsyncMock(return_value=ChatResponse(
            content="SKIP", model="test", input_tokens=0, output_tokens=0
        ))
        result = await evaluate_message(
            mock_provider,
            "My name is Riyo and I work as Head of IT at a reinsurance company"
        )
        # Should have called LLM (even if result is SKIP)
        mock_provider.chat.assert_called_once()


class TestEvaluatorLLMResponse:
    """Test parsing of LLM evaluation responses."""

    @pytest.mark.asyncio
    async def test_parse_store_response(self, mock_provider):
        mock_provider.chat = AsyncMock(return_value=ChatResponse(
            content="STORE|fact|0.7|User's name is Riyo",
            model="test", input_tokens=0, output_tokens=0,
        ))
        result = await evaluate_message(
            mock_provider,
            "My name is Riyo"
        )
        assert result is not None
        assert result["category"] == "fact"
        assert result["importance"] == 0.7
        assert result["content"] == "User's name is Riyo"

    @pytest.mark.asyncio
    async def test_parse_skip_response(self, mock_provider):
        mock_provider.chat = AsyncMock(return_value=ChatResponse(
            content="SKIP", model="test", input_tokens=0, output_tokens=0,
        ))
        result = await evaluate_message(mock_provider, "How are you today?")
        assert result is None

    @pytest.mark.asyncio
    async def test_clamp_importance(self, mock_provider):
        mock_provider.chat = AsyncMock(return_value=ChatResponse(
            content="STORE|health|1.5|User has diabetes",
            model="test", input_tokens=0, output_tokens=0,
        ))
        result = await evaluate_message(mock_provider, "I have diabetes")
        assert result["importance"] == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_handle_malformed_response(self, mock_provider):
        mock_provider.chat = AsyncMock(return_value=ChatResponse(
            content="I think this should be stored",
            model="test", input_tokens=0, output_tokens=0,
        ))
        result = await evaluate_message(mock_provider, "Something meaningful")
        assert result is None  # Malformed → skip
