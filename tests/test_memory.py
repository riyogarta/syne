"""Tests for memory engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from syne.memory.engine import MemoryEngine
from syne.llm.provider import EmbeddingResponse


@pytest.fixture
def mock_provider():
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=EmbeddingResponse(
        vector=[0.1] * 768,
        model="test-model",
        dimensions=768,
    ))
    return provider


class TestMemoryEngine:
    """Test memory engine operations."""

    def test_init(self, mock_provider):
        engine = MemoryEngine(mock_provider)
        assert engine.provider == mock_provider

    @pytest.mark.asyncio
    async def test_embed_called_on_store(self, mock_provider):
        """Verify embedding is generated when storing."""
        engine = MemoryEngine(mock_provider)
        # store() will fail without DB, but embed should be called
        with pytest.raises(Exception):
            await engine.store("test content")
        mock_provider.embed.assert_called_once_with("test content")

    @pytest.mark.asyncio
    async def test_embed_called_on_recall(self, mock_provider):
        """Verify embedding is generated when recalling."""
        engine = MemoryEngine(mock_provider)
        with pytest.raises(Exception):
            await engine.recall("search query")
        mock_provider.embed.assert_called_once_with("search query")
