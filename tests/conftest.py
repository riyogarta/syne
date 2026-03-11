"""Shared test fixtures — mock provider, mock DB connection."""

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch


@dataclass
class MockChatResponse:
    content: str
    model: str = "test-model"
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list = None
    thinking: str = None


@dataclass
class MockEmbeddingResponse:
    vector: list
    model: str = "test-embed"
    dimensions: int = 768


@pytest.fixture
def mock_provider():
    """Mock LLM provider that returns configurable responses."""
    provider = AsyncMock()
    provider.name = "test"
    provider.chat.return_value = MockChatResponse(content="test response")
    provider.embed.return_value = MockEmbeddingResponse(vector=[0.0] * 768)
    return provider


@pytest.fixture
def mock_connection():
    """Mock async DB connection context manager."""
    conn = AsyncMock()
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = None
    conn.execute.return_value = None

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    return conn, ctx


@pytest.fixture
def mock_get_config():
    """Patch get_config to return values from a dict."""
    config_store = {}

    async def _get_config(key, default=None):
        return config_store.get(key, default)

    with patch("syne.db.models.get_config", side_effect=_get_config) as mock:
        mock._store = config_store
        yield mock
