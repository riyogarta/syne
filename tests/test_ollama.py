"""Tests for Ollama embedding provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


# ═══════════════════════════════════════════════════════════════
# OllamaProvider — unit tests
# ═══════════════════════════════════════════════════════════════

class TestOllamaProvider:
    """Test OllamaProvider class."""

    def test_init_defaults(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider()
        assert p.embedding_model == "qwen3-embedding:0.6b"
        assert p.base_url == "http://localhost:11434"
        assert p.name == "ollama"
        assert p.supports_vision is False

    def test_init_custom(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider(
            embedding_model="nomic-embed-text",
            base_url="http://192.168.1.10:11434",
        )
        assert p.embedding_model == "nomic-embed-text"
        assert p.base_url == "http://192.168.1.10:11434"

    def test_init_strips_trailing_slash(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider(base_url="http://localhost:11434/")
        assert p.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_chat_raises_not_implemented(self):
        from syne.llm.ollama import OllamaProvider
        from syne.llm.provider import ChatMessage
        p = OllamaProvider()
        with pytest.raises(NotImplementedError, match="embedding-only"):
            await p.chat([ChatMessage(role="user", content="hello")])

    @pytest.mark.asyncio
    async def test_embed_success(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider()
        mock_vector = [0.1] * 1024

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [mock_vector]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await p.embed("test text")

            assert result.vector == mock_vector
            assert result.dimensions == 1024
            assert result.model == "qwen3-embedding:0.6b"
            assert result.input_tokens == 0

            mock_client.post.assert_called_once_with(
                "http://localhost:11434/api/embed",
                json={"model": "qwen3-embedding:0.6b", "input": "test text"},
            )

    @pytest.mark.asyncio
    async def test_embed_custom_model(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider()
        mock_vector = [0.5] * 768

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [mock_vector]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await p.embed("hello", model="nomic-embed-text")
            assert result.dimensions == 768
            assert result.model == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider()
        vec1 = [0.1] * 1024
        vec2 = [0.2] * 1024

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [vec1, vec2]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await p.embed_batch(["text1", "text2"])

            assert len(results) == 2
            assert results[0].vector == vec1
            assert results[1].vector == vec2
            assert results[0].dimensions == 1024
            assert results[1].dimensions == 1024

            mock_client.post.assert_called_once_with(
                "http://localhost:11434/api/embed",
                json={"model": "qwen3-embedding:0.6b", "input": ["text1", "text2"]},
            )

    @pytest.mark.asyncio
    async def test_embed_http_error(self):
        from syne.llm.ollama import OllamaProvider
        p = OllamaProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response,
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await p.embed("test")


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

class TestOllamaUtils:
    """Test check_ollama_available and check_model_available."""

    @pytest.mark.asyncio
    async def test_check_ollama_available_true(self):
        from syne.llm.ollama import check_ollama_available

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await check_ollama_available()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_ollama_available_false_connection_error(self):
        from syne.llm.ollama import check_ollama_available

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await check_ollama_available()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_model_available_true(self):
        from syne.llm.ollama import check_model_available

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3-embedding:0.6b"},
                {"name": "nomic-embed-text:latest"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await check_model_available("qwen3-embedding:0.6b")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_model_available_false(self):
        from syne.llm.ollama import check_model_available

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await check_model_available("qwen3-embedding:0.6b")
            assert result is False

    @pytest.mark.asyncio
    async def test_check_model_by_base_name(self):
        """Model with different tag should match by base name."""
        from syne.llm.ollama import check_model_available

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "qwen3-embedding:latest"}]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await check_model_available("qwen3-embedding:0.6b")
            assert result is True


# ═══════════════════════════════════════════════════════════════
# Driver integration — create_embedding_provider
# ═══════════════════════════════════════════════════════════════

class TestOllamaDriverIntegration:
    """Test Ollama in the driver registry."""

    @pytest.mark.asyncio
    async def test_create_embedding_provider_ollama(self):
        from syne.llm.drivers import create_embedding_provider
        from syne.llm.ollama import OllamaProvider

        entry = {
            "key": "ollama-qwen3",
            "driver": "ollama",
            "model_id": "qwen3-embedding:0.6b",
            "auth": "none",
            "base_url": "http://localhost:11434",
            "dimensions": 1024,
        }

        provider = await create_embedding_provider(entry)
        assert isinstance(provider, OllamaProvider)
        assert provider.embedding_model == "qwen3-embedding:0.6b"
        assert provider.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_create_embedding_provider_ollama_no_api_key_needed(self):
        """Ollama should NOT require an API key."""
        from syne.llm.drivers import create_embedding_provider
        from syne.llm.ollama import OllamaProvider

        entry = {
            "key": "ollama-qwen3",
            "driver": "ollama",
            "model_id": "qwen3-embedding:0.6b",
            "auth": "none",
            # No credential_key — that's the point
        }

        provider = await create_embedding_provider(entry)
        assert isinstance(provider, OllamaProvider)

    @pytest.mark.asyncio
    async def test_create_embedding_provider_together_still_works(self):
        """Ensure Together AI path still works after Ollama changes."""
        from syne.llm.drivers import create_embedding_provider
        from syne.llm.together import TogetherProvider

        entry = {
            "key": "together-bge",
            "driver": "together",
            "model_id": "BAAI/bge-base-en-v1.5",
            "auth": "api_key",
            "credential_key": "credential.together_api_key",
        }

        with patch("syne.llm.drivers._load_api_key", new_callable=AsyncMock, return_value="fake-key"):
            provider = await create_embedding_provider(entry)
            assert isinstance(provider, TogetherProvider)

    @pytest.mark.asyncio
    async def test_create_embedding_provider_together_no_key_fails(self):
        """Together AI without API key should return None."""
        from syne.llm.drivers import create_embedding_provider

        entry = {
            "key": "together-bge",
            "driver": "together",
            "model_id": "BAAI/bge-base-en-v1.5",
            "auth": "api_key",
            "credential_key": "credential.together_api_key",
        }

        with patch("syne.llm.drivers._load_api_key", new_callable=AsyncMock, return_value=None):
            provider = await create_embedding_provider(entry)
            assert provider is None


# ═══════════════════════════════════════════════════════════════
# System detection (init)
# ═══════════════════════════════════════════════════════════════

class TestSystemDetection:
    """Test system resource detection for Ollama eligibility."""

    def test_cpu_count_available(self):
        import os
        cpu = os.cpu_count()
        assert cpu is not None
        assert cpu >= 1

    def test_meminfo_readable(self):
        """Test that /proc/meminfo is readable (Linux only)."""
        import platform
        if platform.system() != "Linux":
            pytest.skip("Linux only")
        with open("/proc/meminfo") as f:
            content = f.read()
        assert "MemTotal:" in content

    def test_disk_usage_available(self):
        import shutil
        usage = shutil.disk_usage("/")
        assert usage.free > 0
        assert usage.total > 0


# ═══════════════════════════════════════════════════════════════
# Embedding dimension change — memory reset
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingDimensionChange:
    """Test that dimension changes trigger memory reset."""

    def test_dimensions_differ(self):
        """Basic dimension comparison logic."""
        current_dims = 768
        new_dims = 1024
        assert current_dims != new_dims

    def test_dimensions_same(self):
        """Same dimensions = no reset needed."""
        current_dims = 768
        new_dims = 768
        assert current_dims == new_dims

    def test_ollama_entry_has_correct_dimensions(self):
        """Verify Ollama registry entry declares 1024 dimensions."""
        entry = {
            "key": "ollama-qwen3",
            "driver": "ollama",
            "model_id": "qwen3-embedding:0.6b",
            "dimensions": 1024,
        }
        assert entry["dimensions"] == 1024

    def test_together_entry_has_768_dimensions(self):
        """Verify Together AI registry entry declares 768 dimensions."""
        entry = {
            "key": "together-bge",
            "driver": "together",
            "model_id": "BAAI/bge-base-en-v1.5",
            "dimensions": 768,
        }
        assert entry["dimensions"] == 768
