"""Tests for voice message support."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from syne.tools.voice import (
    get_groq_api_key,
    transcribe_audio,
    send_voice_handler,
    SEND_VOICE_TOOL,
    DEFAULT_STT_PROVIDER,
    DEFAULT_STT_MODEL,
)


class TestGroqApiKey:
    """Test Groq API key retrieval."""

    @pytest.mark.asyncio
    async def test_key_from_db_first(self):
        """Test that DB credential.groq_api_key is checked first."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.return_value = "db_key_123"
            
            key = await get_groq_api_key()
            
            assert key == "db_key_123"
            mock_config.assert_called_once_with("credential.groq_api_key", None)

    @pytest.mark.asyncio
    async def test_key_from_env_fallback(self):
        """Test environment variable fallback."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [None, None]  # DB returns None
            with patch.dict("os.environ", {"GROQ_API_KEY": "env_key_456"}):
                key = await get_groq_api_key()
                
                assert key == "env_key_456"

    @pytest.mark.asyncio
    async def test_key_from_openai_compat_fallback(self):
        """Test openai_compat_api_key fallback."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            # First call returns None (groq_api_key), second returns the fallback
            mock_config.side_effect = [None, "compat_key_789"]
            with patch.dict("os.environ", {}, clear=True):
                # Remove GROQ_API_KEY from env
                import os
                orig = os.environ.pop("GROQ_API_KEY", None)
                try:
                    key = await get_groq_api_key()
                    assert key == "compat_key_789"
                finally:
                    if orig:
                        os.environ["GROQ_API_KEY"] = orig

    @pytest.mark.asyncio
    async def test_no_key_available(self):
        """Test when no key is available."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.return_value = None
            with patch.dict("os.environ", {}, clear=True):
                import os
                orig = os.environ.pop("GROQ_API_KEY", None)
                try:
                    key = await get_groq_api_key()
                    assert key is None
                finally:
                    if orig:
                        os.environ["GROQ_API_KEY"] = orig


class TestTranscribeAudio:
    """Test audio transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_no_api_key(self):
        """Test transcription fails gracefully without API key."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [
                "groq",  # stt_provider
                "whisper-large-v3",  # stt_model
                None,  # groq_api_key
                None,  # openai_compat_api_key
            ]
            with patch.dict("os.environ", {}, clear=True):
                import os
                orig = os.environ.pop("GROQ_API_KEY", None)
                try:
                    success, result = await transcribe_audio(b"fake audio data")
                    
                    assert not success
                    assert "not configured" in result.lower()
                finally:
                    if orig:
                        os.environ["GROQ_API_KEY"] = orig

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        """Test successful transcription."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello, this is a test."}
        
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [
                "groq",  # stt_provider
                "whisper-large-v3",  # stt_model
                "test_api_key",  # groq_api_key
            ]
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                success, result = await transcribe_audio(b"fake audio data", "test.ogg")
                
                assert success
                assert result == "Hello, this is a test."
                mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_api_error(self):
        """Test transcription API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid audio format"
        
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [
                "groq",
                "whisper-large-v3",
                "test_api_key",
            ]
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                success, result = await transcribe_audio(b"fake audio data")
                
                assert not success
                assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_transcribe_timeout(self):
        """Test transcription timeout handling."""
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [
                "groq",
                "whisper-large-v3",
                "test_api_key",
            ]
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                success, result = await transcribe_audio(b"fake audio data")
                
                assert not success
                assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_transcribe_empty_result(self):
        """Test handling of empty transcription result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": ""}
        
        with patch("syne.tools.voice.get_config", new_callable=AsyncMock) as mock_config:
            mock_config.side_effect = [
                "groq",
                "whisper-large-v3",
                "test_api_key",
            ]
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client
                
                success, result = await transcribe_audio(b"fake audio data")
                
                assert not success
                assert "empty" in result.lower()


class TestSendVoiceTool:
    """Test the send_voice tool."""

    def test_tool_schema_structure(self):
        """Test that tool schema has required fields."""
        assert SEND_VOICE_TOOL["name"] == "send_voice"
        assert "description" in SEND_VOICE_TOOL
        assert "parameters" in SEND_VOICE_TOOL
        assert "handler" in SEND_VOICE_TOOL
        assert SEND_VOICE_TOOL["requires_access_level"] == "owner"

    def test_tool_parameters_schema(self):
        """Test that parameters schema is valid."""
        params = SEND_VOICE_TOOL["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert "text" in params["required"]

    @pytest.mark.asyncio
    async def test_handler_returns_not_implemented(self):
        """Test that handler returns not implemented message."""
        result = await send_voice_handler("Hello world", chat_id="123")
        
        assert "not" in result.lower()
        assert "configured" in result.lower() or "implemented" in result.lower()


class TestVoiceConfig:
    """Test voice configuration defaults."""

    def test_default_stt_provider(self):
        """Test default STT provider is groq."""
        assert DEFAULT_STT_PROVIDER == "groq"

    def test_default_stt_model(self):
        """Test default STT model is whisper-large-v3."""
        assert DEFAULT_STT_MODEL == "whisper-large-v3"
