"""Tests for reaction support."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from syne.tools.reactions import (
    validate_emoji,
    send_reaction_handler,
    SUPPORTED_EMOJIS,
    SEND_REACTION_TOOL,
)


class TestEmojiValidation:
    """Test reaction emoji validation."""

    def test_valid_common_emojis(self):
        """Test that common reaction emojis are valid."""
        for emoji in ["👍", "👎", "❤️", "🔥", "😁", "🤔", "🎉", "👏"]:
            valid, msg = validate_emoji(emoji)
            assert valid, f"Emoji {emoji} should be valid: {msg}"

    def test_valid_extended_emojis(self):
        """Test that extended reaction emojis are valid."""
        for emoji in ["🤯", "🥰", "💯", "🤣", "👀", "🤝", "😈", "👻"]:
            valid, msg = validate_emoji(emoji)
            assert valid, f"Emoji {emoji} should be valid: {msg}"

    def test_invalid_emojis(self):
        """Test that unsupported emojis are rejected."""
        # Use emojis that are definitely NOT in the Telegram reaction set
        invalid_emojis = ["🧀", "🦑", "🎮", "🚀", "🌈", "🐕", "🍕", "🎸", "⭐"]
        for emoji in invalid_emojis:
            valid, msg = validate_emoji(emoji)
            assert not valid, f"Emoji {emoji} should be invalid"
            assert "not supported" in msg.lower()

    def test_empty_emoji(self):
        """Test that empty emoji is rejected."""
        valid, msg = validate_emoji("")
        assert not valid
        assert "required" in msg.lower()

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        valid, msg = validate_emoji("  👍  ")
        assert valid

    def test_supported_emojis_count(self):
        """Verify we have a reasonable number of supported emojis."""
        assert len(SUPPORTED_EMOJIS) >= 60, "Should have at least 60 supported emojis"
        assert len(SUPPORTED_EMOJIS) <= 100, "Sanity check: not too many emojis"


class TestSendReactionTool:
    """Test the send_reaction tool schema and handler."""

    def test_tool_schema_structure(self):
        """Test that tool schema has required fields."""
        assert SEND_REACTION_TOOL["name"] == "send_reaction"
        assert "description" in SEND_REACTION_TOOL
        assert "parameters" in SEND_REACTION_TOOL
        assert "handler" in SEND_REACTION_TOOL
        assert SEND_REACTION_TOOL["requires_access_level"] == "owner"

    def test_tool_parameters_schema(self):
        """Test that parameters schema is valid."""
        params = SEND_REACTION_TOOL["parameters"]
        assert params["type"] == "object"
        assert "message_id" in params["properties"]
        assert "emoji" in params["properties"]
        assert "message_id" in params["required"]

    def test_message_id_parameter(self):
        """Test message_id parameter definition."""
        params = SEND_REACTION_TOOL["parameters"]["properties"]
        assert params["message_id"]["type"] == "integer"

    def test_emoji_parameter_default(self):
        """Test emoji parameter has default."""
        params = SEND_REACTION_TOOL["parameters"]["properties"]
        assert params["emoji"]["default"] == "👍"


class TestSendReactionHandler:
    """Test the send_reaction handler function."""

    @pytest.mark.asyncio
    async def test_handler_no_channel(self):
        """Test handler returns error when channel not available."""
        with patch("syne.tools.reactions._telegram_channel", None):
            result = await send_reaction_handler(message_id=123, emoji="👍", chat_id="456")
            assert "error" in result.lower()
            assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_invalid_emoji(self):
        """Test handler validates emoji."""
        mock_channel = MagicMock()
        with patch("syne.tools.reactions._telegram_channel", mock_channel):
            result = await send_reaction_handler(message_id=123, emoji="🧀", chat_id="456")
            assert "error" in result.lower()
            assert "not supported" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_missing_chat_id(self):
        """Test handler requires chat_id."""
        mock_channel = MagicMock()
        with patch("syne.tools.reactions._telegram_channel", mock_channel):
            result = await send_reaction_handler(message_id=123, emoji="👍", chat_id="")
            assert "error" in result.lower()
            assert "chat_id" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_success(self):
        """Test handler success case."""
        mock_channel = MagicMock()
        mock_channel.send_reaction = AsyncMock(return_value=True)
        
        with patch("syne.tools.reactions._telegram_channel", mock_channel):
            result = await send_reaction_handler(message_id=123, emoji="👍", chat_id="456")
            assert "reacted" in result.lower()
            assert "👍" in result
            mock_channel.send_reaction.assert_called_once_with(
                chat_id=456,
                message_id=123,
                emoji="👍",
            )

    @pytest.mark.asyncio
    async def test_handler_failure(self):
        """Test handler failure case."""
        mock_channel = MagicMock()
        mock_channel.send_reaction = AsyncMock(return_value=False)
        
        with patch("syne.tools.reactions._telegram_channel", mock_channel):
            result = await send_reaction_handler(message_id=123, emoji="👍", chat_id="456")
            assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_handler_exception(self):
        """Test handler handles exceptions gracefully."""
        mock_channel = MagicMock()
        mock_channel.send_reaction = AsyncMock(side_effect=Exception("Network error"))
        
        with patch("syne.tools.reactions._telegram_channel", mock_channel):
            result = await send_reaction_handler(message_id=123, emoji="👍", chat_id="456")
            assert "error" in result.lower()
            assert "network" in result.lower()
