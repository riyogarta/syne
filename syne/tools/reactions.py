"""Reactions Tool — send emoji reactions to messages."""

import logging

logger = logging.getLogger("syne.tools.reactions")

# Supported Telegram reaction emojis (as of Bot API 7.0+)
SUPPORTED_EMOJIS = {
    "👍", "👎", "❤️", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😢",
    "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴",
    "😍", "🐳", "❤️‍🔥", "🌚", "🌭", "💯", "🤣", "⚡️", "🍌", "🏆",
    "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈", "😴", "😭",
    "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍️",
    "🤗", "🫡", "🎅", "🎄", "☃️", "💅", "🤪", "🗿", "🆒", "💘",
    "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂️", "🤷", "🤷‍♀️", "😡",
}


def validate_emoji(emoji: str) -> tuple[bool, str]:
    """Validate that an emoji is supported by Telegram reactions.
    
    Args:
        emoji: The emoji to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not emoji:
        return False, "Emoji is required"
    
    # Normalize some common variants
    normalized = emoji.strip()
    
    if normalized in SUPPORTED_EMOJIS:
        return True, ""
    
    # Check for close matches (some emojis have variants)
    # e.g., ❤ vs ❤️
    for supported in SUPPORTED_EMOJIS:
        if normalized in supported or supported in normalized:
            return True, supported  # Return the normalized version
    
    return False, f"Emoji '{emoji}' is not supported for Telegram reactions"


# Global reference to telegram channel (set by channel on start)
_telegram_channel = None


def set_telegram_channel(channel):
    """Set the telegram channel reference for reaction sending."""
    global _telegram_channel
    _telegram_channel = channel


async def send_reaction_handler(
    message_id: int,
    emoji: str = "👍",
    chat_id: str = "",
) -> str:
    """Send a reaction to a message.
    
    Args:
        message_id: The message ID to react to
        emoji: The reaction emoji (default: 👍)
        chat_id: Optional chat ID (uses current conversation if empty)
        
    Returns:
        Success or error message
    """
    global _telegram_channel
    
    if not _telegram_channel:
        return "Error: Telegram channel not available"
    
    # Validate emoji
    valid, result = validate_emoji(emoji)
    if not valid:
        return f"Error: {result}"
    
    # Use normalized emoji if returned
    if result:
        emoji = result
    
    try:
        # Get chat_id from current conversation context if not provided
        if not chat_id:
            # The tool execution context should provide this
            return "Error: chat_id is required (context not available)"
        
        success = await _telegram_channel.send_reaction(
            chat_id=int(chat_id),
            message_id=int(message_id),
            emoji=emoji,
        )
        
        if success:
            return f"Reacted with {emoji} to message {message_id}"
        else:
            return f"Failed to send reaction (bot may lack permission)"
            
    except Exception as e:
        logger.error(f"Error sending reaction: {e}")
        return f"Error: {str(e)}"


# Tool metadata for registration
SEND_REACTION_TOOL = {
    "name": "send_reaction",
    "description": "Send an emoji reaction to a message. Use to acknowledge, express sentiment, or provide feedback on messages without typing a full response.",
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "integer",
                "description": "The message ID to react to (from recent message context)",
            },
            "emoji": {
                "type": "string",
                "description": "Reaction emoji. Supported: 👍 👎 ❤️ 🔥 🥰 👏 😁 🤔 🤯 😢 🎉 🤩 🙏 👌 💯 🤣 etc.",
                "default": "👍",
            },
            "chat_id": {
                "type": "string",
                "description": "Chat ID (usually provided automatically from context)",
            },
        },
        "required": ["message_id"],
    },
    "handler": send_reaction_handler,
    "requires_access_level": "owner",  # Rule 700 — reactions represent the bot's voice
}
