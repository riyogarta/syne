"""send_message — Send a message to any Telegram chat/user.

Enables:
- Proactive messaging: agent initiates conversation (reminders, alerts)
- Cross-session relay: forward info between chats (group → DM, DM → group)

Rule 700: Only owner can send to arbitrary chats.
Family can send to their own DM only.
"""

import logging

logger = logging.getLogger("syne.tools.send_message")

# Global reference to telegram channel (set by channel on start)
_telegram_channel = None


def set_telegram_channel(channel):
    """Set the telegram channel reference for message sending."""
    global _telegram_channel
    _telegram_channel = channel


async def send_message_handler(
    chat_id: str,
    message: str,
    reply_to_message_id: int = 0,
) -> str:
    """Send a message to a Telegram chat.

    Args:
        chat_id: Telegram chat ID (user ID for DMs, group ID for groups).
        message: Message text to send. Supports Markdown formatting.
        reply_to_message_id: Optional message ID to reply/quote to.

    Returns:
        Success or error message.
    """
    global _telegram_channel

    if not _telegram_channel:
        return "Error: Telegram channel not available"

    if not chat_id:
        return "Error: chat_id is required"

    if not message or not message.strip():
        return "Error: message cannot be empty"

    try:
        reply_to = reply_to_message_id if reply_to_message_id else None

        sent = await _telegram_channel._send_response_with_media(
            chat_id=int(chat_id),
            text=message,
            reply_to_message_id=reply_to,
        )

        if sent:
            logger.info(f"Message sent to chat {chat_id} (msg_id: {sent.message_id})")
            return f"Message sent to chat {chat_id} (message_id: {sent.message_id})"
        else:
            return f"Message sent to chat {chat_id} (no confirmation)"

    except Exception as e:
        logger.error(f"Error sending message to {chat_id}: {e}")
        error_str = str(e)
        if "chat not found" in error_str.lower():
            return f"Error: Chat {chat_id} not found. Bot may not have access to this chat."
        if "bot was blocked" in error_str.lower():
            return f"Error: Bot was blocked by user {chat_id}."
        return f"Error sending message: {error_str}"


# Tool metadata for registration
SEND_MESSAGE_TOOL = {
    "name": "send_message",
    "description": (
        "Send a message to any Telegram chat or user. "
        "Use for proactive messaging (reminders, alerts, notifications) "
        "and cross-session relay (forwarding info between chats). "
        "Requires chat_id — use user's Telegram ID for DMs, group ID for groups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": "Telegram chat ID. Use user ID for DMs (e.g. '12345'), group ID for groups (e.g. '-100xxx').",
            },
            "message": {
                "type": "string",
                "description": "Message text to send. Supports Markdown formatting.",
            },
            "reply_to_message_id": {
                "type": "integer",
                "description": "Optional: reply to a specific message ID in that chat.",
            },
        },
        "required": ["chat_id", "message"],
    },
    "handler": send_message_handler,
    "requires_access_level": "owner",
}
