"""send_message — Send a message to any chat/user on any live channel.

Enables:
- Proactive messaging: agent initiates conversation (reminders, alerts)
- Cross-session relay: forward info between chats (group → DM, DM → group)
- Cross-CHANNEL send: Telegram (default), WhatsApp, Discord

Telegram keeps its original module-level reference, untouched. Every other
platform is reached through communication.registry, which is exactly what
that registry exists for: each channel publishes an identical
`async send_to_chat(chat_id, text) -> bool`, so routing is one lookup and
one call with no per-channel special casing here.

Permission 770: Owner and family can send messages.
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
    relay_from: str = "",
    platform: str = "telegram",
    _scheduled: bool = False,
) -> str:
    """Send a message to a chat on any connected channel.

    Args:
        chat_id: Destination id in that platform's own format — Telegram chat
            ID, WhatsApp JID (628...@s.whatsapp.net / ...@g.us), Discord
            channel id.
        message: Message text to send. Supports Markdown formatting.
        reply_to_message_id: Optional message ID to reply/quote to. Telegram
            only; silently ignored elsewhere since send_to_chat has no
            equivalent.
        platform: "telegram" (default), "whatsapp", or "discord".
        _scheduled: Injected by tool registry when called from scheduled context.

    Returns:
        Success or error message.
    """
    global _telegram_channel

    if not chat_id:
        return "Error: chat_id is required"

    if not message or not message.strip():
        return "Error: message cannot be empty"

    # Deterministic relay attribution: if this is a forwarded/relayed message,
    # the code (not the LLM) prepends the sender line so the recipient always
    # knows who it's from. Proactive Molt messages leave relay_from empty.
    if relay_from and relay_from.strip():
        message = f"Pesan dari {relay_from.strip()}:\n\n{message}"

    plat = (platform or "telegram").strip().lower() or "telegram"
    dest = str(chat_id)

    # Guard: sending to groups requires scheduled context. Same rule on every
    # platform, only the id shape differs — Telegram group ids start with '-',
    # WhatsApp group JIDs end with '@g.us'. Without the second test a WhatsApp
    # group would sail straight past a guard that was written when '-' was the
    # only group id Syne could see.
    if (dest.startswith("-") or dest.endswith("@g.us")) and not _scheduled:
        return (
            "Error: Cannot send to groups from a regular conversation. "
            "Use manage_schedule to create a scheduled task instead."
        )

    # Non-Telegram platforms go through the channel registry. Returning the
    # channel's own failure reason matters: "not connected" (WhatsApp unpaired
    # or disabled) and "send failed" are different problems for the owner.
    if plat != "telegram":
        from ..communication.registry import get_channel

        channel = get_channel(plat)
        if channel is None:
            return (
                f"Error: {plat} channel is not connected. "
                f"Nothing can be sent until it registers."
            )
        send_to_chat = getattr(channel, "send_to_chat", None)
        if send_to_chat is None:
            return f"Error: {plat} channel does not support sending."
        try:
            ok = await send_to_chat(dest, message)
        except Exception as e:
            logger.error(f"Error sending via {plat} to {dest}: {e}")
            return f"Error sending message via {plat}: {type(e).__name__}: {e}"
        if ok:
            logger.info(f"Message sent via {plat} to {dest}")
            return f"Message sent to {plat} chat {dest}"
        return f"Error: {plat} refused the send to {dest} (see logs)."

    if not _telegram_channel:
        return "Error: Telegram channel not available"

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
        "Send a message to any chat or user on a connected channel "
        "(Telegram, WhatsApp, or Discord). "
        "Use for proactive messaging (reminders, alerts, notifications) "
        "and cross-session relay (forwarding info between chats). "
        "Requires chat_id in the target platform's own format, plus "
        "platform='whatsapp'/'discord' when it is not Telegram."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "chat_id": {
                "type": "string",
                "description": (
                    "Destination id, in the target platform's own format. "
                    "Telegram: user ID for DMs (e.g. '12345'), group ID for "
                    "groups (e.g. '-100xxx'). WhatsApp: full JID "
                    "(e.g. '628111681624@s.whatsapp.net', group '...@g.us'). "
                    "Discord: channel id."
                ),
            },
            "platform": {
                "type": "string",
                "enum": ["telegram", "whatsapp", "discord"],
                "description": (
                    "Which channel to send through. Defaults to 'telegram'. "
                    "Set 'whatsapp' or 'discord' to reach those channels; the "
                    "channel must be connected."
                ),
            },
            "message": {
                "type": "string",
                "description": "Message text to send. Supports Markdown formatting.",
            },
            "reply_to_message_id": {
                "type": "integer",
                "description": "Optional: reply to a specific message ID in that chat.",
            },
            "relay_from": {
                "type": "string",
                "description": (
                    "Set this to the SENDER's name ONLY when relaying/forwarding "
                    "someone else's message (e.g. 'Agha'). The system will "
                    "automatically prefix the delivered text with 'Pesan dari "
                    "<name>:' so the recipient knows who it's from. Leave EMPTY "
                    "for your own proactive messages (reminders, alerts)."
                ),
            },
        },
        "required": ["chat_id", "message"],
    },
    "handler": send_message_handler,
    "permission": 0o770,
}
