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

import json
import logging
import os
import re

logger = logging.getLogger("syne.tools.send_message")

# Global reference to telegram channel (set by channel on start)
_telegram_channel = None


def set_telegram_channel(channel):
    """Set the telegram channel reference for message sending."""
    global _telegram_channel
    _telegram_channel = channel


# ── Attachments ──────────────────────────────────────────────
# Files ride on the same MEDIA: protocol every channel already parses
# (extract_all_media). The difference from send_file / memory_get_file:
# those hand their MEDIA: line to the CURRENT turn's final reply, which a
# scheduled task never delivers (scheduler.echo_response_to_creator) and
# which can never reach a different chat. Here the trailer is part of the
# message itself, so it lands wherever chat_id points.

_TELEGRAM_CAPTION_MAX = 1024
_MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024  # Telegram Bot API upload limit
_MEMORY_REF_RE = re.compile(r"^memory:\s*#?(\d+)$", re.IGNORECASE)


def _workspace_root() -> str:
    """Resolved workspace/ directory (same one file_write is sandboxed to)."""
    try:
        from .file_ops import _workspace_dir
    except Exception:
        _workspace_dir = None
    root = _workspace_dir or os.path.join(os.getcwd(), "workspace")
    return os.path.realpath(root)


def _normalize_attachments(attachments) -> list[str]:
    """Accept a list, a JSON-array string, or a single string."""
    if not attachments:
        return []
    if isinstance(attachments, str):
        s = attachments.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if str(a).strip()]
        return [s]
    if isinstance(attachments, (list, tuple)):
        return [str(a).strip() for a in attachments if str(a).strip()]
    return [str(attachments).strip()]


async def _fetch_memory_blob(memory_id: int):
    """Category + attached file of one memory (file columns NULL if none)."""
    from ..db.connection import get_connection

    async with get_connection() as conn:
        return await conn.fetchrow(
            """
            SELECT m.category, b.filename, b.content
            FROM memory m
            LEFT JOIN memory_blobs b ON b.memory_id = m.id
            WHERE m.id = $1
            """,
            memory_id,
        )


async def _materialize_memory_file(memory_id: int, access_level: str) -> tuple[str | None, str]:
    """Write a memory's attached file to workspace/temp/. Returns (path, error).

    Same permission gate (Rule 760) and same file naming as memory_get_file.
    """
    row = await _fetch_memory_blob(memory_id)
    if not row:
        return None, f"memory #{memory_id} not found"

    from ..security import check_rule_760

    allowed, reason = check_rule_760(row["category"] or "", access_level)
    if not allowed:
        return None, reason
    if row["content"] is None:
        return None, f"memory #{memory_id} has no attached file"

    temp_dir = os.path.join(_workspace_root(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    fname = row["filename"] or f"memory_{memory_id}.bin"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", fname)[:120] or f"memory_{memory_id}"
    out_path = os.path.join(temp_dir, f"mem{memory_id}_{safe}")
    with open(out_path, "wb") as f:
        f.write(bytes(row["content"]))
    return out_path, ""


async def _resolve_attachments(items: list[str], access_level: str) -> tuple[list[str], str]:
    """Turn attachment items into existing file paths. Returns (paths, error).

    'memory:<id>' -> the memory's file, written to workspace/temp/.
    Anything else -> a path that must resolve inside workspace/ (relative
    paths are taken from workspace/). Any bad item fails the whole send, so
    a message never goes out silently missing its file.
    """
    root = _workspace_root()
    paths: list[str] = []
    for item in items:
        m = _MEMORY_REF_RE.match(item)
        if m:
            path, err = await _materialize_memory_file(int(m.group(1)), access_level)
            if err:
                return [], f"Error: attachment '{item}': {err}"
        else:
            p = os.path.expanduser(item)
            if not os.path.isabs(p):
                p = os.path.join(root, p)
            path = os.path.realpath(p)
            if not path.startswith(root + os.sep):
                return [], (
                    f"Error: attachment '{item}' is outside workspace/. Only files "
                    f"inside workspace/ or memory:<id> can be attached."
                )
        if not os.path.isfile(path):
            return [], f"Error: attachment '{item}' not found."
        size = os.path.getsize(path)
        if size == 0:
            return [], f"Error: attachment '{item}' is empty."
        if size > _MAX_ATTACHMENT_SIZE:
            return [], (
                f"Error: attachment '{item}' is too large "
                f"({size / (1024 * 1024):.1f} MB, limit 50 MB)."
            )
        if path not in paths:
            paths.append(path)
    return paths, ""


async def send_message_handler(
    chat_id: str,
    message: str,
    reply_to_message_id: int = 0,
    relay_from: str = "",
    platform: str = "telegram",
    attachments: list | str | None = None,
    _scheduled: bool = False,
    _access_level: str = "public",
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
        attachments: Optional files sent with the message (the text becomes
            the caption): 'memory:<id>' or a path inside workspace/.
        _scheduled: Injected by tool registry when called from scheduled context.
        _access_level: Injected by tool registry -- the caller's access level.

    Returns:
        Success or error message.
    """
    global _telegram_channel

    if not chat_id:
        return "Error: chat_id is required"

    items = _normalize_attachments(attachments)
    message = message or ""
    if not message.strip() and not items:
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

    # Attachments become MEDIA: trailers on the message itself. Resolved only
    # after the group guard, so a refused send never writes files.
    media_paths: list[str] = []
    if items:
        if plat == "telegram" and len(message.strip()) > _TELEGRAM_CAPTION_MAX:
            return (
                f"Error: with attachments the message becomes the caption, and "
                f"Telegram captions are limited to {_TELEGRAM_CAPTION_MAX} characters "
                f"(this one is {len(message.strip())}). Shorten it, or send the long "
                f"text in a separate send_message call without attachments."
            )
        media_paths, err = await _resolve_attachments(items, _access_level)
        if err:
            return err
        trailer = "\n".join(f"MEDIA: {p}" for p in media_paths)
        message = f"{message.rstrip()}\n\n{trailer}" if message.strip() else trailer
    _att = f" with {len(media_paths)} attachment(s)" if media_paths else ""

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

        # Discord channel ids carry no group marker, so ask the channel.
        # Same rule as above: a server channel only from a scheduled task.
        if plat == "discord" and not _scheduled:
            is_group = getattr(channel, "is_group_target", None)
            if is_group is None or await is_group(dest):
                return (
                    "Error: Cannot send to a Discord server channel from a "
                    "regular conversation. Use manage_schedule to create a "
                    "scheduled task instead."
                )

        # Non-owners may only reach WhatsApp chats that already talk to Syne.
        # Messaging arbitrary numbers from the bot's account is how WhatsApp
        # numbers get banned; the owner keeps that call for themselves.
        if plat == "whatsapp" and _access_level != "owner":
            is_known = getattr(channel, "is_known_chat", None)
            if is_known is None or not await is_known(dest):
                return (
                    "Error: This WhatsApp number or group has not chatted with "
                    "Syne yet. Only the owner can message new WhatsApp contacts."
                )
        try:
            ok = await send_to_chat(dest, message)
        except Exception as e:
            logger.error(f"Error sending via {plat} to {dest}: {e}")
            return f"Error sending message via {plat}: {type(e).__name__}: {e}"
        if ok:
            logger.info(f"Message sent via {plat} to {dest}")
            return f"Message sent to {plat} chat {dest}{_att}"
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
            return f"Message sent to chat {chat_id} (message_id: {sent.message_id}){_att}"
        else:
            return f"Message sent to chat {chat_id} (no confirmation){_att}"

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
        "platform='whatsapp'/'discord' when it is not Telegram. "
        "Can carry files/images via attachments."
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
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional files sent with the message; the message text "
                    "becomes the caption. Each item: 'memory:<id>' (the file "
                    "stored with that memory) or a path inside workspace/ "
                    "(e.g. 'outputs/chart.png'). Use this, not send_file or "
                    "memory_get_file, whenever the file must reach a chat other "
                    "than the current one, and in every scheduled task. "
                    "Telegram: caption max 1024 characters when attaching."
                ),
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
