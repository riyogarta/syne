"""Platform message ids of delivered assistant answers.

Each channel records the platform ids of every message it delivers on the
assistant row (metadata tg_message_ids / wa_message_ids / dc_message_ids).
WhatsApp and Discord use them to tell a reaction on one of Syne's replies
from chatter between people, and to show the agent which reply was
reacted to.

Moved out of syne/ratification.py when ratification was removed
(Riyo, 13 Nov 2026). Behaviour unchanged.
"""

import json
import logging
from typing import Optional

from .db.connection import get_connection

logger = logging.getLogger("syne.sent_ids")

_ID_KEYS = {
    "telegram": "tg_message_ids",
    "whatsapp": "wa_message_ids",
    "discord": "dc_message_ids",
}


def ids_key(platform: Optional[str]) -> Optional[str]:
    """metadata key holding the delivered message ids for this platform."""
    return _ID_KEYS.get((platform or "").lower())


def id_value(platform: Optional[str], raw):
    """Platform message id as stored: WhatsApp ids are strings, Telegram and
    Discord ids are integers. None when unusable."""
    if (platform or "").lower() == "whatsapp":
        s = str(raw or "").strip()
        return s or None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _find_sent(conn, platform: str, chat_id: str, platform_message_id):
    """Assistant row delivered as this platform message in this chat."""
    key, val = ids_key(platform), id_value(platform, platform_message_id)
    if key is None or val is None:
        return None
    return await conn.fetchrow(
        "SELECT m.id, m.session_id, m.content "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.platform = $1 AND s.platform_chat_id = $2 AND m.role = 'assistant' "
        f"AND m.metadata->'{key}' @> $3::jsonb "
        "ORDER BY m.id DESC LIMIT 1",
        platform, str(chat_id), json.dumps([val]),
    )


async def record_sent_ids(platform: str, chat_id: str, ids: list) -> None:
    """Attach the platform message ids of a delivered answer to the newest
    assistant row of that chat, so a later reaction can find its content."""
    key = ids_key(platform)
    vals = [v for v in (id_value(platform, i) for i in ids or []) if v is not None]
    if key is None or not vals:
        return
    try:
        async with get_connection() as conn:
            await conn.execute(
                f"""
                UPDATE messages SET metadata = jsonb_set(
                    coalesce(metadata, '{{}}'::jsonb), '{{{key}}}',
                    coalesce(metadata->'{key}', '[]'::jsonb) || $3::jsonb)
                WHERE id = (
                    SELECT m.id FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE s.platform = $1 AND s.platform_chat_id = $2
                      AND m.role = 'assistant' AND btrim(coalesce(m.content, '')) <> ''
                      AND m.created_at > now() - interval '10 minutes'
                    ORDER BY m.created_at DESC, m.id DESC LIMIT 1)
                """,
                platform, str(chat_id), json.dumps(vals),
            )
    except Exception as e:
        logger.warning(f"record_sent_ids failed for {platform}:{chat_id}: {e}")


async def lookup_sent_preview(platform: str, chat_id: str, platform_message_id) -> Optional[str]:
    """Content of the assistant answer delivered as this platform message,
    or None when it is not a mapped Syne reply."""
    try:
        async with get_connection() as conn:
            a = await _find_sent(conn, platform, chat_id, platform_message_id)
        return (a["content"] or "") if a else None
    except Exception as e:
        logger.debug(f"lookup_sent_preview failed for {platform}:{chat_id}: {e}")
        return None
