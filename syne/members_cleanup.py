"""Automatic revoke of idle members.

Runs once a day from the scheduler's maintenance pass. A member who has not
DMed Syne for `members.idle_revoke_days` days (default 60, 0 disables) is
handled exactly like the Delete button in the Telegram members menus:

  * Telegram / Discord: access is revoked (back to public + pending approval,
    hidden from /telemembers and /dcmembers). Their next DM is held for the
    owner to approve again.
  * WhatsApp: the DM entry is removed from whatsapp.allowed_chat_jids.

Nothing is erased: chat history and memories stay. Owner and family users,
blocked entries and groups are never touched.

Activity = the latest of (last DM session activity, users.updated_at). The
users.updated_at term means a user the owner just approved or edited is not
revoked again the next day before they had a chance to write. Group messages
cannot be counted: group sessions are stored per group, not per sender.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

logger = logging.getLogger("syne.members_cleanup")

DEFAULT_IDLE_DAYS = 60
_PROTECTED_LEVELS = ("owner", "family")


def pick_idle(rows: Iterable[dict], now: datetime, days: int) -> list[dict]:
    """Rows whose last_active is older than `days` before `now`.

    Owner / family / blocked rows and rows already pending approval are
    skipped. A row without last_active is skipped too (activity unknown).
    """
    if days <= 0:
        return []
    cutoff = now - timedelta(days=days)
    out = []
    for r in rows:
        if (r.get("access_level") or "public") in _PROTECTED_LEVELS:
            continue
        if r.get("access_level") == "blocked":
            continue
        if (r.get("preferences") or {}).get("pending_approval"):
            continue
        last = r.get("last_active")
        if last is None or last >= cutoff:
            continue
        out.append(r)
    return out


async def _activity_rows(conn, platforms: list[str]) -> list[dict]:
    import json
    from .tools.history import group_session_sql

    rows = await conn.fetch(
        "SELECT u.id, u.platform, u.platform_id, u.name, u.display_name, "
        "u.access_level, u.preferences, "
        "GREATEST(u.updated_at, u.created_at, ("
        "  SELECT MAX(s.updated_at) FROM sessions s WHERE s.user_id = u.id "
        f"  AND NOT {group_session_sql('s')} "
        "  AND s.platform_chat_id NOT LIKE 'scheduled:%'"
        ")) AS last_active "
        "FROM users u WHERE u.active = true AND u.platform = ANY($1::text[])",
        platforms,
    )
    result = []
    for row in rows:
        d = dict(row)
        if isinstance(d.get("preferences"), str):
            try:
                d["preferences"] = json.loads(d["preferences"])
            except Exception:
                d["preferences"] = {}
        result.append(d)
    return result


async def _revoke_tg_dc(rows: list[dict], days: int) -> list[str]:
    """Revoke idle Telegram / Discord users. Returns report lines."""
    from .communication.registry import (
        CHANNEL_DISCORD, CHANNEL_TELEGRAM, get_channel,
    )

    tg = get_channel(CHANNEL_TELEGRAM)
    if tg is None or not hasattr(tg, "_revoke_member"):
        if rows:
            logger.warning("[idle] Telegram channel not live; revoke skipped")
        return []
    dc = get_channel(CHANNEL_DISCORD)
    now = datetime.now(timezone.utc)
    lines = []
    for r in pick_idle(rows, now, days):
        platform, pid = r["platform"], str(r["platform_id"])
        try:
            ok = await tg._revoke_member(platform, pid)
        except Exception as e:
            logger.error(f"[idle] revoke failed for {platform}:{pid}: {e}")
            continue
        if not ok:
            continue
        if platform == "telegram":
            getattr(tg, "_pending_notified", set()).discard(f"pending_notify:{pid}")
        elif dc is not None:
            getattr(dc, "_pending_notified", set()).discard(f"dm:{pid}")
        name = r.get("display_name") or r.get("name") or pid
        lines.append(
            f"• {platform.capitalize()}: {html.escape(str(name))} "
            f"(last active {r['last_active']:%Y-%m-%d})"
        )
        logger.info(f"[idle] revoked {platform}:{pid} ({name})")
    return lines


async def _remove_idle_wa(conn, days: int) -> list[str]:
    """Drop idle DM entries from whatsapp.allowed_chat_jids."""
    import json
    from .db.models import get_config, set_config

    raw = await get_config("whatsapp.allowed_chat_jids", default=[])
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list) or not raw:
        return []

    users = {str(r["platform_id"]): r for r in await _activity_rows(conn, ["whatsapp"])}
    now = datetime.now(timezone.utc)

    to_remove: dict[str, tuple[str, dict]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        jid = str(item.get("jid") or "").strip()
        if not jid or jid.endswith("@g.us") or item.get("blocked"):
            continue
        local = jid.lstrip("+").split("@", 1)[0].split(":", 1)[0]
        user = users.get(local)
        if user is None:
            # Entry stored under a LID: resolve it to the phone number.
            try:
                pn = await conn.fetchval(
                    "SELECT pn FROM whatsmeow_lid_map WHERE lid = $1", local
                )
            except Exception:
                pn = None
            if pn:
                user = users.get(str(pn).split("@", 1)[0])
        if user is None:
            continue  # activity unknown: never remove blind
        if pick_idle([user], now, days):
            to_remove[jid] = (item.get("name") or user.get("name") or local, user)

    if not to_remove:
        return []

    async def _write():
        fresh = await get_config("whatsapp.allowed_chat_jids", default=[])
        if isinstance(fresh, str):
            fresh = json.loads(fresh)
        kept = [
            it for it in fresh
            if not (isinstance(it, dict) and str(it.get("jid") or "").strip() in to_remove)
        ]
        await set_config("whatsapp.allowed_chat_jids", kept)

    # Same lock whatsapp.py uses for auto-registration, so a member registered
    # by a message arriving right now is not overwritten.
    from .communication.registry import CHANNEL_WHATSAPP, get_channel
    wa = get_channel(CHANNEL_WHATSAPP)
    lock = getattr(wa, "_members_lock", None)
    if lock is not None:
        async with lock:
            await _write()
    else:
        await _write()

    lines = []
    for jid, (name, user) in to_remove.items():
        logger.info(f"[idle] removed WhatsApp allowlist entry {jid} ({name})")
        lines.append(
            f"• WhatsApp: {html.escape(str(name))} "
            f"(last active {user['last_active']:%Y-%m-%d})"
        )
    return lines


async def revoke_idle_members() -> int:
    """Daily pass. Returns how many members were revoked/removed."""
    from .db.connection import get_connection
    from .db.models import get_config

    try:
        days = int(await get_config("members.idle_revoke_days", DEFAULT_IDLE_DAYS) or 0)
    except (TypeError, ValueError):
        days = DEFAULT_IDLE_DAYS
    if days <= 0:
        return 0

    async with get_connection() as conn:
        tg_dc_rows = await _activity_rows(conn, ["telegram", "discord"])
        wa_lines = await _remove_idle_wa(conn, days)
    lines = await _revoke_tg_dc(tg_dc_rows, days) + wa_lines
    if lines:
        await _notify_owner(days, lines)
    return len(lines)


async def _notify_owner(days: int, lines: list[str]) -> None:
    from .communication.registry import CHANNEL_TELEGRAM, get_channel
    from .db.models import get_first_owner

    tg = get_channel(CHANNEL_TELEGRAM)
    owner = await get_first_owner("telegram")
    bot = getattr(tg, "_bot", None)
    if bot is None or not owner:
        return
    text = (
        f"🧹 <b>Idle members removed</b> (no DM for {days}+ days)\n\n"
        + "\n".join(lines)
        + "\n\nChat history and memories are kept. Telegram/Discord users "
        "must be approved again; WhatsApp users re-register on their next "
        "triggered message."
    )
    try:
        await bot.send_message(chat_id=int(owner["platform_id"]), text=text, parse_mode="HTML")
    except Exception as e:
        logger.warning(f"[idle] owner notify failed: {e}")
