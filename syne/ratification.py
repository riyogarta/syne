"""User ratification of assistant replies (Riyo, 10 Nov 2026).

history_search anchors on USER messages only, so a fact that only the
assistant wrote is unreachable when the user merely answered "ok". Following
grounding (Clark & Brennan, 1991), a reply becomes shared knowledge once the
user marks acceptance. When a user message is classified as approval, its
embedding is replaced by the embedding of that message + the assistant reply
it approves. The anchor stays the user row, so an assistant reply is only
retrievable once the user ratified it.

Two channels:
  * text     - a short reply made only of approval / acknowledgement words.
               Clear approval ("setuju", "betul") is decided by the lexicon;
               a bare "ok" is ambiguous (agree vs. just received - Jurafsky
               et al., 1997) and is decided by the evaluator model.
  * reaction - an emoji on a specific bot message. Approval emojis ratify,
               rejection emojis mark the reply disputed, and removing the
               emoji revokes it. Bot messages are mapped to their DB rows via
               metadata.platform_message_ids, recorded when they are sent.

State lives in messages.metadata (no schema change):
  user row:      ratification = {kind, label, ratifies, decided_by, status, at}
  assistant row: ratification = {status: ratified|disputed, agree, reject}
                 platform_message_ids = [telegram message ids of its chunks]

Ratification is not verification: a ratified reply can still be wrong.
Switches: ratification.enabled / text_enabled / reaction_enabled / text_llm.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx

from .db.connection import get_connection

logger = logging.getLogger("syne.ratification")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "enabled": True,
    "text": True,
    "reaction": True,
    "text_llm": True,
    "max_chars": 120,
    "agree_emojis": ["👍", "❤️", "👌", "💯"],
    "reject_emojis": ["👎"],
}
_KEYS = {
    "enabled": "ratification.enabled",
    "text": "ratification.text_enabled",
    "reaction": "ratification.reaction_enabled",
    "text_llm": "ratification.text_llm",
    "max_chars": "ratification.max_chars",
    "agree_emojis": "ratification.agree_emojis",
    "reject_emojis": "ratification.reject_emojis",
}


async def settings() -> dict:
    """Read on every call so a config change applies without a restart."""
    from .db.models import get_config
    out = dict(_DEFAULTS)
    for name, key in _KEYS.items():
        try:
            val = await get_config(key, out[name])
        except Exception:
            continue
        if val is not None:
            out[name] = val
    for name in ("enabled", "text", "reaction", "text_llm"):
        out[name] = bool(out[name])
    try:
        out["max_chars"] = int(out["max_chars"])
    except (TypeError, ValueError):
        out["max_chars"] = _DEFAULTS["max_chars"]
    return out


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

# Channels prepend untrusted JSON blocks ("Conversation info (untrusted
# metadata):", "Replied message (untrusted content, ...):") to the stored
# user message. Strip them to get what the person actually typed.
_PREFIX_RE = re.compile(
    r"\A(?:[^\n]*\(untrusted[^\n]*:\n```json\n.*?\n```\s*)+", re.DOTALL
)


def strip_context_prefix(text: str) -> str:
    return _PREFIX_RE.sub("", text or "").strip()


_AGREE = {
    "setuju", "sepakat", "betul", "benar", "bener", "tepat", "persis",
    "valid", "acc", "approve", "approved", "agree", "agreed", "correct",
    "exactly", "right", "mantap", "mantab", "cocok", "sesuai",
}
_ACK = {
    "ok", "oke", "okay", "okey", "okeh", "okee", "oce", "sip", "sipp", "siap",
    "noted", "baik", "baiklah", "ya", "iya", "yaa", "iyaa", "yes", "yup", "yep",
    "lanjut", "lanjutkan", "gas", "good", "nice", "cool", "thanks", "thank",
    "you", "makasih", "terima", "kasih", "thx", "tq", "done", "beres", "paham",
    "mengerti", "ngerti", "understood", "got", "it",
}
_FILLER = {
    "nah", "deh", "dong", "sudah", "udah", "aja", "saja", "itu", "ini", "kok",
    "banget", "sekali", "lah", "kak", "mas", "bro", "molt", "yah", "wah", "oh",
    "jadi", "sih", "kan", "juga", "semua", "yang", "aku", "saya",
}
_VOCAB = _AGREE | _ACK | _FILLER
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)


def _norm_emoji(e: str) -> str:
    return (e or "").replace("\ufe0f", "").strip()


def classify_lexical(text: str, max_chars: int = 120, agree_emojis=None) -> Optional[str]:
    """'agree' | 'ambiguous' | None (not an approval-type reply at all).

    Only short replies made entirely of approval/acknowledgement words (plus
    fillers) qualify; anything with other content ("setuju, tapi ...",
    "tidak setuju") is None and left alone.
    """
    t = (text or "").strip()
    if not t or len(t) > max_chars:
        return None
    emojis = {_norm_emoji(e) for e in (agree_emojis or _DEFAULTS["agree_emojis"])}
    squeezed = _norm_emoji("".join(t.split()))
    if squeezed and all(ch in "".join(emojis) for ch in squeezed) and any(
        e and e in squeezed for e in emojis
    ):
        return "agree"
    toks = _NONWORD.sub(" ", t.lower()).split()
    if not toks or any(tok not in _VOCAB for tok in toks):
        return None
    if any(tok in _AGREE for tok in toks):
        return "agree"
    if any(tok in _ACK for tok in toks):
        return "ambiguous"
    return None


def is_acknowledgement(text: str) -> bool:
    """True for approval/ack replies and reaction events - never memory-worthy."""
    t = strip_context_prefix(text)
    if t.lower().startswith("[reaction:"):
        return True
    return classify_lexical(t) is not None


def reaction_label(emojis, cfg: Optional[dict] = None) -> Optional[str]:
    cfg = cfg or _DEFAULTS
    agree = {_norm_emoji(e) for e in cfg.get("agree_emojis") or []}
    reject = {_norm_emoji(e) for e in cfg.get("reject_emojis") or []}
    norm = {_norm_emoji(e) for e in emojis or []}
    if norm & reject:
        return "reject"
    if norm & agree:
        return "agree"
    return None


def _meta(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Ambiguous "ok": agree or just received?
# ---------------------------------------------------------------------------

_DECIDE_PROMPT = (
    "You label one chat reply. The assistant wrote a message and the user "
    "answered briefly.\n"
    "AGREE = the user accepts the assistant's message as correct, or approves "
    "what it proposed.\n"
    "ACK = the user only acknowledges receipt, thanks, or asks to go on, "
    "without endorsing the content.\n"
    "Answer with exactly one word: AGREE or ACK."
)


async def _llm_decide(provider, user_text: str, assistant_text: str) -> tuple:
    """Returns (label, decided_by). Any failure falls back to 'ack'."""
    from .db.models import get_config
    try:
        driver = await get_config("memory.evaluator_driver", "ollama")
        model = await get_config("memory.evaluator_model", "qwen3:0.6b")
    except Exception:
        driver, model = "ollama", "qwen3:0.6b"
    prompt = f"Assistant message:\n{(assistant_text or '')[:1500]}\n\nUser reply: {user_text}"
    try:
        if driver == "ollama":
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _DECIDE_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                )
                resp.raise_for_status()
                out = resp.json().get("message", {}).get("content", "")
        elif provider is not None:
            from .llm.provider import ChatMessage
            resp = await provider.chat(
                messages=[
                    ChatMessage(role="system", content=_DECIDE_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0,
                max_tokens=None,
                thinking_budget=0,
            )
            out = resp.content or ""
        else:
            return "ack", "fallback"
    except Exception as e:
        logger.warning(f"ratification: classifier call failed: {type(e).__name__}: {e}")
        return "ack", "fallback"
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.DOTALL).strip().upper()
    if out.startswith("AGREE"):
        return "agree", "llm"
    if out.startswith("ACK"):
        return "ack", "llm"
    return "ack", "fallback"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _set_meta_key(conn, message_id: int, key: str, value) -> None:
    await conn.execute(
        "UPDATE messages SET metadata = jsonb_set(coalesce(metadata, '{}'::jsonb), "
        "$2::text[], $3::jsonb, true) WHERE id = $1",
        int(message_id), [key], json.dumps(value, ensure_ascii=False),
    )


async def previous_assistant(conn, session_id: int, message_id: int) -> Optional[dict]:
    """The assistant reply this user message answers: the latest non-empty
    assistant row before it, with no other user message in between."""
    row = await conn.fetchrow(
        """
        WITH me AS (SELECT created_at FROM messages WHERE id = $2)
        SELECT a.id, a.content, a.created_at FROM messages a, me
        WHERE a.session_id = $1 AND a.role = 'assistant'
          AND btrim(coalesce(a.content, '')) <> ''
          AND a.created_at <= me.created_at AND a.id <> $2
        ORDER BY a.created_at DESC, a.id DESC
        LIMIT 1
        """,
        int(session_id), int(message_id),
    )
    if row is None:
        return None
    between = await conn.fetchval(
        """
        SELECT count(*) FROM messages
        WHERE session_id = $1 AND role = 'user' AND id <> $2
          AND created_at > $3
          AND created_at <= (SELECT created_at FROM messages WHERE id = $2)
        """,
        int(session_id), int(message_id), row["created_at"],
    )
    if between:
        return None
    return dict(row)


async def find_by_platform_id(conn, platform: str, chat_id: str, platform_message_id) -> Optional[dict]:
    """Assistant row that was delivered as this platform message."""
    try:
        pmid = int(platform_message_id)
    except (TypeError, ValueError):
        return None
    row = await conn.fetchrow(
        """
        SELECT m.id, m.session_id, m.content, m.created_at
        FROM messages m JOIN sessions s ON s.id = m.session_id
        WHERE s.platform = $1 AND s.platform_chat_id = $2
          AND m.role = 'assistant'
          AND m.metadata -> 'platform_message_ids' @> $3::jsonb
        ORDER BY m.id DESC
        LIMIT 1
        """,
        platform, str(chat_id), json.dumps([pmid]),
    )
    return dict(row) if row else None


async def refresh_status(conn, assistant_id: int, since) -> Optional[str]:
    """Recompute the assistant row's status from its active ratifications."""
    rows = await conn.fetch(
        """
        SELECT metadata -> 'ratification' ->> 'label' AS label, count(*) AS n
        FROM messages
        WHERE role = 'user' AND created_at >= $2
          AND metadata -> 'ratification' ->> 'ratifies' = $1
          AND metadata -> 'ratification' ->> 'status' = 'active'
          AND metadata -> 'ratification' ->> 'label' IN ('agree', 'reject')
        GROUP BY 1
        """,
        str(assistant_id), since,
    )
    counts = {r["label"]: int(r["n"]) for r in rows}
    agree, reject = counts.get("agree", 0), counts.get("reject", 0)
    status = "disputed" if reject else ("ratified" if agree else None)
    if status:
        await _set_meta_key(conn, assistant_id, "ratification",
                            {"status": status, "agree": agree, "reject": reject})
    else:
        await conn.execute(
            "UPDATE messages SET metadata = coalesce(metadata, '{}'::jsonb) - 'ratification' "
            "WHERE id = $1",
            int(assistant_id),
        )
    return status


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def resolve(
    *,
    provider,
    session_id: int,
    message_id: int,
    content: str,
    hint: Optional[dict],
    max_chars: int = 4000,
) -> Optional[str]:
    """Classify a just-saved user row. Returns the text to embed in place of
    the row content when it ratifies (or disputes) an assistant reply, else
    None (embed the row as usual). Called from the background embed task."""
    if not hint:
        return None
    cfg = await settings()
    if not cfg["enabled"]:
        return None
    now = datetime.now(timezone.utc).isoformat()

    if hint.get("is_reaction"):
        if not cfg["reaction"]:
            return None
        emojis = [e for e in (hint.get("reaction_emojis") or []) if e]
        label = reaction_label(emojis, cfg)
        if not label or hint.get("platform_message_id") is None:
            return None
        async with get_connection() as conn:
            asst = await find_by_platform_id(
                conn, hint.get("platform") or "telegram",
                str(hint.get("chat_id") or ""), hint.get("platform_message_id"),
            )
            if not asst:
                logger.info(
                    f"ratification: reaction on platform msg {hint.get('platform_message_id')} "
                    f"has no mapped assistant row (sent before mapping, or not a bot reply)"
                )
                return None
            kind, user_text, decided_by = "reaction", "Reaksi " + " ".join(emojis), "emoji"
            rec = {"kind": kind, "label": label, "ratifies": asst["id"],
                   "decided_by": decided_by, "status": "active", "at": now}
            await _set_meta_key(conn, message_id, "ratification", rec)
            status = await refresh_status(conn, asst["id"], asst["created_at"])
    else:
        if not cfg["text"]:
            return None
        text = hint.get("original_text") or strip_context_prefix(content)
        cand = classify_lexical(text, cfg["max_chars"], cfg["agree_emojis"])
        if cand is None:
            return None
        async with get_connection() as conn:
            asst = await previous_assistant(conn, session_id, message_id)
        if not asst:
            return None
        if cand == "agree":
            label, decided_by = "agree", "lexicon"
        elif cfg["text_llm"]:
            label, decided_by = await _llm_decide(provider, text, asst["content"])
        else:
            label, decided_by = "ack", "lexicon"
        kind, user_text = "text", text.strip()
        rec = {"kind": kind, "label": label, "ratifies": asst["id"],
               "decided_by": decided_by, "status": "active", "at": now}
        async with get_connection() as conn:
            # Kept for 'ack' too: the label is what gets compared with
            # human annotation (agreement classification kappa).
            await _set_meta_key(conn, message_id, "ratification", rec)
            if label == "ack":
                logger.info(f"ratification: msg {message_id} = ack ({decided_by}), embedded as-is")
                return None
            status = await refresh_status(conn, asst["id"], asst["created_at"])

    logger.info(
        f"ratification: msg {message_id} {kind}/{label} ({decided_by}) -> "
        f"assistant {asst['id']} now {status}"
    )
    body = f"{user_text}\n\n{asst['content'] or ''}"
    return body[:max_chars]


async def record_sent_ids(platform: str, chat_id: str, platform_message_ids: list) -> None:
    """Attach the platform ids of a just-delivered reply to its assistant row."""
    ids = [int(i) for i in platform_message_ids or [] if i is not None]
    if not ids:
        return
    try:
        async with get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT m.id FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE s.platform = $1 AND s.platform_chat_id = $2
                  AND m.role = 'assistant' AND btrim(coalesce(m.content, '')) <> ''
                  AND m.created_at > now() - interval '30 minutes'
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT 1
                """,
                platform, str(chat_id),
            )
            if row is None:
                logger.debug(f"record_sent_ids: no recent assistant row for {platform}:{chat_id}")
                return
            await conn.execute(
                """
                UPDATE messages SET metadata = jsonb_set(
                    coalesce(metadata, '{}'::jsonb), '{platform_message_ids}',
                    coalesce(metadata -> 'platform_message_ids', '[]'::jsonb) || $2::jsonb, true)
                WHERE id = $1
                """,
                row["id"], json.dumps(ids),
            )
        logger.debug(f"record_sent_ids: assistant {row['id']} <- {platform} msgs {ids}")
    except Exception as e:
        logger.warning(f"record_sent_ids failed for {platform}:{chat_id}: {type(e).__name__}: {e}")


async def lookup_sent_preview(platform: str, chat_id: str, platform_message_id) -> Optional[str]:
    """Content of the assistant reply delivered as this platform message."""
    try:
        async with get_connection() as conn:
            asst = await find_by_platform_id(conn, platform, chat_id, platform_message_id)
        return (asst or {}).get("content")
    except Exception as e:
        logger.debug(f"lookup_sent_preview failed: {e}")
        return None


async def _reembed_plain(conn, message_id: int, content: str) -> None:
    from .tools import history as _h
    from .db.models import get_config
    prov = getattr(_h, "_embedding_provider", None)
    if prov is None:
        return
    try:
        max_chars = int(await get_config("history_search.max_content_chars", 4000))
    except Exception:
        max_chars = 4000
    resp = await prov.embed((content or "")[:max_chars])
    vec = getattr(resp, "vector", None)
    if vec:
        await conn.execute(
            "UPDATE messages SET embedding = $1::vector WHERE id = $2", str(vec), int(message_id)
        )


async def revoke_reaction(platform: str, chat_id: str, platform_message_id, removed_emojis) -> int:
    """A removed reaction revokes the ratification it created. Returns the
    number of rows revoked."""
    try:
        cfg = await settings()
        if not cfg["enabled"]:
            return 0
        labels = {lb for lb in (reaction_label([e], cfg) for e in removed_emojis or []) if lb}
        if not labels:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        async with get_connection() as conn:
            asst = await find_by_platform_id(conn, platform, chat_id, platform_message_id)
            if not asst:
                return 0
            rows = await conn.fetch(
                """
                SELECT id, content FROM messages
                WHERE role = 'user' AND created_at >= $2
                  AND metadata -> 'ratification' ->> 'ratifies' = $1
                  AND metadata -> 'ratification' ->> 'kind' = 'reaction'
                  AND metadata -> 'ratification' ->> 'status' = 'active'
                  AND metadata -> 'ratification' ->> 'label' = ANY($3::text[])
                """,
                str(asst["id"]), asst["created_at"], sorted(labels),
            )
            for r in rows:
                await conn.execute(
                    """
                    UPDATE messages SET metadata = jsonb_set(
                        jsonb_set(metadata, '{ratification,status}', '"revoked"'::jsonb),
                        '{ratification,revoked_at}', to_jsonb($2::text))
                    WHERE id = $1
                    """,
                    r["id"], now,
                )
                await _reembed_plain(conn, r["id"], r["content"])
            status = await refresh_status(conn, asst["id"], asst["created_at"])
        logger.info(
            f"ratification: revoked {len(rows)} reaction(s) {sorted(labels)} on assistant "
            f"{asst['id']}, now {status}"
        )
        return len(rows)
    except Exception as e:
        logger.warning(f"revoke_reaction failed: {type(e).__name__}: {e}")
        return 0
