"""User ratification of assistant answers (Riyo, 10 Nov 2026).

history_search anchors only on USER messages, so a fact that only the
assistant wrote is invisible to search. When the user explicitly accepts
that answer - a short approval ("setuju", "betul") or an approving
emoji reaction - the user's row is re-embedded as user text + the approved
assistant answer. The anchor stays a user row: an assistant answer only
becomes searchable once the user has endorsed it (grounding, Clark &
Brennan 1991). Ratification is not verification - an approved answer can
still be wrong.

  * text:     lexicon first. Clear approval words ratify directly; bare
              acknowledgements ("ok", "ya") are ambiguous (agree vs. just
              "received", Jurafsky et al. 1997) and go to the local
              evaluator model, falling back to "ack" (no ratification).
              Continuation requests ("lanjutkan", "gas") are labelled
              "continue": asking the agent to go on is not an endorsement,
              so they are never combined and never a history_search anchor.
  * reaction: approving emoji -> ratified, 👎 -> disputed, a reaction that
              is taken back -> revoked (own embedding restored).

Channels: Telegram, WhatsApp and Discord. Each channel records the platform
ids of every message it delivers on the assistant row (metadata
tg_message_ids / wa_message_ids / dc_message_ids), so a reaction on any
chunk - even an old one - resolves to its content. A reaction reaches the
agent as "[Reaction: <emoji> from <name> on your message #<id>: '...']".

The label is written to messages.metadata.ratification on the user row and
the status on the assistant row, so history_expand can show it and the
classification can be scored against human annotation.

Switches (all default on): history_search.ratification (master),
history_search.ratification_text, history_search.ratification_reaction.
"""

import json
import logging
import re
from typing import Optional

import httpx

from .db.connection import get_connection
from .db.models import get_config

logger = logging.getLogger("syne.ratification")

# ---------------------------------------------------------------------------
# Inbound prefix (Conversation info / Sender / Replied message JSON blocks)
# ---------------------------------------------------------------------------

_PREFIX_BLOCK = re.compile(r"\A\s*[^\n`]*:\n```json\n.*?\n```[ \t]*\n*", re.S)
# WhatsApp prepends a bracketed channel instruction before the JSON blocks.
_CHANNEL_INSTRUCTION = re.compile(r"\A\s*\[[A-Za-z ]*channel instruction:[^\]]*\][ \t]*\n*")


def split_context_prefix(text: str) -> tuple:
    """(prefix, body): peel the channel instruction and the untrusted JSON
    context blocks that channels prepend to every user message."""
    text = text or ""
    rest = text
    while True:
        m = _CHANNEL_INSTRUCTION.match(rest) or _PREFIX_BLOCK.match(rest)
        if not m:
            break
        rest = rest[m.end():]
    return text[: len(text) - len(rest)], rest.strip()


def strip_context_prefix(text: str) -> str:
    return split_context_prefix(text)[1]


def anchor_text(content: str) -> str:
    """Text a user row is embedded from: the message without the channel
    instruction and the untrusted JSON context blocks. The header is the same
    on every row, so embedding it pulls unrelated messages together. Falls
    back to the full content when nothing else is left."""
    return strip_context_prefix(content) or (content or "").strip()


# ---------------------------------------------------------------------------
# Emoji
# ---------------------------------------------------------------------------


def norm_emoji(e: str) -> str:
    return (e or "").replace("\ufe0f", "").strip()


APPROVE_EMOJIS = {norm_emoji(e) for e in (
    "👍", "❤️", "👌", "💯", "🔥", "👏", "🙏", "🤝", "✅", "🥰", "😍", "🤩", "🎉",
)}
DISPUTE_EMOJIS = {norm_emoji(e) for e in ("👎", "❌")}

_REACTION_RE = re.compile(
    r"\A\[Reaction: (?P<emojis>.+?) from (?P<who>.+?) on your message #(?P<mid>[^\s:]+)",
    re.S,
)

# ---------------------------------------------------------------------------
# Platform message ids
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text lexicon
# ---------------------------------------------------------------------------

_STRONG = {
    "setuju", "sepakat", "betul", "bener", "benar", "tepat", "sip", "mantap",
    "mantab", "good", "great", "perfect", "agree", "agreed", "correct",
    "exactly", "persis", "cocok", "acc", "approved", "approve",
    "yes", "valid", "setujui", "disetujui",
}
# Continuation requests: the user asks the agent to go on. That is not an
# endorsement of what it said, so they are labelled "continue", never
# combined with the answer, and excluded as history_search anchors.
_CONTINUE = {
    "lanjut", "lanjutkan", "lanjutin", "terus", "teruskan", "gas", "gaskan",
    "go", "next", "continue", "proceed",
}
_AMBIG = {
    "ok", "oke", "okay", "okey", "okeh", "oce", "ya", "iya", "iyaa", "yaa",
    "yup", "yap", "sure", "noted", "siap", "baik", "baiklah", "hmm", "oh",
    "paham", "understood", "ngerti", "makasih", "thanks", "thx", "terima",
    "kasih", "alright", "fine",
}
_FILLER = {
    "deh", "dong", "aja", "saja", "kak", "molt", "bro", "sih", "nih", "yah",
    "lah", "pak", "mas", "banget", "sekali", "juga", "udah", "sudah", "itu",
    "ini", "kalau", "gitu", "nah", "yuk", "ayo", "begitu", "kok", "then", "please", "aku", "saya",
}
_NEG = {
    "tidak", "tak", "nggak", "ngga", "gak", "ga", "enggak", "engga", "bukan",
    "jangan", "no", "not", "belum", "salah", "kurang", "tapi", "but", "nope",
}


def _normalize(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1", s)  # okeee -> oke, sippp -> sip
    return re.sub(r"\s+", " ", s).strip()


def _only_approve_emojis(text: str) -> bool:
    t = norm_emoji(text).replace(" ", "")
    if not t:
        return False
    for e in sorted(APPROVE_EMOJIS, key=len, reverse=True):
        t = t.replace(e, "")
    return t == ""


def classify_lexicon(text: str) -> Optional[str]:
    """'approve' | 'continue' | 'ambiguous' | None (not an approval message
    at all). An approval word wins over a continuation word ("setuju,
    lanjutkan" approves); "oke lanjutkan" is a continuation."""
    if _only_approve_emojis(text):
        return "approve"
    s = _normalize(text)
    if not s or len(s) > 60:
        return None
    toks = s.split()
    if len(toks) > 6:
        return None
    if any(t in _NEG for t in toks):
        return None
    if not all(t in _STRONG or t in _CONTINUE or t in _AMBIG or t in _FILLER for t in toks):
        return None
    if any(t in _STRONG for t in toks):
        return "approve"
    if any(t in _CONTINUE for t in toks):
        return "continue"
    if any(t in _AMBIG for t in toks):
        return "ambiguous"
    return None


_LLM_SYSTEM = (
    "Classify the user's short reply to the assistant's previous message.\n"
    "AGREE = the user accepts, endorses or approves what the assistant said or proposed.\n"
    "ACK = the user only signals they received/read it, says thanks, or moves on without endorsing.\n"
    "Answer with exactly one word: AGREE or ACK."
)


def parse_llm_label(raw: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip().upper()
    return "approve" if t.startswith("AGREE") else "ack"


async def classify_llm(reply: str, assistant_text: str) -> tuple:
    """(label, method) for an ambiguous reply, via the local evaluator model."""
    try:
        model = await get_config("memory.evaluator_model", "qwen3:0.6b")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM},
                        {"role": "user", "content": (
                            f"Assistant said:\n{(assistant_text or '')[:1500]}\n\n"
                            f"User replied: {reply}\n/no_think"
                        )},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
        return parse_llm_label(raw), "llm"
    except Exception as e:
        logger.warning(f"ratification LLM classify failed, treating as ack: {e}")
        return "ack", "fallback"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def _flag(key: str) -> bool:
    try:
        return bool(await get_config(key, True))
    except Exception:
        return True


def _combine(user_text: str, assistant_text: str) -> str:
    return f"{user_text}\n\n{assistant_text or ''}".strip()


async def anchor_text_for_row(conn, content: str, metadata) -> str:
    """Embedding text for a stored user row, as at save time: anchor_text,
    combined with the approved assistant answer when the row ratified one.
    Used by `syne memory reembed-history --all`."""
    text = anchor_text(content)
    meta = json.loads(metadata) if isinstance(metadata, str) else (metadata or {})
    r = meta.get("ratification") or {}
    if isinstance(r, str):
        r = json.loads(r)
    if r.get("label") == "approve" and r.get("assistant_id"):
        a = await conn.fetchval(
            "SELECT content FROM messages WHERE id = $1", int(r["assistant_id"]),
        )
        if a:
            return _combine(text, a)
    return text


async def _mark(user_row_id: int, assistant_row_id: int, info: dict) -> None:
    async with get_connection() as conn:
        await conn.execute(
            "UPDATE messages SET metadata = coalesce(metadata, '{}'::jsonb) "
            "|| jsonb_build_object('ratification', $2::jsonb) WHERE id = $1",
            user_row_id, json.dumps(info, ensure_ascii=False),
        )
        if info.get("label") in ("approve", "dispute"):
            status = {
                "status": "ratified" if info["label"] == "approve" else "disputed",
                "type": info.get("type"),
                "user_message_id": user_row_id,
            }
            if info.get("emoji"):
                status["emoji"] = info["emoji"]
            await conn.execute(
                "UPDATE messages SET metadata = coalesce(metadata, '{}'::jsonb) "
                "|| jsonb_build_object('ratification', $2::jsonb) WHERE id = $1",
                assistant_row_id, json.dumps(status, ensure_ascii=False),
            )


async def _find_sent(conn, platform: str, chat_id: str, platform_message_id):
    """Assistant row delivered as this platform message in this chat."""
    key, val = ids_key(platform), id_value(platform, platform_message_id)
    if key is None or val is None:
        return None
    return await conn.fetchrow(
        "SELECT m.id, m.session_id, m.content, m.metadata->'ratification' AS r "
        "FROM messages m JOIN sessions s ON s.id = m.session_id "
        "WHERE s.platform = $1 AND s.platform_chat_id = $2 AND m.role = 'assistant' "
        f"AND m.metadata->'{key}' @> $3::jsonb "
        "ORDER BY m.id DESC LIMIT 1",
        platform, str(chat_id), json.dumps([val]),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def ratify_user_row(session_id: int, message_id: int, content: str) -> Optional[str]:
    """Classify a freshly saved user row. Returns the text to embed instead of
    the row's own content when it ratifies an assistant answer, else None."""
    if not await _flag("history_search.ratification"):
        return None
    prefix, clean = split_context_prefix(content)
    if not clean:
        return None

    m = _REACTION_RE.match(clean)
    if m:
        if not await _flag("history_search.ratification_reaction"):
            return None
        return await _ratify_reaction_row(session_id, message_id, clean, m)

    if not await _flag("history_search.ratification_text"):
        return None
    # A reply points at a specific (possibly older) message; the previous
    # assistant row is then not necessarily the one approved.
    if "Replied message (" in prefix:
        return None
    kind = classify_lexicon(clean)
    if kind is None:
        return None

    async with get_connection() as conn:
        a = await conn.fetchrow(
            "SELECT id, content FROM messages WHERE session_id = $1 AND id < $2 "
            "AND role = 'assistant' AND btrim(coalesce(content, '')) <> '' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            session_id, message_id,
        )
    if not a:
        return None

    if kind == "continue":
        label, method = "continue", "lexicon"
    elif kind == "ambiguous":
        label, method = await classify_llm(clean, a["content"])
    else:
        label, method = "approve", "lexicon"
    info = {"type": "text", "label": label, "method": method, "assistant_id": a["id"]}
    await _mark(message_id, a["id"], info)
    logger.info(f"ratification: user row {message_id} -> {label} ({method}) on assistant row {a['id']}")
    return _combine(clean, a["content"]) if label == "approve" else None


async def _ratify_reaction_row(session_id: int, message_id: int, clean: str, m) -> Optional[str]:
    mid = m.group("mid")
    emojis = [norm_emoji(e) for e in m.group("emojis").split()]
    label, emoji = None, None
    for e in emojis:
        if e in DISPUTE_EMOJIS:
            label, emoji = "dispute", e
            break
        if e in APPROVE_EMOJIS and label is None:
            label, emoji = "approve", e
    if label is None:
        return None

    async with get_connection() as conn:
        platform = await conn.fetchval("SELECT platform FROM sessions WHERE id = $1", session_id)
        key, val = ids_key(platform), id_value(platform, mid)
        if key is None or val is None:
            return None
        a = await conn.fetchrow(
            "SELECT id, content FROM messages WHERE session_id = $1 AND role = 'assistant' "
            f"AND metadata->'{key}' @> $2::jsonb "
            "ORDER BY id DESC LIMIT 1",
            session_id, json.dumps([val]),
        )
    if not a:
        logger.info(f"ratification: reaction on {platform} msg {mid} has no mapped assistant row")
        return None

    info = {
        "type": "reaction", "label": label, "emoji": emoji, "reactor": m.group("who"),
        "assistant_id": a["id"], "platform": platform, "platform_message_id": val,
    }
    await _mark(message_id, a["id"], info)
    logger.info(f"ratification: reaction {emoji} row {message_id} -> {label} on assistant row {a['id']}")
    return _combine(clean, a["content"]) if label == "approve" else None


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


def same_reactor(stored: Optional[str], reactor: str) -> bool:
    """WhatsApp and Discord name a reactor as 'Name (stable-id)'; match on
    the id so a changed display name still revokes."""
    if not stored:
        return True
    return stored == reactor or stored.endswith(f"({reactor})")


async def _embed_text(text: str) -> Optional[list]:
    from .tools import history as _h
    prov = getattr(_h, "_embedding_provider", None)
    if prov is None:
        return None
    max_chars = int(await get_config("history_search.max_content_chars", 4000))
    resp = await prov.embed(text[:max_chars])
    return getattr(resp, "vector", None) or None


async def revoke_reaction(
    platform: str,
    chat_id: str,
    platform_message_id,
    emojis: Optional[list],
    reactor: Optional[str] = None,
) -> int:
    """A reaction was taken back: restore each affected user row's own
    embedding and mark it revoked. emojis=None revokes every emoji (WhatsApp
    removals and replacements do not say which one is gone); reactor limits
    it to that person's reactions (their stable id, see same_reactor).
    Returns the number of rows revoked."""
    removed = None if emojis is None else {norm_emoji(e) for e in emojis}
    revoked_ids = []
    try:
        async with get_connection() as conn:
            a = await _find_sent(conn, platform, chat_id, platform_message_id)
            if not a:
                return 0
            rows = await conn.fetch(
                "SELECT id, content, metadata->'ratification' AS r FROM messages "
                "WHERE session_id = $1 AND role = 'user' "
                "AND metadata->'ratification'->>'type' = 'reaction' "
                "AND metadata->'ratification'->>'assistant_id' = $2 "
                "AND metadata->'ratification'->>'label' IN ('approve', 'dispute')",
                a["session_id"], str(a["id"]),
            )
        for row in rows:
            r = row["r"]
            r = json.loads(r) if isinstance(r, str) else (r or {})
            if removed is not None and norm_emoji(r.get("emoji", "")) not in removed:
                continue
            if reactor and not same_reactor(r.get("reactor"), reactor):
                continue
            vec = None
            if r.get("label") == "approve":
                vec = await _embed_text(anchor_text(row["content"] or ""))
            r = dict(r, label="revoked", revoked_label=r.get("label"))
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE messages SET metadata = coalesce(metadata, '{}'::jsonb) "
                    "|| jsonb_build_object('ratification', $2::jsonb) WHERE id = $1",
                    row["id"], json.dumps(r, ensure_ascii=False),
                )
                if vec:
                    await conn.execute(
                        "UPDATE messages SET embedding = $1::vector WHERE id = $2",
                        str(vec), row["id"],
                    )
            revoked_ids.append(row["id"])

        cur = a["r"]
        cur = json.loads(cur) if isinstance(cur, str) else (cur or {})
        if revoked_ids and cur.get("user_message_id") in revoked_ids:
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE messages SET metadata = coalesce(metadata, '{}'::jsonb) "
                    "|| jsonb_build_object('ratification', $2::jsonb) WHERE id = $1",
                    a["id"], json.dumps(dict(cur, status="revoked"), ensure_ascii=False),
                )
        if revoked_ids:
            logger.info(f"ratification: revoked rows {revoked_ids} on assistant row {a['id']}")
    except Exception as e:
        logger.warning(
            f"revoke_reaction failed for {platform}:{chat_id} msg {platform_message_id}: {e}"
        )
    return len(revoked_ids)
