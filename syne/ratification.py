"""User ratification of assistant answers (Riyo, 10 Nov 2026).

history_search anchors only on USER messages, so a fact that only the
assistant wrote is invisible to search. When the user explicitly accepts
that answer - a short approval ("setuju", "oke lanjutkan") or an approving
emoji reaction - the user's row is re-embedded as user text + the approved
assistant answer. The anchor stays a user row: an assistant answer only
becomes searchable once the user has endorsed it (grounding, Clark &
Brennan 1991). Ratification is not verification - an approved answer can
still be wrong.

  * text:     lexicon first. Clear approval words ratify directly; bare
              acknowledgements ("ok", "ya") are ambiguous (agree vs. just
              "received", Jurafsky et al. 1997) and go to the local
              evaluator model, falling back to "ack" (no ratification).
  * reaction: approving emoji -> ratified, 👎 -> disputed, a reaction that
              is taken back -> revoked (own embedding restored).

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


def split_context_prefix(text: str) -> tuple:
    """(prefix, body): peel the untrusted JSON context blocks that channels
    prepend to every user message."""
    text = text or ""
    rest = text
    while True:
        m = _PREFIX_BLOCK.match(rest)
        if not m:
            break
        rest = rest[m.end():]
    return text[: len(text) - len(rest)], rest.strip()


def strip_context_prefix(text: str) -> str:
    return split_context_prefix(text)[1]


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
    r"\A\[Reaction: (?P<emojis>.+?) from .+? on your message #(?P<mid>\d+)", re.S
)

# ---------------------------------------------------------------------------
# Text lexicon
# ---------------------------------------------------------------------------

_STRONG = {
    "setuju", "sepakat", "betul", "bener", "benar", "tepat", "sip", "mantap",
    "mantab", "good", "great", "perfect", "agree", "agreed", "correct",
    "exactly", "persis", "cocok", "acc", "approved", "approve", "lanjut",
    "lanjutkan", "gas", "gaskan", "yes", "valid", "setujui", "disetujui",
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
    """'approve' | 'ambiguous' | None (not an approval message at all)."""
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
    if not all(t in _STRONG or t in _AMBIG or t in _FILLER for t in toks):
        return None
    if any(t in _STRONG for t in toks):
        return "approve"
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
    # A Telegram reply points at a specific (possibly older) message; the
    # previous assistant row is then not necessarily the one approved.
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

    if kind == "ambiguous":
        label, method = await classify_llm(clean, a["content"])
    else:
        label, method = "approve", "lexicon"
    info = {"type": "text", "label": label, "method": method, "assistant_id": a["id"]}
    await _mark(message_id, a["id"], info)
    logger.info(f"ratification: user row {message_id} -> {label} ({method}) on assistant row {a['id']}")
    return _combine(clean, a["content"]) if label == "approve" else None


async def _ratify_reaction_row(session_id: int, message_id: int, clean: str, m) -> Optional[str]:
    tg_id = int(m.group("mid"))
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
        a = await conn.fetchrow(
            "SELECT id, content FROM messages WHERE session_id = $1 AND role = 'assistant' "
            "AND metadata->'tg_message_ids' @> to_jsonb($2::bigint) "
            "ORDER BY id DESC LIMIT 1",
            session_id, tg_id,
        )
    if not a:
        logger.info(f"ratification: reaction on tg msg {tg_id} has no mapped assistant row")
        return None

    info = {
        "type": "reaction", "label": label, "emoji": emoji,
        "assistant_id": a["id"], "tg_message_id": tg_id,
    }
    await _mark(message_id, a["id"], info)
    logger.info(f"ratification: reaction {emoji} row {message_id} -> {label} on assistant row {a['id']}")
    return _combine(clean, a["content"]) if label == "approve" else None


async def record_sent_ids(platform: str, chat_id: str, ids: list) -> None:
    """Attach the platform message ids of a delivered answer to the newest
    assistant row of that chat, so a later reaction can find its content."""
    if not ids:
        return
    try:
        async with get_connection() as conn:
            await conn.execute(
                """
                UPDATE messages SET metadata = jsonb_set(
                    coalesce(metadata, '{}'::jsonb), '{tg_message_ids}',
                    coalesce(metadata->'tg_message_ids', '[]'::jsonb) || $3::jsonb)
                WHERE id = (
                    SELECT m.id FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE s.platform = $1 AND s.platform_chat_id = $2
                      AND m.role = 'assistant' AND btrim(coalesce(m.content, '')) <> ''
                      AND m.created_at > now() - interval '10 minutes'
                    ORDER BY m.created_at DESC, m.id DESC LIMIT 1)
                """,
                platform, str(chat_id), json.dumps([int(i) for i in ids]),
            )
    except Exception as e:
        logger.warning(f"record_sent_ids failed for {platform}:{chat_id}: {e}")


async def _embed_text(text: str) -> Optional[list]:
    from .tools import history as _h
    prov = getattr(_h, "_embedding_provider", None)
    if prov is None:
        return None
    max_chars = int(await get_config("history_search.max_content_chars", 4000))
    resp = await prov.embed(text[:max_chars])
    return getattr(resp, "vector", None) or None


async def revoke_reaction(platform: str, chat_id: str, tg_message_id: int, emojis: list) -> int:
    """A reaction was taken back: restore each affected user row's own
    embedding and mark it revoked. Returns the number of rows revoked."""
    removed = {norm_emoji(e) for e in emojis}
    revoked_ids = []
    try:
        async with get_connection() as conn:
            a = await conn.fetchrow(
                "SELECT m.id, m.session_id, m.metadata->'ratification' AS r "
                "FROM messages m JOIN sessions s ON s.id = m.session_id "
                "WHERE s.platform = $1 AND s.platform_chat_id = $2 AND m.role = 'assistant' "
                "AND m.metadata->'tg_message_ids' @> to_jsonb($3::bigint) "
                "ORDER BY m.id DESC LIMIT 1",
                platform, str(chat_id), int(tg_message_id),
            )
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
            if norm_emoji(r.get("emoji", "")) not in removed:
                continue
            vec = None
            if r.get("label") == "approve":
                vec = await _embed_text(row["content"] or "")
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
        logger.warning(f"revoke_reaction failed for {platform}:{chat_id} msg {tg_message_id}: {e}")
    return len(revoked_ids)
