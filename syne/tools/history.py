"""history_search + history_expand — semantic recall over the chat log.

Two-primitive design (Riyo, 13 Jul 2026):

  * ``history_search`` returns anchor PREVIEWS only, ordered by similarity or
    recency, with filters (session, time range, keyword co-occurrence). Cheap
    on tokens so the model can skim like a human scanning search results.
  * ``history_expand`` fetches the surrounding turns for a specific anchor.
    Called SELECTIVELY after previews — the model only "opens" candidates
    that look promising, not the whole top-N.

Anchors are user messages ONLY (embedded at save-time in Conversation.
save_message). Assistant / tool / system rows are the context you scroll
around each anchor. This keeps the semantic space clean (one canonical
anchor per topic) and mirrors the human intuition: "when did I ASK about
X?" → jump to that user message → read around.

Both tools are owner+family (0o440): raw messages include tool outputs, other
family members' turns, redaction contexts, etc. Owner explicitly widened
access to family level; public is still excluded. The dedicated
``memory_search`` remains available to family for curated, permission-gated
recall from the memory table.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from ..db.connection import get_connection
from ..db.models import get_config

logger = logging.getLogger("syne.tools.history")


# ---------------------------------------------------------------------------
# Provider hook — set by agent.py at boot so the tools can embed queries
# without importing conversation/memory internals directly. Kept as a
# module-level slot so the schema (used for LLM schema registration) does
# not require the provider to already exist at import time.
# ---------------------------------------------------------------------------

_embedding_provider = None  # type: ignore[assignment]


def set_embedding_provider(provider) -> None:
    """Called from agent boot to wire the embedding provider used by search.

    Passing None is a valid "no embedding provider configured" state — the
    handler returns an explanatory error rather than crashing.
    """
    global _embedding_provider
    _embedding_provider = provider


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _preview(text: str, n: int = 160) -> str:
    """Collapse whitespace + truncate for cheap-token previews."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1] + "…"


def _iso_or_none(raw: Optional[str]) -> Optional[datetime]:
    """Accept 'YYYY-MM-DD' or full ISO. Return None on parse failure so the
    handler can reject with a clear message instead of a raw exception."""
    if not raw:
        return None
    try:
        # Postgres will happily coerce timestamptz from these formats but
        # asyncpg needs a real datetime for parameter binding.
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# history_search
# ---------------------------------------------------------------------------


async def history_search_handler(
    query: str,
    limit: int = 10,
    offset: int = 0,
    session_id: Optional[int] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    also_contains: Optional[list] = None,
    sort_by: str = "similarity",
) -> str:
    """Return an ordered list of anchor previews, JSON-formatted for the LLM."""
    if not query or not query.strip():
        return "Error: query is required (non-empty string)."

    # Bounds — defensive against pathological calls.
    try:
        default_limit = int(await get_config("history_search.default_limit", 10))
    except Exception:
        default_limit = 10
    limit = max(1, min(50, int(limit) if limit is not None else default_limit))
    offset = max(0, int(offset) if offset is not None else 0)

    if sort_by not in ("similarity", "recency"):
        return f"Error: sort_by must be 'similarity' or 'recency', got {sort_by!r}."

    since_dt = _iso_or_none(since)
    if since and since_dt is None:
        return f"Error: unparseable 'since' timestamp: {since!r} (use ISO like '2026-07-01')."
    until_dt = _iso_or_none(until)
    if until and until_dt is None:
        return f"Error: unparseable 'until' timestamp: {until!r}."

    also = [str(k).strip() for k in (also_contains or []) if str(k).strip()]

    if _embedding_provider is None:
        return (
            "Error: embedding provider not configured — history_search cannot "
            "produce a query vector. Configure an embedding provider "
            "(see /embedding) and try again."
        )

    try:
        resp = await _embedding_provider.embed(query)
    except Exception as e:
        return f"Error: embedding provider failed: {e}"
    vector = getattr(resp, "vector", None)
    if not vector:
        return "Error: embedding provider returned an empty vector."

    # Assemble SQL. Sort direction depends on mode; recency mode still
    # applies the vector similarity as a tie-breaker so top matches still
    # semantically-close if two rows share the same second.
    conditions = ["m.role = 'user'", "m.embedding IS NOT NULL"]
    params: list = [str(vector)]
    p = 2

    if session_id is not None:
        conditions.append(f"m.session_id = ${p}")
        params.append(int(session_id))
        p += 1

    if since_dt is not None:
        conditions.append(f"m.created_at >= ${p}")
        params.append(since_dt)
        p += 1

    if until_dt is not None:
        conditions.append(f"m.created_at <= ${p}")
        params.append(until_dt)
        p += 1

    for kw in also:
        conditions.append(f"m.content ILIKE ${p}")
        params.append(f"%{kw}%")
        p += 1

    where = " AND ".join(conditions)

    if sort_by == "recency":
        order = "m.created_at DESC, (m.embedding <=> $1::vector) ASC"
    else:
        order = "(m.embedding <=> $1::vector) ASC, m.created_at DESC"

    limit_param = p
    offset_param = p + 1
    params.append(limit)
    params.append(offset)

    sql = f"""
        SELECT
            m.id, m.session_id, m.content, m.created_at,
            1 - (m.embedding <=> $1::vector) AS similarity,
            s.platform_chat_id AS chat_id
        FROM messages m
        LEFT JOIN sessions s ON s.id = m.session_id
        WHERE {where}
        ORDER BY {order}
        LIMIT ${limit_param} OFFSET ${offset_param}
    """

    try:
        async with get_connection() as conn:
            rows = await conn.fetch(sql, *params)
    except Exception as e:
        logger.exception("history_search DB error")
        return f"Error: database query failed: {e}"

    if not rows:
        return "No matches. Try a broader query, a different keyword, or wider since/until range."

    hits = []
    for r in rows:
        hits.append({
            "anchor_id": r["id"],
            "session_id": r["session_id"],
            "chat_id": r["chat_id"] or "",
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "similarity": round(float(r["similarity"] or 0.0), 4),
            "preview": _preview(r["content"]),
        })

    return json.dumps({
        "sort_by": sort_by,
        "limit": limit,
        "offset": offset,
        "returned": len(hits),
        "hits": hits,
        "next_offset": offset + len(hits),
        "hint": (
            "Call history_expand([anchor_id, ...]) to read full context around "
            "promising previews. Reject obvious no-matches without expanding."
        ),
    }, ensure_ascii=False)


HISTORY_SEARCH_TOOL = {
    "name": "history_search",
    "description": (
        "Semantic search over USER messages across the entire chat log. Indexing "
        "is user-only BY DESIGN: user messages are ground truth (what the owner "
        "actually said) so results can never be seeded by a past assistant "
        "hallucination. Returns anchor PREVIEWS only — a snippet of the USER "
        "message, WITHOUT the surrounding assistant reply or context. "
        "MANDATORY: after finding a relevant anchor you MUST call history_expand "
        "to read the full turn context (which includes ALL roles: user, "
        "assistant, tool) BEFORE concluding or answering. NEVER draw a conclusion "
        "from the preview alone — the preview is only the question, not the "
        "answer. Iterate like scrolling through search results: broad query "
        "first, then refine with more specific terms, or narrow with since/until "
        "(temporal), also_contains (keyword AND-filter), or session_id (scope to "
        "one chat). Sort by similarity (default) or recency (for 'terakhir kali' "
        "/ 'kemarin' style questions). Owner-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic query — natural language.",
            },
            "limit": {
                "type": "integer",
                "description": "Max previews to return (default 10, cap 50).",
            },
            "offset": {
                "type": "integer",
                "description": "Skip N results — for pagination.",
            },
            "session_id": {
                "type": "integer",
                "description": "Restrict to one session (chat). Omit for cross-session search.",
            },
            "since": {
                "type": "string",
                "description": "ISO date/datetime lower bound (e.g. '2026-07-01').",
            },
            "until": {
                "type": "string",
                "description": "ISO date/datetime upper bound.",
            },
            "also_contains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "AND-filter: each keyword must appear (case-insensitive) in "
                    "the message content. Use to narrow when semantic returns too "
                    "many candidates."
                ),
            },
            "sort_by": {
                "type": "string",
                "enum": ["similarity", "recency"],
                "description": (
                    "Default 'similarity'. Use 'recency' for temporal queries "
                    "('terakhir kali', 'kemarin', 'minggu lalu')."
                ),
            },
        },
        "required": ["query"],
    },
    "handler": history_search_handler,
    "permission": 0o440,  # owner+family read-only (raw log incl. tool outputs)
}


# ---------------------------------------------------------------------------
# history_expand
# ---------------------------------------------------------------------------


async def history_expand_handler(
    anchor_ids: list,
    context_before: Optional[int] = None,
    context_after: Optional[int] = None,
) -> str:
    """Return context blocks (turns around each anchor), JSON-formatted."""
    if not anchor_ids:
        return "Error: anchor_ids is required (list of message ids from history_search)."

    try:
        ids = [int(x) for x in anchor_ids]
    except (TypeError, ValueError):
        return "Error: anchor_ids must be a list of integers."

    if len(ids) > 20:
        return "Error: expand at most 20 anchors at once (be selective — expand only promising previews)."

    try:
        default_before = int(await get_config("history_search.context_before", 2))
        default_after = int(await get_config("history_search.context_after", 5))
    except Exception:
        default_before, default_after = 2, 5

    before = max(0, int(context_before) if context_before is not None else default_before)
    after = max(0, int(context_after) if context_after is not None else default_after)

    # Cap so a rogue call can't dump gigabytes.
    before = min(before, 20)
    after = min(after, 40)

    blocks = []
    try:
        async with get_connection() as conn:
            # Fetch anchor metadata first so we know each anchor's session +
            # created_at. We then do a per-anchor window fetch — this is
            # simpler than a lateral join and each window is small.
            anchors = await conn.fetch(
                "SELECT id, session_id, created_at FROM messages "
                "WHERE id = ANY($1::int[])",
                ids,
            )
            anchor_map = {r["id"]: r for r in anchors}
            missing = [i for i in ids if i not in anchor_map]

            for aid in ids:
                if aid not in anchor_map:
                    continue
                a = anchor_map[aid]
                sid = a["session_id"]

                # ROW_NUMBER by (session, created_at, id) so we can slice
                # the neighbourhood deterministically.
                window_rows = await conn.fetch(
                    """
                    WITH ordered AS (
                        SELECT id, role, content, metadata, created_at,
                               ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
                        FROM messages
                        WHERE session_id = $1
                    ),
                    anchor AS (
                        SELECT rn FROM ordered WHERE id = $2
                    )
                    SELECT o.id, o.role, o.content, o.metadata, o.created_at,
                           (o.id = $2) AS is_anchor
                    FROM ordered o, anchor a
                    WHERE o.rn BETWEEN a.rn - $3 AND a.rn + $4
                    ORDER BY o.rn ASC
                    """,
                    sid, aid, before, after,
                )
                turns = []
                for r in window_rows:
                    turns.append({
                        "id": r["id"],
                        "role": r["role"],
                        "content": r["content"],
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                        "is_anchor": bool(r["is_anchor"]),
                    })
                blocks.append({
                    "anchor_id": aid,
                    "session_id": sid,
                    "context_before": before,
                    "context_after": after,
                    "turns": turns,
                })
    except Exception as e:
        logger.exception("history_expand DB error")
        return f"Error: database query failed: {e}"

    payload = {"blocks": blocks}
    if missing:
        payload["missing_anchor_ids"] = missing
    return json.dumps(payload, ensure_ascii=False)


HISTORY_EXPAND_TOOL = {
    "name": "history_expand",
    "description": (
        "Fetch full turn context around one or more anchor message IDs found by "
        "history_search. This is the ONLY way to see the ASSISTANT replies and "
        "tool results around a hit — history_search previews are user-only. "
        "REQUIRED before concluding from any relevant anchor: never answer from "
        "the search preview alone; expand to read what was actually said around "
        "it. Call SELECTIVELY — only expand previews that look promising, not the "
        "whole top-N. context_after can/should be larger than context_before "
        "because you want to see the assistant response + follow-up around each "
        "anchor. Owner-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "anchor_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Message IDs from history_search hits.",
            },
            "context_before": {
                "type": "integer",
                "description": "Turns BEFORE the anchor to include (default 2, cap 20).",
            },
            "context_after": {
                "type": "integer",
                "description": "Turns AFTER the anchor to include (default 5, cap 40).",
            },
        },
        "required": ["anchor_ids"],
    },
    "handler": history_expand_handler,
    "permission": 0o440,
}
