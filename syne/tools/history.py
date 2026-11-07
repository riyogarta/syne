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

Visibility (Riyo, 7 Nov 2026): both tools are open to everyone (0o444), but
every query is scoped in SQL by who is asking - see scope_sql():

  * public        -> the chat they are asking from, nothing else
  * owner/family  -> that chat plus their own DMs on the same platform
                     (reminder sessions count as DM)
  * owner in a DM -> may widen to any chat on any platform, explicitly

The scope is built by ToolRegistry.execute from the live conversation and
passed as ``_scope``; whatever the model puts under that name is overwritten.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Optional

from ..db.connection import get_connection
from ..db.models import get_config

logger = logging.getLogger("syne.tools.history")

# Per-step limit for history_expand (pool acquire + each query). A stall
# surfaces as an explicit error instead of a silent multi-minute hang.
_STEP_TIMEOUT = 60

# Rows history_expand returns: only what was actually said between the
# owner and the assistant. Tool results and empty assistant tool-call
# shells are left out entirely (Riyo, 7 Nov 2026): the owner never saw
# them, and a cut-off tool result invites wrong conclusions.
_CONVO_FILTER = "role IN ('user', 'assistant') AND btrim(coalesce(content, '')) <> ''"


# ---------------------------------------------------------------------------
# Visibility scope (Riyo, 7 Nov 2026)
# ---------------------------------------------------------------------------

_NO_SCOPE_ERROR = (
    "Error: no chat context for this call, so the allowed scope cannot be "
    "determined. History tools only run inside a live chat."
)


def build_history_scope(conv, access_level: str) -> Optional[dict]:
    """Who is asking, from where - taken from the live conversation.

    Returns None when the chat cannot be identified; callers treat that as
    "deny" (fail closed). In a group the sender is the person who typed
    (inbound.sender_id); in a DM it is the session's user.
    """
    if conv is None:
        return None
    inbound = getattr(conv, "inbound", None)
    user = getattr(conv, "user", None) or {}
    is_group = bool(getattr(conv, "is_group", False))
    platform = getattr(inbound, "platform", None) if inbound is not None else None
    if not platform or platform == "unknown":
        platform = user.get("platform")
    chat_id = (getattr(inbound, "chat_id", None) if inbound is not None else None) \
        or getattr(conv, "chat_id", None)
    sender = None
    if inbound is not None and getattr(inbound, "sender_id", None):
        sender = str(inbound.sender_id)
    elif not is_group:
        sender = str(user.get("platform_id") or "") or None
    if not platform or not chat_id:
        return None
    return {
        "access": access_level or "public",
        "platform": str(platform),
        "chat_id": str(chat_id),
        "is_group": is_group,
        "sender_pid": sender,
        "session_id": getattr(conv, "session_id", None),
    }


def group_session_sql(alias: str = "s") -> str:
    """SQL predicate: this session is a group chat (or a group's reminder).

    Group sessions are owned by whoever created them, so "owned by X" alone
    would count a group as X's DM. Telegram groups have negative ids,
    WhatsApp groups end in @g.us, Discord guilds are known only from the
    groups table - which also catches any registered group on any platform.
    """
    a = alias
    return (
        f"({a}.platform_chat_id LIKE '-%'"
        f" OR {a}.platform_chat_id LIKE '%@g.us'"
        f" OR {a}.platform_chat_id LIKE 'scheduled:-%'"
        f" OR {a}.platform_chat_id LIKE 'scheduled:%@g.us:%'"
        f" OR EXISTS (SELECT 1 FROM groups g WHERE g.platform = {a}.platform"
        f" AND g.platform_group_id = {a}.platform_chat_id))"
    )


def scope_sql(scope: dict, start: int) -> tuple:
    """Default visibility as a SQL predicate on sessions ``s``.

    Placeholders are numbered from ``start``. Returns (predicate, params).
      * public: the current chat only
      * owner/family: the current chat OR their own DMs on this platform.
        A DM is recognised by the session OWNER (sessions.user_id ->
        users.platform_id), not by platform_chat_id - on WhatsApp (@lid) and
        Discord the DM chat id is not the person's id. Reminder sessions
        (scheduled:<id>:<task>) are owned by their person, so they count.
    """
    params = [str(scope.get("platform") or ""), str(scope.get("chat_id") or "")]
    current = f"(s.platform = ${start} AND s.platform_chat_id = ${start + 1})"
    sender = scope.get("sender_pid")
    if scope.get("access") not in ("owner", "family") or not sender:
        return current, params
    params.append(str(sender))
    own_dm = (
        f"(s.platform = ${start} AND NOT {group_session_sql('s')}"
        f" AND EXISTS (SELECT 1 FROM users u WHERE u.id = s.user_id"
        f" AND u.platform = s.platform AND u.platform_id = ${start + 2}))"
    )
    return f"({current} OR {own_dm})", params


def _is_owner_dm(scope: Optional[dict]) -> bool:
    return bool(scope) and scope.get("access") == "owner" and not scope.get("is_group")


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
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    all_chats: bool = False,
    _scope: Optional[dict] = None,
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
    # Visibility - enforced here, never left to the model (see scope_sql).
    if not _scope:
        return _NO_SCOPE_ERROR
    owner_dm = _is_owner_dm(_scope)
    explicit = bool(all_chats) or bool(platform) or bool(chat_id)
    if explicit and not owner_dm:
        return (
            "Error: platform / chat_id / all_chats are for the owner in a direct "
            "message only. Omit them to search the chats this person may see."
        )

    conditions = ["m.role = 'user'", "m.embedding IS NOT NULL"]
    params: list = [str(vector)]
    p = 2

    if owner_dm and (explicit or session_id is not None):
        # Owner asked about a specific place (or everything) from a DM.
        if platform:
            conditions.append(f"s.platform = ${p}")
            params.append(str(platform).strip().lower())
            p += 1
        if chat_id:
            conditions.append(f"s.platform_chat_id = ${p}")
            params.append(str(chat_id).strip())
            p += 1
    else:
        sc_sql, sc_params = scope_sql(_scope, p)
        conditions.append(sc_sql)
        params += sc_params
        p += len(sc_params)

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
            s.platform_chat_id AS chat_id,
            s.platform AS platform
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
            "platform": r.get("platform") or "",
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
        "/ 'kemarin' style questions). SCOPE is enforced by the system from who "
        "is asking: public = this chat only; owner/family = this chat plus their "
        "own DMs on this platform. Only the owner in a direct message may widen it "
        "with platform / chat_id / all_chats (or session_id) - use them when the "
        "owner asks about another chat or platform; anywhere else they are refused."
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
            "platform": {
                "type": "string",
                "enum": ["telegram", "whatsapp", "discord", "cli"],
                "description": "Owner in a DM only: search this platform.",
            },
            "chat_id": {
                "type": "string",
                "description": "Owner in a DM only: search this chat (platform chat id, as shown in hits).",
            },
            "all_chats": {
                "type": "boolean",
                "description": "Owner in a DM only: search every chat on every platform.",
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
    "permission": 0o444,  # everyone; rows scoped per requester (scope_sql)
}


# ---------------------------------------------------------------------------
# history_expand
# ---------------------------------------------------------------------------


async def history_expand_handler(
    anchor_ids: list,
    context_before: Optional[int] = None,
    context_after: Optional[int] = None,
    _scope: Optional[dict] = None,
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
        default_before = int(await get_config("history_search.context_before", 10))
        default_after = int(await get_config("history_search.context_after", 10))
    except Exception:
        default_before, default_after = 10, 10

    before = max(0, int(context_before) if context_before is not None else default_before)
    after = max(0, int(context_after) if context_after is not None else default_after)

    # Cap so a rogue call can't dump gigabytes.
    before = min(before, 20)
    after = min(after, 40)

    # Visibility: the owner in a DM may open anything; everyone else only
    # anchors inside their scope. Out-of-scope ids are reported as missing.
    if not _scope:
        return _NO_SCOPE_ERROR
    unrestricted = _is_owner_dm(_scope)

    blocks = []
    missing = []
    _t0 = time.monotonic()
    _stage = "acquire connection"
    logger.info(f"history_expand: start anchors={ids} before={before} after={after}")
    try:
        async with get_connection(timeout=_STEP_TIMEOUT) as conn:
            logger.info(f"history_expand: connection acquired in {time.monotonic() - _t0:.2f}s")
            _stage = "fetch anchors"
            if unrestricted:
                anchors = await conn.fetch(
                    "SELECT id, session_id, created_at FROM messages "
                    "WHERE id = ANY($1::int[])",
                    ids,
                    timeout=_STEP_TIMEOUT,
                )
            else:
                sc_sql, sc_params = scope_sql(_scope, 2)
                anchors = await conn.fetch(
                    "SELECT m.id, m.session_id, m.created_at FROM messages m "
                    "JOIN sessions s ON s.id = m.session_id "
                    "WHERE m.id = ANY($1::int[]) AND " + sc_sql,
                    ids, *sc_params,
                    timeout=_STEP_TIMEOUT,
                )
            anchor_map = {r["id"]: r for r in anchors}
            missing = [i for i in ids if i not in anchor_map]

            for aid in ids:
                if aid not in anchor_map:
                    continue
                a = anchor_map[aid]
                sid = a["session_id"]
                a_at = a["created_at"]
                _tw = time.monotonic()

                # Step 1 — find the window edges. Take the `before`
                # conversation messages right before the anchor and the
                # `after` ones right after it. Each is a short range scan on
                # idx_messages_session (session_id, created_at), so the cost
                # does not grow with the size of the session. (The old query
                # numbered EVERY row of the session with ROW_NUMBER() and the
                # planner could turn that into a ~N^2 nested loop: 7 Nov 2026,
                # a 34k-row session took >10 minutes.)
                _stage = f"fetch rows before anchor {aid}"
                before_rows = await conn.fetch(
                    f"""
                    /* history_expand:before */
                    SELECT id, created_at FROM messages
                    WHERE session_id = $1
                      AND created_at <= $2
                      AND (created_at < $2 OR id < $3)
                      AND {_CONVO_FILTER}
                    ORDER BY created_at DESC, id DESC
                    LIMIT $4
                    """,
                    sid, a_at, aid, before,
                    timeout=_STEP_TIMEOUT,
                )
                _stage = f"fetch rows after anchor {aid}"
                after_rows = await conn.fetch(
                    f"""
                    /* history_expand:after */
                    SELECT id, created_at FROM messages
                    WHERE session_id = $1
                      AND created_at >= $2
                      AND (created_at > $2 OR id > $3)
                      AND {_CONVO_FILTER}
                    ORDER BY created_at ASC, id ASC
                    LIMIT $4
                    """,
                    sid, a_at, aid, after,
                    timeout=_STEP_TIMEOUT,
                )

                # Fewer rows than asked = reached the start/end of the
                # session, so leave that side open (also picks up trailing
                # tool rows of a turn still in progress).
                if len(before_rows) < before:
                    lo = None
                elif before_rows:
                    lo = before_rows[-1]
                else:
                    lo = a  # before == 0
                if len(after_rows) < after:
                    hi = None
                elif after_rows:
                    hi = after_rows[-1]
                else:
                    hi = a  # after == 0

                # Step 2 — the conversation between the edges, in order.
                conds = ["session_id = $1", _CONVO_FILTER]
                params = [sid]
                if lo is not None:
                    params += [lo["created_at"], lo["id"]]
                    n = len(params)
                    conds.append(f"created_at >= ${n - 1} AND (created_at > ${n - 1} OR id >= ${n})")
                if hi is not None:
                    params += [hi["created_at"], hi["id"]]
                    n = len(params)
                    conds.append(f"created_at <= ${n - 1} AND (created_at < ${n - 1} OR id <= ${n})")
                _stage = f"fetch window for anchor {aid}"
                window_rows = await conn.fetch(
                    "/* history_expand:window */ "
                    "SELECT id, role, content, created_at FROM messages WHERE "
                    + " AND ".join(conds)
                    + " ORDER BY created_at ASC, id ASC",
                    *params,
                    timeout=_STEP_TIMEOUT,
                )
                logger.info(
                    f"history_expand: anchor {aid} window {len(window_rows)} rows "
                    f"({len(before_rows)} before / {len(after_rows)} after) "
                    f"in {time.monotonic() - _tw:.2f}s"
                )

                turns = []
                for r in window_rows:
                    turns.append({
                        "id": r["id"],
                        "role": r["role"],
                        "content": r["content"] or "",
                        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                        "is_anchor": r["id"] == aid,
                    })
                blocks.append({
                    "anchor_id": aid,
                    "session_id": sid,
                    "context_before": before,
                    "context_after": after,
                    "turns": turns,
                })
    except TimeoutError:
        _el = time.monotonic() - _t0
        logger.error(f"history_expand: timeout at '{_stage}' after {_el:.1f}s")
        return (
            f"Error: history_expand timed out at '{_stage}' "
            f"(>{_STEP_TIMEOUT}s, total {_el:.1f}s)."
        )
    except Exception as e:
        logger.exception(f"history_expand DB error at '{_stage}'")
        return f"Error: database query failed at '{_stage}': {e}"

    logger.info(f"history_expand: done in {time.monotonic() - _t0:.2f}s")
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
        "whole top-N. Returns the conversation only (user + assistant text); "
        "tool results are not included. Anchors outside the asker's scope "
        "(same rule as history_search) are returned as missing."
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
                "description": "Conversation messages (user + assistant text) BEFORE the anchor (default 10, cap 20).",
            },
            "context_after": {
                "type": "integer",
                "description": "Conversation messages (user + assistant text) AFTER the anchor (default 10, cap 40).",
            },
        },
        "required": ["anchor_ids"],
    },
    "handler": history_expand_handler,
    "permission": 0o444,  # everyone; anchors outside the scope are withheld
}
