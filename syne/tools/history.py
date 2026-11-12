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
import math
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
        # Size of the live context window (X). history_expand uses it to
        # flag turns the model already has instead of repeating them.
        "window_size": len(getattr(conv, "_message_cache", None) or []),
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
    _exclude_session: Optional[int] = None,
    _query_vector: Optional[list] = None,
) -> str:
    """Return an ordered list of anchor previews, JSON-formatted for the LLM.

    ``_exclude_session`` / ``_query_vector`` are internal (history.auto_recall,
    Riyo 12 Nov 2026): leave one session out, reuse the turn's query vector.
    The registry strips both from model calls.
    """
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

    if _query_vector:
        vector = list(_query_vector)
    else:
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

    # A continuation request ("lanjutkan") is never an anchor on its own;
    # the exchange it belongs to is still reached through its neighbours.
    conditions = [
        "m.role = 'user'",
        "m.embedding IS NOT NULL",
        "COALESCE(m.metadata->'ratification'->>'label', '') <> 'continue'",
    ]
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

    if _exclude_session is not None:
        conditions.append(f"m.session_id <> ${p}")
        params.append(int(_exclude_session))
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


# ---------------------------------------------------------------------------
# history_expand: adaptive radius + dedup (Riyo, 10 Nov 2026)
#
#   * Adaptive radius: context_before / context_after are only the MAXIMUM.
#     Inside that window the read stops at the first exchange where the topic
#     shifts (TextTiling-style: cosine between blocks of user-message
#     embeddings falls below a threshold) or where a long time gap separates
#     two exchanges. The anchor's own exchange is always kept. The reason each
#     side stopped is returned and logged.
#   * Dedup: overlapping windows of several anchors are merged into one block,
#     and turns already in the live context window are flagged, not repeated.
#
# Both are on by default. history_search.adaptive_radius / history_search.dedup
# switch them off for experiments (fixed-radius condition).
# ---------------------------------------------------------------------------


def _parse_vec(raw) -> Optional[list]:
    """pgvector text form '[0.1,0.2,...]' (or a list) -> list of floats."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw] or None
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return None
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        return None


def _mean(vecs: list) -> list:
    n = len(vecs)
    return [sum(col) / n for col in zip(*vecs)]


def _cos(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def adaptive_bounds(
    rows: list,
    anchor_idx: int,
    vecs: dict,
    *,
    threshold: float,
    gap_seconds: float,
    block: int,
    edge_before: str = "max_radius",
    edge_after: str = "max_radius",
) -> tuple:
    """Choose how far to read on each side of the anchor.

    ``rows`` is the maximum window in time order (dicts with id, role,
    created_at); ``vecs`` maps user-row id -> embedding. Rows are grouped into
    exchanges (a user row plus the assistant rows after it). Exchanges are
    added outward from the anchor's own exchange until one of:
      * time_gap    - the gap to the next exchange exceeds ``gap_seconds``
      * topic_shift - cosine(block of user embeddings already included,
                      next block) < ``threshold``
    otherwise the side ends at the window edge (``edge_before``/``edge_after``:
    "session_edge" or "max_radius").

    Returns (lo_index, hi_index, stop_before, stop_after), indices inclusive.
    """
    block = max(1, int(block))
    groups: list = []
    for i, r in enumerate(rows):
        if r["role"] == "user" or not groups:
            groups.append([i, i])
        else:
            groups[-1][1] = i
    gi = next(k for k, g in enumerate(groups) if g[0] <= anchor_idx <= g[1])

    def uvec(k):
        r = rows[groups[k][0]]
        return vecs.get(r["id"]) if r["role"] == "user" else None

    def bvec(ks):
        vs = [v for v in (uvec(k) for k in ks) if v is not None]
        return _mean(vs) if vs else None

    def gap(a, b):
        ta, tb = rows[a].get("created_at"), rows[b].get("created_at")
        if ta is None or tb is None:
            return 0.0
        return (tb - ta).total_seconds()

    included = [gi]
    stop_after = edge_after
    for k in range(gi + 1, len(groups)):
        if gap_seconds > 0 and gap(groups[k - 1][1], groups[k][0]) > gap_seconds:
            stop_after = "time_gap"
            break
        ref = bvec(included[-block:])
        cand = bvec(range(k, min(len(groups), k + block)))
        if ref is not None and cand is not None and _cos(ref, cand) < threshold:
            stop_after = "topic_shift"
            break
        included.append(k)

    stop_before = edge_before
    for k in range(gi - 1, -1, -1):
        if gap_seconds > 0 and gap(groups[k][1], groups[k + 1][0]) > gap_seconds:
            stop_before = "time_gap"
            break
        ref = bvec(included[:block])
        cand = bvec(range(max(0, k - block + 1), k + 1))
        if ref is not None and cand is not None and _cos(ref, cand) < threshold:
            stop_before = "topic_shift"
            break
        included.insert(0, k)

    return groups[included[0]][0], groups[included[-1]][1], stop_before, stop_after


async def _expand_settings() -> dict:
    """history_expand knobs, read on every call (a config change applies on
    the next call, no restart)."""
    out = {
        "before": 10, "after": 10, "dedup": True, "adaptive": True,
        "threshold": 0.5, "gap_minutes": 120.0, "block": 2,
    }
    keys = (
        ("before", "history_search.context_before", int),
        ("after", "history_search.context_after", int),
        ("dedup", "history_search.dedup", bool),
        ("adaptive", "history_search.adaptive_radius", bool),
        ("threshold", "history_search.adaptive_threshold", float),
        ("gap_minutes", "history_search.adaptive_gap_minutes", float),
        ("block", "history_search.adaptive_block", int),
    )
    for name, key, cast in keys:
        try:
            out[name] = cast(await get_config(key, out[name]))
        except Exception:
            pass
    out["block"] = max(1, out["block"])
    return out


async def _context_window_ids(conn, scope: Optional[dict]) -> set:
    """Ids of the rows currently in the live context window (X).

    The in-memory cache mirrors the tail of the session: load_history() reads
    the newest active rows and save_message() appends each saved row, so the
    newest ``window_size`` active rows are exactly what the model holds.
    """
    sid = (scope or {}).get("session_id")
    try:
        n = int((scope or {}).get("window_size") or 0)
    except (TypeError, ValueError):
        n = 0
    if not sid or n <= 0:
        return set()
    rows = await conn.fetch(
        "/* history_expand:in_context */ SELECT id FROM messages "
        "WHERE session_id = $1 AND status = 'active' "
        "ORDER BY created_at DESC, id DESC LIMIT $2",
        int(sid), n,
        timeout=_STEP_TIMEOUT,
    )
    return {r["id"] for r in rows}


def _row_key(r):
    return (r["created_at"], r["id"])


def _build_blocks(pieces: list, in_context: set, merge: bool, before: int, after: int) -> list:
    """Turn per-anchor windows into output blocks (merged when ``merge``)."""
    groups: list = []
    if merge:
        with_rows = sorted(
            [p for p in pieces if p["rows"]],
            key=lambda p: (p["session_id"], _row_key(p["rows"][0])),
        )
        for p in with_rows + [p for p in pieces if not p["rows"]]:
            last = groups[-1] if groups else None
            if (
                last is not None and p["rows"] and last["rows"]
                and last["session_id"] == p["session_id"]
                and _row_key(p["rows"][0]) <= _row_key(last["rows"][-1])
            ):
                seen = {r["id"] for r in last["rows"]}
                last["rows"] = sorted(
                    last["rows"] + [r for r in p["rows"] if r["id"] not in seen],
                    key=_row_key,
                )
                last["anchors"].append(p["anchor_id"])
                last["radius"][str(p["anchor_id"])] = p["radius"]
            else:
                groups.append({
                    "session_id": p["session_id"], "rows": list(p["rows"]),
                    "anchors": [p["anchor_id"]],
                    "radius": {str(p["anchor_id"]): p["radius"]},
                })
    else:
        groups = [{
            "session_id": p["session_id"], "rows": p["rows"],
            "anchors": [p["anchor_id"]],
            "radius": {str(p["anchor_id"]): p["radius"]},
        } for p in pieces]

    blocks = []
    for g in groups:
        anchors = set(g["anchors"])
        turns = []
        for r in g["rows"]:
            t = {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"] or "",
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "is_anchor": r["id"] in anchors,
            }
            rat = r.get("ratification")
            if isinstance(rat, str):
                try:
                    rat = json.loads(rat)
                except ValueError:
                    rat = None
            if r["role"] == "assistant" and isinstance(rat, dict) and rat.get("status"):
                t["ratification"] = {k: rat[k] for k in ("status", "type", "emoji") if rat.get(k)}
            if r["id"] in in_context:
                t["content"] = ""
                t["already_in_context"] = True
            turns.append(t)
        blocks.append({
            "anchor_id": g["anchors"][0],
            "anchor_ids": g["anchors"],
            "session_id": g["session_id"],
            "context_before": before,
            "context_after": after,
            "radius": g["radius"],
            "turns": turns,
        })
    return blocks


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
    ids = list(dict.fromkeys(ids))  # same anchor twice = one anchor

    cfg = await _expand_settings()
    before = max(0, int(context_before) if context_before is not None else cfg["before"])
    after = max(0, int(context_after) if context_after is not None else cfg["after"])

    # Cap so a rogue call can't dump gigabytes.
    before = min(before, 20)
    after = min(after, 40)

    # Visibility: the owner in a DM may open anything; everyone else only
    # anchors inside their scope. Out-of-scope ids are reported as missing.
    if not _scope:
        return _NO_SCOPE_ERROR
    unrestricted = _is_owner_dm(_scope)

    pieces = []
    missing = []
    in_context: set = set()
    _t0 = time.monotonic()
    _stage = "acquire connection"
    logger.info(
        f"history_expand: start anchors={ids} before={before} after={after} "
        f"adaptive={cfg['adaptive']} dedup={cfg['dedup']}"
    )
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
                    "SELECT id, role, content, created_at, "
                    "metadata->'ratification' AS ratification FROM messages WHERE "
                    + " AND ".join(conds)
                    + " ORDER BY created_at ASC, id ASC",
                    *params,
                    timeout=_STEP_TIMEOUT,
                )
                rows = [dict(r) for r in window_rows]

                # Step 3 — adaptive radius inside the maximum window.
                radius = {
                    "adaptive": bool(cfg["adaptive"]),
                    "before": None,
                    "after": None,
                    "stop_before": "session_edge" if len(before_rows) < before else "max_radius",
                    "stop_after": "session_edge" if len(after_rows) < after else "max_radius",
                }
                a_idx = next((i for i, r in enumerate(rows) if r["id"] == aid), None)
                if cfg["adaptive"] and a_idx is not None and len(rows) > 1:
                    _stage = f"fetch embeddings for anchor {aid}"
                    user_ids = [r["id"] for r in rows if r["role"] == "user"]
                    emb_rows = await conn.fetch(
                        "/* history_expand:emb */ SELECT id, embedding::text AS emb "
                        "FROM messages WHERE id = ANY($1::int[]) AND embedding IS NOT NULL",
                        user_ids,
                        timeout=_STEP_TIMEOUT,
                    )
                    vecs = {}
                    for er in emb_rows:
                        v = _parse_vec(er["emb"])
                        if v:
                            vecs[er["id"]] = v
                    lo_i, hi_i, sb, sa = adaptive_bounds(
                        rows, a_idx, vecs,
                        threshold=cfg["threshold"],
                        gap_seconds=cfg["gap_minutes"] * 60.0,
                        block=cfg["block"],
                        edge_before=radius["stop_before"],
                        edge_after=radius["stop_after"],
                    )
                    rows = rows[lo_i:hi_i + 1]
                    a_idx -= lo_i
                    radius["stop_before"], radius["stop_after"] = sb, sa
                if a_idx is not None:
                    radius["before"] = a_idx
                    radius["after"] = len(rows) - a_idx - 1

                logger.info(
                    f"history_expand: anchor {aid} window {len(window_rows)} rows "
                    f"({len(before_rows)} before / {len(after_rows)} after) -> kept "
                    f"{len(rows)} (before={radius['before']} stop={radius['stop_before']}, "
                    f"after={radius['after']} stop={radius['stop_after']}) "
                    f"in {time.monotonic() - _tw:.2f}s"
                )
                pieces.append({"anchor_id": aid, "session_id": sid, "rows": rows, "radius": radius})

            if cfg["dedup"] and pieces:
                _stage = "fetch context window ids"
                in_context = await _context_window_ids(conn, _scope)
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

    blocks = _build_blocks(pieces, in_context, bool(cfg["dedup"]), before, after)
    flagged = sum(1 for b in blocks for t in b["turns"] if t.get("already_in_context"))
    logger.info(
        f"history_expand: done in {time.monotonic() - _t0:.2f}s, "
        f"{len(pieces)} anchor(s) -> {len(blocks)} block(s), {flagged} turn(s) already in context"
    )
    payload = {"blocks": blocks}
    if flagged:
        payload["note"] = (
            "Turns marked already_in_context are in your current context window; "
            "their text is omitted here - read them there."
        )
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
        "(same rule as history_search) are returned as missing. "
        "context_before/context_after are a MAXIMUM: the window may stop early "
        "at a topic shift or a long time gap (see radius.stop_before / "
        "stop_after per anchor). Overlapping windows are merged into one block; "
        "turns already in your context window are marked already_in_context "
        "with their text omitted. An assistant turn may carry ratification "
        "(status ratified / disputed / revoked): the user approved it (text or "
        "reaction), rejected it with a reaction, or took the reaction back. "
        "Ratified is not verified - it can still be wrong."
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
