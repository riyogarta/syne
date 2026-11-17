"""Memory guard: check a final draft against stored decisions before it is sent.

Riyo, 15 Nov 2026. Memory recall at the start of a turn is driven by the
user's message only. A topic the assistant raises on its own therefore never
pulls in the memories about it. Case that started this: the assistant
re-opened session.history_limit although memory #57566 says the value 100 is
final and must not be raised again; the user's message was about something
else, so #57566 was never recalled.

This guard embeds the DRAFT, fetches the closest memories in the guarded
categories (decision / preference / lesson by default) that were not already
in the turn's context, and asks a judge model whether the draft conflicts
with them. On a conflict the caller rewrites the draft and judges the
rewrite again, up to memory.guard_max_retries rewrites (default 2).

Similarity alone cannot decide this: in a test on real data the decisive
memory ranked 3rd (0.60) behind an unrelated one (0.82), and half of the last
30 replies had a guarded memory at >= 0.60. So a model judges every draft
that has candidates; the prompt carries only the draft, the user message and
the candidates (about 1-2k tokens), never the full context.

Judge model: memory.guard_model, picked via /guard on Telegram, built by
syne.judge_model (shared with the rule checker). Empty = the default model
(provider.active_model). Any other value is a key from /models; an Ollama
entry is called directly on its base_url (/api/chat), every other driver is
built with llm.drivers.create_provider. A key that is missing or cannot be
built skips the check — another model is never substituted.

Fail-open: any error returns None and is logged. The guard must never stop a
reply from being sent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("syne.memory_guard")


# key -> (default value, description). Seeded by schema.sql and migration 46.
DEFAULTS: dict = {
    "memory.guard_enabled": (
        True,
        "Memory guard on/off. When true, every final draft is embedded and "
        "checked against the closest stored memories in memory.guard_categories "
        "that were not already recalled this turn; the judge model decides "
        "whether the draft conflicts with them. A conflict triggers a rewrite, "
        "and every rewrite is judged again, up to memory.guard_max_retries "
        "rewrites. Fail-open. Set via /guard.",
    ),
    "memory.guard_model": (
        "",
        "Model that judges drafts for the memory guard: a key from /models "
        "(provider.models). Empty = the default model (provider.active_model). "
        "An Ollama entry is called directly on its base_url. A key that is "
        "missing or cannot be built skips the check. Set via /guard.",
    ),
    "memory.guard_categories": (
        ["decision", "preference", "lesson"],
        "Memory categories the memory guard checks a draft against.",
    ),
    "memory.guard_limit": (
        5,
        "Max memories the memory guard sends to the judge per draft.",
    ),
    "memory.guard_min_similarity": (
        0.5,
        "Minimum cosine similarity between the draft and a memory for that "
        "memory to be judged. Lower = more memories judged.",
    ),
    "memory.guard_timeout": (
        30,
        "Max seconds for one memory-guard judge call (clamped 5-120). On "
        "timeout the draft is sent unchanged.",
    ),
    "memory.guard_max_retries": (
        2,
        "How many times the memory guard asks the main model to rewrite a draft "
        "that conflicts with stored memories. Every rewrite is judged again. "
        "0 = check only, no rewrite; default 2 = up to 3 drafts in total. "
        "Clamped 0-5. If the last draft still conflicts it is sent as is and logged.",
    ),
}

# Drafts shorter than this ("Iya.", "Oke.") are not checked.
_MIN_WORDS = 3
_DRAFT_CHARS = 4000
_MEMORY_CHARS = 1500
_USER_CHARS = 1500

Judge = Callable[[str, float], Awaitable[str]]


@dataclass
class GuardResult:
    conflict: bool
    memory_ids: list = field(default_factory=list)
    reason: str = ""
    memories: list = field(default_factory=list)


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _as_list(v) -> list:
    if isinstance(v, str):
        import json
        try:
            v = json.loads(v)
        except Exception:
            v = [p for p in (s.strip() for s in v.split(",")) if p]
    if not isinstance(v, (list, tuple)):
        return []
    return [str(c).strip() for c in v if str(c).strip()]


async def load_config(get_config: Callable[..., Awaitable]) -> dict:
    """Read memory.guard_* with safe fallbacks to DEFAULTS."""
    async def _get(key):
        try:
            return await get_config(key, DEFAULTS[key][0])
        except Exception:
            return DEFAULTS[key][0]

    def _num(v, key, cast):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return DEFAULTS[key][0]

    enabled = _as_bool(await _get("memory.guard_enabled"))
    model = await _get("memory.guard_model")
    model = model.strip() if isinstance(model, str) else ""
    categories = _as_list(await _get("memory.guard_categories"))
    limit = max(1, min(20, _num(await _get("memory.guard_limit"), "memory.guard_limit", int)))
    min_sim = _num(
        await _get("memory.guard_min_similarity"), "memory.guard_min_similarity", float
    )
    timeout = max(5.0, min(120.0, _num(
        await _get("memory.guard_timeout"), "memory.guard_timeout", float
    )))
    max_retries = max(0, min(5, _num(
        await _get("memory.guard_max_retries"), "memory.guard_max_retries", int
    )))
    return {
        "enabled": enabled,
        "model": model,
        "categories": categories,
        "limit": limit,
        "min_similarity": min_sim,
        "timeout": timeout,
        "max_retries": max_retries,
    }


async def build_judge(
    model_key: str,
    *,
    default_provider,
    get_config: Callable[..., Awaitable],
) -> Optional[Judge]:
    """Judge callable for memory.guard_model, or None (check skipped).

    See syne.judge_model.build_judge.
    """
    from .judge_model import build_judge as _build
    return await _build(
        model_key, default_provider=default_provider, get_config=get_config,
        who="Memory guard",
    )


async def find_candidates(
    vector,
    *,
    exclude_ids,
    categories,
    limit: int,
    min_similarity: float,
    access_level: str,
    user_id=None,
) -> list:
    """Closest guarded memories to the draft vector, read-only.

    Deliberately does NOT touch access_count: the guard is a check, not a
    recall, and must not count as one.
    """
    from .db.connection import get_connection
    from .security import check_rule_760

    if not categories or limit <= 0:
        return []
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content, category, created_at, updated_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memory
            WHERE embedding IS NOT NULL
              AND LOWER(category) = ANY($2::text[])
              AND NOT (id = ANY($3::int[]))
              AND ($5::int IS NULL OR user_id = $5::int OR user_id IS NULL)
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            str(vector),
            [c.lower() for c in categories],
            [int(i) for i in (exclude_ids or [])],
            int(limit),
            user_id,
        )
    out = []
    for r in rows:
        sim = r["similarity"]
        if sim is None or sim < min_similarity:
            continue
        allowed, _ = check_rule_760(r["category"] or "", access_level)
        if not allowed:
            continue
        out.append(dict(r))
    return out


_PROMPT = """You check an assistant's draft reply against the owner's stored memories before the reply is sent.

STORED MEMORIES (decisions, preferences and lessons the owner confirmed earlier):
{memories}

OWNER'S MESSAGE:
{user_message}

ASSISTANT DRAFT:
{draft}

A CONFLICT is when the draft does any of these against one of the memories above:
- states something that the memory says is different or wrong;
- re-opens, questions, or offers to revisit something the memory marks as decided, final, or not to be raised again;
- proposes an action that the memory forbids or that the owner already rejected.
A memory that is only on the same topic, or that the draft agrees with, is NOT a conflict. If unsure, answer CLEAN.

Reply with exactly one line and nothing else:
CLEAN
or
CONFLICT|<memory ids, comma-separated>|<short reason>
"""


def _fmt_memory(m: dict) -> str:
    when = m.get("updated_at") or m.get("created_at")
    when_s = f" ({when:%Y-%m-%d})" if hasattr(when, "strftime") else ""
    content = (m.get("content") or "").strip().replace("\n", " ")
    return f"#{m.get('id')} [{m.get('category')}]{when_s} {content[:_MEMORY_CHARS]}"


def build_prompt(draft: str, user_message: str, memories: list) -> str:
    return _PROMPT.format(
        memories="\n".join(_fmt_memory(m) for m in memories),
        user_message=(user_message or "").strip()[:_USER_CHARS],
        draft=(draft or "").strip()[:_DRAFT_CHARS],
    )


_LINE = re.compile(r"^\s*\**\s*(CLEAN|CONFLICT)\b(.*)$", re.I)


def parse_verdict(raw: str, candidate_ids) -> Optional[GuardResult]:
    """CLEAN / CONFLICT|ids|reason -> GuardResult; None when unparseable.

    Ids the judge names that were not among the candidates are dropped; a
    CONFLICT left with no valid id is treated as CLEAN (logged).
    """
    from .rule_checker import _strip_think

    text = _strip_think(raw or "")
    valid = {int(i) for i in candidate_ids}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        if m.group(1).upper() == "CLEAN":
            return GuardResult(conflict=False)
        parts = m.group(2).lstrip(" :|").split("|", 1)
        ids = [int(x) for x in re.findall(r"\d+", parts[0] if parts else "")]
        ids = [i for i in ids if i in valid]
        reason = parts[1].strip() if len(parts) > 1 else ""
        if not ids:
            logger.warning(f"Memory guard: CONFLICT without a valid id, ignored: {line[:200]!r}")
            return GuardResult(conflict=False)
        return GuardResult(conflict=True, memory_ids=ids, reason=reason)
    return None


async def check_draft(
    *,
    draft: str,
    user_message: str,
    embed: Callable[[str], Awaitable],
    judge: Judge,
    exclude_ids,
    access_level: str,
    cfg: dict,
    user_id=None,
) -> Optional[GuardResult]:
    """Run the guard on one draft. None = guard could not decide (fail-open)."""
    text = (draft or "").strip()
    if len([w for w in text.split() if len(w) > 1]) < _MIN_WORDS:
        return GuardResult(conflict=False)
    try:
        resp = await embed(text[:_DRAFT_CHARS])
        vector = getattr(resp, "vector", None)
    except Exception as e:
        logger.warning(f"Memory guard: embed failed: {type(e).__name__}: {e}")
        return None
    if not vector:
        logger.warning("Memory guard: empty draft vector")
        return None

    try:
        candidates = await find_candidates(
            vector,
            exclude_ids=exclude_ids,
            categories=cfg["categories"],
            limit=cfg["limit"],
            min_similarity=cfg["min_similarity"],
            access_level=access_level,
            user_id=user_id,
        )
    except Exception as e:
        logger.warning(f"Memory guard: candidate query failed: {type(e).__name__}: {e}")
        return None
    if not candidates:
        logger.info("Memory guard: no candidates")
        return GuardResult(conflict=False)
    logger.info(
        f"Memory guard [{getattr(judge, 'label', '?')}]: judging against "
        + ", ".join(f"#{c['id']}({c['similarity']:.3f})" for c in candidates)
    )

    try:
        raw = await judge(build_prompt(text, user_message, candidates), cfg["timeout"])
    except Exception as e:
        logger.warning(f"Memory guard: judge failed: {type(e).__name__}: {e}")
        return None

    result = parse_verdict(raw, [c["id"] for c in candidates])
    if result is None:
        logger.warning(f"Memory guard: unparseable verdict: {(raw or '')[:200]!r}")
        return None
    result.memories = [c for c in candidates if c["id"] in result.memory_ids]
    logger.info(
        f"Memory guard: {'CONFLICT ' + str(result.memory_ids) if result.conflict else 'CLEAN'}"
        + (f" — {result.reason}" if result.reason else "")
    )
    return result


def build_nudge(result: GuardResult) -> str:
    lines = [
        "Before sending, your draft was checked against the owner's stored "
        "memories and it conflicts with:",
    ]
    lines += [f"- {_fmt_memory(m)}" for m in result.memories]
    if result.reason:
        lines.append(f"Reason: {result.reason}")
    lines.append(
        "Rewrite your reply so it respects these memories. Drop any part that "
        "re-opens what they settle. Do not mention this check."
    )
    return "\n".join(lines)
