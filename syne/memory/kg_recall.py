"""KG recall (Riyo, 18 Nov 2026) - KG-RAG stage 2, steps 2 and 5.

Reads only relations written by the KG worker (kg_extract.py) in the new
era: a relation counts when it has a kg_relation_sources row with origin
'llm' created at or after kg.extract_era_start. Older LLM guesses in the
same tables are ignored without deleting them.

  step 2 - entities named in the message: word n-grams of the message,
           normalised like extraction does, matched on kg_entities.norm_name;
           Qur'an verses named with "QS ..." are matched on the 'ayah' entities;
  step 5 - expansion: seeds are those entities plus the entities of
           relations taken from the memories recalled this turn (their
           chunk IDs in kg_relation_sources). Hop 1 = relations of the
           seeds, hop 2 = relations of the entities reached (kg.graph_hops).

Privacy (decision, Riyo 18 Nov 2026): owner/family see every relation. A
public requester sees a relation only when one of its sources is a memory
of a public category or a message of the chat being answered.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import timezone
from typing import Optional

from ..db.connection import get_connection
from ..db.models import get_config
from . import kg_refs as _refs
from .kg_extract import REFERENCE_TYPES, ayah_index, ngrams, parse_era

logger = logging.getLogger("syne.memory.kg_recall")

# key -> (default, description). Seeded by migration 54 and schema.sql.
DEFAULTS: dict = {
    "kg.graph_recall_limit": (
        10,
        "KG lines (relations from the KG worker, newer than kg.extract_era_start) added to the context per turn, with source and date. 0 = off. Clamped 0-50.",
    ),
    "kg.graph_hops": (
        2,
        "KG recall depth: 1 = relations of the entities named in the message and of the recalled memories; 2 = also relations of the entities reached. Clamped 1-2.",
    ),
    "kg.graph_per_entity": (
        8,
        "Most relations read per entity in KG recall (newest first), so an entity with many relations does not crowd out the rest. Clamped 1-50.",
    ),
    "kg.graph_max_seeds": (
        10,
        "Most starting entities of KG recall: entities named in the message first, then entities from the recalled memories. Clamped 1-50.",
    ),
}

_CLAMPS = {
    "kg.graph_recall_limit": (0, 50),
    "kg.graph_hops": (1, 2),
    "kg.graph_per_entity": (1, 50),
    "kg.graph_max_seeds": (1, 50),
}


def _clamp(key: str, v) -> int:
    default = DEFAULTS[key][0]
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(default)
    if math.isnan(x):
        x = float(default)
    lo, hi = _CLAMPS[key]
    return int(max(lo, min(hi, x)))


async def load_settings() -> dict:
    out = {}
    for key, (default, _) in DEFAULTS.items():
        try:
            v = await get_config(key, default)
        except Exception:
            v = default
        out[key] = _clamp(key, v)
    try:
        out["era"] = parse_era(await get_config("kg.extract_era_start", None))
    except Exception:
        out["era"] = None
    try:
        out["tz"] = str(await get_config("system.timezone", "UTC") or "UTC")
    except Exception:
        out["tz"] = "UTC"
    return out


def mention_grams(text: str) -> list:
    """Normalised 1-3 word n-grams, longest first (a full name beats a part)."""
    grams = ngrams(text or "")
    return sorted(grams, key=lambda g: (-len(g.split()), grams.index(g)))


# ── SQL ───────────────────────────────────────────────────────────────────

_MATCH_SQL = """
SELECT e.id, e.norm_name
FROM kg_entities e
WHERE e.norm_name = ANY($1::text[])
  AND e.entity_type <> ALL($2::text[])
  AND EXISTS (
      SELECT 1 FROM kg_relations r
      JOIN kg_relation_sources s ON s.relation_id = r.id
      WHERE (r.subject_id = e.id OR r.object_id = e.id)
        AND r.updated_at >= $3
        AND s.origin = 'llm' AND s.created_at >= $3)
"""

_MEMORY_SEEDS_SQL = """
SELECT s.source_id, r.subject_id, r.object_id
FROM kg_relation_sources s
JOIN kg_relations r ON r.id = s.relation_id
WHERE s.source_type = 'memory' AND s.source_id = ANY($1::int[])
  AND s.origin = 'llm' AND s.created_at >= $2
ORDER BY s.source_id, r.id
"""


def expand_sql(public: bool) -> str:
    """Relations of the entities in $1, newest source first, $3 per entity.

    $2 era, $4 relation ids to skip; for a public requester $5 public
    categories and $6 the session being answered.
    """
    priv = ""
    if public:
        priv = """
            AND ((s.source_type = 'memory' AND EXISTS (
                     SELECT 1 FROM memory m WHERE m.id = s.source_id
                       AND LOWER(TRIM(m.category)) = ANY($5::text[])))
              OR (s.source_type = 'message' AND EXISTS (
                     SELECT 1 FROM messages g WHERE g.id = s.source_id
                       AND g.session_id = $6)))"""
    return f"""
SELECT x.rid, x.subject_id, x.object_id, x.predicate, x.eid,
       x.src_type, x.src_id, x.src_at,
       se.name AS subject, se.entity_type AS s_type,
       oe.name AS object, oe.entity_type AS o_type
FROM unnest($1::int[]) AS c(eid)
CROSS JOIN LATERAL (
    SELECT r.id AS rid, r.subject_id, r.object_id, r.predicate, c.eid AS eid,
           src.source_type AS src_type, src.source_id AS src_id,
           src.created_at AS src_at
    FROM kg_relations r
    CROSS JOIN LATERAL (
        SELECT s.source_type, s.source_id, s.created_at
        FROM kg_relation_sources s
        WHERE s.relation_id = r.id AND s.origin = 'llm' AND s.created_at >= $2{priv}
        ORDER BY s.created_at DESC
        LIMIT 1
    ) src
    WHERE (r.subject_id = c.eid OR r.object_id = c.eid)
      AND r.updated_at >= $2
      AND r.id <> ALL($4::int[])
    ORDER BY src.created_at DESC, r.id DESC
    LIMIT $3
) x
JOIN kg_entities se ON se.id = x.subject_id
JOIN kg_entities oe ON oe.id = x.object_id
"""


_AYAH_SQL = """
SELECT e.id, e.name
FROM kg_entities e
WHERE e.entity_type = 'ayah' AND e.name = ANY($1::text[])
  AND EXISTS (
      SELECT 1 FROM kg_relations r
      JOIN kg_relation_sources s ON s.relation_id = r.id
      WHERE (r.subject_id = e.id OR r.object_id = e.id)
        AND r.updated_at >= $2
        AND s.origin = 'llm' AND s.created_at >= $2)
"""

_AYAH_MAX_RANGE = 10   # a longer "QS 2:1-286" mention seeds nothing


async def _ayah_seeds(conn, query: str, era) -> list:
    """'ayah' entity IDs of the "QS ..." mentions in the message that have
    relations from the KG worker, in the order mentioned."""
    try:
        index, pattern = await ayah_index(conn)
        refs, _rej = _refs.parse_mentions(query, index, pattern)
    except Exception as e:
        logger.debug(f"KG recall ayah mentions skipped: {e}")
        return []
    names: list = []
    for ref in refs:
        if ref.end - ref.start + 1 > _AYAH_MAX_RANGE:
            continue
        for a in ref.ayahs():
            n = _refs.ayah_entity_name(ref.surah, a)
            if n not in names:
                names.append(n)
    if not names:
        return []
    rows = await conn.fetch(_AYAH_SQL, names, era)
    order = {n: i for i, n in enumerate(names)}
    return [int(r["id"]) for r in sorted(rows, key=lambda r: order.get(r["name"], 1 << 20))]


# ── ranking + formatting (pure) ───────────────────────────────────────────

def rank(hop1: list, hop2: list, seeds: list, limit: int) -> list:
    """Hop 1 before hop 2. In hop 1: relations joining two seeds first, then
    by the rank of the seed (named in the message before memory seeds),
    then newest source. Each relation once."""
    pos = {eid: n for n, eid in enumerate(seeds)}
    big = len(seeds) + 1

    def k1(r):
        both = int(r["subject_id"]) in pos and int(r["object_id"]) in pos
        p = min(pos.get(int(r["subject_id"]), big), pos.get(int(r["object_id"]), big))
        return (0 if both else 1, p, -_ts(r["src_at"]))

    out, seen = [], set()
    for r in sorted(hop1, key=k1) + sorted(hop2, key=lambda r: -_ts(r["src_at"])):
        rid = int(r["rid"])
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def _ts(dt) -> float:
    try:
        return dt.timestamp()
    except Exception:
        return 0.0


def fmt_date(dt, tz_name: str) -> str:
    try:
        from zoneinfo import ZoneInfo
        dt = dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        try:
            dt = dt.astimezone(timezone.utc)
        except Exception:
            return "?"
    return f"{dt.day} {dt.strftime('%b %Y')}"


def fmt_line(r, tz_name: str) -> str:
    kind = "pesan" if r["src_type"] == "message" else "memori"
    return (f"- {r['subject']} [{r['s_type']}] --{r['predicate']}--> "
            f"{r['object']} [{r['o_type']}] "
            f"(sumber: {kind} #{r['src_id']}, {fmt_date(r['src_at'], tz_name)})")


# ── entry point ───────────────────────────────────────────────────────────

async def recall_kg(
    query: str,
    memory_ids=None,
    *,
    session_id: Optional[int] = None,
    requester_access_level: str = "owner",
) -> list:
    """KG lines for this turn; [] when off, nothing matches, or on failure."""
    s = await load_settings()
    limit = s["kg.graph_recall_limit"]
    era = s["era"]
    if limit <= 0 or era is None:
        return []

    public_cats = None
    if requester_access_level not in ("owner", "family"):
        from ..security import _load_public_categories
        public_cats = await _load_public_categories() or []

    mids: list = []
    for i in memory_ids or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if i not in mids:
            mids.append(i)

    grams = mention_grams(query)
    try:
        async with get_connection() as conn:
            seeds: list = []
            if grams:
                rows = await conn.fetch(_MATCH_SQL, grams, REFERENCE_TYPES, era)
                order = {g: n for n, g in enumerate(grams)}
                for r in sorted(rows, key=lambda r: (order.get(r["norm_name"], 1 << 20), r["id"])):
                    if int(r["id"]) not in seeds:
                        seeds.append(int(r["id"]))
            if re.search(r"\d", query or ""):
                for eid in await _ayah_seeds(conn, query, era):
                    if eid not in seeds:
                        seeds.append(eid)
            if mids:
                mrank = {m: n for n, m in enumerate(mids)}
                rows = await conn.fetch(_MEMORY_SEEDS_SQL, mids, era)
                for r in sorted(rows, key=lambda r: mrank.get(int(r["source_id"]), 1 << 20)):
                    for eid in (int(r["subject_id"]), int(r["object_id"])):
                        if eid not in seeds:
                            seeds.append(eid)
            seeds = seeds[: s["kg.graph_max_seeds"]]
            if not seeds:
                return []

            sql = expand_sql(public_cats is not None)
            extra = [public_cats, session_id] if public_cats is not None else []
            per = s["kg.graph_per_entity"]
            hop1 = await conn.fetch(sql, seeds, era, per, [], *extra)
            hop2: list = []
            if s["kg.graph_hops"] >= 2 and hop1:
                reached: list = []
                for r in rank(hop1, [], seeds, len(hop1)):
                    for eid in (int(r["subject_id"]), int(r["object_id"])):
                        if eid not in seeds and eid not in reached:
                            reached.append(eid)
                reached = reached[: s["kg.graph_max_seeds"]]
                if reached:
                    hop2 = await conn.fetch(
                        sql, reached, era, per, [int(r["rid"]) for r in hop1], *extra)
    except Exception as e:
        logger.error(f"KG recall failed: {e}")
        return []

    picked = rank(list(hop1), list(hop2), seeds, limit)
    return [fmt_line(r, s["tz"]) for r in picked]
