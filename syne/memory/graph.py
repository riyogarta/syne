"""Knowledge Graph — entity/relation storage and graph recall.

Entities and relations live in kg_entities / kg_relations; every source
of a relation is kept in kg_relation_sources. AI extraction from memories
was removed (Riyo, 14 Nov 2026): links are built from written references.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, TYPE_CHECKING

from ..db.connection import get_connection
from ..db.models import get_config

if TYPE_CHECKING:
    from ..llm.provider import LLMProvider

logger = logging.getLogger("syne.memory.graph")

async def _resolve_entity(name: str, entity_type: str, description: str = "") -> int:
    """Find existing entity or create new one. Returns entity ID.

    Resolution order:
    1. Exact name + type match (case-insensitive)
    2. Alias match
    3. Create new entity
    """
    async with get_connection() as conn:
        # 1. Exact match
        row = await conn.fetchrow(
            "SELECT id FROM kg_entities WHERE LOWER(name) = LOWER($1) AND entity_type = $2",
            name, entity_type,
        )
        if row:
            return row["id"]

        # 2. Alias match
        row = await conn.fetchrow(
            "SELECT id FROM kg_entities WHERE entity_type = $1 AND LOWER($2) = ANY(SELECT LOWER(unnest(aliases)))",
            entity_type, name,
        )
        if row:
            return row["id"]

        # 3. Create new
        row = await conn.fetchrow(
            """INSERT INTO kg_entities (name, entity_type, description)
               VALUES ($1, $2, $3)
               ON CONFLICT (LOWER(name), entity_type) DO UPDATE SET updated_at = NOW()
               RETURNING id""",
            name, entity_type, description or "",
        )
        return row["id"]


async def _store_graph(extracted: dict, memory_id: int) -> None:
    """Store extracted entities and relations into the graph tables.

    Only creates entities that are referenced in relations — no orphans.
    """
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])

    # Build entity lookup by name (from LLM extraction)
    entity_info = {}
    for e in entities:
        name = e["name"].strip().lower()
        entity_info[name] = {
            "name": e["name"].strip(),
            "type": e["type"].strip().lower(),
            "desc": e.get("description", "").strip(),
        }

    # Determine which entities are actually needed by relations
    needed = set()
    for r in relations:
        needed.add(r["subject"].strip().lower())
        needed.add(r["object"].strip().lower())

    # Only resolve/create entities that have relations
    entity_map = {}  # name -> entity_id
    for name in needed:
        info = entity_info.get(name)
        if info:
            entity_id = await _resolve_entity(info["name"], info["type"], info["desc"])
            entity_map[name] = entity_id

    # Store relations
    for r in relations:
        subj_name = r["subject"].strip().lower()
        obj_name = r["object"].strip().lower()
        predicate = r["predicate"].strip().lower()

        subj_id = entity_map.get(subj_name)
        obj_id = entity_map.get(obj_name)

        if not subj_id or not obj_id:
            logger.debug(f"Skipping relation: entity not found ({r['subject']} -> {r['object']})")
            continue

        async with get_connection() as conn:
            rel_id = await conn.fetchval(
                """INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (subject_id, predicate, object_id)
                   DO UPDATE SET source_memory_id = $4, updated_at = NOW()
                   RETURNING id""",
                subj_id, predicate, obj_id, memory_id,
            )
            # Every source is kept (kg_relation_sources), not only the last
            # one. Fail-quiet: a missing table never blocks the relation.
            if rel_id is not None:
                try:
                    await conn.execute(
                        "INSERT INTO kg_relation_sources "
                        "(relation_id, source_type, source_id, origin) "
                        "VALUES ($1, 'memory', $2, 'llm') ON CONFLICT DO NOTHING",
                        rel_id, memory_id,
                    )
                except Exception as e:
                    logger.debug(f"Relation source not recorded: {e}")


# ── Graph recall: keyword candidates, ranked (Riyo, 14 Nov 2026) ─────────
# Entities are found by keyword: kw = how many query words (>= 4 chars, no
# stop-word filter) occur in the entity name; score = kw / number_of_words.
# A relation scores by its best end, ties broken by its other end, then
# oldest first. No threshold: the top `limit` relations are injected.
# Entity embeddings are not used (Riyo, 14 Nov 2026): semantic search runs
# once, over the chunks themselves; the graph only follows links.
# Every call with a log_scope is written to kg_recall_log.
_KG_KW_CANDIDATES = 100     # keyword candidates
_KG_REL_ENTITIES = 40       # top-scored entities whose relations are fetched
_KG_REL_PER_ENTITY = 20     # relations fetched per entity (hubs have 20k+)
_KG_LOG_CANDIDATES = 30     # candidates kept in one log row


def _query_words(query: str) -> list[str]:
    """Words matched against entity names: >= 4 chars, else > 2 chars."""
    words = [w for w in query.lower().split() if len(w) >= 4]
    if not words:
        words = [w for w in query.lower().split() if len(w) > 2]
    return words


def _combine_entity_scores(kw_rows, sem_rows, n_words: int) -> dict:
    """{entity_id: {"kw": int, "sem": float|None, "score": float}}."""
    scores: dict = {}
    for r in kw_rows:
        scores[int(r["id"])] = {"kw": int(r["kw"] or 0), "sem": None}
    for r in sem_rows:
        d = scores.setdefault(int(r["id"]), {"kw": 0, "sem": None})
        d["sem"] = float(r["sim"]) if r["sim"] is not None else None
    n = max(1, int(n_words))
    for d in scores.values():
        d["score"] = max(d["sem"] or 0.0, d["kw"] / n)
    return scores


def _rank_relations(rows, scores: dict, limit: int) -> list:
    """Top `limit` relations as [(row, score)], best end first.

    Order: best-end score DESC, other-end score DESC, relation id ASC.
    A relation reached from both of its ends is counted once.
    """
    best: dict = {}
    for r in rows:
        rid = int(r["rid"])
        if rid in best:
            continue
        s = scores.get(int(r["subject_id"]), {}).get("score", 0.0)
        o = scores.get(int(r["object_id"]), {}).get("score", 0.0)
        best[rid] = (max(s, o), min(s, o), -rid, r)
    ranked = sorted(best.values(), key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [(x[3], x[0]) for x in ranked[:max(0, int(limit))]]


async def _log_graph_recall(log_scope: dict, query: str, access_level: str,
                            words: list, query_embedded: bool,
                            embedded, total, kw_rows, sem_rows,
                            scores: dict, ranked: list, latency_ms: int) -> None:
    """One kg_recall_log row per graph recall. Fail-quiet."""
    try:
        names: dict = {}
        for r in kw_rows:
            names[int(r["id"])] = r["name"]
        for r in sem_rows:
            names[int(r["id"])] = r["name"]
        top = sorted(scores.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
        candidates = [
            {
                "id": eid, "name": names.get(eid), "kw": d["kw"],
                "sem": round(d["sem"], 4) if d["sem"] is not None else None,
                "score": round(d["score"], 4),
            }
            for eid, d in top[:_KG_LOG_CANDIDATES]
        ]
        relations = []
        for row, sc in ranked:
            s = scores.get(int(row["subject_id"]), {})
            o = scores.get(int(row["object_id"]), {})
            relations.append({
                "rid": int(row["rid"]),
                "subject_id": int(row["subject_id"]),
                "object_id": int(row["object_id"]),
                "predicate": row["predicate"],
                "score": round(sc, 4),
                "s_kw": s.get("kw", 0),
                "s_sem": round(s["sem"], 4) if s.get("sem") is not None else None,
                "o_kw": o.get("kw", 0),
                "o_sem": round(o["sem"], 4) if o.get("sem") is not None else None,
            })
        sid = log_scope.get("session_id")
        async with get_connection() as conn:
            await conn.execute(
                "INSERT INTO kg_recall_log (session_id, platform, chat_id, "
                "access_level, query, words, query_embedded, entities_embedded, "
                "entities_total, kw_candidates, sem_candidates, candidates, "
                "relations, latency_ms) VALUES ($1, $2, $3, $4, $5, $6::text[], "
                "$7, $8, $9, $10, $11, $12::jsonb, $13::jsonb, $14)",
                int(sid) if sid is not None else None,
                str(log_scope.get("platform") or ""),
                str(log_scope.get("chat_id") or ""),
                access_level, query, list(words), bool(query_embedded),
                embedded, total, len(kw_rows), len(sem_rows),
                json.dumps(candidates, ensure_ascii=False),
                json.dumps(relations, ensure_ascii=False),
                int(latency_ms),
            )
    except Exception as e:
        logger.warning(f"kg_recall_log write failed: {e}")


async def recall_graph(
    query: str,
    limit: int = 10,
    requester_access_level: str = "owner",
    log_scope: Optional[dict] = None,
) -> list[str]:
    """Query the knowledge graph for relevant entity-relation context.

    1. Keyword candidates: query words (>= 4 chars) as substrings of entity
       names, scored by how many words match (GIN trigram index).
    2. Relations of the top-scored entities are fetched (btree indexes on
       subject_id/object_id) and ranked by _rank_relations.

    Entity embeddings are not used (Riyo, 14 Nov 2026).

    ``log_scope`` ({session_id, platform, chat_id}) writes one
    kg_recall_log row. Returns formatted lines (empty on no-match/failure).
    """
    if not query or not query.strip():
        return []

    # --- Rule 760/765 for the knowledge graph ---
    # kg_entities carries no category of its own, so the only provenance marker
    # in the graph is kg_relations.source_memory_id -> memory.category. A
    # relation is therefore releasable to a non-owner/family requester only
    # when it can be PROVEN to come from a public category. Relations whose
    # source memory is absent -- never recorded, or the memory row was deleted
    # (the FK is ON DELETE SET NULL) -- cannot be proven public and are
    # withheld. Without this, graph recall bypassed memory privacy entirely:
    # a keyword (or a semantic match) could drag a private relation into a
    # public requester's context through an unrelated entity.
    public_cats: list[str] | None = None
    if requester_access_level not in ("owner", "family"):
        from ..security import _load_public_categories
        public_cats = await _load_public_categories() or []
        if not public_cats:
            logger.debug(
                "Graph recall: public requester + empty public_categories -> skip"
            )
            return []

    words = _query_words(query)
    if not words:
        return []

    t0 = time.monotonic()
    kw_rows: list = []
    sem_rows: list = []
    scores: dict = {}
    ranked: list = []
    embedded = total = None
    try:
        async with get_connection() as conn:
            # Check if tables exist
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'kg_entities')"
            )
            if not exists:
                return []

            # Step 1: keyword candidates, ranked by keyword overlap
            conditions = []
            score_parts = []
            params: list = []
            for i, word in enumerate(words):
                idx = i + 1
                conditions.append(f"LOWER(name) LIKE ${idx}")
                score_parts.append(f"CASE WHEN LOWER(name) LIKE ${idx} THEN 1 ELSE 0 END")
                params.append(f"%{word}%")
            where_clause = " OR ".join(conditions)
            score_expr = " + ".join(score_parts)
            kw_rows = await conn.fetch(
                f"SELECT id, name, ({score_expr}) AS kw FROM kg_entities "
                f"WHERE ({where_clause}) AND entity_type NOT IN ('chunk', 'ayah') "
                f"ORDER BY kw DESC, id "
                f"LIMIT {_KG_KW_CANDIDATES}",
                *params,
            )

            scores = _combine_entity_scores(kw_rows, sem_rows, len(words))
            if scores:
                top_ids = [
                    eid for eid, _ in sorted(
                        scores.items(), key=lambda kv: (-kv[1]["score"], kv[0])
                    )[:_KG_REL_ENTITIES]
                ]

                # Step 2: relations of the top entities. Per entity, relations
                # whose other end is also a top entity come first, so a hub
                # entity (20k+ relations) still yields its relevant ones.
                # For a non-owner/family requester the category filter is
                # pushed down before LIMIT (same reasoning as Rule 765 in
                # memory/engine.py); the INNER JOIN to memory also drops
                # relations with no traceable source.
                _cat_join = ""
                _cat_cond = ""
                rel_params: list = [top_ids, _KG_REL_PER_ENTITY]
                if public_cats is not None:
                    _cat_join = "JOIN memory m ON m.id = r.source_memory_id"
                    _cat_cond = "AND LOWER(TRIM(m.category)) = ANY($3)"
                    rel_params.append(public_cats)
                rows = await conn.fetch(
                    f"""
                    SELECT x.rid, x.subject_id, x.object_id, x.predicate,
                           s.name AS subject, s.entity_type AS s_type,
                           o.name AS object, o.entity_type AS o_type
                    FROM unnest($1::int[]) AS c(eid)
                    CROSS JOIN LATERAL (
                        SELECT r.id AS rid, r.subject_id, r.object_id, r.predicate
                        FROM kg_relations r
                        {_cat_join}
                        WHERE (r.subject_id = c.eid OR r.object_id = c.eid)
                        {_cat_cond}
                        ORDER BY ((CASE WHEN r.subject_id = c.eid
                                        THEN r.object_id ELSE r.subject_id END)
                                  = ANY($1::int[])) DESC, r.id
                        LIMIT $2
                    ) x
                    JOIN kg_entities s ON s.id = x.subject_id
                    JOIN kg_entities o ON o.id = x.object_id
                    """,
                    *rel_params,
                )
                ranked = _rank_relations(rows, scores, limit)
    except Exception as e:
        logger.error(f"Graph recall failed: {e}")
        ranked = []

    lines = [
        f"- {row['subject']} [{row['s_type']}] --{row['predicate']}--> "
        f"{row['object']} [{row['o_type']}]"
        for row, _score in ranked
    ]

    if log_scope is not None:
        await _log_graph_recall(
            log_scope, query, requester_access_level, words,
            False, embedded, total, kw_rows, sem_rows,
            scores, ranked, int((time.monotonic() - t0) * 1000),
        )
    return lines


# Links of recalled reference chunks (rule-built KG, v1.22.65).
_KG_LINK_HOP1 = 200        # hop-1 relations fetched for the recalled chunks
_KG_LINK_PER_AYAH = 5      # hop-2 relations fetched per ayah reached
_KG_LINK_FIRST = ("explains", "asbab_of", "cites")


async def recall_chunk_links(
    memory_ids,
    limit: int = 10,
    requester_access_level: str = "owner",
) -> list[str]:
    """KG lines from the rule-built links of the memories recalled this turn.

    Entry is the recalled memories, not query words: each linked reference
    chunk has one 'chunk' entity whose description is 'memory:<id>'.
      hop 1: the chunk's own relations (tafsir/asbab/cites -> ayah, Qur'an
             chunk -> its ayahs, part -> next part);
      hop 2: for every ayah reached, other chunks linked to that ayah (the
             Qur'an chunk holding it first, then tafsir, hadith citing it).
    Order: hop-1 explains/asbab_of/cites, then hop 2, then the rest of hop 1;
    within hop 1 by the recall rank of the seed chunk. Rule 760/765: for a
    non-owner/family requester every relation must come from a public
    category, same gate as recall_graph. Returns [] on no-match/failure.
    """
    ids: list[int] = []
    for i in memory_ids or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if i not in ids:
            ids.append(i)
    try:
        limit = int(limit or 0)
    except (TypeError, ValueError):
        limit = 0
    if not ids or limit <= 0:
        return []

    public_cats: list[str] | None = None
    if requester_access_level not in ("owner", "family"):
        from ..security import _load_public_categories
        public_cats = await _load_public_categories() or []
        if not public_cats:
            return []
    cat_join = "JOIN memory m ON m.id = r.source_memory_id" if public_cats is not None else ""
    extra = [public_cats] if public_cats is not None else []

    def _line(r) -> str:
        return (f"- {r['subject']} [{r['s_type']}] --{r['predicate']}--> "
                f"{r['object']} [{r['o_type']}]")

    try:
        async with get_connection() as conn:
            seeds = await conn.fetch(
                "SELECT id, description FROM kg_entities "
                "WHERE entity_type = 'chunk' AND description = ANY($1::text[])",
                [f"memory:{i}" for i in ids],
            )
            if not seeds:
                return []
            rank = {f"memory:{i}": n for n, i in enumerate(ids)}
            seed_rank = {int(r["id"]): rank.get(r["description"], len(ids)) for r in seeds}
            seed_ids = list(seed_rank)

            cond1 = "AND LOWER(TRIM(m.category)) = ANY($3)" if extra else ""
            hop1 = await conn.fetch(
                f"""
                SELECT r.id AS rid, r.subject_id, r.object_id, r.predicate,
                       s.name AS subject, s.entity_type AS s_type,
                       o.name AS object, o.entity_type AS o_type
                FROM kg_relations r
                {cat_join}
                JOIN kg_entities s ON s.id = r.subject_id
                JOIN kg_entities o ON o.id = r.object_id
                WHERE (r.subject_id = ANY($1::int[]) OR r.object_id = ANY($1::int[]))
                {cond1}
                ORDER BY r.id
                LIMIT $2
                """,
                seed_ids, _KG_LINK_HOP1, *extra,
            )

            def _srank(r) -> int:
                big = 1 << 30
                return min(seed_rank.get(int(r["subject_id"]), big),
                           seed_rank.get(int(r["object_id"]), big))

            hop1 = sorted(hop1, key=lambda r: (_srank(r), int(r["rid"])))
            ayah_ids: list[int] = []
            for r in hop1:
                for eid, et in ((r["subject_id"], r["s_type"]), (r["object_id"], r["o_type"])):
                    if et == "ayah" and int(eid) not in ayah_ids:
                        ayah_ids.append(int(eid))

            hop2: list = []
            if ayah_ids:
                cond2 = "AND LOWER(TRIM(m.category)) = ANY($4)" if extra else ""
                hop2 = await conn.fetch(
                    f"""
                    SELECT x.rid, x.subject_id, x.object_id, x.predicate,
                           s.name AS subject, s.entity_type AS s_type,
                           o.name AS object, o.entity_type AS o_type
                    FROM unnest($1::int[]) WITH ORDINALITY AS a(eid, ord)
                    CROSS JOIN LATERAL (
                        SELECT r.id AS rid, r.subject_id, r.object_id, r.predicate,
                               a.ord AS ord
                        FROM kg_relations r
                        {cat_join}
                        WHERE r.object_id = a.eid AND r.subject_id <> ALL($3::int[])
                        {cond2}
                        ORDER BY (r.predicate = 'contains') DESC, r.id
                        LIMIT $2
                    ) x
                    JOIN kg_entities s ON s.id = x.subject_id
                    JOIN kg_entities o ON o.id = x.object_id
                    ORDER BY x.ord, (x.predicate = 'contains') DESC, x.rid
                    """,
                    ayah_ids, _KG_LINK_PER_AYAH, seed_ids, *extra,
                )
    except Exception as e:
        logger.error(f"Chunk link recall failed: {e}")
        return []

    first = [r for r in hop1 if r["predicate"] in _KG_LINK_FIRST]
    rest = [r for r in hop1 if r["predicate"] not in _KG_LINK_FIRST]
    out: list[str] = []
    seen: set = set()
    for r in first + list(hop2) + rest:
        rid = int(r["rid"])
        if rid in seen:
            continue
        seen.add(rid)
        out.append(_line(r))
        if len(out) >= limit:
            break
    return out


async def get_graph_stats() -> dict:
    """Get knowledge graph statistics."""
    try:
        async with get_connection() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'kg_entities')"
            )
            if not exists:
                return {"entities": 0, "relations": 0, "types": []}

            entity_count = await conn.fetchval("SELECT COUNT(*) FROM kg_entities")
            relation_count = await conn.fetchval("SELECT COUNT(*) FROM kg_relations")
            types = await conn.fetch(
                "SELECT entity_type, COUNT(*) as c FROM kg_entities GROUP BY entity_type ORDER BY c DESC"
            )
            return {
                "entities": entity_count or 0,
                "relations": relation_count or 0,
                "types": [{"type": r["entity_type"], "count": r["c"]} for r in types],
            }
    except Exception:
        return {"entities": 0, "relations": 0, "types": []}


async def search_entities(query: str, limit: int = 20) -> list[dict]:
    """Search entities by name. Returns list of entity dicts."""
    try:
        async with get_connection() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'kg_entities')"
            )
            if not exists:
                return []

            rows = await conn.fetch(
                """SELECT e.id, e.name, e.entity_type, e.description,
                          (SELECT COUNT(*) FROM kg_relations WHERE subject_id = e.id OR object_id = e.id) as rel_count
                   FROM kg_entities e
                   WHERE LOWER(e.name) LIKE LOWER($1)
                   ORDER BY rel_count DESC, e.name
                   LIMIT $2""",
                f"%{query}%", limit,
            )
            return [dict(r) for r in rows]
    except Exception:
        return []
