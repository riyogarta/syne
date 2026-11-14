"""Knowledge Graph — entity/relation extraction and graph-enhanced recall.

Extracts entities and relationships from permanent memories and stores them
in PostgreSQL tables (kg_entities, kg_relations). Enhances memory recall
with graph traversal for multi-hop reasoning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional, TYPE_CHECKING

import httpx

from ..db.connection import get_connection
from ..db.models import get_config

if TYPE_CHECKING:
    from ..llm.provider import LLMProvider

logger = logging.getLogger("syne.memory.graph")

# ── KG extraction throttle ──
# Limits concurrent KG extractions to prevent flooding the LLM provider
# when many memories are stored rapidly (e.g. bulk import).
_KG_SEMAPHORE = asyncio.Semaphore(2)  # max 2 concurrent extractions
_KG_DELAY = 1.0  # seconds between extractions



EXTRACT_PROMPT = """You extract knowledge graphs from personal memory statements. Always call store_knowledge_graph with the entities and relations you find.

LANGUAGE
Memory may be in any language (often Indonesian or English). Extract entity names verbatim — do not translate.

SPEAKER
When the speaker is identified (e.g. "The speaker is Riyo"), replace first-person references ("I", "me", "my", "saya", "aku", "ku") with the speaker's real name. Never emit "User", "he", "she", "it" as entity names.

ENTITIES
Extract specific named things — people, places, organizations, roles, events, items, dates, skills, animals, abstract concepts. Always extract at least the people, places, organizations, and key items mentioned.

RELATIONS
Express facts as subject-predicate-object triples. Prefer common predicates in snake_case:
  lives_in, works_at, married_to, child_of, parent_of, sibling_of, friend_of,
  owns, member_of, studies_at, has_role, located_in, part_of, prefers,
  has_condition, takes_medication, knows, likes, dislikes, born_in, visited,
  speaks, plays, follows, supports, founded
Create new snake_case predicates for facts that don't fit the list above. Always extract at least one relation if any subject-predicate-object can be inferred from the memory.

EXAMPLE
Speaker: Riyo
Memory: "Aku tinggal di Jakarta dan bekerja sebagai engineer di Surelden. Suka kopi pagi dan punya kucing bernama Mochi."

entities:
  - Riyo (person)
  - Jakarta (place)
  - Surelden (org)
  - engineer (role)
  - kopi (item)
  - Mochi (animal)
relations:
  - Riyo lives_in Jakarta
  - Riyo works_at Surelden
  - Riyo has_role engineer
  - Riyo prefers kopi
  - Riyo owns Mochi

EMPTY OUTPUT
Only call the tool with empty arrays if the memory is purely emotional or abstract with no concrete entities at all (e.g. "I felt sad today", "Hari ini melelahkan"). Any memory mentioning a name, place, or fact must produce at least one entity."""

# Tool definition for structured extraction — OpenAI format (works for all drivers)
# Anthropic driver has _convert_tools() that converts this to Anthropic format.
_KG_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "store_knowledge_graph",
        "description": (
            "Store entities and their relationships extracted from a memory statement. "
            "Always extract at least the people, places, organizations, and key items "
            "mentioned, even if relations between them are weak."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "description": "All named things mentioned in the memory.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Exact entity name as it appears (do not translate).",
                            },
                            "type": {
                                "type": "string",
                                "enum": [
                                    "person", "place", "org", "concept", "role",
                                    "event", "item", "date", "skill", "animal",
                                ],
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description under 15 words. Leave empty if not informative.",
                            },
                        },
                        "required": ["name", "type"],
                    },
                },
                "relations": {
                    "type": "array",
                    "description": "Subject-predicate-object triples expressing facts. Subject and object names must appear in entities[].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Subject entity name (must match an entity in entities[]).",
                            },
                            "predicate": {
                                "type": "string",
                                "description": "Relation verb in snake_case. Prefer common verbs like lives_in, works_at, owns, prefers; create new snake_case verbs as needed.",
                            },
                            "object": {
                                "type": "string",
                                "description": "Object entity name (must match an entity in entities[]).",
                            },
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
            },
            "required": ["entities", "relations"],
        },
    },
}


async def _mark_kg_processed(memory_id: int) -> None:
    """Mark a memory as KG-processed, regardless of extraction result."""
    try:
        async with get_connection() as conn:
            await conn.execute(
                "UPDATE memory SET kg_processed = true WHERE id = $1", memory_id
            )
    except Exception as e:
        logger.debug(f"Failed to mark kg_processed for memory #{memory_id}: {e}")


async def extract_and_store(
    provider: LLMProvider,
    content: str,
    memory_id: int,
    speaker_name: str = "",
    _use_semaphore: bool = True,
) -> bool:
    """Extract entities/relations from a permanent memory and store in graph.

    This is the main entry point, called after a permanent memory is stored.
    Reads graph config to decide provider vs ollama extraction.

    Uses a semaphore to limit concurrent extractions — prevents flooding
    the LLM provider when many memories are stored rapidly (bulk import).
    Reprocess bypasses the semaphore since it's already sequential.

    Returns True if extraction succeeded, False otherwise.
    """
    return await _do_extract_and_store(provider, content, memory_id, speaker_name, _use_semaphore)


async def _do_extract_and_store(
    provider: LLMProvider,
    content: str,
    memory_id: int,
    speaker_name: str,
    use_semaphore: bool,
) -> bool:
    """Internal extraction — optionally throttled by semaphore."""
    if use_semaphore:
        async with _KG_SEMAPHORE:
            result = await _run_extraction(provider, content, memory_id, speaker_name)
        # Delay OUTSIDE semaphore — don't waste the slot on sleeping
        await asyncio.sleep(_KG_DELAY)
        return result
    else:
        return await _run_extraction(provider, content, memory_id, speaker_name)


async def _run_extraction(
    provider: LLMProvider,
    content: str,
    memory_id: int,
    speaker_name: str,
) -> bool:
    """Core extraction logic without throttling."""
    try:
        enabled = await get_config("graph.enabled", True)
        if not enabled:
            return False

        driver = await get_config("graph.extractor_driver", "provider")

        if driver == "ollama":
            model = await get_config("graph.extractor_model", "qwen3:8b")
            extracted = await _extract_via_ollama(content, model=model, speaker_name=speaker_name)
        else:
            extracted = await _extract_via_provider(provider, content, speaker_name=speaker_name)

        # None = error (provider returned empty, network error, etc.)
        # Do NOT mark processed — keep as pending for retry
        if extracted is None:
            logger.warning(f"KG extraction returned None for memory #{memory_id} (will retry)")
            return False

        # Empty but valid extraction — genuinely no entities/relations
        # Mark processed so it won't be retried
        if not extracted.get("entities") and not extracted.get("relations"):
            logger.debug(f"No entities/relations extracted from memory #{memory_id}")
            await _mark_kg_processed(memory_id)
            return False

        await _store_graph(extracted, memory_id, provider=provider)
        await _mark_kg_processed(memory_id)
        logger.info(
            f"Graph: stored {len(extracted.get('entities', []))} entities, "
            f"{len(extracted.get('relations', []))} relations from memory #{memory_id}"
        )
        return True

    except Exception as e:
        # Error — do NOT mark processed, keep as pending for retry
        logger.error(f"Graph extraction failed for memory #{memory_id}: {e}")
        return False


async def _extract_via_provider(provider: LLMProvider, content: str, speaker_name: str = "") -> Optional[dict]:
    """Extract entities/relations using the main LLM provider via tool call.

    Uses structured tool output instead of raw JSON parsing — the API
    guarantees valid structure, no truncation, no preamble text issues.
    """
    from ..llm.provider import ChatMessage

    speaker_ctx = f"The speaker is {speaker_name}. " if speaker_name else ""
    messages = [
        ChatMessage(role="system", content=EXTRACT_PROMPT),
        ChatMessage(role="user", content=f'{speaker_ctx}Memory: "{content}"'),
    ]

    response = await provider.chat(
        messages, temperature=0.1, thinking_budget=None,
        tools=[_KG_EXTRACT_TOOL],
    )

    logger.debug(
        f"KG extract response: tool_calls={response.tool_calls is not None and len(response.tool_calls) if response.tool_calls else 0}, "
        f"content={len(response.content or '')} chars, provider={provider.name}"
    )

    # Extract from tool call response — handle both driver formats:
    #   1. Anthropic/OpenAI/Codex: {"name": ..., "args": <dict>, "id": ...}
    #   2. Vertex/Google: {"id": ..., "function": {"name": ..., "arguments": <json_string>}}
    if response.tool_calls:
        tc = response.tool_calls[0]
        logger.debug(f"KG tool_call raw keys: {list(tc.keys())}")
        args = tc.get("args") or tc.get("input")
        if args is None:
            # Vertex/Google format: function.arguments is a JSON string
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            logger.debug(f"KG tool_call function.arguments type={type(raw_args).__name__}, value={str(raw_args)[:300]}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
        if not isinstance(args, dict):
            args = {}
        entities = args.get("entities", [])
        relations = args.get("relations", [])

        # Validate structure
        valid_entities = [
            e for e in entities
            if isinstance(e, dict) and e.get("name") and e.get("type")
        ]
        valid_relations = [
            r for r in relations
            if isinstance(r, dict) and r.get("subject") and r.get("predicate") and r.get("object")
        ]
        logger.debug(f"KG parsed: {len(valid_entities)} entities, {len(valid_relations)} relations")
        return {"entities": valid_entities, "relations": valid_relations}

    # Fallback: LLM responded with text instead of tool call — try parsing
    if response.content and response.content.strip():
        logger.debug(f"KG no tool_call, falling back to text parse: {response.content[:300]}")
        return _parse_extraction(response.content.strip())

    return None


async def _extract_via_ollama(
    content: str,
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    speaker_name: str = "",
) -> Optional[dict]:
    """Extract entities/relations using Ollama."""
    speaker_ctx = f"The speaker is {speaker_name}. " if speaker_name else ""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": EXTRACT_PROMPT},
                        {"role": "user", "content": f'{speaker_ctx}Memory: "{content}"'},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data.get("message", {}).get("content", "").strip()
        # Strip thinking tags (qwen3)
        if "<think>" in raw:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        return _parse_extraction(raw)

    except ValueError:
        raise  # Parse failure — let caller handle (don't mark kg_processed)
    except Exception as e:
        logger.error(f"Ollama graph extraction failed: {e}")
        return None


def _parse_extraction(raw: str) -> Optional[dict]:
    """Parse LLM extraction output into structured dict.

    Returns None on parse failure — caller should NOT mark kg_processed
    for parse errors (they are retryable, unlike genuinely empty results).
    Raises ValueError on parse failure so caller can distinguish from empty.
    """
    if not raw:
        return None

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = cleaned.strip()

    # Extract JSON from response — find outermost { ... }
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not json_match:
        logger.warning(f"No JSON found in extraction: {raw[:500]}")
        raise ValueError(f"No JSON found in extraction output")

    json_str = json_match.group()

    # Try parsing as-is first
    data = _try_parse_json(json_str)
    if data is None:
        # Fix common LLM JSON issues: trailing commas, unescaped newlines
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)  # trailing commas
        fixed = fixed.replace('\n', ' ')  # flatten newlines inside strings
        data = _try_parse_json(fixed)

    if data is None:
        logger.warning(f"Invalid JSON in extraction: {raw[:500]}")
        raise ValueError(f"Invalid JSON in extraction output")

    entities = data.get("entities", [])
    relations = data.get("relations", [])

    # Validate structure
    valid_entities = []
    for e in entities:
        if isinstance(e, dict) and e.get("name") and e.get("type"):
            valid_entities.append(e)

    valid_relations = []
    for r in relations:
        if isinstance(r, dict) and r.get("subject") and r.get("predicate") and r.get("object"):
            valid_relations.append(r)

    return {"entities": valid_entities, "relations": valid_relations}


def _try_parse_json(s: str) -> Optional[dict]:
    """Try to parse JSON string, return None on failure."""
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


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


def _entity_embed_text(name: str, entity_type: str = "", description: str = "") -> str:
    """Text embedded for one entity: "name (type): description".

    Used both at extraction time and by `syne memory reembed-kg`, so every
    entity embedding is built from the same shape of text.
    """
    head = (name or "").strip()
    et = (entity_type or "").strip()
    if et:
        head = f"{head} ({et})"
    desc = (description or "").strip()
    return f"{head}: {desc}" if desc else head


async def _embed_entities(provider: "LLMProvider", entity_ids: list[int]) -> int:
    """Embed the given entities that have no embedding yet. Fail-quiet.

    Returns how many were embedded. A failure on one entity (provider error,
    dimension mismatch) skips that entity only; extraction never breaks.
    """
    if not entity_ids:
        return 0
    try:
        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT id, name, entity_type, description FROM kg_entities "
                "WHERE id = ANY($1::int[]) AND embedding IS NULL",
                list(entity_ids),
            )
    except Exception as e:
        logger.debug(f"Entity embed lookup failed: {e}")
        return 0
    done = 0
    for r in rows:
        try:
            resp = await provider.embed(
                _entity_embed_text(r["name"], r["entity_type"], r["description"])
            )
            vec = getattr(resp, "vector", None)
            if not vec:
                continue
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE kg_entities SET embedding = $1::vector WHERE id = $2",
                    str(list(vec)), r["id"],
                )
            done += 1
        except Exception as e:
            logger.debug(f"Entity #{r['id']} embed failed: {e}")
    return done


async def _store_graph(extracted: dict, memory_id: int, provider: "Optional[LLMProvider]" = None) -> None:
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

    # Entities this memory touched that have no embedding yet get one now,
    # so semantic graph recall can find them (old entities are filled by
    # `syne memory reembed-kg`). Fail-quiet: never blocks the relations.
    if provider is not None and entity_map:
        try:
            await _embed_entities(provider, sorted(set(entity_map.values())))
        except Exception as e:
            logger.debug(f"Entity embedding skipped: {e}")

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
            await conn.execute(
                """INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (subject_id, predicate, object_id)
                   DO UPDATE SET source_memory_id = $4, updated_at = NOW()""",
                subj_id, predicate, obj_id, memory_id,
            )


# ── Graph recall: keyword + semantic, ranked (Riyo, 14 Nov 2026) ──────────
# Entities are found two ways and every candidate keeps both scores:
#   kw  = how many query words occur in the entity name (substring match,
#         words >= 4 chars; no stop-word filter - that needs research first)
#   sem = cosine similarity between the query vector and the entity
#         embedding ("name (type): description"); None when the entity is
#         not among the nearest _KG_SEM_CANDIDATES, has no embedding yet,
#         or the turn has no query vector
# score = max(sem, kw / number_of_words). A relation scores by its best
# end, ties broken by its other end, then oldest first. No threshold: the
# top `limit` relations are injected. Every call with a log_scope is
# written to kg_recall_log (candidates with both raw scores, the injected
# relations, how many entities were embedded at the time) so the ranking
# formula and any threshold can be chosen from data. The max() formula is
# interim; analysis should use the raw kw/sem values.
_KG_KW_CANDIDATES = 100     # keyword candidates (same cap as before)
_KG_SEM_CANDIDATES = 50     # nearest entities by embedding
_KG_REL_ENTITIES = 40       # top-scored entities whose relations are fetched
_KG_REL_PER_ENTITY = 20     # relations fetched per entity (hubs have 20k+)
_KG_LOG_CANDIDATES = 30     # candidates kept in one log row
_KG_COUNT_TTL = 300.0       # seconds the embedded-entity count is cached
_kg_count_cache: dict = {"t": 0.0, "embedded": None, "total": None}


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


async def _entity_embed_counts(conn) -> tuple:
    """(entities with an embedding, all entities), cached _KG_COUNT_TTL s."""
    now = time.monotonic()
    c = _kg_count_cache
    if c["embedded"] is not None and now - c["t"] < _KG_COUNT_TTL:
        return c["embedded"], c["total"]
    row = await conn.fetchrow(
        "SELECT count(embedding) AS embedded, count(*) AS total FROM kg_entities"
    )
    c.update(t=now, embedded=int(row["embedded"] or 0), total=int(row["total"] or 0))
    return c["embedded"], c["total"]


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
    query_vector: Optional[list] = None,
    log_scope: Optional[dict] = None,
) -> list[str]:
    """Query the knowledge graph for relevant entity-relation context.

    1. Keyword candidates: query words (>= 4 chars) as substrings of entity
       names, scored by how many words match (GIN trigram index).
    2. Semantic candidates: nearest entities to ``query_vector`` by
       embedding, only when a vector is given and some entity is embedded.
       A failure here (no index, dimension mismatch) falls back to keyword
       candidates only.
    3. Relations of the top-scored entities are fetched (btree indexes on
       subject_id/object_id) and ranked by _rank_relations.

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
                f"WHERE {where_clause} "
                f"ORDER BY kw DESC, id "
                f"LIMIT {_KG_KW_CANDIDATES}",
                *params,
            )

            # Step 2: semantic candidates (skipped while nothing is embedded)
            try:
                embedded, total = await _entity_embed_counts(conn)
            except Exception as e:
                logger.debug(f"Graph embed count failed: {e}")
            if query_vector and embedded:
                try:
                    sem_rows = await conn.fetch(
                        "SELECT id, name, 1 - (embedding <=> $1::vector) AS sim "
                        "FROM kg_entities WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> $1::vector LIMIT $2",
                        str(list(query_vector)), _KG_SEM_CANDIDATES,
                    )
                except Exception as e:
                    logger.warning(f"Graph semantic recall skipped: {e}")
                    sem_rows = []

            scores = _combine_entity_scores(kw_rows, sem_rows, len(words))
            if scores:
                top_ids = [
                    eid for eid, _ in sorted(
                        scores.items(), key=lambda kv: (-kv[1]["score"], kv[0])
                    )[:_KG_REL_ENTITIES]
                ]

                # Step 3: relations of the top entities. Per entity, relations
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
            bool(query_vector), embedded, total, kw_rows, sem_rows,
            scores, ranked, int((time.monotonic() - t0) * 1000),
        )
    return lines


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


async def reprocess_permanent_memories(provider: "LLMProvider", force: bool = False, conversations_mgr=None) -> dict:
    """Re-extract graph from permanent memories that haven't been KG-processed.

    Useful after fixing extraction bugs — processes memories that were
    stored before graph extraction was working.

    Args:
        provider: LLM provider for extraction
        conversations_mgr: ConversationManager — if provided, pauses extraction
            when any chat is active to avoid concurrent API calls
        force: If True, reset kg_processed flag for memories that have no
               KG relations yet, then reprocess them. Useful when bulk imports
               incorrectly set kg_processed=True.

    Returns dict with counts of processed/succeeded/failed.
    """
    from ..db.connection import get_connection

    stats = {"processed": 0, "succeeded": 0, "failed": 0, "reset": 0}

    try:
        async with get_connection() as conn:
            # Ensure kg_processed column exists
            col_exists = await conn.fetchval(
                """SELECT EXISTS(
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'memory' AND column_name = 'kg_processed'
                )"""
            )
            if not col_exists:
                await conn.execute(
                    "ALTER TABLE memory ADD COLUMN kg_processed BOOLEAN DEFAULT false"
                )
                # Backfill: mark memories that already have KG relations as processed
                await conn.execute("""
                    UPDATE memory SET kg_processed = true
                    WHERE id IN (SELECT DISTINCT source_memory_id FROM kg_relations)
                """)

            # Force mode: reset kg_processed for memories marked processed
            # but that have NO actual KG relations (e.g. bulk import set True incorrectly)
            if force:
                reset_result = await conn.execute("""
                    UPDATE memory SET kg_processed = false
                    WHERE permanent = true
                    AND kg_processed = true
                    AND id NOT IN (SELECT DISTINCT source_memory_id FROM kg_relations)
                """)
                reset_count = int(reset_result.split()[-1]) if reset_result else 0
                stats["reset"] = reset_count
                if reset_count:
                    logger.info(f"Force reprocess: reset kg_processed on {reset_count} memories")

            # Find permanent memories that haven't been KG-processed yet
            rows = await conn.fetch("""
                SELECT m.id, m.content, u.display_name, u.name
                FROM memory m
                LEFT JOIN users u ON m.user_id = u.id
                WHERE m.permanent = true
                AND (m.kg_processed IS NOT TRUE)
                ORDER BY m.id
            """)

        logger.info(f"KG reprocess: {len(rows)} memories to process")

        # Read config once — avoid repeated DB queries in loop
        driver = await get_config("graph.extractor_driver", "provider")
        kg_model = await get_config("graph.extractor_model", "qwen3:8b") if driver == "ollama" else None

        import time as _time

        for row in rows:
            # Pause if any chat is active — avoid concurrent API calls on same provider
            if conversations_mgr and hasattr(conversations_mgr, 'is_any_chat_active'):
                while conversations_mgr.is_any_chat_active():
                    await asyncio.sleep(1.0)

            stats["processed"] += 1
            speaker = row["display_name"] or row["name"] or ""
            t0 = _time.monotonic()
            try:
                # Extract directly — bypass semaphore and per-call config reads
                if driver == "ollama":
                    extracted = await _extract_via_ollama(
                        row["content"], model=kg_model, speaker_name=speaker,
                    )
                else:
                    extracted = await _extract_via_provider(
                        provider, row["content"], speaker_name=speaker,
                    )

                elapsed = _time.monotonic() - t0

                # None = error → don't mark, keep pending for retry
                if extracted is None:
                    logger.warning(f"Reprocess #{row['id']}: extraction returned None ({elapsed:.1f}s)")
                    stats["failed"] += 1
                    continue

                # Empty but valid → genuinely no entities, mark processed
                if not extracted.get("entities") and not extracted.get("relations"):
                    logger.debug(f"Reprocess #{row['id']}: no entities ({elapsed:.1f}s)")
                    await _mark_kg_processed(row["id"])
                    stats["succeeded"] += 1
                    continue

                await _store_graph(extracted, row["id"], provider=provider)
                await _mark_kg_processed(row["id"])
                stats["succeeded"] += 1
                logger.info(
                    f"Reprocess #{row['id']}: {len(extracted.get('entities', []))} entities, "
                    f"{len(extracted.get('relations', []))} relations ({elapsed:.1f}s)"
                )

            except ValueError as e:
                elapsed = _time.monotonic() - t0
                # Parse failure → don't mark, keep pending for retry
                logger.warning(f"Reprocess #{row['id']}: parse error ({elapsed:.1f}s): {e}")
                stats["failed"] += 1
            except Exception as e:
                elapsed = _time.monotonic() - t0
                # Error → don't mark, keep pending for retry
                logger.error(f"Reprocess #{row['id']} failed ({elapsed:.1f}s): {e}")
                stats["failed"] += 1
            # Delay between extractions — Gemini CCA has strict rate limits
            _delay = 3.0 if provider.name in ("google", "vertex") else 0.3
            await asyncio.sleep(_delay)

    except Exception as e:
        logger.error(f"Reprocess permanent memories failed: {e}")

    return stats


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
