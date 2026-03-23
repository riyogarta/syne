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

EXTRACT_PROMPT = """You are a knowledge graph extractor. Given a memory statement, extract entities and their relationships using the store_knowledge_graph tool.

Rules:
- Entity names must be specific (not "he", "she", "it", "User")
- When the speaker is identified, use their real name instead of "User"
- Use consistent predicate verbs: lives_in, works_at, married_to, child_of, parent_of, sibling_of, friend_of, owns, member_of, studies_at, has_role, located_in, part_of, prefers, has_condition, takes_medication
- If no clear entities or relations, call the tool with empty arrays
- Keep descriptions brief (under 15 words)"""

# Tool definition for structured extraction — OpenAI format (works for all drivers)
# Anthropic driver has _convert_tools() that converts this to Anthropic format.
_KG_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "store_knowledge_graph",
        "description": "Store extracted entities and relations from a memory statement",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Exact entity name"},
                            "type": {
                                "type": "string",
                                "enum": ["person", "place", "org", "concept", "role", "event", "item"],
                            },
                            "description": {"type": "string", "description": "One-line description"},
                        },
                        "required": ["name", "type"],
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "Subject entity name"},
                            "predicate": {"type": "string", "description": "Relation verb"},
                            "object": {"type": "string", "description": "Object entity name"},
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


def _is_kg_worthy(content: str) -> bool:
    """Check if content has enough substance for KG extraction.

    Filters out short messages, commands, and low-value content
    that would waste an LLM call and always produce 0 relations.
    """
    text = content.strip()
    if len(text) < 30:
        return False
    # Count words with substance (> 2 chars)
    words = [w for w in text.split() if len(w) > 2]
    if len(words) < 5:
        return False
    return True


async def _run_extraction(
    provider: LLMProvider,
    content: str,
    memory_id: int,
    speaker_name: str,
) -> bool:
    """Core extraction logic without throttling."""
    try:
        # Skip content too short/trivial for KG — mark processed immediately
        if not _is_kg_worthy(content):
            logger.debug(f"KG skip: content too short for memory #{memory_id}")
            await _mark_kg_processed(memory_id)
            return False

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

        await _store_graph(extracted, memory_id)
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
        messages, temperature=0.1, thinking_budget=0,
        tools=[_KG_EXTRACT_TOOL],
    )

    # Extract from tool call response
    if response.tool_calls:
        tc = response.tool_calls[0]
        args = tc.get("args") or tc.get("input") or {}
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
        return {"entities": valid_entities, "relations": valid_relations}

    # Fallback: LLM responded with text instead of tool call — try parsing
    if response.content and response.content.strip():
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
            await conn.execute(
                """INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (subject_id, predicate, object_id)
                   DO UPDATE SET source_memory_id = $4, updated_at = NOW()""",
                subj_id, predicate, obj_id, memory_id,
            )


async def recall_graph(query: str, limit: int = 10) -> list[str]:
    """Query the knowledge graph for relevant entity-relation context.

    Searches entities by name substring, then traverses 1-hop relations.
    Returns formatted context lines for injection into conversation.
    """
    if not query or not query.strip():
        return []

    try:
        async with get_connection() as conn:
            # Check if tables exist
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'kg_entities')"
            )
            if not exists:
                return []

            # Find matching entities by word overlap
            words = [w for w in query.lower().split() if len(w) > 2]
            if not words:
                return []

            # Build OR conditions for each word
            conditions = []
            params = []
            for i, word in enumerate(words):
                conditions.append(f"LOWER(e.name) LIKE ${ i + 1}")
                params.append(f"%{word}%")

            where_clause = " OR ".join(conditions)

            # Get matching entities and their 1-hop relations
            rows = await conn.fetch(
                f"""SELECT DISTINCT
                        s.name AS subject, s.entity_type AS s_type,
                        r.predicate,
                        o.name AS object, o.entity_type AS o_type
                    FROM kg_relations r
                    JOIN kg_entities s ON r.subject_id = s.id
                    JOIN kg_entities o ON r.object_id = o.id
                    JOIN kg_entities e ON (e.id = s.id OR e.id = o.id)
                    WHERE {where_clause}
                    LIMIT ${ len(params) + 1 }""",
                *params, limit,
            )

            if not rows:
                return []

            lines = []
            for row in rows:
                lines.append(
                    f"- {row['subject']} [{row['s_type']}] --{row['predicate']}--> "
                    f"{row['object']} [{row['o_type']}]"
                )
            return lines

    except Exception as e:
        logger.error(f"Graph recall failed: {e}")
        return []


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


async def reprocess_permanent_memories(provider: "LLMProvider", force: bool = False) -> dict:
    """Re-extract graph from permanent memories that haven't been KG-processed.

    Useful after fixing extraction bugs — processes memories that were
    stored before graph extraction was working.

    Args:
        provider: LLM provider for extraction
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
                    stats["failed"] += 1
                    continue

                await _store_graph(extracted, row["id"])
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
