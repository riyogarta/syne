"""Memory engine — store and recall with semantic search."""

import logging
from datetime import datetime, timezone
from typing import Optional
from ..db.connection import get_connection
from ..llm.provider import LLMProvider
from ..security import check_rule_760

logger = logging.getLogger("syne.memory.engine")


def format_relative_time(dt, now: Optional[datetime] = None, locale: str = "id") -> str:
    """Render a past datetime as a short relative-time phrase.

    Returns a phrase like "baru saja" / "3 jam lalu" / "5 hari lalu" /
    "2 bulan lalu" / "1 tahun lalu". Locale 'id' (default) returns Indonesian;
    anything else returns English ("3 hours ago", etc.).

    Returns "" if dt is None or not a datetime, so callers can drop the prefix
    cleanly when no timestamp is available (e.g. legacy memories).
    """
    if not isinstance(dt, datetime):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta = now - dt
    seconds = delta.total_seconds()
    # Treat slight clock skew (memory stamped microseconds in the "future") as now.
    if seconds < 60:
        return "baru saja" if locale == "id" else "just now"

    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} menit lalu" if locale == "id" else f"{minutes} min ago"

    hours = int(seconds // 3600)
    if hours < 24:
        return f"{hours} jam lalu" if locale == "id" else f"{hours} hr ago"

    days = int(seconds // 86400)
    if days < 30:
        return f"{days} hari lalu" if locale == "id" else f"{days} day{'s' if days != 1 else ''} ago"

    months = int(days // 30)
    if months < 12:
        return f"{months} bulan lalu" if locale == "id" else f"{months} mo ago"

    years = int(days // 365)
    return f"{years} tahun lalu" if locale == "id" else f"{years} yr{'s' if years != 1 else ''} ago"


class MemoryEngine:
    """Handles storing and recalling memories using PostgreSQL + pgvector."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def store(
        self,
        content: str,
        category: str = "fact",
        source: str = "user_confirmed",
        user_id: Optional[int] = None,
        importance: float = 0.5,
        permanent: bool = False,
    ) -> int:
        """Store a memory with its embedding vector.

        Args:
            permanent: If True, memory never decays (explicit "remember this").
                      If False (default), memory has recall_count that decays over conversations.

        Raises:
            RuntimeError: if embedding generation returns an empty vector. We refuse
                to insert zombie memory rows (memories without embedding can't be
                recalled and crash similarity comparisons).
        """
        from ..db.models import get_config

        # Generate embedding
        embedding_resp = await self.provider.embed(content)
        vector = embedding_resp.vector
        if not vector:
            raise RuntimeError(
                "Refusing to store memory: embedding provider returned empty vector. "
                "Check Ollama/provider status."
            )

        initial_count = int(await get_config("memory.initial_recall_count", "5")) if not permanent else 0

        async with get_connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO memory (content, category, embedding, source, user_id, importance, permanent, recall_count)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8)
                RETURNING id
            """, content, category, str(vector), source, user_id, importance, permanent, initial_count)

            # Decay v2: store-event tick — a brand-new non-permanent memory
            # nudges all OTHER non-permanent memories down by 1 (permanent
            # immune). New row keeps its initial recall_count. Life/death is
            # decided by the cap in run_decay(), not by hitting 0.
            if not permanent:
                await conn.execute("""
                    UPDATE memory
                    SET recall_count = recall_count - 1
                    WHERE COALESCE(permanent, false) = false
                      AND id <> $1
                """, row["id"])

            return row["id"]

    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        category: Optional[str] = None,
        categories: Optional[list[str]] = None,
        user_id: Optional[int] = None,
        requester_access_level: str = "public",
    ) -> list[dict]:
        """Recall memories by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold
            category: Filter by single category (legacy, still supported)
            categories: Filter by multiple categories (e.g. ["fact", "preference"])
            user_id: Filter by user
            requester_access_level: Access level of the requester (for Rule 760 filtering)

        Returns:
            List of matching memories (filtered by Rule 760 for privacy)
        """
        # ═══════════════════════════════════════════════════════
        # RULE 760/765 — PRIVACY PROTECTION
        # Owner/family: access all. Public: only allowed categories (Rule 765).
        # Load public categories cache for Rule 765 checks below.
        # ═══════════════════════════════════════════════════════
        if requester_access_level not in ("owner", "family"):
            from ..security import _load_public_categories
            await _load_public_categories()  # refresh cache for Rule 765

        # Skip recall for very short queries (1 word) — no meaningful semantic match
        words = [w for w in query.strip().split() if len(w) > 1]
        if len(words) < 2:
            logger.debug(f"Recall skipped: query too short ({query!r})")
            return []

        # Generate query embedding
        embedding_resp = await self.provider.embed(query)
        vector = embedding_resp.vector

        async with get_connection() as conn:
            # Build query with optional filters. All columns prefixed with m. so
            # that the LEFT JOIN to memory_blobs (alias b) doesn't conflict.
            conditions = ["1 = 1"]
            params = [str(vector), limit]
            param_idx = 3

            # Category filter: categories (list) takes priority over category (single)
            _cats = categories if categories else ([category] if category else None)
            if _cats:
                if len(_cats) == 1:
                    conditions.append(f"m.category = ${param_idx}")
                    params.append(_cats[0])
                else:
                    conditions.append(f"m.category = ANY(${param_idx})")
                    params.append(_cats)
                param_idx += 1

            if user_id is not None:
                conditions.append(f"(m.user_id = ${param_idx} OR m.user_id IS NULL)")
                params.append(user_id)
                param_idx += 1

            where = " AND ".join(conditions)

            # LEFT JOIN memory_blobs is cheap (only metadata, no BYTEA content)
            # so semantic search reveals which results have attachments.
            rows = await conn.fetch(f"""
                SELECT
                    m.id, m.content, m.category, m.source, m.importance,
                    m.access_count, m.created_at,
                    COALESCE(m.permanent, false) as permanent,
                    COALESCE(m.recall_count, 1) as recall_count,
                    1 - (m.embedding <=> $1::vector) as similarity,
                    (b.memory_id IS NOT NULL) AS has_attachment,
                    b.filename AS attachment_filename,
                    b.mime_type AS attachment_mime,
                    b.size_bytes AS attachment_size
                FROM memory m
                LEFT JOIN memory_blobs b ON b.memory_id = m.id
                WHERE {where}
                  AND m.embedding IS NOT NULL
                ORDER BY m.embedding <=> $1::vector
                LIMIT $2
            """, *params)

            # Filter by minimum similarity and update access stats
            results = []
            ids_to_update = []
            for row in rows:
                # Null guard — defensive; SQL filter should make this impossible
                if row["similarity"] is None:
                    continue
                if row["similarity"] >= min_similarity:
                    # ═══════════════════════════════════════════════════════
                    # RULE 760 CHECK — FAMILY PRIVACY PROTECTION
                    # Filter out private memory categories for non-owner/family
                    # ═══════════════════════════════════════════════════════
                    mem_category = row.get("category", "")
                    allowed, reason = check_rule_760(mem_category, requester_access_level)
                    if not allowed:
                        logger.debug(f"Rule 760 filtered memory id={row['id']}: {reason}")
                        continue  # Skip this memory
                    
                    results.append(dict(row))
                    ids_to_update.append(row["id"])

            # Sort by combined score: similarity (primary) + recall_count boost
            # Higher recall_count = more frequently relevant = slight priority boost
            if results:
                for r in results:
                    rc = r.get("recall_count", 1)
                    # Boost: log2(recall_count+1) * 0.02 — subtle but meaningful
                    # rc=1 → +0.02, rc=5 → +0.05, rc=20 → +0.09, rc=100 → +0.13
                    import math
                    r["_score"] = r["similarity"] + math.log2(max(rc, 1) + 1) * 0.02
                results.sort(key=lambda r: r["_score"], reverse=True)

            # Update access stats + recall_count (for non-permanent memories)
            if ids_to_update:
                await conn.execute("""
                    UPDATE memory
                    SET access_count = access_count + 1,
                        accessed_at = NOW(),
                        recall_count = CASE WHEN permanent = true THEN recall_count ELSE recall_count + 1 END
                    WHERE id = ANY($1)
                """, ids_to_update)

                # Decay v2: event-driven tick — every recall nudges all
                # OTHER non-permanent memories down by 1 (permanent immune).
                # recall_count may go 0/negative; that's just ordering score.
                # Life/death is decided by the cap in run_decay(), not by 0.
                await conn.execute("""
                    UPDATE memory
                    SET recall_count = recall_count - 1
                    WHERE COALESCE(permanent, false) = false
                      AND NOT (id = ANY($1))
                """, ids_to_update)

            # ═══════════════════════════════════════════════════════════
            # CONFLICT DETECTION — flag contradictory memories
            # If multiple high-similarity results are about the same
            # topic but have different content, mark them as conflicting.
            # This is CODE-ENFORCED so the LLM gets structured metadata
            # instead of relying on prompt instructions to detect conflicts.
            # ═══════════════════════════════════════════════════════════
            results = self._detect_conflicts(results)

            return results

    def _detect_conflicts(self, results: list[dict]) -> list[dict]:
        """Detect and flag conflicting memories in recall results.

        Two memories conflict when they:
        1. Are in the same category
        2. Have high mutual similarity (same topic)
        3. But different content (different values)

        When conflicts are found:
        - Higher authority (user_confirmed > auto_captured) gets "authoritative" flag
        - Lower authority gets "conflicted" flag with reference to the authoritative one
        - Same authority: newer gets "authoritative", older gets "conflicted"

        This gives the LLM structured metadata to work with, instead of
        relying on prompt instructions to detect contradictions.
        """
        if len(results) < 2:
            return results

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for r in results:
            cat = r.get("category", "fact")
            by_category.setdefault(cat, []).append(r)

        # Check for conflicts within each category
        conflict_pairs = set()  # (lower_id, higher_id) to avoid double-marking
        for cat, mems in by_category.items():
            if len(mems) < 2:
                continue
            for i in range(len(mems)):
                for j in range(i + 1, len(mems)):
                    a, b = mems[i], mems[j]
                    # Both high similarity to query means same topic
                    # But they're different memories → potential conflict
                    # Only flag if both have similarity > 0.5 to the query
                    # and their content is meaningfully different
                    if a.get("similarity", 0) < 0.5 or b.get("similarity", 0) < 0.5:
                        continue
                    if a["content"].strip() == b["content"].strip():
                        continue  # Same content, not a conflict

                    # Determine winner
                    a_pri = self._source_priority(a.get("source", "system"))
                    b_pri = self._source_priority(b.get("source", "system"))

                    if a_pri > b_pri:
                        winner, loser = a, b
                    elif b_pri > a_pri:
                        winner, loser = b, a
                    else:
                        # Same priority → newer wins
                        a_time = a.get("created_at")
                        b_time = b.get("created_at")
                        if a_time and b_time and a_time > b_time:
                            winner, loser = a, b
                        else:
                            winner, loser = b, a

                    # Flag
                    winner["_conflict_status"] = "authoritative"
                    loser["_conflict_status"] = "conflicted"
                    loser["_conflicts_with"] = winner["id"]
                    conflict_pairs.add((loser["id"], winner["id"]))

        if conflict_pairs:
            logger.info(f"Detected {len(conflict_pairs)} memory conflict(s) during recall")

        return results

    async def find_similar(self, content: str, threshold: float = 0.85) -> Optional[dict]:
        """Check if a similar memory already exists (for dedup)."""
        results = await self.recall(content, limit=1, min_similarity=threshold)
        return results[0] if results else None

    # ================================================================
    # SOURCE PRIORITY (for conflict resolution)
    # Higher number = higher authority. Used when deciding whether
    # a new memory should overwrite an existing one.
    # ================================================================
    SOURCE_PRIORITY = {
        "user_confirmed": 3,  # User explicitly stated or confirmed
        "observed": 2,        # Inferred from user behavior/messages
        "system": 1,          # System-generated (auto_capture, etc.)
        "auto_captured": 1,   # Alias for system
    }

    def _source_priority(self, source: str) -> int:
        """Get numeric priority for a memory source. Higher = more authoritative."""
        return self.SOURCE_PRIORITY.get(source, 0)

    async def store_if_new(
        self,
        content: str,
        category: str = "fact",
        source: str = "user_confirmed",
        user_id: Optional[int] = None,
        importance: float = 0.5,
        similarity_threshold: float = None,
        conflict_threshold: float = None,
        permanent: bool = False,
    ) -> Optional[int]:
        """Store a memory with conflict resolution.

        Three zones based on similarity to existing memories:
        - >= similarity_threshold (0.85): Exact duplicate → SKIP
        - >= conflict_threshold (0.70):   Same topic, different info → RESOLVE
        - < conflict_threshold:           New topic → INSERT

        Conflict resolution rules (enforced by code, not prompt):
        1. user_confirmed ALWAYS wins over auto_captured/system/observed
        2. Same source priority → newer wins (time-based data changes)
        3. Higher importance wins as tiebreaker

        Returns memory ID (new or updated) or None if duplicate/skipped.
        """
        from ..db.models import get_config

        # Read thresholds from config (allow per-call override)
        if similarity_threshold is None:
            similarity_threshold = float(await get_config("memory.similarity_threshold", "0.85"))
        if conflict_threshold is None:
            conflict_threshold = float(await get_config("memory.conflict_threshold", "0.70"))

        # Embed ONCE — reuse vector for similarity search + store/update
        embedding_resp = await self.provider.embed(content)
        vector = embedding_resp.vector
        if not vector:
            raise RuntimeError(
                "Refusing to store memory: embedding provider returned empty vector."
            )

        # Find most similar existing memory using pre-computed vector
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT id, content, source, importance,
                       COALESCE(permanent, false) as permanent,
                       1 - (embedding <=> $1::vector) as similarity
                FROM memory
                WHERE 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT 1
            """, str(vector), conflict_threshold)

        if rows:
            existing = dict(rows[0])
            sim = existing["similarity"]

            if sim >= similarity_threshold:
                logger.debug(f"Duplicate (sim={sim:.3f}), skipping: {content[:80]}")
                return None

            # ═══════════════════════════════════════════════════════════
            # CONFLICT ZONE (0.70–0.85): same topic, different info
            # Resolve by source priority, then recency, then importance.
            # This is CODE-ENFORCED — model-agnostic, deterministic.
            # ═══════════════════════════════════════════════════════════
            old_id = existing["id"]
            old_content = existing["content"]
            old_source = existing.get("source", "system")
            old_importance = existing.get("importance", 0.5)

            old_permanent = existing.get("permanent", False)

            # Decay v2 — Rule 0: permanent ALWAYS wins. A non-permanent
            # (auto-captured / observed) memory must never silently
            # overwrite a permanent one the user explicitly locked in.
            # Skip the overwrite and log so the LLM can flag it to the user.
            if old_permanent and not permanent:
                logger.info(
                    f"Conflict: new non-permanent memory contradicts PERMANENT "
                    f"#{old_id}. Keeping permanent, skipping new. "
                    f"(new='{content[:50]}')"
                )
                return None

            new_priority = self._source_priority(source)
            old_priority = self._source_priority(old_source)

            # Rule 1: Higher source priority wins
            if new_priority > old_priority:
                logger.info(
                    f"Conflict resolved by source priority: "
                    f"new '{source}'({new_priority}) > old '{old_source}'({old_priority}). "
                    f"Updating #{old_id}."
                )
                return await self._update_memory(
                    old_id, content, category, source, importance, vector=vector,
                )

            if new_priority < old_priority:
                logger.info(
                    f"Conflict resolved by source priority: "
                    f"old '{old_source}'({old_priority}) > new '{source}'({new_priority}). "
                    f"Keeping #{old_id}, skipping new."
                )
                return None

            # Rule 2: Same priority → newer wins (data changes over time)
            logger.info(
                f"Conflict resolved by recency (same source priority): "
                f"old='{old_content[:50]}' → new='{content[:50]}'. "
                f"Updating #{old_id}."
            )
            return await self._update_memory(
                old_id, content, category, source, importance, vector=vector,
            )

        # No similar memory at all — insert new (reuse vector)
        return await self._store_with_vector(
            content, vector, category, source, user_id, importance, permanent
        )

    async def _store_with_vector(
        self,
        content: str,
        vector: list,
        category: str = "fact",
        source: str = "user_confirmed",
        user_id: Optional[int] = None,
        importance: float = 0.5,
        permanent: bool = False,
    ) -> int:
        """Store a memory with a pre-computed embedding vector."""
        from ..db.models import get_config
        if not vector:
            raise RuntimeError("Refusing to store memory with empty vector.")
        initial_count = int(await get_config("memory.initial_recall_count", "5")) if not permanent else 0

        async with get_connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO memory (content, category, embedding, source, user_id, importance, permanent, recall_count)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8)
                RETURNING id
            """, content, category, str(vector), source, user_id, importance, permanent, initial_count)

            # Decay v2: store-event tick — a brand-new non-permanent memory
            # nudges all OTHER non-permanent memories down by 1 (permanent
            # immune). New row keeps its initial recall_count. Life/death is
            # decided by the cap in run_decay(), not by hitting 0.
            if not permanent:
                await conn.execute("""
                    UPDATE memory
                    SET recall_count = recall_count - 1
                    WHERE COALESCE(permanent, false) = false
                      AND id <> $1
                """, row["id"])

            return row["id"]

    async def _update_memory(
        self,
        memory_id: int,
        content: str,
        category: str,
        source: str,
        importance: float,
        vector: Optional[list] = None,
    ) -> int:
        """Update an existing memory with new content and embedding.

        If vector is provided, reuses it instead of re-embedding.
        """
        if vector is None:
            embedding_resp = await self.provider.embed(content)
            vector = embedding_resp.vector

        async with get_connection() as conn:
            await conn.execute("""
                UPDATE memory
                SET content = $1, embedding = $2::vector, category = $3,
                    source = $4, importance = $5, updated_at = NOW()
                WHERE id = $6
            """, content, str(vector), category, source, importance, memory_id)

        return memory_id

    async def delete(self, memory_id: int):
        """Delete a memory by ID."""
        async with get_connection() as conn:
            await conn.execute("DELETE FROM memory WHERE id = $1", memory_id)

    # ─────────────────────────────────────────────────────────────────────
    # Binary attachments — memory_blobs table
    # ─────────────────────────────────────────────────────────────────────

    # 50 MB limit per blob (matches Telegram bot API cap).
    MAX_BLOB_BYTES = 50 * 1024 * 1024

    async def attach_file(
        self,
        memory_id: int,
        content: bytes,
        mime_type: str = "application/octet-stream",
        filename: str = "",
    ) -> None:
        """Attach a binary file to an existing memory.

        Raises ValueError if file too large or memory_id doesn't exist.
        Overwrites any existing blob for the same memory_id.
        """
        if not isinstance(content, (bytes, bytearray)):
            raise ValueError("content must be bytes")
        size = len(content)
        if size == 0:
            raise ValueError("content is empty")
        if size > self.MAX_BLOB_BYTES:
            mb = size / 1024 / 1024
            limit_mb = self.MAX_BLOB_BYTES / 1024 / 1024
            raise ValueError(
                f"File too large ({mb:.1f} MB). Maximum is {limit_mb:.0f} MB per attachment."
            )

        async with get_connection() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM memory WHERE id = $1", memory_id,
            )
            if not exists:
                raise ValueError(f"Memory #{memory_id} not found")
            await conn.execute(
                """
                INSERT INTO memory_blobs (memory_id, mime_type, filename, size_bytes, content)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (memory_id) DO UPDATE
                SET mime_type = EXCLUDED.mime_type,
                    filename = EXCLUDED.filename,
                    size_bytes = EXCLUDED.size_bytes,
                    content = EXCLUDED.content,
                    created_at = NOW()
                """,
                memory_id, mime_type, filename, size, bytes(content),
            )

    async def get_file(self, memory_id: int) -> Optional[dict]:
        """Retrieve attachment for a memory.

        Returns dict {filename, mime_type, size_bytes, content} or None if no blob.
        """
        async with get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT filename, mime_type, size_bytes, content
                FROM memory_blobs
                WHERE memory_id = $1
                """,
                memory_id,
            )
        if not row:
            return None
        return {
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "size_bytes": row["size_bytes"],
            "content": bytes(row["content"]),
        }

    async def has_file(self, memory_id: int) -> bool:
        """Quick check if a memory has an attached blob."""
        async with get_connection() as conn:
            return bool(await conn.fetchval(
                "SELECT 1 FROM memory_blobs WHERE memory_id = $1", memory_id,
            ))

    async def count(self) -> int:
        """Get total memory count."""
        async with get_connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM memory")
            return row["count"]

    async def dedup(self, similarity_threshold: float = 0.95, dry_run: bool = False) -> dict:
        """Remove duplicate memories based on embedding similarity.
        
        Compares all pairs of memories and removes newer duplicates.
        Keeps the memory with the higher importance, or the older one if equal.
        
        Args:
            similarity_threshold: Similarity above which memories are considered duplicates
            dry_run: If True, only report duplicates without deleting
            
        Returns:
            dict with keys: duplicates_found, deleted_ids, kept_ids
        """
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT id, content, embedding, importance, created_at
                FROM memory
                WHERE embedding IS NOT NULL
                ORDER BY id
            """)

        if len(rows) < 2:
            return {"duplicates_found": 0, "deleted_ids": [], "kept_ids": []}

        # Compare each pair using cosine similarity via DB
        duplicates_found = []
        deleted_ids = set()

        for i in range(len(rows)):
            if rows[i]["id"] in deleted_ids:
                continue
            for j in range(i + 1, len(rows)):
                if rows[j]["id"] in deleted_ids:
                    continue

                # Compute similarity via DB
                async with get_connection() as conn:
                    sim_row = await conn.fetchrow(
                        "SELECT 1 - ($1::vector <=> $2::vector) as similarity",
                        str(rows[i]["embedding"]),
                        str(rows[j]["embedding"]),
                    )
                sim = sim_row["similarity"]

                if sim >= similarity_threshold:
                    # Determine which to keep: higher importance wins, then older wins
                    imp_i = rows[i]["importance"] or 0
                    imp_j = rows[j]["importance"] or 0
                    if imp_i >= imp_j:
                        keep, remove = rows[i], rows[j]
                    else:
                        keep, remove = rows[j], rows[i]

                    duplicates_found.append({
                        "keep_id": keep["id"],
                        "remove_id": remove["id"],
                        "similarity": round(sim, 3),
                        "keep_preview": keep["content"][:60],
                        "remove_preview": remove["content"][:60],
                    })
                    deleted_ids.add(remove["id"])

        if not dry_run and deleted_ids:
            async with get_connection() as conn:
                await conn.execute(
                    "DELETE FROM memory WHERE id = ANY($1::int[])",
                    list(deleted_ids),
                )
            logger.info(f"Dedup: removed {len(deleted_ids)} duplicate memories")

        return {
            "duplicates_found": len(duplicates_found),
            "deleted_ids": list(deleted_ids),
            "kept_ids": [d["keep_id"] for d in duplicates_found],
            "details": duplicates_found,
        }

    async def run_decay(self):
        """Run memory decay based on conversation count.
        
        Called periodically (e.g. after every conversation).
        Increments global conversation counter. Every N conversations,
        decays non-permanent memories by reducing recall_count.
        Memories with recall_count <= 0 are deleted.
        """
        from ..db.models import get_config, set_config
        
        # Increment global conversation counter
        current = int(await get_config("memory.conversation_counter", "0"))
        current += 1
        await set_config("memory.conversation_counter", str(current))
        
        # Check if it's time to decay
        interval = int(await get_config("memory.decay_interval", "50"))
        if current % interval != 0:
            return  # Not time yet
        
        cap = int(await get_config("memory.max_records", "50"))
        promo = int(await get_config("memory.promotion_threshold", "1000"))

        async with get_connection() as conn:
            # Decay v2 — Fase 2 (dormant): auto-promote a non-permanent
            # memory to permanent once its recall_count crosses the (high)
            # promotion threshold. Threshold defaults very high so this is
            # effectively inert until tuned against real recall data.
            await conn.execute("""
                UPDATE memory
                SET permanent = true, updated_at = NOW()
                WHERE COALESCE(permanent, false) = false
                  AND recall_count >= $1
            """, promo)

            # Decay v2 — cap-based eviction (LRU/LFU hybrid). Life/death is
            # decided HERE, not by recall_count<=0. When non-permanent count
            # exceeds the cap, evict the lowest-recall (LFU), oldest-id (LRU)
            # rows until back at cap. Permanent memories are never touched.
            n = await conn.fetchval("""
                SELECT COUNT(*) FROM memory WHERE COALESCE(permanent, false) = false
            """)
            over = int(n) - cap
            if over > 0:
                deleted = await conn.fetch("""
                    DELETE FROM memory
                    WHERE id IN (
                        SELECT id FROM memory
                        WHERE COALESCE(permanent, false) = false
                        ORDER BY recall_count ASC, id ASC
                        LIMIT $1
                    )
                    RETURNING id, content
                """, over)
                if deleted:
                    logger.info(f"Memory decay v2: evicted {len(deleted)} over-cap memories (cap={cap})")
                    for row in deleted:
                        logger.debug(f"  Evicted: [{row['id']}] {row['content'][:60]}...")
