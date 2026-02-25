"""Memory engine — store and recall with semantic search."""

import logging
from typing import Optional
from ..db.connection import get_connection
from ..llm.provider import LLMProvider
from ..security import check_rule_760

logger = logging.getLogger("syne.memory.engine")


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
        """
        from ..db.models import get_config
        
        # Generate embedding
        embedding_resp = await self.provider.embed(content)
        vector = embedding_resp.vector
        
        initial_count = int(await get_config("memory.initial_recall_count", "1")) if not permanent else 0

        async with get_connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO memory (content, category, embedding, source, user_id, importance, permanent, recall_count)
                VALUES ($1, $2, $3::vector, $4, $5, $6, $7, $8)
                RETURNING id
            """, content, category, str(vector), source, user_id, importance, permanent, initial_count)

            return row["id"]

    async def recall(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        category: Optional[str] = None,
        user_id: Optional[int] = None,
        requester_access_level: str = "public",
    ) -> list[dict]:
        """Recall memories by semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold
            category: Filter by category
            user_id: Filter by user
            requester_access_level: Access level of the requester (for Rule 760 filtering)
            
        Returns:
            List of matching memories (filtered by Rule 760 for privacy)
        """
        # Generate query embedding
        embedding_resp = await self.provider.embed(query)
        vector = embedding_resp.vector

        async with get_connection() as conn:
            # Build query with optional filters
            conditions = ["1 = 1"]
            params = [str(vector), limit]
            param_idx = 3

            if category:
                conditions.append(f"category = ${param_idx}")
                params.append(category)
                param_idx += 1

            if user_id is not None:
                conditions.append(f"(user_id = ${param_idx} OR user_id IS NULL)")
                params.append(user_id)
                param_idx += 1

            where = " AND ".join(conditions)

            rows = await conn.fetch(f"""
                SELECT
                    id, content, category, source, importance,
                    access_count, created_at,
                    COALESCE(permanent, false) as permanent,
                    COALESCE(recall_count, 1) as recall_count,
                    1 - (embedding <=> $1::vector) as similarity
                FROM memory
                WHERE {where}
                  AND (COALESCE(permanent, false) = true OR COALESCE(recall_count, 1) > 0)
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, *params)

            # Filter by minimum similarity and update access stats
            results = []
            ids_to_update = []
            for row in rows:
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
                        recall_count = CASE WHEN permanent = true THEN recall_count ELSE recall_count + 2 END
                    WHERE id = ANY($1)
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
        similarity_threshold: float = 0.85,
        conflict_threshold: float = 0.70,
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
        # Find the most similar existing memory
        results = await self.recall(content, limit=1, min_similarity=conflict_threshold)

        if results:
            existing = results[0]
            sim = existing["similarity"]

            if sim >= similarity_threshold:
                # Exact duplicate — skip
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
                    old_id, content, category, source, importance
                )

            if new_priority < old_priority:
                # Existing memory has higher authority — don't overwrite
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
                old_id, content, category, source, importance
            )

        # No similar memory at all — insert new
        return await self.store(content, category, source, user_id, importance, permanent=permanent)

    async def _update_memory(
        self,
        memory_id: int,
        content: str,
        category: str,
        source: str,
        importance: float,
    ) -> int:
        """Update an existing memory with new content and embedding."""
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

    async def count(self) -> int:
        """Get total memory count."""
        async with get_connection() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM memory")
            return row["count"]

    async def dedup(self, similarity_threshold: float = 0.85, dry_run: bool = False) -> dict:
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
        
        decay_amount = int(await get_config("memory.decay_amount", "1"))
        
        async with get_connection() as conn:
            # Decay non-permanent memories
            await conn.execute("""
                UPDATE memory
                SET recall_count = recall_count - $1,
                    updated_at = NOW()
                WHERE permanent = false
                  AND recall_count > 0
            """, decay_amount)
            
            # Delete memories with recall_count <= 0
            deleted = await conn.fetch("""
                DELETE FROM memory
                WHERE permanent = false
                  AND recall_count <= 0
                RETURNING id, content
            """)
            
            if deleted:
                logger.info(f"Memory decay: deleted {len(deleted)} forgotten memories")
                for row in deleted:
                    logger.debug(f"  Forgotten: [{row['id']}] {row['content'][:60]}...")
