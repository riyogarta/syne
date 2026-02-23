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
    ) -> int:
        """Store a memory with its embedding vector."""
        # Generate embedding
        embedding_resp = await self.provider.embed(content)
        vector = embedding_resp.vector

        async with get_connection() as conn:
            row = await conn.fetchrow("""
                INSERT INTO memory (content, category, embedding, source, user_id, importance)
                VALUES ($1, $2, $3::vector, $4, $5, $6)
                RETURNING id
            """, content, category, str(vector), source, user_id, importance)

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
                    1 - (embedding <=> $1::vector) as similarity
                FROM memory
                WHERE {where}
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

            # Update access stats
            if ids_to_update:
                await conn.execute("""
                    UPDATE memory
                    SET access_count = access_count + 1, accessed_at = NOW()
                    WHERE id = ANY($1)
                """, ids_to_update)

            return results

    async def find_similar(self, content: str, threshold: float = 0.85) -> Optional[dict]:
        """Check if a similar memory already exists (for dedup)."""
        results = await self.recall(content, limit=1, min_similarity=threshold)
        return results[0] if results else None

    async def store_if_new(
        self,
        content: str,
        category: str = "fact",
        source: str = "user_confirmed",
        user_id: Optional[int] = None,
        importance: float = 0.5,
        similarity_threshold: float = 0.85,
        conflict_threshold: float = 0.70,
    ) -> Optional[int]:
        """Store a memory with conflict resolution.

        Three zones based on similarity to existing memories:
        - >= similarity_threshold (0.85): Exact duplicate → SKIP
        - >= conflict_threshold (0.70):   Same topic, possibly updated info → UPDATE old
        - < conflict_threshold:           New topic → INSERT
        
        Returns memory ID (new or updated) or None if duplicate.
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

            # Conflict zone (0.70–0.85): same topic but different info → update
            old_id = existing["id"]
            old_content = existing["content"]
            logger.info(
                f"Conflict detected (sim={sim:.3f}): "
                f"old='{old_content[:60]}' → new='{content[:60]}'. "
                f"Updating memory #{old_id}."
            )

            # Generate new embedding for the updated content
            embedding_resp = await self.provider.embed(content)
            vector = embedding_resp.vector

            async with get_connection() as conn:
                await conn.execute("""
                    UPDATE memory
                    SET content = $1, embedding = $2::vector, category = $3,
                        source = $4, importance = $5, updated_at = NOW()
                    WHERE id = $6
                """, content, str(vector), category, source, importance, old_id)

            return old_id

        # No similar memory at all — insert new
        return await self.store(content, category, source, user_id, importance)

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
