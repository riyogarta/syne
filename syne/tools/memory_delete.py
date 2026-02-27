"""memory_delete — Delete memory entries by ID.

Allows deletion of specific memory entries by ID.
Use with caution as this is destructive.
"""

import logging
from syne.db.connection import get_pool

logger = logging.getLogger("syne.tools.memory_delete")


async def memory_delete(memory_ids: str) -> str:
    """Delete memory entries by ID.
    
    Args:
        memory_ids: Comma-separated memory IDs to delete (e.g., "35,57")
    
    Returns:
        Result message
    """
    try:
        # Parse memory IDs
        ids = [int(id_str.strip()) for id_str in memory_ids.split(",")]
        
        if not ids:
            return "❌ No memory IDs provided"
        
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Delete memories
            result = await conn.execute(
                "DELETE FROM memory WHERE id = ANY($1)",
                ids
            )
            
            # Parse result to get count
            deleted_count = int(result.split()[-1]) if result.startswith("DELETE") else 0
            
            if deleted_count > 0:
                return f"✅ Deleted {deleted_count} memory entries (IDs: {memory_ids})"
            else:
                return f"❌ No memory entries found with IDs: {memory_ids}"
                
    except ValueError:
        return f"❌ Invalid memory IDs format: {memory_ids}"
    except Exception as e:
        logger.error(f"Error deleting memories: {e}")
        return f"❌ Error deleting memories: {e}"