"""Memory deletion functionality."""

import logging
from typing import List
from ..engine.memory import MemoryEngine

logger = logging.getLogger("syne.tools.memory_delete")

async def memory_delete(memory_ids: str) -> str:
    """
    Delete memory entries by comma-separated IDs.
    
    Args:
        memory_ids: Comma-separated memory IDs (e.g., "35,57")
        
    Returns:
        Status message
    """
    try:
        # Parse IDs
        ids = []
        for id_str in memory_ids.split(","):
            id_str = id_str.strip()
            if id_str:
                ids.append(int(id_str))
        
        if not ids:
            return "Error: No valid memory IDs provided"
        
        memory_engine = MemoryEngine()
        deleted_count = 0
        errors = []
        
        for mem_id in ids:
            try:
                # Check if memory exists first
                existing = await memory_engine.search(f"id:{mem_id}", limit=1)
                if not existing:
                    errors.append(f"Memory ID {mem_id} not found")
                    continue
                
                # Delete the memory
                await memory_engine.delete(mem_id)
                deleted_count += 1
                logger.info(f"Deleted memory ID {mem_id}")
                
            except Exception as e:
                errors.append(f"Error deleting memory ID {mem_id}: {str(e)}")
        
        # Build result message
        result_parts = []
        if deleted_count > 0:
            result_parts.append(f"Successfully deleted {deleted_count} memory entries")
        
        if errors:
            result_parts.append(f"Errors: {'; '.join(errors)}")
        
        return ". ".join(result_parts) if result_parts else "No action taken"
        
    except ValueError as e:
        return f"Error: Invalid memory ID format - {str(e)}"
    except Exception as e:
        logger.error(f"Error in memory_delete: {e}")
        return f"Error: {str(e)}"