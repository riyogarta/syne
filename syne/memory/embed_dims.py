"""Embedding dimension check (Riyo, 18 Nov 2026) - v1.23.07.

Switching the embedding model never deletes memories. Vectors of different
dimensions cannot be mixed (memory.embedding is a fixed vector(N) column),
so a model whose dimension differs from the stored vectors is refused.
"""
import re
from typing import Optional

COLUMN_TYPE_SQL = (
    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
    "WHERE attrelid = 'memory'::regclass AND attname = 'embedding' AND NOT attisdropped"
)


def parse_vector_dims(type_name) -> Optional[int]:
    """'vector(1024)' -> 1024; 'vector' or None -> None."""
    m = re.fullmatch(r"vector\((\d+)\)", (type_name or "").strip())
    return int(m.group(1)) if m else None


async def stored_embedding_dims(conn) -> tuple[Optional[int], int]:
    """(dimension of stored memory vectors, number of memories with a vector).

    The dimension is the memory.embedding column type; when the column has
    no fixed dimension it is read from a stored vector. None = no constraint.
    """
    count = int(await conn.fetchval(
        "SELECT count(*) FROM memory WHERE embedding IS NOT NULL") or 0)
    dims = parse_vector_dims(await conn.fetchval(COLUMN_TYPE_SQL))
    if dims is None and count:
        dims = await conn.fetchval(
            "SELECT vector_dims(embedding) FROM memory WHERE embedding IS NOT NULL LIMIT 1")
    return dims, count


def refuse_switch_message(stored_dims, new_dims, mem_count: int) -> Optional[str]:
    """HTML refusal text when new_dims cannot be used, else None."""
    if stored_dims is None or not new_dims:
        return None
    if int(new_dims) == int(stored_dims):
        return None
    return (
        "⛔ <b>Embedding switch refused</b>\n\n"
        f"This model produces {int(new_dims)}-dimension vectors; "
        f"stored memories use {int(stored_dims)}.\n"
        f"Memories stored: <b>{mem_count}</b>\n\n"
        "Vectors of different dimensions cannot be mixed, and Syne never deletes "
        "memories to switch models. Nothing was changed. "
        f"Choose a model with {int(stored_dims)} dimensions."
    )
