"""Embedding dimension check (Riyo, 18 Nov 2026) - v1.23.07.

Switching the embedding model never deletes memories. Vectors of different
dimensions cannot be mixed (memory.embedding is a fixed vector(N) column),
so a model whose dimension differs from the stored vectors is refused.
"""
import html
import json
import logging
import re
from typing import Optional

logger = logging.getLogger("syne.memory.embed_dims")

# pgvector builds an HNSW index on a `vector` column of at most 2000
# dimensions (v1.23.34, review A).
HNSW_MAX_DIMS = 2000

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


# ── v1.23.14: every embedding column, empty columns may switch ────────────

# kg_entities is not checked: its column is untyped and unindexed, so
# vectors of several dimensions can sit side by side, and the KG worker
# compares a name only with vectors of its own dimension.
TABLES = ("memory", "messages")
LABELS = {"memory": "memories", "messages": "chat history (history_search)"}
HNSW_INDEX = {"memory": "idx_memory_embedding_hnsw",
              "messages": "idx_messages_embedding_hnsw"}
_COL_SQL = ("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped")


async def column_state(conn, table: str) -> tuple:
    """(column dimension or None, stored vector dimension or None, vectors stored)."""
    assert table in TABLES
    col = parse_vector_dims(await conn.fetchval(_COL_SQL, table))
    n = int(await conn.fetchval(
        f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL") or 0)
    vec = None
    if n:
        vec = col or await conn.fetchval(
            f"SELECT vector_dims(embedding) FROM {table} WHERE embedding IS NOT NULL LIMIT 1")
    return col, vec, n


async def plan_switch(conn, new_dims) -> dict:
    """What a switch to new_dims needs. Read-only.

    blocked: [(table, stored dims, vectors)] - vectors of another dimension.
    relax:   [table] - holding no vector, and not already vector(new_dims):
             relax_columns() types it vector(new_dims). Before v1.23.14 a
             vector(N) column refused the switch even with no memory at
             all; before v1.23.22 an untyped `vector` column (fresh
             install) was skipped, so a late write of the old provider
             could still mix two dimensions in it.
    """
    out = {"blocked": [], "relax": []}
    if not new_dims:
        return out
    nd = int(new_dims)
    for t in TABLES:
        col, vec, n = await column_state(conn, t)
        if n and vec and int(vec) != nd:
            out["blocked"].append((t, int(vec), n))
        elif not n and col != nd:
            out["relax"].append(t)
    return out


def refusal_for_plan(plan: dict, new_dims) -> Optional[str]:
    """HTML refusal when stored vectors block new_dims, else None."""
    if not plan["blocked"]:
        return None
    mem = next((b for b in plan["blocked"] if b[0] == "memory"), None)
    if mem:
        return refuse_switch_message(mem[1], new_dims, mem[2])
    lines = "\n".join(f"• {LABELS[t]}: <b>{n}</b> vectors of {d} dimensions"
                      for t, d, n in plan["blocked"])
    out = (
        "⛔ <b>Embedding switch refused</b>\n\n"
        f"This model produces {int(new_dims)}-dimension vectors. Stored vectors of "
        f"another dimension:\n{lines}\n\n"
        "Vectors of different dimensions cannot be mixed. Nothing was changed."
    )
    if history_only(plan):
        out += (
            "\n\nOnly chat history blocks the switch. <b>Clear history vectors and "
            "switch</b> keeps every message and empties their vectors: old messages "
            "stay findable by keyword only, new messages are embedded with the "
            "new model."
        )
    return out


def history_only(plan: dict) -> bool:
    """True when chat history alone blocks the switch (v1.23.18)."""
    return bool(plan["blocked"]) and all(b[0] == "messages" for b in plan["blocked"])


async def _rebuild_empty_column(conn, table: str, nd: int) -> None:
    """Drop and re-add an embedding column as vector(nd) (v1.23.37/38): two
    catalog changes, no row rewrite. Other indexes on the column (the
    partial idx_messages_embedding_pending) go with it and are rebuilt;
    then the HNSW index (build_hnsw skips it above HNSW_MAX_DIMS)."""
    keep = await conn.fetch(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
        "AND tablename = $1 AND indexname <> $2 AND indexdef ~ $3",
        table, HNSW_INDEX[table], r"\membedding\M")
    await conn.execute(f"DROP INDEX IF EXISTS {HNSW_INDEX[table]}")
    await conn.execute(f"ALTER TABLE {table} DROP COLUMN embedding")
    await conn.execute(f"ALTER TABLE {table} ADD COLUMN embedding vector({nd})")
    for r in keep:
        await conn.execute(r["indexdef"])
    await build_hnsw(conn, table, nd)


async def clear_history_vectors(conn, new_dims, n: int = None) -> int:
    """Empty every messages.embedding and type the column vector(new_dims)
    with a fresh HNSW index (v1.23.18, review #14). Messages are kept; old
    ones are not embedded again (Riyo's decision #57758), new ones get
    vectors of the new model. Before, a switch was refused as long as any
    history vector existed, with no way out.
    Returns the number of vectors cleared.

    v1.23.38 (review B): n is the count the caller already read under the
    same locks (apply_switch's plan); it is counted here only when not given.
    """
    nd = int(new_dims)
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '10s'")
        await conn.execute("LOCK TABLE messages IN ACCESS EXCLUSIVE MODE")
        if n is None:
            n = int(await conn.fetchval(
                "SELECT count(*) FROM messages WHERE embedding IS NOT NULL") or 0)
        # v1.23.37 (review B): drop and re-add the column, no row rewrite.
        await _rebuild_empty_column(conn, "messages", nd)
    return int(n)


async def build_hnsw(conn, table: str, dims: int) -> bool:
    """Create the HNSW index of an embedding column when pgvector can index
    that dimension. Above HNSW_MAX_DIMS the column stays unindexed (search
    still works, scanning the table). v1.23.34: building it anyway failed
    the whole switch to qwen3-embedding:4b/8b. True when built."""
    assert table in TABLES
    if int(dims) > HNSW_MAX_DIMS:
        logger.warning(f"{table}.embedding: {int(dims)} dimensions, above the HNSW limit "
                       f"of {HNSW_MAX_DIMS}; left without an index")
        return False
    await conn.execute(
        f"CREATE INDEX {HNSW_INDEX[table]} ON {table} USING hnsw "
        "(embedding vector_cosine_ops) WITH (m = 24, ef_construction = 200)")
    return True


def clear_confirm_text(plan: dict, label) -> str:
    """Confirmation asked before chat history vectors are cleared (v1.23.21,
    review v1.23.16-19 bug 6: one tap used to clear them at once)."""
    n = sum(b[2] for b in plan["blocked"] if b[0] == "messages")
    return (
        "🧹 <b>Clear chat history vectors?</b>\n\n"
        f"Switching to <b>{html.escape(str(label))}</b> empties the vectors of "
        f"<b>{n}</b> chat messages. The messages are kept and stay findable by "
        "keyword in history_search; old messages are not embedded again "
        "automatically. This cannot be undone.\n\n"
        "The disk space of the old vectors comes back only after "
        "<code>VACUUM FULL messages</code>, which locks the table while it runs; "
        "Syne never runs it on its own."
    )


class SwitchRefused(Exception):
    """Stored vectors block the switch; .html is the refusal shown."""

    def __init__(self, text: str):
        super().__init__("embedding switch refused")
        self.html = text


_CONFIG_UPSERT = (
    "INSERT INTO config (key, value) VALUES ($1, $2::jsonb) "
    "ON CONFLICT (key) DO UPDATE SET value = $2::jsonb, updated_at = NOW()")


async def apply_switch(conn, new_dims, config: dict, clear_history: bool = False) -> dict:
    """Switch the embedding columns and write the provider config in ONE
    transaction (v1.23.21, review v1.23.16-19 bug 3). Before, the history
    vectors were cleared and the columns retyped in their own transactions
    and the config written after: a failure in between left the vectors
    gone and the columns of the new dimension while the old provider stayed
    active. Now any failure rolls everything back.

    The plan is checked again under the table locks. Chat history vectors
    are cleared only when clear_history is set and history alone blocks the
    switch. Raises SwitchRefused (nothing changed) when vectors still block.
    Returns {"cleared": vectors cleared, "relaxed": tables retyped}.
    """
    nd = int(new_dims) if new_dims else None
    out = {"cleared": 0, "relaxed": []}
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '10s'")
        plan = await plan_switch(conn, nd)
        if plan["blocked"] or plan["relax"]:
            for t in sorted(TABLES):
                await conn.execute(f"LOCK TABLE {t} IN ACCESS EXCLUSIVE MODE")
            plan = await plan_switch(conn, nd)
        if plan["blocked"] and clear_history and history_only(plan):
            n_hist = sum(b[2] for b in plan["blocked"] if b[0] == "messages")
            out["cleared"] = await clear_history_vectors(conn, nd, n_hist)
            plan = await plan_switch(conn, nd)
        refusal = refusal_for_plan(plan, nd)
        if refusal:
            raise SwitchRefused(refusal)
        if plan["relax"]:
            out["relaxed"] = await relax_columns(conn, plan["relax"], nd)
        for key, value in config.items():
            await conn.execute(_CONFIG_UPSERT, key, json.dumps(value))
    return out


# Date the active embedding model took over (v1.23.30, Riyo #57757 point 11,
# #57758): old rows are not re-embedded; a later re-embed, if ever wanted,
# only takes rows created before this date.
ERA_KEY = "embedding.era_start"


async def reembed_hint(conn, plan: dict) -> str:
    """HTML note for the switch screen: the tables that keep vectors of the
    OLD model after the switch. v1.23.30: no re-embed command is suggested
    (v1.23.29 did, against the owner's decision); the switch records its
    date in embedding.era_start instead. A table that is retyped (empty) or
    blocked is not listed."""
    blocked = {b[0] for b in plan["blocked"]}
    parts = []
    for t in TABLES:
        if t in plan["relax"] or t in blocked:
            continue
        n = int(await conn.fetchval(
            f"SELECT count(*) FROM {t} WHERE embedding IS NOT NULL") or 0)
        if n:
            parts.append(f"{n} {LABELS[t]}")
    if not parts:
        return ""
    return ("Kept as they are, with vectors of the old model: " + ", ".join(parts)
            + f". The switch date is recorded in <code>{ERA_KEY}</code>; rows created "
            "before it are the ones to re-embed if that is ever wanted.")


def switch_note(plan: dict, new_dims) -> str:
    if plan["relax"]:
        names = ", ".join(LABELS[t] for t in plan["relax"])
        note = (f"New dimension ({int(new_dims)}): {names} hold no vectors yet and "
                "will be switched to it. Memories are preserved.")
        if int(new_dims) > HNSW_MAX_DIMS:
            note += (f" Above {HNSW_MAX_DIMS} dimensions pgvector builds no HNSW index: "
                     "search works but scans the whole table.")
        return note
    if new_dims:
        return f"Same dimensions ({int(new_dims)}), memories will be preserved."
    return "Memories will be preserved."


async def relax_columns(conn, tables: list, new_dims) -> list:
    """Switch empty embedding columns to vector(new_dims) and rebuild their
    HNSW index (empty, so instant). Refuses a column that holds a vector.

    v1.23.17: the tables are locked BEFORE counting (a write between count
    and ALTER slipped in), and the column gets the NEW dimension instead of
    an untyped `vector`: until reload_provider() swaps the provider, a write
    from the old one is rejected by the column type instead of mixing two
    dimensions (which broke every `<=>` query). The index is back at once.

    v1.23.38 (review B): the empty column is dropped and re-added (catalog
    only; ALTER TYPE USING rewrote the table under the lock) and emptiness
    is checked with EXISTS instead of counting every row. Still checked
    under the lock, so a writer that committed first refuses the switch.
    """
    nd = int(new_dims)
    order = sorted(set(tables))
    for t in order:
        assert t in TABLES
    done = []
    async with conn.transaction():
        await conn.execute("SET LOCAL lock_timeout = '10s'")
        for t in order:
            await conn.execute(f"LOCK TABLE {t} IN ACCESS EXCLUSIVE MODE")
        for t in order:
            if await conn.fetchval(
                    f"SELECT EXISTS (SELECT 1 FROM {t} WHERE embedding IS NOT NULL)"):
                raise RuntimeError(f"{t} holds vectors now; nothing was changed")
            await _rebuild_empty_column(conn, t, nd)
            done.append(t)
    return done
