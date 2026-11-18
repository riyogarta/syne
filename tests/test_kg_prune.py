"""KG prune on delete (Riyo, 18 Nov 2026) - v1.23.04.

The PostgreSQL tests run when SYNE_TEST_DATABASE_URL points at a throwaway
database; each test works in its own schema and drops it.
"""
import os
import re
import uuid

import pytest

import syne.db.migrations as M
from syne.cli.helpers import _split_sql_statements
from syne.db import kg_prune_ddl as P
from syne.memory import kg_extract as K

_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "syne", "db", "schema.sql",
)


def test_schema_sql_carries_ddl():
    with open(_SCHEMA) as f:
        assert P.KG_PRUNE_SCHEMA_SQL in f.read()


def test_kept_types_are_reference_types():
    m = re.search(r"ARRAY\[([^\]]*)\]::text\[\]", P.KG_PRUNE_SCHEMA_SQL)
    assert re.findall(r"'([^']+)'", m.group(1)) == K.REFERENCE_TYPES


def test_ddl_splits_into_statements():
    stmts = [s.strip() for s in _split_sql_statements(P.KG_PRUNE_SCHEMA_SQL)]
    assert len(stmts) == 4
    assert stmts[0].startswith("CREATE INDEX IF NOT EXISTS idx_kg_entities_chunk_description")
    assert stmts[1].startswith("CREATE OR REPLACE FUNCTION kg_prune_sources")
    assert stmts[2].startswith("CREATE OR REPLACE FUNCTION kg_prune_on_delete")
    assert all(s.endswith("$$ LANGUAGE plpgsql;") for s in stmts[1:3])
    assert stmts[3].startswith("DO $$") and stmts[3].endswith("END $$;")


def test_migration_56_registered():
    assert (56, M._m56_kg_prune_on_delete, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 56


async def test_migration_56_runs_ddl_and_prunes_orphans():
    calls = []

    class C:
        async def execute(self, sql, *a):
            calls.append((sql, a))

        async def fetchval(self, sql, *a):
            return [7, 8] if a == ("memory",) else None
    await M._m56_kg_prune_on_delete(C())
    sqls = [c[0] for c in calls]
    assert any("trg_memory_kg_prune" in s and "trg_messages_kg_prune" in s for s in sqls)
    prunes = [c for c in calls if c[0].startswith("SELECT kg_prune_sources")]
    assert prunes == [("SELECT kg_prune_sources($1, $2::int[])", ("memory", [7, 8]))]


# ── real PostgreSQL ───────────────────────────────────────────────────────

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

_BASE = """
CREATE TABLE memory (id INT PRIMARY KEY, content TEXT);
CREATE TABLE messages (id INT PRIMARY KEY, content TEXT);
CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, name TEXT NOT NULL,
    entity_type VARCHAR(50), description TEXT);
CREATE TABLE kg_relations (
    id SERIAL PRIMARY KEY,
    subject_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100) NOT NULL,
    object_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    source_memory_id INT REFERENCES memory(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(subject_id, predicate, object_id));
CREATE TABLE kg_relation_sources (
    relation_id INT NOT NULL REFERENCES kg_relations(id) ON DELETE CASCADE,
    source_type VARCHAR(20) NOT NULL, source_id INT NOT NULL,
    origin VARCHAR(20) NOT NULL DEFAULT 'llm',
    PRIMARY KEY (relation_id, source_type, source_id));
CREATE TABLE kg_extract_log (source_type VARCHAR(20) NOT NULL, source_id INT NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'done', PRIMARY KEY (source_type, source_id));
"""


async def _db():
    import asyncpg
    conn = await asyncpg.connect(_DB)
    schema = "kgprune_" + uuid.uuid4().hex[:10]
    await conn.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    await conn.execute(_BASE)
    for stmt in _split_sql_statements(P.KG_PRUNE_SCHEMA_SQL):
        await conn.execute(stmt)
    return conn, schema


async def _close(conn, schema):
    await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await conn.close()


async def _ent(c, name, typ, desc=None):
    return await c.fetchval(
        "INSERT INTO kg_entities (name, entity_type, description) VALUES ($1, $2, $3) RETURNING id",
        name, typ, desc)


async def _rel(c, s, p, o, sources, mem=None):
    rid = await c.fetchval(
        "INSERT INTO kg_relations (subject_id, predicate, object_id, source_memory_id) "
        "VALUES ($1, $2, $3, $4) RETURNING id", s, p, o, mem)
    for t, i in sources:
        await c.execute("INSERT INTO kg_relation_sources (relation_id, source_type, source_id) "
                        "VALUES ($1, $2, $3)", rid, t, i)
    return rid


async def _names(c):
    return {r["name"] for r in await c.fetch("SELECT name FROM kg_entities")}


async def _count(c, table):
    return await c.fetchval(f"SELECT count(*) FROM {table}")


@pg
async def test_delete_memory_prunes_its_graph():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) VALUES (1), (2)")
        riyo = await _ent(c, "Riyo", "person")
        rumah = await _ent(c, "Rumah", "item")
        bintaro = await _ent(c, "Bintaro", "place")
        ch1 = await _ent(c, "c1", "chunk", "memory:1")
        ch2 = await _ent(c, "c2", "chunk", "memory:2")
        ayah = await _ent(c, "QS 1:1", "ayah")
        ra = await _rel(c, riyo, "decided", rumah, [("memory", 1)], mem=1)
        rb = await _rel(c, riyo, "mentions", bintaro, [("memory", 1), ("memory", 2)], mem=1)
        await _rel(c, ch1, "explains", ayah, [("memory", 1)], mem=1)
        await _rel(c, ch2, "explains", ayah, [("memory", 2)], mem=2)
        await c.execute("INSERT INTO kg_extract_log (source_type, source_id) "
                        "VALUES ('memory', 1), ('memory', 2)")

        await c.execute("DELETE FROM memory WHERE id = 1")
        assert await _names(c) == {"Riyo", "Bintaro", "c2", "QS 1:1"}
        assert await c.fetchval("SELECT count(*) FROM kg_relations WHERE id = $1", ra) == 0
        # the relation kept by memory 2 points at memory 2 again
        assert await c.fetchval("SELECT source_memory_id FROM kg_relations WHERE id = $1", rb) == 2
        assert await c.fetchval("SELECT array_agg(source_id) FROM kg_extract_log") == [2]

        await c.execute("DELETE FROM memory WHERE id = 2")
        # Riyo is left with no relation and goes; the ayah node is a reference type
        assert await _names(c) == {"QS 1:1"}
        for t in ("kg_relations", "kg_relation_sources", "kg_extract_log"):
            assert await _count(c, t) == 0, t
    finally:
        await _close(c, sch)


@pg
async def test_delete_messages_and_mixed_sources():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) VALUES (5)")
        await c.execute("INSERT INTO messages (id) VALUES (10), (11)")
        riyo = await _ent(c, "Riyo", "person")
        kopi = await _ent(c, "Kopi", "concept")
        yba = await _ent(c, "YBA", "org")
        await _rel(c, riyo, "prefers", kopi, [("message", 10)])
        rf = await _rel(c, riyo, "member_of", yba, [("memory", 5), ("message", 11)], mem=5)
        await c.execute("INSERT INTO kg_extract_log (source_type, source_id) VALUES ('message', 10)")

        await c.execute("DELETE FROM messages WHERE id = 10")
        assert await _names(c) == {"Riyo", "YBA"}
        assert await _count(c, "kg_extract_log") == 0

        await c.execute("DELETE FROM messages WHERE id = 11")
        assert await c.fetchval("SELECT source_memory_id FROM kg_relations WHERE id = $1", rf) == 5

        await c.execute("DELETE FROM memory WHERE id = 5")
        assert await _names(c) == set() and await _count(c, "kg_relations") == 0
    finally:
        await _close(c, sch)


@pg
async def test_bulk_delete_one_statement():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) SELECT generate_series(1, 4)")
        org = await _ent(c, "Org", "org")
        for i in range(1, 5):
            p = await _ent(c, f"P{i}", "person")
            await _rel(c, p, "member_of", org, [("memory", i)], mem=i)
        await c.execute("DELETE FROM memory WHERE id = ANY($1::int[])", [1, 2, 3])
        assert await _names(c) == {"P4", "Org"}
        assert await _count(c, "kg_relation_sources") == 1
    finally:
        await _close(c, sch)


@pg
async def test_prune_failure_never_blocks_the_delete():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) VALUES (1)")
        a = await _ent(c, "A", "person")
        b = await _ent(c, "B", "item")
        await _rel(c, a, "created", b, [("memory", 1)], mem=1)
        await c.execute("DROP TABLE kg_extract_log")
        await c.execute("DELETE FROM memory WHERE id = 1")
        assert await _count(c, "memory") == 0
        # the prune rolled back as a whole, nothing half-done
        assert await _count(c, "kg_relation_sources") == 1
    finally:
        await _close(c, sch)


@pg
async def test_prune_orphan_sources_once():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) VALUES (3)")
        x = await _ent(c, "X", "person")
        y = await _ent(c, "Y", "item")
        z = await _ent(c, "Z", "item")
        await _ent(c, "c98", "chunk", "memory:98")
        await _rel(c, x, "created", y, [("memory", 99)])
        await _rel(c, x, "uses", z, [("memory", 3)], mem=3)
        await c.execute("INSERT INTO kg_extract_log (source_type, source_id) "
                        "VALUES ('message', 77), ('memory', 3)")
        assert await P.prune_orphan_sources(c) == {"memory": 2, "message": 1}
        assert await _names(c) == {"X", "Z"}
        assert await c.fetchval("SELECT array_agg(source_id) FROM kg_extract_log") == [3]
        assert await P.prune_orphan_sources(c) == {"memory": 0, "message": 0}
    finally:
        await _close(c, sch)
