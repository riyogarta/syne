"""Memory guard + embedding switch (Riyo, 18 Nov 2026) - v1.23.07.

The PostgreSQL tests run when SYNE_TEST_DATABASE_URL points at a throwaway
database; each test works in its own schema and drops it.
"""
import os
import uuid
from contextlib import asynccontextmanager

import pytest

import syne.db.migrations as M
from syne.cli.helpers import _split_sql_statements
from syne.db import kg_prune_ddl as P
from syne.db import memory_guard_ddl as G
from syne.memory import embed_dims as ED

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    with open(os.path.join(_ROOT, *p)) as f:
        return f.read()


def test_schema_sql_carries_ddl():
    assert G.MEMORY_GUARD_SCHEMA_SQL in _read("syne", "db", "schema.sql")


def test_ddl_splits_into_statements():
    stmts = [s.strip() for s in _split_sql_statements(G.MEMORY_GUARD_SCHEMA_SQL)]
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE OR REPLACE FUNCTION memory_guard_on_delete")
    assert stmts[1].startswith("DO $$") and stmts[1].endswith("END $$;")


def test_guard_fires_before_kg_prune():
    # same timing/event: PostgreSQL fires triggers in name order
    assert G.GUARD_TRIGGER < "trg_memory_kg_prune"


def test_migration_57_registered():
    assert (57, M._m57_memory_guard, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 57


async def test_migration_57_runs_ddl():
    sqls = []

    class C:
        async def execute(self, sql, *a):
            sqls.append(sql)
    await M._m57_memory_guard(C())
    assert len(sqls) == 2 and "trg_memory_guard" in sqls[1]


class _GuardErr(Exception):
    sqlstate = "23001"
    hint = "Delete the chunks that reference them first."


def test_guard_refusal():
    e = _GuardErr("memory guard: 1 memories are referenced ...: #5 is referenced by 1 other memories (e.g. #6)")
    assert G.guard_refusal(e).endswith("Delete the chunks that reference them first.")
    assert G.guard_refusal(ValueError("x")) is None

    class FK(Exception):
        sqlstate = "23001"
    assert G.guard_refusal(FK("other restrict")) is None


# ── memory_delete tool ───────────────────────────────────────────────────

def _patch_conn(monkeypatch, conn):
    import syne.db.connection as DC

    @asynccontextmanager
    async def gc(timeout=None):
        yield conn
    monkeypatch.setattr(DC, "get_connection", gc)


async def test_tool_delete_one_statement(monkeypatch):
    from syne.agent import SyneAgent
    seen = []

    class Conn:
        async def fetch(self, sql, ids):
            seen.append((sql, ids))
            return [{"id": 5}]
    _patch_conn(monkeypatch, Conn())
    out = await SyneAgent._tool_memory_delete(object(), "5, 6")
    assert seen == [("DELETE FROM memory WHERE id = ANY($1::int[]) RETURNING id", [5, 6])]
    assert out == "Deleted 1 memory entries. Not found: 6"


async def test_tool_delete_refused(monkeypatch):
    from syne.agent import SyneAgent

    class Conn:
        async def fetch(self, sql, ids):
            raise _GuardErr("memory guard: 1 memories are referenced by other chunks")
    _patch_conn(monkeypatch, Conn())
    out = await SyneAgent._tool_memory_delete(object(), "5")
    assert out.startswith("Refused, nothing deleted. memory guard:")


# ── embedding dimension ──────────────────────────────────────────────────

def test_parse_vector_dims():
    assert ED.parse_vector_dims("vector(1024)") == 1024
    assert ED.parse_vector_dims("vector") is None
    assert ED.parse_vector_dims(None) is None


def test_refuse_switch_message():
    assert ED.refuse_switch_message(1024, 1024, 5) is None
    assert ED.refuse_switch_message(None, 768, 0) is None
    assert ED.refuse_switch_message(1024, None, 5) is None
    msg = ED.refuse_switch_message(1024, 768, 31273)
    assert "refused" in msg and "768" in msg and "1024" in msg and "31273" in msg
    # the column type is fixed, so a different dimension is refused even when empty
    assert ED.refuse_switch_message(1024, 768, 0)


async def test_stored_embedding_dims():
    class Conn:
        def __init__(self, typ, count, row_dims=None):
            self.typ, self.count, self.row_dims = typ, count, row_dims

        async def fetchval(self, sql, *a):
            if "count(*)" in sql:
                return self.count
            if "format_type" in sql:
                return self.typ
            return self.row_dims
    assert await ED.stored_embedding_dims(Conn("vector(1024)", 9)) == (1024, 9)
    assert await ED.stored_embedding_dims(Conn("vector", 3, 768)) == (768, 3)
    assert await ED.stored_embedding_dims(Conn("vector", 0)) == (None, 0)


async def test_probe_embedding_measures_vector(monkeypatch):
    from syne.llm import drivers as D

    class R:
        vector = [0.1] * 7

    class Prov:
        async def embed(self, text, model=None):
            return R()

    async def fake(entry):
        return Prov()
    monkeypatch.setattr(D, "create_embedding_provider", fake)
    assert await D.probe_embedding({"model_id": "x", "dimensions": 1024}) == (True, "", 7)
    assert await D.test_embedding({"model_id": "x"}) == (True, "")


def test_no_mass_delete_on_embedding_switch():
    src = _read("syne", "communication", "telegram.py")
    assert "DELETE FROM memory WHERE embedding" not in src
    assert "memories cleared" not in src
    assert "new_active = models[0]" not in src


# ── real PostgreSQL ──────────────────────────────────────────────────────

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
    schema = "memguard_" + uuid.uuid4().hex[:10]
    await conn.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    await conn.execute(_BASE)
    for ddl in (P.KG_PRUNE_SCHEMA_SQL, G.MEMORY_GUARD_SCHEMA_SQL):
        for stmt in _split_sql_statements(ddl):
            await conn.execute(stmt)
    return conn, schema


async def _close(conn, schema):
    await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await conn.close()


async def _chunk(c, mid):
    await c.execute("INSERT INTO memory (id) VALUES ($1)", mid)
    return await c.fetchval("INSERT INTO kg_entities (name, entity_type, description) "
                            "VALUES ($1, 'chunk', $2) RETURNING id", f"c{mid}", f"memory:{mid}")


async def _rel(c, s, p, o, mid):
    rid = await c.fetchval("INSERT INTO kg_relations (subject_id, predicate, object_id, "
                           "source_memory_id) VALUES ($1, $2, $3, $4) RETURNING id", s, p, o, mid)
    await c.execute("INSERT INTO kg_relation_sources (relation_id, source_type, source_id, "
                    "origin) VALUES ($1, 'memory', $2, 'rule')", rid, mid)


async def _ids(c):
    return [r["id"] for r in await c.fetch("SELECT id FROM memory ORDER BY id")]


async def _refused(c, sql, *a):
    import asyncpg
    with pytest.raises(asyncpg.exceptions.RestrictViolationError) as ei:
        await c.execute(sql, *a)
    assert G.guard_refusal(ei.value)
    return str(ei.value)


@pg
async def test_chain_deletes_from_the_front():
    c, sch = await _db()
    try:
        c1, c2, c3 = await _chunk(c, 1), await _chunk(c, 2), await _chunk(c, 3)
        await _rel(c, c1, "next_part", c2, 1)
        await _rel(c, c2, "next_part", c3, 2)
        nrel = await c.fetchval("SELECT count(*) FROM kg_relations")

        msg = await _refused(c, "DELETE FROM memory WHERE id = 2")
        assert "#2 is referenced by 1 other memories (e.g. #1)" in msg
        await _refused(c, "DELETE FROM memory WHERE id = ANY($1::int[])", [2, 3])
        # refused delete is rolled back whole; the KG prune never ran
        assert await _ids(c) == [1, 2, 3]
        assert await c.fetchval("SELECT count(*) FROM kg_relations") == nrel

        await c.execute("DELETE FROM memory WHERE id = 1")
        await _refused(c, "DELETE FROM memory WHERE id = 3")
        await c.execute("DELETE FROM memory WHERE id = ANY($1::int[])", [2, 3])
        assert await _ids(c) == []
        assert await c.fetchval("SELECT count(*) FROM kg_relations") == 0
    finally:
        await _close(c, sch)


@pg
async def test_whole_chain_in_one_statement():
    c, sch = await _db()
    try:
        c1, c2 = await _chunk(c, 1), await _chunk(c, 2)
        await _rel(c, c1, "next_part", c2, 1)
        await c.execute("DELETE FROM memory WHERE id = ANY($1::int[])", [1, 2])
        assert await _ids(c) == []
    finally:
        await _close(c, sch)


@pg
async def test_ayah_referenced_through_contains():
    c, sch = await _db()
    try:
        quran, tafsir, hadits = await _chunk(c, 10), await _chunk(c, 11), await _chunk(c, 12)
        ayah = await c.fetchval("INSERT INTO kg_entities (name, entity_type) "
                                "VALUES ('QS 2:255', 'ayah') RETURNING id")
        await _rel(c, quran, "contains", ayah, 10)
        await _rel(c, tafsir, "explains", ayah, 11)
        await _rel(c, hadits, "cites", ayah, 12)

        msg = await _refused(c, "DELETE FROM memory WHERE id = 10")
        assert "#10 is referenced by 2 other memories (e.g. #11)" in msg
        # the referencing chunks are not referenced themselves
        await c.execute("DELETE FROM memory WHERE id = 11")
        await _refused(c, "DELETE FROM memory WHERE id = 10")
        await c.execute("DELETE FROM memory WHERE id = ANY($1::int[])", [10, 12])
        assert await _ids(c) == []
    finally:
        await _close(c, sch)


@pg
async def test_plain_memories_delete_freely():
    c, sch = await _db()
    try:
        await c.execute("INSERT INTO memory (id) VALUES (20), (21)")
        await _chunk(c, 22)  # a chunk nobody refers to
        await c.execute("DELETE FROM memory WHERE id = 20")
        await c.execute("DELETE FROM memory WHERE id = ANY($1::int[])", [21, 22])
        assert await _ids(c) == []
    finally:
        await _close(c, sch)


@pg
async def test_tool_message_on_real_refusal(monkeypatch):
    from syne.agent import SyneAgent
    c, sch = await _db()
    try:
        c1, c2 = await _chunk(c, 1), await _chunk(c, 2)
        await _rel(c, c1, "next_part", c2, 1)
        _patch_conn(monkeypatch, c)
        out = await SyneAgent._tool_memory_delete(object(), "2")
        assert out.startswith("Refused, nothing deleted. memory guard:") and "#2" in out
        assert await SyneAgent._tool_memory_delete(object(), "1,2") == "Deleted 2 memory entries"
    finally:
        await _close(c, sch)
