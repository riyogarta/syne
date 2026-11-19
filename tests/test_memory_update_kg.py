"""memory_update keeps the KG in step (Riyo, 19 Nov 2026) - v1.23.08.

A memory of the KG era whose content or category changes loses what the KG
worker extracted from it, in the same transaction, and is extracted again.
Older memories keep their graph. The PostgreSQL tests run when
SYNE_TEST_DATABASE_URL points at a throwaway database; each test works in
its own schema and drops it.
"""
import inspect
import json
import os
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest

from syne.memory import engine as E
from syne.memory import kg_extract as K

ERA = "2026-11-18T04:45:16+00:00"
_ERA = K.parse_era(ERA)
_AFTER = _ERA + timedelta(hours=1)
_BEFORE = _ERA - timedelta(days=1)


def _s():
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    s["kg.extract_predicates"] = dict(K.DEFAULT_PREDICATES)
    return s


def _row(content="a", category="fact", created=_AFTER):
    return {"content": content, "category": category, "created_at": created}


def test_needs_reextract():
    assert K.needs_reextract(_row(), "b", "fact", _ERA)
    assert K.needs_reextract(_row(), "a", "decision", _ERA)
    assert not K.needs_reextract(_row(), "a", "fact", _ERA)                 # nothing changed
    assert not K.needs_reextract(_row(created=_BEFORE), "b", "fact", _ERA)  # before the era
    assert not K.needs_reextract(_row(), "b", "fact", None)                 # no era
    assert not K.needs_reextract(None, "b", "fact", _ERA)
    assert K.needs_reextract(_row(created=_AFTER.replace(tzinfo=None)), "b", "fact", _ERA)


def test_pending_reads_updated_at():
    assert "x.updated_at" in inspect.getsource(K.pending)


def test_worker_does_not_log_twice():
    assert 'if not res.get("_logged"):' in inspect.getsource(K.run_worker)


# ── real PostgreSQL ──────────────────────────────────────────────────────

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

# 'vector' is a text domain here: the test database has no pgvector.
_BASE = """
CREATE DOMAIN vector AS text;
CREATE TABLE memory (id INT PRIMARY KEY, content TEXT, embedding vector, category TEXT,
    source TEXT, importance REAL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, name TEXT NOT NULL, entity_type VARCHAR(50),
    description TEXT, norm_name TEXT, embedding vector, aliases TEXT[] DEFAULT '{}',
    scope TEXT, updated_at TIMESTAMPTZ DEFAULT now());
CREATE UNIQUE INDEX kg_entities_name_type ON kg_entities (LOWER(name), entity_type, coalesce(scope, ''));
CREATE TABLE kg_relations (
    id SERIAL PRIMARY KEY,
    subject_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100) NOT NULL,
    object_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    source_memory_id INT REFERENCES memory(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(subject_id, predicate, object_id));
CREATE TABLE kg_relation_sources (
    relation_id INT NOT NULL REFERENCES kg_relations(id) ON DELETE CASCADE,
    source_type VARCHAR(20) NOT NULL, source_id INT NOT NULL,
    origin VARCHAR(20) NOT NULL DEFAULT 'llm',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (relation_id, source_type, source_id));
CREATE TABLE kg_extract_log (
    source_type VARCHAR(20) NOT NULL, source_id INT NOT NULL, layer SMALLINT, session_id INT,
    status VARCHAR(10) NOT NULL CHECK (status IN ('done', 'skipped', 'error')),
    reason TEXT, model TEXT, n_entities INT NOT NULL DEFAULT 0,
    n_relations INT NOT NULL DEFAULT 0, latency_ms INT, attempts INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source_type, source_id));
"""


class _CM:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


class _Provider:
    async def embed(self, text):
        return SimpleNamespace(vector=[0.5, 0.25])


async def _db(monkeypatch, era=ERA):
    import asyncpg
    conn = await asyncpg.connect(_DB)
    schema = "memupd_" + uuid.uuid4().hex[:10]
    await conn.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    await conn.execute(_BASE)
    monkeypatch.setattr(E, "get_connection", lambda: _CM(conn))
    monkeypatch.setattr(K, "get_connection", lambda: _CM(conn))

    async def get_config(key, default=None):
        return era if key == "kg.extract_era_start" else default
    monkeypatch.setattr(K, "get_config", get_config)
    return conn, schema


async def _close(conn, schema):
    await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    await conn.close()


async def _mem(c, mid, created, content="lama"):
    await c.execute("INSERT INTO memory (id, content, category, created_at, updated_at) "
                    "VALUES ($1, $2, 'fact', $3, $3)", mid, content, created)


async def _ent(c, name, etype, desc=None):
    return await c.fetchval("INSERT INTO kg_entities (name, entity_type, description) "
                            "VALUES ($1, $2, $3) RETURNING id", name, etype, desc)


async def _rel(c, a, pred, b, sources, mem=None):
    rid = await c.fetchval("INSERT INTO kg_relations (subject_id, predicate, object_id, "
                           "source_memory_id) VALUES ($1, $2, $3, $4) RETURNING id",
                           a, pred, b, mem)
    for st, sid, origin in sources:
        await c.execute("INSERT INTO kg_relation_sources (relation_id, source_type, source_id, "
                        "origin) VALUES ($1, $2, $3, $4)", rid, st, sid, origin)
    return rid


async def _graph(c, created):
    """Memory 1 with an own relation, one shared with memory 2, and a rule link."""
    await _mem(c, 1, created)
    await _mem(c, 2, created)
    agha = await _ent(c, "Agha", "person")
    paket = await _ent(c, "Paket C", "concept")
    riyo = await _ent(c, "Riyo", "person")
    chunk = await _ent(c, "c1", "chunk", "memory:1")
    ayah = await _ent(c, "QS 1:1", "ayah")
    r1 = await _rel(c, agha, "studies", paket, [("memory", 1, "llm")], mem=1)
    r2 = await _rel(c, agha, "child_of", riyo, [("memory", 1, "llm"), ("memory", 2, "llm")], mem=1)
    r3 = await _rel(c, chunk, "contains", ayah, [("memory", 1, "rule")], mem=1)
    await c.execute("INSERT INTO kg_extract_log (source_type, source_id, status) "
                    "VALUES ('memory', 1, 'done'), ('memory', 2, 'done')")
    return r1, r2, r3


async def _names(c):
    return {r["name"] for r in await c.fetch("SELECT name FROM kg_entities")}


async def _snapshot(c):
    return (
        sorted(tuple(r) for r in await c.fetch(
            "SELECT relation_id, source_type, source_id, origin FROM kg_relation_sources")),
        sorted(tuple(r) for r in await c.fetch("SELECT id, source_memory_id FROM kg_relations")),
        await _names(c),
        sorted(tuple(r) for r in await c.fetch("SELECT source_type, source_id FROM kg_extract_log")),
    )


@pg
async def test_update_resets_extraction_of_era_memory(monkeypatch):
    c, sch = await _db(monkeypatch)
    try:
        _r1, r2, r3 = await _graph(c, _AFTER)
        eng = E.MemoryEngine(_Provider())
        assert await eng._update_memory(1, "baru", "fact", "user_confirmed", 0.5) == 1
        row = await c.fetchrow("SELECT content, embedding FROM memory WHERE id = 1")
        assert row["content"] == "baru" and row["embedding"] == "[0.5, 0.25]"
        # own relation gone; the shared one stays and points at memory 2;
        # the rule link of the chunk stays with memory 1
        rels = {r["id"]: r["source_memory_id"]
                for r in await c.fetch("SELECT id, source_memory_id FROM kg_relations")}
        assert rels == {r2: 2, r3: 1}
        assert await c.fetchval("SELECT array_agg(origin) FROM kg_relation_sources "
                                "WHERE source_type = 'memory' AND source_id = 1") == ["rule"]
        # Paket C has no relation left; Agha keeps the shared one
        assert await _names(c) == {"Agha", "Riyo", "c1", "QS 1:1"}
        # no log row: the worker extracts memory 1 again
        assert await c.fetchval("SELECT array_agg(source_id) FROM kg_extract_log") == [2]
    finally:
        await _close(c, sch)


@pg
async def test_category_change_resets_too(monkeypatch):
    c, sch = await _db(monkeypatch)
    try:
        await _graph(c, _AFTER)
        await E.MemoryEngine(_Provider())._update_memory(1, "lama", "decision", "user_confirmed", 0.5)
        assert await c.fetchval("SELECT array_agg(source_id) FROM kg_extract_log") == [2]
    finally:
        await _close(c, sch)


@pg
@pytest.mark.parametrize("case", ["before_era", "no_era", "unchanged"])
async def test_update_keeps_graph(monkeypatch, case):
    c, sch = await _db(monkeypatch, era=None if case == "no_era" else ERA)
    try:
        await _graph(c, _BEFORE if case == "before_era" else _AFTER)
        before = await _snapshot(c)
        content = "lama" if case == "unchanged" else "baru"
        await E.MemoryEngine(_Provider())._update_memory(1, content, "fact", "user_confirmed", 0.5)
        assert await _snapshot(c) == before
        assert await c.fetchval("SELECT content FROM memory WHERE id = 1") == content
    finally:
        await _close(c, sch)


_RAW = json.dumps({
    "entities": [{"name": "Agha", "type": "person"}, {"name": "Paket C", "type": "concept"}],
    "relations": [{"subject": "Agha", "predicate": "studies", "object": "Paket C"}],
})


def _judge(raw=_RAW):
    async def judge(prompt, timeout):
        return raw
    judge.label = "fake"
    return judge


async def _item(c, mid):
    r = await c.fetchrow("SELECT id, content, category, updated_at FROM memory WHERE id = $1", mid)
    return {"type": "memory", **dict(r)}


@pg
async def test_worker_writes_graph_and_log_together(monkeypatch):
    c, sch = await _db(monkeypatch)
    try:
        await _mem(c, 1, _AFTER, content="Agha belajar Paket C")
        res = await K.process_one(await _item(c, 1), _judge(), None, _s())
        assert res["status"] == "done" and res["_logged"] and res["n_relations"] == 1
        log = await c.fetchrow("SELECT status, n_relations FROM kg_extract_log "
                               "WHERE source_type = 'memory' AND source_id = 1")
        assert dict(log) == {"status": "done", "n_relations": 1}
        assert await c.fetchval("SELECT count(*) FROM kg_relation_sources "
                                "WHERE source_id = 1 AND origin = 'llm'") == 1
    finally:
        await _close(c, sch)


@pg
async def test_worker_drops_extraction_of_changed_memory(monkeypatch):
    c, sch = await _db(monkeypatch)
    try:
        await _mem(c, 1, _AFTER, content="Agha belajar Paket C")
        item = await _item(c, 1)          # picked up by the worker ...
        await E.MemoryEngine(_Provider())._update_memory(   # ... then corrected
            1, "Agha lulus Paket C", "fact", "user_confirmed", 0.5)
        res = await K.process_one(item, _judge(), None, _s())
        assert res["status"] == "stale" and res["_logged"]
        for t in ("kg_relations", "kg_relation_sources", "kg_extract_log", "kg_entities"):
            assert await c.fetchval(f"SELECT count(*) FROM {t}") == 0, t
        # the next round extracts the new content
        res = await K.process_one(await _item(c, 1), _judge(), None, _s())
        assert res["status"] == "done"
        assert await c.fetchval("SELECT status FROM kg_extract_log WHERE source_id = 1") == "done"
    finally:
        await _close(c, sch)
