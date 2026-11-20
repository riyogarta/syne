"""v1.23.31: titled person entities join the untitled one (#57775 option B)."""
import os
import uuid

import pytest

import syne.db.migrations as M
from syne.memory import kg_extract as K


def test_person_norm_drops_titles():
    assert K.person_norm("Pak Riyo") == "riyo"
    assert K.person_norm("Prof. Dr. Hari") == "hari"
    assert K.person_norm("Pak") == "pak"
    assert K.entity_norm("Pak Riyo", "person") == "riyo"
    assert K.entity_norm("Bapak Pucung", "concept") == "bapak pucung"


def test_migration_64_registered():
    assert M.MIGRATIONS[-1][0] == 64 and M.CURRENT_SCHEMA_VERSION == 64


_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

_BASE = """
CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, name TEXT NOT NULL, entity_type VARCHAR(50),
    aliases TEXT[] DEFAULT '{}', norm_name TEXT, scope TEXT, updated_at TIMESTAMPTZ DEFAULT NOW());
CREATE TABLE kg_relations (id SERIAL PRIMARY KEY,
    subject_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    predicate VARCHAR(100) NOT NULL,
    object_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
    updated_at TIMESTAMPTZ DEFAULT NOW(), UNIQUE(subject_id, predicate, object_id));
CREATE TABLE kg_relation_sources (
    relation_id INT NOT NULL REFERENCES kg_relations(id) ON DELETE CASCADE,
    source_type VARCHAR(20) NOT NULL, source_id INT NOT NULL,
    origin VARCHAR(20) NOT NULL DEFAULT 'llm', created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (relation_id, source_type, source_id));
"""


@pg
async def test_merge_on_real_postgres():
    import asyncpg
    c = await asyncpg.connect(_DB)
    schema = "kgmerge_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    try:
        await c.execute(_BASE)

        async def ent(name, norm, scope=None):
            return await c.fetchval(
                "INSERT INTO kg_entities (name, entity_type, norm_name, scope) "
                "VALUES ($1, 'person', $2, $3) RETURNING id", name, norm, scope)

        async def rel(s, p, o, src):
            rid = await c.fetchval("INSERT INTO kg_relations (subject_id, predicate, object_id) "
                                   "VALUES ($1, $2, $3) RETURNING id", s, p, o)
            await c.execute("INSERT INTO kg_relation_sources (relation_id, source_type, source_id) "
                            "VALUES ($1, 'message', $2)", rid, src)
            return rid

        riyo, pak = await ent("Riyo", "riyo"), await ent("Pak Riyo", "pak riyo")
        molt, danu = await ent("Molt", "molt"), await ent("Danu", "danu")
        prof = await ent("Prof Hari", "prof hari")
        other = await ent("Pak Riyo", "pak riyo", "chat:telegram:-1")   # other scope
        same = await rel(molt, "mentions", riyo, 1)
        await rel(molt, "mentions", pak, 2)          # duplicates `same`
        await rel(danu, "friend_of", pak, 3)         # moves
        await rel(pak, "mentions", riyo, 4)          # becomes a self-loop

        res = await K.merge_titled_people(c)
        assert res == {"merged": 1, "renamed": 2}
        assert await c.fetchval("SELECT count(*) FROM kg_entities WHERE id = $1", pak) == 0
        assert await c.fetchval("SELECT aliases FROM kg_entities WHERE id = $1", riyo) == ["Pak Riyo"]
        rows = {(r["subject_id"], r["predicate"], r["object_id"])
                for r in await c.fetch("SELECT * FROM kg_relations")}
        assert rows == {(molt, "mentions", riyo), (danu, "friend_of", riyo)}
        assert sorted(await c.fetchval(
            "SELECT array_agg(source_id) FROM kg_relation_sources WHERE relation_id = $1", same)) == [1, 2]
        assert await c.fetchval("SELECT norm_name FROM kg_entities WHERE id = $1", prof) == "hari"
        assert await c.fetchval("SELECT norm_name FROM kg_entities WHERE id = $1", other) == "riyo"
        assert await K.merge_titled_people(c) == {"merged": 0, "renamed": 0}   # idempotent
    finally:
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()
