"""v1.23.32: one KG entity per owner/family person."""
import json
import os
import uuid

import pytest

import syne.db.migrations as M
from syne.memory import kg_extract as K

DIRECTORY = [
    {"names": ["Riyo", "Riyogarta", "Yahyo"], "access": "owner", "chat_id": "3478983", "platform": "telegram"},
    {"names": ["Yuli", "Yuliazmi", "Bunli"], "access": "family", "chat_id": "1544010", "platform": "telegram"},
    {"names": ["Agha", "Gamitra"], "access": "family", "chat_id": "41005565", "platform": "telegram"},
]
USERS = [
    {"platform": "telegram", "platform_id": "3478983", "name": "Riyogarta",
     "display_name": "Riyogarta Pratikto", "aliases": "{}"},
    {"platform": "whatsapp", "platform_id": "628111681624", "name": "Pak Riyo",
     "display_name": "Pak Riyo", "aliases": "{}"},
    {"platform": "telegram", "platform_id": "1544010", "name": "Yuliazmi",
     "display_name": "Yuliazmi", "aliases": json.dumps({"default": "Bu Yuli"})},
]


class _Conn:
    async def fetchval(self, q, *a):
        return json.dumps(DIRECTORY)

    async def fetch(self, q, *a):
        return USERS


async def test_person_groups_join_directory_users_and_shared_names():
    g = await K.person_groups(_Conn(), fresh=True)
    riyo = g["riyo"]
    assert riyo == {"riyo", "riyogarta", "yahyo", "riyogarta pratikto"}
    assert g["yahyo"] is riyo and g["riyogarta pratikto"] is riyo
    assert g["bunli"] == {"yuli", "yuliazmi", "bunli"}
    assert "pak riyo" not in g and "pak" not in g


async def test_person_key():
    g = await K.person_groups(_Conn(), fresh=True)
    assert K.person_key("pak riyo", g) == g["riyo"]
    assert K.person_key("agha gamitra", g) == g["agha"]
    assert K.person_key("yuli riyo", g) is None
    assert K.person_key("riyo suharto", g) is None
    assert K.person_key("riyo", {}) is None


async def test_person_groups_without_db_is_empty(monkeypatch):
    monkeypatch.setitem(K._PG_CACHE, "val", None)
    assert await K.person_groups() == {}


async def test_select_chunk_matches_other_names():
    class C:
        def __init__(self):
            self.grams = None

        async def fetchval(self, q, grams, *a):
            self.grams = grams
            return 1 if "riyogarta pratikto" in grams else None
    g = await K.person_groups(_Conn(), fresh=True)
    c = C()
    assert await K.select_chunk(c, "yahyo", 10, g) == (True, "known_entity")
    assert await K.select_chunk(C(), "yahyo", 10) == (False, "short_no_entity")


def test_migration_65_registered():
    assert M.MIGRATIONS[-1][0] == 65 and M.CURRENT_SCHEMA_VERSION == 65


_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

_BASE = """
CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB);
CREATE TABLE users (id SERIAL PRIMARY KEY, platform TEXT, platform_id TEXT, name TEXT,
    display_name TEXT, aliases JSONB DEFAULT '{}', access_level TEXT);
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
async def test_merge_known_people_on_real_postgres():
    import asyncpg
    c = await asyncpg.connect(_DB)
    schema = "kgpeople_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    try:
        await c.execute(_BASE)
        await c.execute("INSERT INTO config VALUES ('family.directory', $1::jsonb)", json.dumps(DIRECTORY))
        for u in USERS:
            await c.execute("INSERT INTO users (platform, platform_id, name, display_name, aliases, "
                            "access_level) VALUES ($1, $2, $3, $4, $5::jsonb, 'owner')",
                            u["platform"], u["platform_id"], u["name"], u["display_name"], u["aliases"])

        async def ent(name, scope=None):
            return await c.fetchval(
                "INSERT INTO kg_entities (name, entity_type, norm_name, scope) "
                "VALUES ($1, 'person', $2, $3) RETURNING id", name, K.person_norm(name), scope)

        async def rel(s, p, o):
            await c.execute("INSERT INTO kg_relations (subject_id, predicate, object_id) "
                            "VALUES ($1, $2, $3)", s, p, o)

        full, riyo, yahyo = await ent("Riyogarta Pratikto"), await ent("Riyo"), await ent("Yahyo")
        agha, gam, both = await ent("Agha"), await ent("Gamitra"), await ent("Agha Gamitra")
        molt, danu = await ent("Molt"), await ent("Danu")
        scoped = await ent("Riyo", "chat:telegram:-1")
        await rel(full, "decided", molt)
        await rel(full, "uses", danu)
        await rel(riyo, "decided", molt)
        await rel(molt, "mentions", yahyo)
        await rel(gam, "friend_of", danu)

        assert await K.merge_known_people(c) == {"merged": 4, "people": 2}
        ids = {r["id"] for r in await c.fetch("SELECT id FROM kg_entities")}
        assert ids == {full, gam, molt, danu, scoped}
        al = await c.fetchval("SELECT aliases FROM kg_entities WHERE id = $1", full)
        assert sorted(al) == ["Riyo", "Yahyo"]
        assert await c.fetchval("SELECT count(*) FROM kg_relations") == 4
        g = await K.person_groups(c, fresh=True)
        assert await K.person_entity(c, K.person_key("yahyo", g)) == full
        assert await K.person_entity(c, K.person_key("agha", g)) == gam
        assert await K.merge_known_people(c) == {"merged": 0, "people": 0}
    finally:
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()
