"""v1.23.44: kg-build removes stale chunk entities (Riyo, 22 Nov 2026).

A chunk entity is named "<title> [#id]". Renaming a chunk's title used to
leave the old entity, with its old links, next to the new one.
"""
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from syne.memory import kg_build as B

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

SOURCES = {"alquran": {"kind": "quran"},
           "asbabnuzul": B.DEFAULT_REF_SOURCES["asbabnuzul"]}


def _q(mid, surah, name, a1, a2):
    return {"id": mid, "category": "alquran",
            "content": f"Al-Quran - {name} (x) - y\nSurat ke-{surah}, Makkiyah, Ayat {a1}-{a2}\n\n..."}


def _rows(title):
    return [_q(1, 80, "'Abasa", 1, 42),
            {"id": 7, "category": "asbabnuzul", "content": f"{title}\n\nisi"},
            {"id": 8, "category": "asbabnuzul", "content": "Tanpa rujukan\n\nisi"}]


def test_plan_records_every_chunk_read():
    plan = B.plan_links(_rows("Asbab Nuzul — QS. 'Abasa 80:1-10"), SOURCES)
    assert plan.chunk_ids == {1, 7, 8}
    assert not any("[#8]" in k[0] for k in plan.entities)


async def test_remove_stale_chunks_sql():
    plan = B.plan_links(_rows("Asbab Nuzul — QS. 'Abasa 80:1-10"), SOURCES)
    conn = MagicMock()
    conn.execute = AsyncMock(return_value="DELETE 2")
    assert await B.remove_stale_chunks(conn, plan) == 2
    sql, descs, keep = conn.execute.call_args.args
    assert "DELETE FROM kg_entities" in sql and "entity_type = 'chunk'" in sql
    assert descs == ["memory:1", "memory:7", "memory:8"]
    assert "asbab nuzul — qs. 'abasa 80:1-10 [#7]" in keep
    assert all(not k.startswith("tanpa rujukan") for k in keep)


async def test_remove_stale_chunks_nothing_read():
    conn = MagicMock()
    conn.execute = AsyncMock()
    assert await B.remove_stale_chunks(conn, B.BuildPlan()) == 0
    conn.execute.assert_not_called()


@pg
async def test_rename_leaves_one_entity():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = "v44_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}; SET search_path TO {s}")
    try:
        await c.execute("""
            CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, name TEXT NOT NULL,
                entity_type TEXT, description TEXT, scope TEXT,
                updated_at TIMESTAMPTZ DEFAULT now());
            CREATE UNIQUE INDEX ON kg_entities (LOWER(name), entity_type, coalesce(scope, ''));
            CREATE TABLE kg_relations (id SERIAL PRIMARY KEY,
                subject_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
                predicate TEXT,
                object_id INT REFERENCES kg_entities(id) ON DELETE CASCADE,
                source_memory_id INT, updated_at TIMESTAMPTZ DEFAULT now(),
                UNIQUE (subject_id, predicate, object_id));
            CREATE TABLE kg_relation_sources (
                relation_id INT REFERENCES kg_relations(id) ON DELETE CASCADE,
                source_type TEXT, source_id INT, origin TEXT,
                PRIMARY KEY (relation_id, source_type, source_id));
        """)
        await B.write_plan(c, B.plan_links(_rows("Asbab Nuzul — QS. 'Abasa 80:1-9"), SOURCES))
        plan = B.plan_links(_rows("Asbab Nuzul — QS. 'Abasa 80:1-10"), SOURCES)
        await B.write_plan(c, plan)
        assert plan.stale_removed == 1
        names = [r["name"] for r in await c.fetch(
            "SELECT name FROM kg_entities WHERE description = 'memory:7'")]
        assert names == ["Asbab Nuzul — QS. 'Abasa 80:1-10 [#7]"]
        ayat = {r["name"] for r in await c.fetch(
            "SELECT o.name FROM kg_relations r JOIN kg_entities s ON s.id = r.subject_id "
            "JOIN kg_entities o ON o.id = r.object_id "
            "WHERE s.description = 'memory:7' AND r.predicate = 'asbab_of'")}
        assert ayat == {f"QS 80:{a}" for a in range(1, 11)}
        await B.write_plan(c, plan)                  # idempotent
        assert plan.stale_removed == 0
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
