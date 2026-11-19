"""v1.23.18: KG daily quota counts model calls (#6), wrong embedding merges
across numbers are prevented and split (#10), chat history alone no longer
blocks an embedding switch with no way out (#14)."""
import inspect
import os
import uuid

import pytest

from syne.memory import kg_extract as K
from syne.memory import embed_dims as D

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


# ── #6 ────────────────────────────────────────────────────────────────────

def test_negative_daily_limit_is_default_not_unlimited():
    assert K._clamp("kg.extract_daily_limit", -1) == 300
    assert K._clamp("kg.extract_daily_limit", 0) == 0
    assert K._clamp("kg.extract_daily_limit", 50) == 50
    # other keys keep the plain clamp
    assert K._clamp("kg.extract_batch", -3) == 1


def test_call_counted_before_the_model_is_called():
    src = inspect.getsource(K.process_one)
    assert src.index("await _count_call()") < src.index("raw = await judge(")


def test_description_matches_behaviour():
    desc = K.DEFAULTS["kg.extract_daily_limit"][1]
    assert "calls" in desc and "stale" in desc and "negative" in desc
    sql = open(os.path.join(os.path.dirname(K.__file__), "..", "db", "schema.sql")).read()
    assert desc in sql
    assert "CREATE TABLE IF NOT EXISTS kg_extract_usage" in sql


# ── #10 ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b,same", [
    ("v1 23 02", "v1 23 06", False),
    ("migrasi 44", "migrasi 45", False),
    ("18 nov 2026", "19 nov 2026", False),
    ("18 nov 2026", "18 november 2026", True),
    ("postgresql", "postgres", True),
    ("keputusan no 11", "keputusan no 12", False),
])
def test_same_numbers(a, b, same):
    assert K.same_numbers(a, b) is same


def test_embedding_merge_requires_same_numbers():
    src = inspect.getsource(K.resolve_entity)
    assert "same_numbers(row.get(\"norm_name\") or \"\", norm)" in src


# ── #14 ───────────────────────────────────────────────────────────────────

def test_history_only_refusal_offers_a_way_out():
    plan = {"blocked": [("messages", 1024, 50)], "relax": []}
    assert D.history_only(plan)
    assert "Clear history vectors" in D.refusal_for_plan(plan, 768)
    assert "reembed" not in D.refusal_for_plan(plan, 768)
    plan = {"blocked": [("memory", 1024, 3), ("messages", 1024, 50)], "relax": []}
    assert not D.history_only(plan)
    assert "Clear history vectors" not in D.refusal_for_plan(plan, 768)


def test_telegram_offers_clear_only_for_history():
    from syne.communication import telegram as T
    src = inspect.getsource(T)
    assert "embed:confirm_clear:" in src
    assert "if history_only(plan):" in src
    assert "clear_history and history_only(plan)" in src


# ── real PostgreSQL ───────────────────────────────────────────────────────

async def _schema(c, vector=False):
    if vector and not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        pytest.skip("pgvector not installed in the test database")
    s = "v18_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}; SET search_path TO {s}, public")
    return s


@pg
async def test_usage_counter_counts_every_call(monkeypatch):
    import asyncpg
    from syne.db.migrations import KG_USAGE_DDL
    c = await asyncpg.connect(_DB)
    s = await _schema(c)

    async def tz(*a, **k):
        return "Asia/Jakarta"
    monkeypatch.setattr(K, "get_config", tz)
    try:
        for stmt in KG_USAGE_DDL:
            await c.execute(stmt)
        assert await K.extracted_today(c) == 0
        for _ in range(3):           # e.g. one chunk tried three times
            await K.count_call(c)
        assert await K.extracted_today(c) == 3
        # an old day does not count
        await c.execute("INSERT INTO kg_extract_usage VALUES (current_date - 3, 99)")
        assert await K.extracted_today(c) == 3
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


_KG = """
CREATE TABLE memory (id SERIAL PRIMARY KEY, content TEXT);
CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, name TEXT, entity_type TEXT,
    aliases TEXT[] DEFAULT '{}', norm_name TEXT, scope TEXT,
    updated_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE kg_relations (id SERIAL PRIMARY KEY, subject_id INT, predicate TEXT,
    object_id INT, source_memory_id INT);
CREATE TABLE kg_relation_sources (relation_id INT, source_type TEXT, source_id INT,
    origin TEXT DEFAULT 'llm');
CREATE TABLE kg_extract_log (source_type TEXT, source_id INT, status TEXT,
    PRIMARY KEY (source_type, source_id));
"""


@pg
async def test_split_bad_merges():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    try:
        await c.execute(_KG)
        await c.execute("""
            INSERT INTO kg_entities (id, name, entity_type, aliases, norm_name) VALUES
              (1, 'v1.23.02', 'project', '{v1.23.06,v1.23.02}', 'v1 23 02'),
              (2, 'Syne', 'project', '{}', 'syne'),
              (3, 'PostgreSQL', 'item', '{Postgres}', 'postgresql'),
              (4, 'Riyo', 'person', '{}', 'riyo');
            INSERT INTO kg_relations (id, subject_id, predicate, object_id) VALUES
              (10, 1, 'part_of', 2), (11, 3, 'used_by', 2), (12, 4, 'created', 2);
            INSERT INTO kg_relation_sources VALUES (10, 'message', 100, 'llm'),
              (12, 'message', 100, 'llm'), (11, 'message', 200, 'llm');
            INSERT INTO kg_extract_log VALUES ('message', 100, 'done'), ('message', 200, 'done');
        """)
        async with c.transaction():
            res = await K.split_bad_merges(c)
        assert res == {"entities": 1}
        assert await c.fetchval("SELECT aliases FROM kg_entities WHERE id = 1") == ["v1.23.02"]
        # nothing goes back to the extraction queue, nothing is deleted (#57758)
        assert await c.fetchval("SELECT count(*) FROM kg_extract_log") == 2
        assert await c.fetchval("SELECT count(*) FROM kg_relations") == 3
        assert await c.fetchval("SELECT count(*) FROM kg_entities") == 4
        # PostgreSQL / Postgres is a right merge and stays
        assert await c.fetchval("SELECT aliases FROM kg_entities WHERE id = 3") == ["Postgres"]
        async with c.transaction():
            assert await K.split_bad_merges(c) == {"entities": 0}
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_clear_history_vectors():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c, vector=True)
    try:
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, content TEXT, "
                        "embedding vector(4))")
        await c.execute("INSERT INTO messages (content, embedding) VALUES "
                        "('a', '[1,2,3,4]'), ('b', '[4,3,2,1]'), ('c', NULL)")
        assert await D.clear_history_vectors(c, 3) == 2
        assert await c.fetchval("SELECT count(*) FROM messages") == 3
        assert await c.fetchval("SELECT count(*) FROM messages WHERE embedding IS NOT NULL") == 0
        assert await c.fetchval(D._COL_SQL, "messages") == "vector(3)"
        assert await c.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                                "AND indexname = $2", s, D.HNSW_INDEX["messages"]) == 1
        await c.execute("INSERT INTO messages (content, embedding) VALUES ('d', '[1,2,3]')")
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
