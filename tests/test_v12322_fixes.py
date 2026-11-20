"""v1.23.22: an untyped empty embedding column is typed to the new
dimension on a switch (review v1.23.16-19 bug 2)."""
import os
import uuid

import pytest

from syne.memory import embed_dims as D

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


class _DimConn:
    def __init__(self, cols):
        self.cols = cols   # table -> (column dims or None, vector dims or None, count)

    async def fetchval(self, q, *a):
        if "format_type" in q:
            col = self.cols[a[0]][0]
            return f"vector({col})" if col else "vector"
        t = next(t for t in self.cols if f"FROM {t} " in q)
        if "count(*)" in q:
            return self.cols[t][2]
        return self.cols[t][1]


async def test_untyped_empty_columns_are_typed():
    c = _DimConn({"memory": (None, None, 0), "messages": (None, None, 0)})
    assert await D.plan_switch(c, 768) == {"blocked": [], "relax": ["memory", "messages"]}


async def test_same_typed_empty_column_left_alone():
    c = _DimConn({"memory": (768, None, 0), "messages": (None, None, 0)})
    assert await D.plan_switch(c, 768) == {"blocked": [], "relax": ["messages"]}


async def test_untyped_column_with_vectors_is_not_retyped():
    # same dimension: nothing to do; another dimension: blocked, not relaxed
    c = _DimConn({"memory": (None, 768, 5), "messages": (None, 1024, 3)})
    plan = await D.plan_switch(c, 768)
    assert plan == {"blocked": [("messages", 1024, 3)], "relax": []}


@pg
async def test_switch_types_untyped_columns_and_refuses_old_writes():
    import asyncpg
    c = await asyncpg.connect(_DB)
    if not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        await c.close()
        pytest.skip("pgvector not installed in the test database")
    s = "v22_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}")
    await c.execute(f"SET search_path TO {s}, public")
    try:
        await c.execute("CREATE TABLE memory (id SERIAL PRIMARY KEY, embedding vector)")
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, content TEXT, "
                        "embedding vector)")
        await c.execute("INSERT INTO messages (content) VALUES ('a')")
        await c.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB, "
                        "updated_at TIMESTAMPTZ DEFAULT now())")
        res = await D.apply_switch(c, 3, {"provider.embedding_dimensions": 3})
        assert res == {"cleared": 0, "relaxed": ["memory", "messages"]}
        for t in ("memory", "messages"):
            assert await c.fetchval(D._COL_SQL, t) == "vector(3)"
            assert await c.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                                    "AND indexname = $2", s, D.HNSW_INDEX[t]) == 1
        with pytest.raises(asyncpg.PostgresError, match="expected 3 dimensions"):
            await c.execute("INSERT INTO memory (embedding) VALUES ('[1,2,3,4]')")
        await c.execute("INSERT INTO memory (embedding) VALUES ('[1,2,3]')")
        assert await c.fetchval("SELECT count(*) FROM messages") == 1
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
