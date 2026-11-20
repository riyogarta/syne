"""v1.23.34 (review A): a switch above 2000 dimensions types the column
and leaves it without an HNSW index instead of failing."""
import os
import uuid

import pytest

from syne.memory import embed_dims as D

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


class _Exec:
    def __init__(self):
        self.sql = []

    async def execute(self, q, *a):
        self.sql.append(q)


async def test_build_hnsw_limit():
    c = _Exec()
    assert await D.build_hnsw(c, "memory", 2000) is True and "USING hnsw" in c.sql[0]
    c = _Exec()
    assert await D.build_hnsw(c, "memory", 2560) is False and c.sql == []


def test_switch_note_warns_above_limit():
    plan = {"blocked": [], "relax": ["memory"]}
    assert "no HNSW index" in D.switch_note(plan, 4096)
    assert "HNSW" not in D.switch_note(plan, 1024)


@pg
@pytest.mark.parametrize("dims,indexed", [(2560, False), (4096, False), (2000, True)])
async def test_switch_on_real_postgres(dims, indexed):
    import asyncpg
    c = await asyncpg.connect(_DB)
    if not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        await c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    s = "v34_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}")
    await c.execute(f"SET search_path TO {s}, public")
    try:
        await c.execute("CREATE TABLE memory (id SERIAL PRIMARY KEY, embedding vector)")
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, content TEXT, embedding vector)")
        await c.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB, "
                        "updated_at TIMESTAMPTZ DEFAULT now())")
        res = await D.apply_switch(c, dims, {"provider.embedding_dimensions": dims})
        assert res["relaxed"] == ["memory", "messages"]
        assert await c.fetchval("SELECT value FROM config") == str(dims)
        for t in ("memory", "messages"):
            assert await c.fetchval(D._COL_SQL, t) == f"vector({dims})"
            n = await c.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                                 "AND indexname = $2", s, D.HNSW_INDEX[t])
            assert n == (1 if indexed else 0)
        vec = "[" + ",".join(["0.1"] * dims) + "]"
        await c.execute("INSERT INTO memory (embedding) VALUES ($1::vector)", vec)
        assert await c.fetchval("SELECT count(*) FROM memory ORDER BY 1 LIMIT 1") == 1
        assert await c.fetchval("SELECT embedding <=> $1::vector FROM memory", vec) == pytest.approx(0, abs=1e-6)
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_pgvector_really_refuses_hnsw_above_2000():
    """The limit build_hnsw relies on, checked against the installed pgvector."""
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = "v34l_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}")
    await c.execute(f"SET search_path TO {s}, public")
    try:
        await c.execute("CREATE TABLE t (embedding vector(2001))")
        with pytest.raises(asyncpg.PostgresError, match="2000"):
            await c.execute("CREATE INDEX ON t USING hnsw (embedding vector_cosine_ops)")
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
