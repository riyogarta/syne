"""v1.23.16: BM25 statistics writers can no longer deadlock across corpora.

Real PostgreSQL (SYNE_TEST_DATABASE_URL). The review of v1.23.15 found
that a backfill batch (user + assistant rows) and a bulk DELETE lock the
'messages:user' / 'messages:assistant' statistics rows in opposite order,
and that the in-function retry cannot break the cycle (the first lock of
each side belongs to the outer transaction). The fix takes one per-table
lock before any row of the statement is touched.
"""
import asyncio
import os
import time
import uuid

import pytest

from syne.cli.helpers import _split_sql_statements
from syne.db.bm25_ddl import BM25_SCHEMA_SQL

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

_BASE = """
CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB, description TEXT);
CREATE TABLE memory (id SERIAL PRIMARY KEY, content TEXT);
CREATE TABLE messages (id SERIAL PRIMARY KEY, role TEXT, content TEXT);
"""


async def _connect(schema):
    import asyncpg
    c = await asyncpg.connect(_DB)
    await c.execute(f"SET search_path TO {schema}")
    return c


async def _setup():
    import asyncpg
    schema = "bm25dl_" + uuid.uuid4().hex[:10]
    c = await asyncpg.connect(_DB)
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    await c.execute(_BASE)
    for stmt in _split_sql_statements(BM25_SCHEMA_SQL):
        if stmt.strip() and not stmt.strip().startswith("--"):
            await c.execute(stmt)
    return c, schema


async def _stats(c):
    corp = {r["corpus"]: (r["n_docs"], r["total_len"])
            for r in await c.fetch("SELECT * FROM bm25_corpus WHERE n_docs <> 0 OR total_len <> 0")}
    terms = {(r["corpus"], r["term"]): r["df"]
             for r in await c.fetch("SELECT * FROM bm25_terms WHERE df <> 0")}
    return corp, terms


async def _drift(c):
    before = await _stats(c)
    await c.execute("SELECT syne_bm25_rebuild()")
    return before != await _stats(c)


@pg
async def test_backfill_and_bulk_delete_do_not_deadlock():
    c, schema = await _setup()
    a = await _connect(schema)
    b = await _connect(schema)
    warnings = []
    for conn in (a, b):
        conn.add_log_listener(lambda _c, m: warnings.append(str(m)))
    try:
        # rows 1 (user) and 2 (assistant) still waiting for the backfill,
        # rows 3 (assistant) and 4 (user) already indexed.
        await c.execute("ALTER TABLE messages DISABLE TRIGGER USER")
        await c.execute("INSERT INTO messages (id, role, content) VALUES "
                        "(1, 'user', 'alpha beta'), (2, 'assistant', 'gamma delta')")
        await c.execute("ALTER TABLE messages ENABLE TRIGGER USER")
        await c.execute("INSERT INTO messages (id, role, content) VALUES "
                        "(3, 'assistant', 'epsilon zeta'), (4, 'user', 'eta theta')")

        # B: bulk delete, assistant row first.
        await b.execute("BEGIN")
        await b.execute("DELETE FROM messages WHERE id = 3")

        async def backfill():
            await a.execute("BEGIN")
            await a.execute("UPDATE messages SET tsv = NULL WHERE id = 1")
            await a.execute("UPDATE messages SET tsv = NULL WHERE id = 2")
            await a.execute("COMMIT")

        async def delete_rest():
            await b.execute("DELETE FROM messages WHERE id = 4")
            await b.execute("COMMIT")

        t0 = time.monotonic()
        ta = asyncio.create_task(backfill())
        await asyncio.sleep(0.3)
        tb = asyncio.create_task(delete_rest())
        await asyncio.wait_for(asyncio.gather(ta, tb), 30)
        took = time.monotonic() - t0

        assert not [w for w in warnings if "bm25" in w], warnings
        assert took < 1.0, f"writers waited {took:.1f}s"
        assert not await _drift(c)
        assert await c.fetchval("SELECT count(*) FROM messages WHERE tsv IS NULL") == 0
    finally:
        for conn in (a, b):
            await conn.close()
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()


@pg
async def test_single_statement_backfill_vs_delete():
    """The real backfill statement (FOR UPDATE SKIP LOCKED subquery) against a
    multi-row DELETE of both roles, run many times concurrently."""
    from syne.memory.bm25_backfill import PENDING
    c, schema = await _setup()
    conns = [await _connect(schema) for _ in range(4)]
    warnings = []
    for conn in conns:
        conn.add_log_listener(lambda _c, m: warnings.append(str(m)))
    try:
        await c.execute("ALTER TABLE messages DISABLE TRIGGER USER")
        await c.execute(
            "INSERT INTO messages (role, content) SELECT "
            "CASE WHEN g % 2 = 0 THEN 'user' ELSE 'assistant' END, 'kata' || (g % 7) || ' umum x' || g "
            "FROM generate_series(1, 400) g")
        await c.execute("ALTER TABLE messages ENABLE TRIGGER USER")
        await c.execute(
            "INSERT INTO messages (role, content) SELECT "
            "CASE WHEN g % 3 = 0 THEN 'user' ELSE 'assistant' END, 'kata' || (g % 5) || ' lain y' || g "
            "FROM generate_series(1, 400) g")
        sql = (f"UPDATE messages SET tsv = NULL WHERE id IN (SELECT id FROM messages "
               f"WHERE {PENDING['messages']} ORDER BY id LIMIT 25 FOR UPDATE SKIP LOCKED)")

        async def backfill(conn):
            while await conn.execute(sql) != "UPDATE 0":
                pass

        async def deleter(conn, lo):
            for start in range(lo, 801, 40):
                await conn.execute(
                    "DELETE FROM messages WHERE id IN (SELECT id FROM messages "
                    "WHERE id BETWEEN $1 AND $2 AND tsv IS NOT NULL ORDER BY id DESC)",
                    start, start + 15)

        await asyncio.wait_for(asyncio.gather(
            backfill(conns[0]), backfill(conns[1]),
            deleter(conns[2], 401), deleter(conns[3], 421)), 60)
        assert not [w for w in warnings if "bm25" in w], warnings[:3]
        assert not await _drift(c)
    finally:
        for conn in conns:
            await conn.close()
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()
