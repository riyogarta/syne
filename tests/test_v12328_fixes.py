"""v1.23.28: delete_user and a memory update no longer deadlock on the BM25
corpus lock (review v1.23.16-19 bug 7)."""
import asyncio
import inspect
import os
import uuid

import pytest

from syne.cli.helpers import _split_sql_statements
from syne.db import models
from syne.db.bm25_ddl import BM25_SCHEMA_SQL

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


def test_delete_user_clears_memory_owner_before_bm25_writes():
    src = inspect.getsource(models.delete_user)
    mem = src.index("UPDATE memory SET user_id = NULL")
    assert src.count("UPDATE memory SET user_id = NULL") == 1
    assert mem < src.index("UPDATE messages SET content")
    assert mem < src.index("DELETE FROM sessions")


_BASE = """
CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB, description TEXT);
CREATE TABLE memory (id SERIAL PRIMARY KEY, content TEXT, user_id INT);
CREATE TABLE messages (id SERIAL PRIMARY KEY, role TEXT, content TEXT);
"""


async def _setup():
    import asyncpg
    schema = "deluser_" + uuid.uuid4().hex[:10]
    c = await asyncpg.connect(_DB)
    await c.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
    await c.execute(_BASE)
    for stmt in _split_sql_statements(BM25_SCHEMA_SQL):
        if stmt.strip() and not stmt.strip().startswith("--"):
            await c.execute(stmt)
    await c.execute("INSERT INTO memory (id, content, user_id) VALUES (1, 'kopi pagi', 7)")
    await c.execute("INSERT INTO messages (id, role, content) VALUES (1, 'user', 'halo grup')")
    return c, schema


async def _connect(schema):
    import asyncpg
    c = await asyncpg.connect(_DB)
    await c.execute(f"SET search_path TO {schema}")
    return c


async def _race(memory_first: bool):
    """B = memory update (row lock, then content UPDATE -> corpus lock).
    A = delete_user statements in the given order. Returns the error A or B
    raised, or None."""
    import asyncpg
    c, schema = await _setup()
    a, b = await _connect(schema), await _connect(schema)
    try:
        await b.execute("BEGIN")
        await b.execute("SELECT 1 FROM memory WHERE id = 1 FOR UPDATE")
        await a.execute("BEGIN")

        async def delete_user():
            stmts = ["UPDATE messages SET content = '[deleted]' WHERE id = 1",
                     "UPDATE memory SET user_id = NULL WHERE user_id = 7"]
            if memory_first:
                stmts.reverse()
            for s in stmts:
                await a.execute(s)
            await a.execute("COMMIT")

        async def memory_update():
            await asyncio.sleep(0.3)        # A has run its first statement
            await b.execute("UPDATE memory SET content = 'teh sore' WHERE id = 1")
            await b.execute("COMMIT")

        res = await asyncio.wait_for(asyncio.gather(
            delete_user(), memory_update(), return_exceptions=True), 30)
        errs = [r for r in res if isinstance(r, asyncpg.PostgresError)]
        if not errs:
            assert await c.fetchval("SELECT user_id FROM memory WHERE id = 1") is None
            assert await c.fetchval("SELECT content FROM memory WHERE id = 1") == "teh sore"
        return errs[0] if errs else None
    finally:
        for conn in (a, b):
            await conn.close()
        await c.execute(f"DROP SCHEMA {schema} CASCADE")
        await c.close()


@pg
async def test_old_order_deadlocks():
    """Proves the scenario is real: messages first, then memory."""
    import asyncpg
    err = await _race(memory_first=False)
    assert isinstance(err, asyncpg.exceptions.DeadlockDetectedError), err


@pg
async def test_memory_first_does_not_deadlock():
    assert await _race(memory_first=True) is None
