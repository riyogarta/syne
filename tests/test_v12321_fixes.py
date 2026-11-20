"""v1.23.21: an embedding switch is one transaction (review v1.23.16-19
bug 3); clearing chat history vectors asks first (bug 6)."""
import inspect
import json
import os
import uuid

import pytest

from syne.memory import embed_dims as D

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")

CFG = {"provider.active_embedding": "new", "provider.embedding_dimensions": 3}


def _tg_src():
    from syne.communication import telegram as T
    return inspect.getsource(T)


def test_apply_embedding_is_one_transaction():
    src = _tg_src()
    seg = src[src.index("async def _apply_embedding"):src.index("async def _apply_model")]
    assert "await apply_switch(conn, dims," in seg
    assert "set_config(" not in seg
    assert "clear_history_vectors" not in seg and "relax_columns" not in seg


def test_clearing_asks_first():
    src = _tg_src()
    assert 'callback_data=f"embed:ask_clear:{embed_key}"' in src
    # the only button that clears is on the confirmation screen
    assert src.count('callback_data=f"embed:confirm_clear:{embed_key}"') == 1
    ask = src[src.index('elif data.startswith("embed:ask_clear:")'):
              src.index('elif data.startswith("embed:confirm_clear:")')]
    assert "embed:confirm_clear:" in ask and "clear_confirm_text(" in ask


def test_clear_confirm_text():
    plan = {"blocked": [("messages", 1024, 50)], "relax": []}
    t = D.clear_confirm_text(plan, "<Big> model")
    assert "<b>50</b>" in t and "cannot be undone" in t and "&lt;Big&gt;" in t


async def _setup(c, memory_vec=False, cfg_check=""):
    if not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        pytest.skip("pgvector not installed in the test database")
    s = "v21_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}")
    await c.execute(f"SET search_path TO {s}, public")
    await c.execute("CREATE TABLE memory (id SERIAL PRIMARY KEY, embedding vector(4))")
    await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, content TEXT, "
                    "embedding vector(4))")
    await c.execute("INSERT INTO messages (content, embedding) VALUES "
                    "('a', '[1,2,3,4]'), ('b', NULL)")
    if memory_vec:
        await c.execute("INSERT INTO memory (embedding) VALUES ('[1,1,1,1]')")
    await c.execute("CREATE TABLE config (key TEXT PRIMARY KEY" + cfg_check +
                    ", value JSONB, updated_at TIMESTAMPTZ DEFAULT now())")
    return s


async def _unchanged(c):
    assert await c.fetchval(D._COL_SQL, "messages") == "vector(4)"
    assert await c.fetchval(D._COL_SQL, "memory") == "vector(4)"
    assert await c.fetchval("SELECT count(*) FROM messages WHERE embedding IS NOT NULL") == 1


@pg
async def test_switch_commits_everything_together():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _setup(c)
    try:
        res = await D.apply_switch(c, 3, CFG, clear_history=True)
        assert res == {"cleared": 1, "relaxed": ["memory"]}
        assert await c.fetchval(D._COL_SQL, "messages") == "vector(3)"
        assert await c.fetchval(D._COL_SQL, "memory") == "vector(3)"
        assert await c.fetchval("SELECT count(*) FROM messages") == 2
        cfg = {r["key"]: json.loads(r["value"]) for r in await c.fetch("SELECT * FROM config")}
        assert cfg == CFG
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_config_failure_rolls_back_the_clear():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _setup(c, cfg_check=" CHECK (key <> 'provider.embedding_dimensions')")
    try:
        with pytest.raises(asyncpg.CheckViolationError):
            await D.apply_switch(c, 3, CFG, clear_history=True)
        await _unchanged(c)
        assert await c.fetchval("SELECT count(*) FROM config") == 0
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
@pytest.mark.parametrize("memory_vec,clear", [(True, True), (False, False)])
async def test_refusal_changes_nothing(memory_vec, clear):
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _setup(c, memory_vec=memory_vec)
    try:
        with pytest.raises(D.SwitchRefused):
            await D.apply_switch(c, 3, CFG, clear_history=clear)
        await _unchanged(c)
        assert await c.fetchval("SELECT count(*) FROM config") == 0
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
