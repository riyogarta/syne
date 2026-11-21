"""v1.23.17: restore gate for the KG worker's DB steps, scheduler gate gap,
relax_columns race, kg_entities.embedding typed column (migration 61)."""
import asyncio
import inspect
import os
import uuid

import pytest

from syne import maintenance as MT
from syne.memory import kg_extract as K
from syne.memory import embed_dims as D
from syne.scheduler import Scheduler

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.setattr(MT, "_reason", None)
    monkeypatch.setattr(MT, "_turns", set())


# ── #2 gated() ────────────────────────────────────────────────────────────

async def test_gated_refuses_when_restore_active(monkeypatch):
    ran = []

    async def step():
        ran.append(1)
    monkeypatch.setattr(MT, "_reason", "restore")
    with pytest.raises(K.GateClosed):
        await K.gated(step())
    assert ran == []


async def test_gated_step_is_cancelled_by_restore():
    started = asyncio.Event()
    seen = {}

    async def step():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
    t = asyncio.create_task(K.gated(step()))
    await started.wait()
    assert len(MT._turns) == 1
    n = await MT.begin("test", timeout=2)
    MT.end()
    with pytest.raises(K.GateClosed):
        await t
    assert n == 1 and seen == {"cancelled": True} and not MT._turns


async def test_gated_returns_result_and_raises_errors():
    async def ok():
        return 7

    async def bad():
        raise ValueError("x")
    assert await K.gated(ok()) == 7
    with pytest.raises(ValueError):
        await K.gated(bad())
    assert not MT._turns


def test_worker_gates_every_db_step():
    src = inspect.getsource(K.run_worker)
    for step in ("gated(maybe_prune())", "gated(load_settings())", "gated(_floors())",
                 "gated(_queue())", "gated(write_log(item, res))"):
        assert step in src, step
    assert "except GateClosed:" in src
    # no ungated connection left in the loop
    assert src.count("async with get_connection() as conn:") == 2   # inside _floors/_queue


# ── bug 4: scheduler ─────────────────────────────────────────────────────

async def test_scheduler_does_not_start_a_turn_during_restore(monkeypatch):
    ran = []

    async def on_exec(*a):
        ran.append(a)
    monkeypatch.setattr(MT, "_reason", "restore")
    assert await Scheduler(on_exec)._execute_isolated(1) is False
    assert ran == []


async def test_scheduler_turn_tracked_before_it_starts():
    """begin() between create_task and the callback's first line still
    cancels the turn (it is tracked from creation)."""
    ran = []

    async def on_exec(*a):
        ran.append(1)
    t = asyncio.create_task(Scheduler(on_exec)._execute_isolated(1))
    await asyncio.sleep(0)          # _execute_isolated created the run task
    assert len(MT._turns) == 1
    await MT.begin("test", timeout=2)
    MT.end()
    assert await t is False and ran == []


def test_scheduler_stops_writing_after_restore_began():
    src = inspect.getsource(Scheduler._check_and_execute)
    i = src.index("run again on a later tick")
    assert "if _maint.active():" in src[i:i + 400]


# ── bug 3 / bug 8: real PostgreSQL + pgvector ─────────────────────────────

async def _schema(c):
    if not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        pytest.skip("pgvector not installed in the test database")
    s = "emb_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}; SET search_path TO {s}, public")
    return s


@pg
async def test_relax_columns_types_to_new_dims_and_reindexes():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    try:
        for t in ("memory", "messages"):
            await c.execute(f"CREATE TABLE {t} (id SERIAL PRIMARY KEY, embedding vector(4))")
            await c.execute(f"CREATE INDEX {D.HNSW_INDEX[t]} ON {t} USING hnsw "
                            "(embedding vector_cosine_ops)")
        assert await D.relax_columns(c, ["messages", "memory"], 3) == ["memory", "messages"]
        for t in ("memory", "messages"):
            assert await c.fetchval(D._COL_SQL, t) == "vector(3)"
            assert await c.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                                    "AND indexname = $2", s, D.HNSW_INDEX[t]) == 1
        # an old-provider write of the old dimension is refused, not mixed
        with pytest.raises(asyncpg.PostgresError, match="expected 3 dimensions"):
            await c.execute("INSERT INTO memory (embedding) VALUES ('[1,2,3,4]')")
        await c.execute("INSERT INTO memory (embedding) VALUES ('[1,2,3]')")
        # a column that holds a vector is refused; nothing changes
        with pytest.raises(RuntimeError):
            await D.relax_columns(c, ["memory"], 5)
        assert await c.fetchval(D._COL_SQL, "memory") == "vector(3)"
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_relax_columns_waits_for_a_running_writer():
    """The count runs under the table lock: a writer whose transaction is
    open when the switch starts either commits first (switch refused) or
    waits for the switch (its old-dimension write refused)."""
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    w = await asyncpg.connect(_DB)
    await w.execute(f"SET search_path TO {s}, public")
    try:
        await c.execute("CREATE TABLE memory (id SERIAL PRIMARY KEY, embedding vector(4))")
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, embedding vector(4))")
        await w.execute("BEGIN")
        await w.execute("INSERT INTO memory (embedding) VALUES ('[1,2,3,4]')")
        sw = asyncio.create_task(D.relax_columns(c, ["memory"], 3))
        await asyncio.sleep(0.3)
        assert not sw.done()            # blocked on the lock
        await w.execute("COMMIT")
        with pytest.raises(RuntimeError):
            await sw
        assert await c.fetchval(D._COL_SQL, "memory") == "vector(4)"
    finally:
        await w.close()
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_migration_61_untypes_kg_entities_embedding():
    import asyncpg
    from syne.db.migrations import _m61_kg_entities_embedding_untyped as m61
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    try:
        await c.execute("CREATE TABLE kg_entities (id SERIAL PRIMARY KEY, embedding vector(4))")
        await c.execute("INSERT INTO kg_entities (embedding) VALUES ('[1,2,3,4]')")
        await c.execute("CREATE INDEX idx_kg_entities_embedding_hnsw ON kg_entities "
                        "USING hnsw (embedding vector_cosine_ops)")
        await m61(c)
        await m61(c)                    # idempotent
        assert await c.fetchval(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'kg_entities'::regclass AND attname = 'embedding'") == "vector"
        await c.execute("INSERT INTO kg_entities (embedding) VALUES ('[1,2,3]')")
        assert await c.fetchval("SELECT count(*) FROM kg_entities") == 2
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
