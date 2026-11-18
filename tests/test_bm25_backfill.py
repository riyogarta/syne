"""BM25 background backfill (Riyo, 18 Nov 2026) - KG-RAG stage 1, step 2."""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

from syne.db.bm25_ddl import BM25_SCHEMA_SQL
from syne.memory import bm25_backfill as BF


class _Conn:
    """Fake connection: `pending` rows per table, indexed by UPDATE ... LIMIT $1."""

    def __init__(self, pending, fail=0):
        self.pending = dict(pending)
        self.fail = fail
        self.updates = []

    async def fetchval(self, sql, *a):
        table = sql.split(" FROM ")[1].split()[0]
        return self.pending[table]

    async def execute(self, sql, *a):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("db down")
        table = sql.split()[1]
        n = min(self.pending[table], a[0])
        self.pending[table] -= n
        self.updates.append((table, sql, a[0]))
        return f"UPDATE {n}"


def _run(conn, cfg=None):
    cfg = {"bm25.backfill_enabled": True, "bm25.backfill_batch": 50,
           "bm25.backfill_pause": 0, **(cfg or {})}

    @asynccontextmanager
    async def gc(*a, **k):
        yield conn

    async def get_config(key, default=None):
        return cfg.get(key, default)

    async def no_sleep(*a, **k):
        return None

    with patch.object(BF, "get_connection", gc), \
            patch.object(BF, "get_config", get_config), \
            patch.object(BF.asyncio, "sleep", no_sleep):
        return asyncio.run(BF.run_backfill())


def test_indexes_every_pending_row_in_batches():
    conn = _Conn({"memory": 120, "messages": 30})
    assert _run(conn) == {"memory": 120, "messages": 30}
    assert conn.pending == {"memory": 0, "messages": 0}
    # memory 50+50+20, then an empty batch ends it; messages 30, then empty.
    assert [u[2] for u in conn.updates] == [50] * 6


def test_update_touches_tsv_and_skips_locked_rows():
    conn = _Conn({"memory": 5, "messages": 5})
    _run(conn)
    for table, sql, _ in conn.updates:
        assert sql.startswith(f"UPDATE {table} SET tsv = NULL")
        assert "FOR UPDATE SKIP LOCKED" in sql
        assert BF.PENDING[table] in sql
    # tool rows are never indexed
    assert "role IN ('user', 'assistant')" in BF.PENDING["messages"]


def test_nothing_pending_does_nothing():
    conn = _Conn({"memory": 0, "messages": 0})
    assert _run(conn) == {"memory": 0, "messages": 0}
    assert conn.updates == []


def test_disabled_does_nothing():
    conn = _Conn({"memory": 10, "messages": 10})
    assert _run(conn, {"bm25.backfill_enabled": False}) == {"memory": 0, "messages": 0}
    assert conn.updates == []


def test_batch_is_clamped():
    conn = _Conn({"memory": 25, "messages": 0})
    _run(conn, {"bm25.backfill_batch": 1})
    assert conn.updates[0][2] == 10


def test_failures_retry_then_continue():
    conn = _Conn({"memory": 10, "messages": 0}, fail=2)
    assert _run(conn)["memory"] == 10


def test_gives_up_after_repeated_failures():
    conn = _Conn({"memory": 10, "messages": 4}, fail=BF._MAX_FAILURES)
    done = _run(conn)
    assert done["memory"] == 0          # gave up on memory ...
    assert done["messages"] == 4        # ... but still indexed messages


def test_config_seeds_match_module_defaults():
    assert "('bm25.backfill_enabled', 'true'" in BM25_SCHEMA_SQL
    assert "('bm25.backfill_batch', '50'" in BM25_SCHEMA_SQL
    assert "('bm25.backfill_pause', '0.5'" in BM25_SCHEMA_SQL
