"""Memory decay removal (1.22.79): migration 49 + no leftover wiring."""

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

from syne.db import migrations as M
from syne.memory.engine import MemoryEngine

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("memory.decay_interval", "memory.initial_recall_count",
        "memory.promotion_threshold", "memory.max_records",
        "memory.conversation_counter")


def test_migration_49_registered():
    assert (49, M._m49_remove_memory_decay, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 49


def test_migration_49_drops_columns_and_config():
    conn = AsyncMock()
    asyncio.run(M._m49_remove_memory_decay(conn))
    sql = " ".join(c.args[0] for c in conn.execute.call_args_list)
    assert "DROP COLUMN IF EXISTS permanent" in sql
    assert "DROP COLUMN IF EXISTS recall_count" in sql
    for key in KEYS:
        assert key in sql


def test_engine_has_no_decay():
    assert not hasattr(MemoryEngine, "run_decay")
    assert "recall_count" not in inspect.getsource(MemoryEngine)
    for fn in (MemoryEngine.store, MemoryEngine.store_if_new, MemoryEngine._store_with_vector):
        assert "permanent" not in inspect.signature(fn).parameters


def test_no_caller_left():
    for rel in ("syne/conversation.py", "syne/agent.py"):
        src = (ROOT / rel).read_text()
        assert "run_decay" not in src
        assert "permanent=True" not in src


def test_schema_memory_table_has_no_decay_columns():
    sql = (ROOT / "syne/db/schema.sql").read_text()
    start = sql.index("CREATE TABLE IF NOT EXISTS memory (")
    block = sql[start:sql.index(");", start)]
    assert "permanent" not in block and "recall_count" not in block
    assert "ADD COLUMN permanent" not in sql and "ADD COLUMN recall_count" not in sql
    for key in KEYS:
        assert key not in sql
