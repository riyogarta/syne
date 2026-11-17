"""memory_auto_archive dropped (1.22.80): migration 50."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from syne.db import migrations as M

ROOT = Path(__file__).resolve().parent.parent


def test_migration_50_registered():
    assert (50, M._m50_drop_memory_auto_archive, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 50


def test_migration_50_drops_table():
    conn = AsyncMock()
    asyncio.run(M._m50_drop_memory_auto_archive(conn))
    sql = " ".join(c.args[0] for c in conn.execute.call_args_list)
    assert "DROP TABLE IF EXISTS memory_auto_archive" in sql


def test_fresh_schema_has_no_archive():
    assert "memory_auto_archive" not in (ROOT / "syne/db/schema.sql").read_text()
