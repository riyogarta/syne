"""Auto-capture removal (1.22.59): migration 41 + no leftover wiring."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

from syne.db import migrations as M

ROOT = Path(__file__).resolve().parent.parent


def test_migration_41_registered():
    entry = next(m for m in M.MIGRATIONS if m[0] == 41)
    assert entry[1] is M._m41_remove_auto_capture
    assert M.CURRENT_SCHEMA_VERSION >= 41


async def _run(raw_category):
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=raw_category)
    await M._m41_remove_auto_capture(conn)
    return [c.args for c in conn.execute.await_args_list]


async def test_migration_41_moves_default_category():
    calls = await _run(None)
    assert "CREATE TABLE IF NOT EXISTS memory_auto_archive" in calls[0][0]
    move_sql, category = calls[1]
    assert category == "auto"
    assert "DELETE FROM memory" in move_sql
    assert "INSERT INTO memory_auto_archive" in move_sql
    # only non-permanent rows without attachments move
    assert "permanent" in move_sql and "memory_blobs" in move_sql
    assert "memory.auto_capture" in calls[2][0]
    assert "memory.auto_evaluate" in calls[2][0]
    assert "memory.auto_capture_category" in calls[2][0]
    assert "evaluator" not in calls[2][0]


async def test_migration_41_honours_custom_category():
    calls = await _run('"quarantine"')
    assert calls[1][1] == "quarantine"


async def test_migration_41_empty_category_falls_back():
    calls = await _run('""')
    assert calls[1][1] == "auto"


def test_no_auto_capture_wiring_left():
    import syne.conversation as conv
    assert "evaluate_and_store" not in inspect.getsource(conv)
    tg = (ROOT / "syne/communication/telegram.py").read_text()
    for needle in ("autocapture", "memory.auto_capture", "toggle_ac"):
        assert needle not in tg, needle
    schema = (ROOT / "syne/db/schema.sql").read_text()
    assert "memory.auto_capture" not in schema
    assert "memory.auto_evaluate" not in schema
    assert "memory.evaluator_model" in schema  # rule checker still needs it
