"""v1.23.43: the KG daily extraction limit is removed (Riyo, 21 Nov 2026)."""
import inspect
import os
import uuid

import pytest

from syne.db import migrations as M
from syne.memory import kg_extract as K

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


def test_no_daily_limit_left():
    assert "kg.extract_daily_limit" not in K.DEFAULTS
    src = inspect.getsource(K)
    for name in ("kg_extract_usage", "extracted_today", "count_chunk", "extract_daily_limit"):
        assert name not in src
    db = os.path.join(os.path.dirname(M.__file__))
    for f in ("schema.sql", "kg_extract_ddl.py"):
        text = open(os.path.join(db, f)).read()
        assert "kg_extract_usage" not in text
        assert "kg.extract_daily_limit" not in text


def test_order_batch_has_no_daily_room():
    assert list(inspect.signature(K.order_batch).parameters) == ["items", "suspects", "batch"]


def test_m69_registered():
    assert (69, M._m69_drop_kg_daily_limit, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 69


@pg
async def test_m69_drops_table_and_config():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = "v43_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}; SET search_path TO {s}")
    try:
        await c.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value JSONB, description TEXT)")
        await c.execute("INSERT INTO config VALUES ('kg.extract_daily_limit', '300', 'x'), "
                        "('kg.extract_batch', '5', 'y')")
        await c.execute("CREATE TABLE kg_extract_usage (day DATE, source_type TEXT, source_id INT)")
        for _ in range(2):                 # idempotent
            async with c.transaction():
                await M._m69_drop_kg_daily_limit(c)
        assert await c.fetchval("SELECT to_regclass('kg_extract_usage')") is None
        assert [r["key"] for r in await c.fetch("SELECT key FROM config")] == ["kg.extract_batch"]
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
