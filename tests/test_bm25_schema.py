"""Migration 51 / BM25 keyword index (Riyo, 18 Nov 2026) - KG-RAG stage 1."""
import os

import syne.db.migrations as M
from syne.cli.helpers import _split_sql_statements
from syne.db.bm25_ddl import BM25_SCHEMA_SQL

_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "syne", "db", "schema.sql",
)


def test_migration_51_registered():
    assert (51, M._m51_bm25_keyword_index, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 51


def test_schema_sql_carries_identical_block():
    with open(_SCHEMA) as f:
        schema = f.read()
    assert BM25_SCHEMA_SQL in schema


def test_block_splits_into_whole_statements():
    stmts = [s.strip() for s in _split_sql_statements(BM25_SCHEMA_SQL)]
    stmts = [s for s in stmts if s and not s.startswith("--")]
    # Every function / DO block must stay one statement.
    funcs = [s for s in stmts if "FUNCTION" in s.split("(")[0] or s.startswith("DO")]
    # 6 functions (syne_tsv, syne_tsv_len, syne_bm25_apply, two trigger
    # functions, syne_bm25_rebuild) + 2 DO blocks creating the triggers.
    assert len(funcs) == 8
    for s in funcs:
        assert s.rstrip().endswith(("$$;", "$$ LANGUAGE plpgsql;"))
    assert any("trg_memory_tsv" in s for s in stmts)
    assert any("trg_messages_tsv" in s for s in stmts)


class _Conn:
    def __init__(self):
        self.sql = []

    async def execute(self, sql, *a):
        self.sql.append(sql)


async def test_migration_51_runs_every_statement():
    conn = _Conn()
    await M._m51_bm25_keyword_index(conn)
    joined = "\n".join(conn.sql)
    assert "ALTER TABLE memory ADD COLUMN IF NOT EXISTS tsv tsvector" in joined
    assert "CREATE TABLE IF NOT EXISTS bm25_terms" in joined
    assert "syne_bm25_rebuild" in joined
    assert not any(s.strip().startswith("--") for s in conn.sql)
