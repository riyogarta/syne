"""Graph recall (keyword only), ranked relations, kg_recall_log, relation sources."""
import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from syne.memory import graph as G

ROOT = Path(__file__).resolve().parents[1]
SCOPE = {"session_id": 51, "platform": "telegram", "chat_id": "3478983"}


def _rel(rid, s, o, pred="rel", sname=None, oname=None):
    return {"rid": rid, "subject_id": s, "object_id": o, "predicate": pred,
            "subject": sname or f"E{s}", "s_type": "concept",
            "object": oname or f"E{o}", "o_type": "concept"}


def _db(kw_rows, rel_rows=None):
    """Mock connection routing each query by its SQL text."""
    cap = {"fetch": [], "execute": []}
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=True)

    async def fetch(sql, *params):
        cap["fetch"].append((sql, params))
        if "kg_relations" in sql:
            return rel_rows or []
        return kw_rows

    async def execute(sql, *params):
        cap["execute"].append((sql, params))

    conn.fetch = AsyncMock(side_effect=fetch)
    conn.execute = AsyncMock(side_effect=execute)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, cap, conn


async def _recall(ctx, query="kopi pagi Bandung", **kw):
    with patch("syne.memory.graph.get_connection", return_value=ctx):
        return await G.recall_graph(query, **kw)


# ── pure helpers ─────────────────────────────────────────────────────────

def test_query_words_prefers_long_words():
    assert G._query_words("aku suka kopi pagi") == ["suka", "kopi", "pagi"]
    assert G._query_words("aku dan dia") == ["aku", "dan", "dia"]
    assert G._query_words("ya ok") == []


def test_combine_scores_keyword_only():
    s = G._combine_entity_scores([{"id": 1, "kw": 2}, {"id": 2, "kw": 1}], [], n_words=4)
    assert s[1] == {"kw": 2, "sem": None, "score": 0.5}
    assert s[2]["score"] == 0.25


def test_rank_relations_best_end_then_other_end_then_oldest():
    scores = {1: {"score": 0.9}, 2: {"score": 0.5}, 3: {"score": 0.1}}
    rows = [_rel(10, 3, 9), _rel(11, 1, 9), _rel(12, 1, 2), _rel(13, 2, 9),
            _rel(12, 1, 2)]  # rid 12 reached from both ends: counted once
    ranked = G._rank_relations(rows, scores, limit=10)
    assert [r["rid"] for r, _ in ranked] == [12, 11, 13, 10]
    assert ranked[0][1] == 0.9
    assert len(G._rank_relations(rows, scores, limit=2)) == 2


def test_rank_relations_tie_prefers_older_relation():
    scores = {1: {"score": 0.5}}
    ranked = G._rank_relations([_rel(30, 1, 7), _rel(20, 1, 8)], scores, limit=5)
    assert [r["rid"] for r, _ in ranked] == [20, 30]


# ── recall_graph: keyword only ───────────────────────────────────────────

async def test_recall_is_keyword_only_and_logs():
    ctx, cap, _ = _db(
        kw_rows=[{"id": 1, "name": "kopi pagi", "kw": 2}, {"id": 2, "name": "Bandung", "kw": 1}],
        rel_rows=[_rel(5, 2, 9), _rel(6, 1, 9)],
    )
    lines = await _recall(ctx, log_scope=SCOPE)
    assert not any("<=>" in sql for sql, _ in cap["fetch"])
    assert lines[0].startswith("- E1 ") and lines[1].startswith("- E2 ")
    ins = [p for sql, p in cap["execute"] if "kg_recall_log" in sql]
    assert len(ins) == 1
    p = ins[0]
    assert p[0] == 51 and p[1] == "telegram" and p[2] == "3478983"
    assert p[6] is False                 # query_embedded
    assert p[7] is None and p[8] is None  # entity embedding counts unused
    assert p[9] == 2 and p[10] == 0      # kw / sem candidate counts
    rels = json.loads(p[12])
    assert [r["rid"] for r in rels] == [6, 5]
    assert rels[0]["s_kw"] == 2 and rels[0]["s_sem"] is None


async def test_no_log_scope_writes_no_log():
    ctx, cap, _ = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}], rel_rows=[_rel(7, 1, 9)])
    await _recall(ctx)
    assert not any("kg_recall_log" in sql for sql, _ in cap["execute"])


async def test_limit_caps_injected_relations():
    ctx, _, _ = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        rel_rows=[_rel(i, 1, 100 + i) for i in range(1, 30)],
    )
    assert len(await _recall(ctx, limit=10)) == 10


async def test_no_candidates_returns_empty_but_still_logs():
    ctx, cap, _ = _db(kw_rows=[])
    assert await _recall(ctx, log_scope=SCOPE) == []
    assert not any("kg_relations" in sql for sql, _ in cap["fetch"])
    assert any("kg_recall_log" in sql for sql, _ in cap["execute"])


async def test_log_failure_never_breaks_recall():
    ctx, _, conn = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}], rel_rows=[_rel(7, 1, 9)])

    async def boom(sql, *p):
        raise RuntimeError("relation kg_recall_log does not exist")

    conn.execute = AsyncMock(side_effect=boom)
    assert len(await _recall(ctx, log_scope=SCOPE)) == 1


# ── privacy (Rule 760/765) ───────────────────────────────────────────────

async def test_public_requester_relations_filtered_by_category():
    ctx, cap, _ = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}], rel_rows=[_rel(7, 1, 9)])
    with patch("syne.security._load_public_categories", AsyncMock(return_value=["fact"])):
        await _recall(ctx, requester_access_level="public")
    rel_sql, rel_params = [(s, p) for s, p in cap["fetch"] if "kg_relations" in s][0]
    assert "JOIN memory m ON m.id = r.source_memory_id" in rel_sql
    assert rel_params[-1] == ["fact"]


async def test_public_requester_without_public_categories_gets_nothing():
    ctx, cap, _ = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}])
    with patch("syne.security._load_public_categories", AsyncMock(return_value=[])):
        assert await _recall(ctx, requester_access_level="public") == []
    assert cap["fetch"] == []


# ── relation sources ─────────────────────────────────────────────────────

EXTRACTED = {
    "entities": [{"name": "Riyo", "type": "person"}, {"name": "Bandung", "type": "place"}],
    "relations": [{"subject": "Riyo", "predicate": "lives_in", "object": "Bandung"}],
}


async def _store(ctx, memory_id=5):
    ids = {"riyo": 11, "bandung": 12}
    with patch("syne.memory.graph.get_connection", return_value=ctx), \
         patch("syne.memory.graph._resolve_entity",
               AsyncMock(side_effect=lambda n, t, d: ids[n.lower()])):
        await G._store_graph(EXTRACTED, memory_id)


async def test_store_graph_records_relation_source():
    ctx, cap, conn = _db(kw_rows=[])
    conn.fetchval = AsyncMock(return_value=77)
    await _store(ctx)
    sql, *args = conn.fetchval.call_args.args
    assert "RETURNING id" in sql
    assert args == [11, "lives_in", 12, 5]
    src = [p for s, p in cap["execute"] if "kg_relation_sources" in s]
    assert src == [(77, 5)]


async def test_store_graph_source_failure_keeps_relation():
    ctx, _, conn = _db(kw_rows=[])
    conn.fetchval = AsyncMock(return_value=77)
    conn.execute = AsyncMock(side_effect=RuntimeError("no table"))
    await _store(ctx)  # must not raise
    conn.fetchval.assert_awaited_once()


# ── entity embedding removed ─────────────────────────────────────────────

def test_entity_embedding_code_removed():
    for name in ("_embed_entities", "_entity_embed_text", "_entity_embed_counts",
                 "_kg_count_cache", "_KG_SEM_CANDIDATES"):
        assert not hasattr(G, name), name
    assert "query_vector" not in inspect.signature(G.recall_graph).parameters
    assert "provider" not in inspect.signature(G._store_graph).parameters
    assert '"reembed-kg"' not in (ROOT / "syne/cli/cmd_memory.py").read_text()
    for p in ("syne/cli/cmd_repair.py", "syne/cli/helpers.py", "syne/db/schema.sql"):
        assert "ensure_kg_entities_hnsw_index" not in (ROOT / p).read_text(), p


# ── migrations ───────────────────────────────────────────────────────────

def test_migration_40_registered():
    from syne.db import migrations as M
    entry = next(m for m in M.MIGRATIONS if m[0] == 40)
    assert entry[1] is M._m40_kg_semantic_recall


def test_migration_42_registered():
    from syne.db import migrations as M
    assert M.MIGRATIONS[-1][0] == 42
    assert M.MIGRATIONS[-1][1] is M._m42_kg_relation_sources
    assert M.CURRENT_SCHEMA_VERSION == 42


async def test_migration_42_creates_backfills_and_drops():
    from syne.db import migrations as M
    conn = AsyncMock()
    await M._m42_kg_relation_sources(conn)
    sqls = " ".join(c.args[0] for c in conn.execute.call_args_list)
    assert "CREATE TABLE IF NOT EXISTS kg_relation_sources" in sqls
    assert "FROM kg_relations" in sqls and "source_memory_id IS NOT NULL" in sqls
    assert "DROP FUNCTION IF EXISTS ensure_kg_entities_hnsw_index()" in sqls


def test_schema_has_relation_sources():
    sql = (ROOT / "syne/db/schema.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS kg_relation_sources" in sql


# ── AI extraction and /graph removed ─────────────────────────────────────

def test_ai_extraction_removed():
    for name in ("extract_and_store", "reprocess_permanent_memories", "_extract_via_provider",
                 "_extract_via_ollama", "_parse_extraction", "EXTRACT_PROMPT", "_KG_SEMAPHORE"):
        assert not hasattr(G, name), name
    assert "extract_and_store" not in (ROOT / "syne/agent.py").read_text()


def test_graph_command_removed():
    src = (ROOT / "syne/communication/telegram.py").read_text()
    for needle in ('CommandHandler("graph"', "_cmd_graph", "_handle_graph_callback",
                   "_process_addgraph_input", "graph.enabled", "kg_processed"):
        assert needle not in src, needle


async def test_migration_42_deletes_graph_config():
    from syne.db import migrations as M
    conn = AsyncMock()
    await M._m42_kg_relation_sources(conn)
    sqls = " ".join(c.args[0] for c in conn.execute.call_args_list)
    assert "DELETE FROM config" in sqls and "'graph.enabled'" in sqls
    assert "graph.enabled" not in (ROOT / "syne/db/schema.sql").read_text()
