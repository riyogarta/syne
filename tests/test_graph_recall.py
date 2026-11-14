"""Graph recall: keyword + semantic candidates, ranked relations, kg_recall_log."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syne.memory import graph as G


@pytest.fixture(autouse=True)
def _reset_count_cache():
    G._kg_count_cache.update(t=0.0, embedded=None, total=None)
    yield
    G._kg_count_cache.update(t=0.0, embedded=None, total=None)


def _rel(rid, s, o, pred="rel", sname=None, oname=None):
    return {"rid": rid, "subject_id": s, "object_id": o, "predicate": pred,
            "subject": sname or f"E{s}", "s_type": "concept",
            "object": oname or f"E{o}", "o_type": "concept"}


def _db(kw_rows, sem_rows=None, rel_rows=None, embedded=0, total=100,
        sem_error=None):
    """Mock connection routing each query by its SQL text."""
    cap = {"fetch": [], "execute": []}
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=True)
    conn.fetchrow = AsyncMock(return_value={"embedded": embedded, "total": total})

    async def fetch(sql, *params):
        cap["fetch"].append((sql, params))
        if "<=>" in sql:
            if sem_error:
                raise sem_error
            return sem_rows or []
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
    return ctx, cap


async def _recall(ctx, query="kopi pagi Bandung", **kw):
    with patch("syne.memory.graph.get_connection", return_value=ctx):
        return await G.recall_graph(query, **kw)


SCOPE = {"session_id": 51, "platform": "telegram", "chat_id": "3478983"}


# ── pure helpers ─────────────────────────────────────────────────────────

def test_query_words_prefers_long_words():
    assert G._query_words("aku suka kopi pagi") == ["suka", "kopi", "pagi"]
    assert G._query_words("aku dan dia") == ["aku", "dan", "dia"]
    assert G._query_words("ya ok") == []


def test_entity_embed_text():
    assert G._entity_embed_text("Mochi", "animal", "kucing Riyo") == "Mochi (animal): kucing Riyo"
    assert G._entity_embed_text("Mochi", "animal", "") == "Mochi (animal)"
    assert G._entity_embed_text("Mochi") == "Mochi"


def test_combine_scores_keeps_both_raw_values():
    s = G._combine_entity_scores(
        [{"id": 1, "kw": 2}, {"id": 2, "kw": 1}],
        [{"id": 2, "sim": 0.9}, {"id": 3, "sim": 0.4}],
        n_words=4,
    )
    assert s[1] == {"kw": 2, "sem": None, "score": 0.5}
    assert s[2]["kw"] == 1 and s[2]["sem"] == 0.9 and s[2]["score"] == 0.9
    assert s[3]["kw"] == 0 and s[3]["score"] == 0.4


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


# ── recall_graph: embedding empty / partial / failing ────────────────────

async def test_no_embeddings_skips_semantic_and_ranks_by_keyword():
    ctx, cap = _db(
        kw_rows=[{"id": 1, "name": "kopi pagi", "kw": 2}, {"id": 2, "name": "Bandung", "kw": 1}],
        rel_rows=[_rel(5, 2, 9), _rel(6, 1, 9)],
        embedded=0,
    )
    lines = await _recall(ctx, query_vector=[0.1] * 4, log_scope=SCOPE)
    assert not any("<=>" in sql for sql, _ in cap["fetch"])
    assert lines[0].startswith("- E1 ") and lines[1].startswith("- E2 ")
    ins = [p for sql, p in cap["execute"] if "kg_recall_log" in sql]
    assert len(ins) == 1
    p = ins[0]
    assert p[0] == 51 and p[1] == "telegram" and p[2] == "3478983"
    assert p[6] is True          # query_embedded
    assert p[7] == 0 and p[8] == 100   # entities_embedded / total
    assert p[9] == 2 and p[10] == 0    # kw / sem candidate counts
    rels = json.loads(p[12])
    assert [r["rid"] for r in rels] == [6, 5]
    assert rels[0]["s_kw"] == 2 and rels[0]["s_sem"] is None


async def test_partial_embeddings_semantic_candidate_can_win():
    ctx, cap = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        sem_rows=[{"id": 4, "name": "Gesha Janson", "sim": 0.82}],
        rel_rows=[_rel(7, 1, 9), _rel(8, 4, 9, sname="Gesha Janson")],
        embedded=10,
    )
    lines = await _recall(ctx, query_vector=[0.1] * 4, log_scope=SCOPE)
    assert any("<=>" in sql for sql, _ in cap["fetch"])
    assert lines[0].startswith("- Gesha Janson ")
    p = [p for sql, p in cap["execute"] if "kg_recall_log" in sql][0]
    assert p[10] == 1
    cands = json.loads(p[11])
    assert cands[0]["id"] == 4 and cands[0]["sem"] == 0.82 and cands[0]["kw"] == 0


async def test_semantic_failure_falls_back_to_keyword():
    ctx, cap = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        rel_rows=[_rel(7, 1, 9)],
        embedded=10,
        sem_error=RuntimeError("different vector dimensions"),
    )
    lines = await _recall(ctx, query_vector=[0.1] * 4)
    assert len(lines) == 1 and lines[0].startswith("- E1 ")


async def test_full_embeddings_without_query_vector_is_keyword_only():
    ctx, cap = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        rel_rows=[_rel(7, 1, 9)],
        embedded=100,
    )
    lines = await _recall(ctx, query_vector=None, log_scope=SCOPE)
    assert not any("<=>" in sql for sql, _ in cap["fetch"])
    assert len(lines) == 1
    p = [p for sql, p in cap["execute"] if "kg_recall_log" in sql][0]
    assert p[6] is False


async def test_no_log_scope_writes_no_log():
    ctx, cap = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}], rel_rows=[_rel(7, 1, 9)])
    await _recall(ctx)
    assert not any("kg_recall_log" in sql for sql, _ in cap["execute"])


async def test_limit_caps_injected_relations():
    ctx, _ = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        rel_rows=[_rel(i, 1, 100 + i) for i in range(1, 30)],
    )
    assert len(await _recall(ctx, limit=10)) == 10


async def test_no_candidates_returns_empty_but_still_logs():
    ctx, cap = _db(kw_rows=[])
    assert await _recall(ctx, log_scope=SCOPE) == []
    assert not any("kg_relations" in sql for sql, _ in cap["fetch"])
    assert any("kg_recall_log" in sql for sql, _ in cap["execute"])


async def test_log_failure_never_breaks_recall():
    ctx, cap = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}], rel_rows=[_rel(7, 1, 9)])
    conn = await ctx.__aenter__()

    async def boom(sql, *p):
        raise RuntimeError("relation kg_recall_log does not exist")

    conn.execute = AsyncMock(side_effect=boom)
    assert len(await _recall(ctx, log_scope=SCOPE)) == 1


# ── privacy (Rule 760/765) still applies to semantic candidates ──────────

async def test_public_requester_relations_filtered_by_category():
    ctx, cap = _db(
        kw_rows=[{"id": 1, "name": "kopi", "kw": 1}],
        sem_rows=[{"id": 4, "name": "x", "sim": 0.9}],
        rel_rows=[_rel(7, 1, 9)],
        embedded=10,
    )
    with patch("syne.security._load_public_categories", AsyncMock(return_value=["fact"])):
        await _recall(ctx, requester_access_level="public", query_vector=[0.1] * 4)
    rel_sql, rel_params = [(s, p) for s, p in cap["fetch"] if "kg_relations" in s][0]
    assert "JOIN memory m ON m.id = r.source_memory_id" in rel_sql
    assert rel_params[-1] == ["fact"]


async def test_public_requester_without_public_categories_gets_nothing():
    ctx, cap = _db(kw_rows=[{"id": 1, "name": "kopi", "kw": 1}])
    with patch("syne.security._load_public_categories", AsyncMock(return_value=[])):
        assert await _recall(ctx, requester_access_level="public") == []
    assert cap["fetch"] == []


# ── entity embedding at extraction time ──────────────────────────────────

async def test_embed_entities_skips_failures(mock_provider):
    ctx, cap = _db(kw_rows=[])
    conn = await ctx.__aenter__()
    conn.fetch = AsyncMock(return_value=[
        {"id": 1, "name": "A", "entity_type": "person", "description": ""},
        {"id": 2, "name": "B", "entity_type": "place", "description": "kota"},
    ])
    good = MagicMock(vector=[0.1, 0.2])
    mock_provider.embed = AsyncMock(side_effect=[RuntimeError("down"), good])
    with patch("syne.memory.graph.get_connection", return_value=ctx):
        n = await G._embed_entities(mock_provider, [1, 2])
    assert n == 1
    upd = [p for sql, p in cap["execute"] if "UPDATE kg_entities" in sql]
    assert upd == [("[0.1, 0.2]", 2)]
    assert mock_provider.embed.call_args_list[1].args[0] == "B (place): kota"


async def test_store_graph_embeds_touched_entities(mock_provider):
    extracted = {
        "entities": [{"name": "Riyo", "type": "person"}, {"name": "Bandung", "type": "place"}],
        "relations": [{"subject": "Riyo", "predicate": "lives_in", "object": "Bandung"}],
    }
    ctx, _ = _db(kw_rows=[])
    ids = {"riyo": 11, "bandung": 12}
    with patch("syne.memory.graph.get_connection", return_value=ctx), \
         patch("syne.memory.graph._resolve_entity",
               AsyncMock(side_effect=lambda n, t, d: ids[n.lower()])), \
         patch("syne.memory.graph._embed_entities", new_callable=AsyncMock) as emb:
        await G._store_graph(extracted, 5, provider=mock_provider)
    emb.assert_awaited_once_with(mock_provider, [11, 12])


async def test_store_graph_without_provider_does_not_embed():
    extracted = {"entities": [{"name": "A", "type": "person"}, {"name": "B", "type": "place"}],
                 "relations": [{"subject": "A", "predicate": "x", "object": "B"}]}
    ctx, _ = _db(kw_rows=[])
    with patch("syne.memory.graph.get_connection", return_value=ctx), \
         patch("syne.memory.graph._resolve_entity", AsyncMock(return_value=1)), \
         patch("syne.memory.graph._embed_entities", new_callable=AsyncMock) as emb:
        await G._store_graph(extracted, 5)
    emb.assert_not_awaited()


# ── migration ────────────────────────────────────────────────────────────

def test_migration_40_registered():
    from syne.db import migrations as M
    assert M.MIGRATIONS[-1][0] == 40
    assert M.MIGRATIONS[-1][1] is M._m40_kg_semantic_recall
    assert M.CURRENT_SCHEMA_VERSION == 40
