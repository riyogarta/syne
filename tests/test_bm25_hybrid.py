"""Hybrid search cosine + BM25 (Riyo, 18 Nov 2026) - KG-RAG stage 1, step 3."""
import json
import math
import os

import syne.db.migrations as M
from syne.memory import hybrid as H

_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "syne", "db", "schema.sql",
)


def _r(i, sim=0.5):
    return {"id": i, "similarity": sim}


def test_fuse_both_lists_win():
    cos = [_r(1, .9), _r(2, .8), _r(3, .7)]
    bm = [{"id": 3, "bm25": 9.0}, {"id": 4, "bm25": 5.0, "similarity": .2}]
    out = H.fuse(cos, bm, 60)
    assert [d["id"] for d in out] == [3, 1, 2, 4]
    assert out[0]["bm25"] == 9.0 and out[0]["rank_cosine"] == 3 and out[0]["rank_bm25"] == 1
    assert out[1]["bm25"] is None and out[1]["rank_bm25"] is None
    assert out[3]["rank_cosine"] is None
    assert math.isclose(out[0]["rrf"], 1 / 63 + 1 / 61)


def test_fuse_limit_and_cosine_only():
    cos = [_r(i, 1 - i / 10) for i in range(1, 6)]
    assert [d["id"] for d in H.fuse(cos, [], 60, 3)] == [1, 2, 3]


def test_fuse_tie_prefers_higher_similarity():
    out = H.fuse([_r(1, .4)], [{"id": 2, "similarity": .6, "bm25": 1.0}], 60)
    assert [d["id"] for d in out] == [2, 1]


def test_idf_rare_beats_common():
    assert H.idf(1000, 1) > H.idf(1000, 100) > H.idf(1000, 900) > 0


def test_tsquery_quotes():
    assert H.tsquery_text(["a", "it's", "b\\c"]) == "'a' | 'it''s' | 'b\\\\c'"


class _Conn:
    def __init__(self, stats, dfs):
        self.stats, self.dfs = stats, dfs

    async def fetchrow(self, sql, *a):
        return self.stats

    async def fetch(self, sql, *a):
        return [{"term": t, "df": d} for t, d in self.dfs.items()]


async def test_query_terms_drops_common_and_unknown():
    s = {k: v[0] for k, v in H.DEFAULTS.items()}
    s["bm25.max_df_ratio"] = 0.05
    conn = _Conn({"n_docs": 1000, "total_len": 50000},
                 {"dan": 900, "tayammum": 10, "air": 40, "zzz": 0, "cara": 60})
    q = await H.query_terms(conn, "x", "memory", s)
    assert q.terms == ["tayammum", "air"]          # rarest first; dan, cara (>5%), zzz (0) out
    assert q.avg_len == 50.0
    assert q.tsquery == "'tayammum' | 'air'"
    assert len(q.params(s)) == 6


async def test_query_terms_empty_corpus():
    s = {k: v[0] for k, v in H.DEFAULTS.items()}
    assert await H.query_terms(_Conn(None, {}), "x", "memory", s) is None
    assert await H.query_terms(_Conn({"n_docs": 0, "total_len": 0}, {}), "x", "memory", s) is None


def test_bm25_cte_placeholders():
    cte = H.bm25_cte("memory m", "m.category = $3", 2, 4)
    for ph in ("$2", "$3", "$4::text[]", "$5::float8[]", "$6::tsquery", "$7::float8", "$8::float8", "$9::float8"):
        assert ph in cte, ph
    assert "$10" not in cte


async def test_settings_clamped(monkeypatch):
    import syne.db.models as models
    store = {"bm25.k1": 99, "bm25.max_terms": "0", "bm25.recall_enabled": "false",
             "bm25.max_df_ratio": "abc"}

    async def gc(key, default=None):
        return store.get(key, default)
    monkeypatch.setattr(models, "get_config", gc)
    s = await H.load_settings()
    assert s["bm25.k1"] == 3.0 and s["bm25.max_terms"] == 1
    assert s["bm25.recall_enabled"] is False
    assert s["bm25.max_df_ratio"] == 0.05
    assert s["bm25.history_enabled"] is True


def test_migration_52_registered_and_seeds():
    assert (52, M._m52_bm25_search_config, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 52


async def test_migration_52_inserts_every_key():
    calls = []

    class C:
        async def execute(self, sql, *a):
            calls.append(a)
    await M._m52_bm25_search_config(C())
    assert {a[0] for a in calls} == set(H.DEFAULTS)
    for key, value, _ in calls:
        assert json.loads(value) == H.DEFAULTS[key][0]


def test_schema_sql_carries_same_seeds():
    with open(_SCHEMA) as f:
        schema = f.read()
    for key, (value, desc) in H.DEFAULTS.items():
        assert f"('{key}', '{json.dumps(value)}', '{desc}')" in schema, key
