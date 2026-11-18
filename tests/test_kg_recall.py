"""KG recall (Riyo, 18 Nov 2026) - KG-RAG stage 2, steps 2 and 5."""
import inspect
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

import syne.db.migrations as M
from syne.memory import kg_extract as K
from syne.memory import kg_recall as R

ERA = datetime(2026, 11, 18, 4, 45, tzinfo=timezone.utc)
_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "syne", "db", "schema.sql",
)


def _rel(rid, s, o, src_at, *, sub="A", obj="B", src_type="message", src_id=1, pred="studies"):
    return {"rid": rid, "subject_id": s, "object_id": o, "predicate": pred, "eid": s,
            "src_type": src_type, "src_id": src_id, "src_at": src_at,
            "subject": sub, "s_type": "person", "object": obj, "o_type": "concept"}


def _at(h):
    return datetime(2026, 11, 18, h, 0, tzinfo=timezone.utc)


# ── pure parts ────────────────────────────────────────────────────────────

def test_mention_grams_longest_first():
    g = R.mention_grams("anak Riyogarta Pratikto sekolah")
    assert g[0].count(" ") == 2
    assert "riyogarta pratikto" in g and "sekolah" in g


def test_rank_hop1_first_seed_pairs_first_unique():
    seeds = [1, 2]
    hop1 = [_rel(10, 1, 9, _at(5)), _rel(11, 1, 2, _at(4)), _rel(12, 2, 8, _at(6))]
    hop2 = [_rel(20, 9, 7, _at(9)), _rel(10, 1, 9, _at(5))]
    out = [int(r["rid"]) for r in R.rank(hop1, hop2, seeds, 10)]
    assert out == [11, 10, 12, 20]
    assert [int(r["rid"]) for r in R.rank(hop1, hop2, seeds, 2)] == [11, 10]


def test_fmt_line_has_source_and_local_date():
    r = _rel(1, 1, 2, datetime(2026, 11, 17, 20, 0, tzinfo=timezone.utc),
             sub="anak Riyogarta Pratikto", obj="Paket C", src_id=96492)
    line = R.fmt_line(r, "Asia/Jakarta")
    assert line == ("- anak Riyogarta Pratikto [person] --studies--> Paket C [concept] "
                    "(sumber: pesan #96492, 18 Nov 2026)")
    assert "memori #5" in R.fmt_line(_rel(1, 1, 2, _at(5), src_type="memory", src_id=5), "UTC")


def test_expand_sql_privacy_only_for_public():
    assert "$5" not in R.expand_sql(False) and "$6" not in R.expand_sql(False)
    pub = R.expand_sql(True)
    assert "m.category" in pub and "g.session_id = $6" in pub
    for sql in (pub, R.expand_sql(False), R._MATCH_SQL):
        assert "origin = 'llm'" in sql


async def test_settings_clamped(monkeypatch):
    vals = {"kg.graph_recall_limit": 999, "kg.graph_hops": 7, "kg.graph_per_entity": "x",
            "kg.graph_max_seeds": -3, "kg.extract_era_start": "2026-11-18T04:45:00+00:00",
            "system.timezone": "Asia/Jakarta"}

    async def gc(key, default=None):
        return vals.get(key, default)
    monkeypatch.setattr(R, "get_config", gc)
    s = await R.load_settings()
    assert (s["kg.graph_recall_limit"], s["kg.graph_hops"], s["kg.graph_per_entity"],
            s["kg.graph_max_seeds"]) == (50, 2, 8, 1)
    assert s["era"] == datetime(2026, 11, 18, 4, 45, tzinfo=timezone.utc)
    assert s["tz"] == "Asia/Jakarta"


# ── recall_kg with a fake connection ──────────────────────────────────────

class FakeConn:
    def __init__(self, match=(), mem=(), hop1=(), hop2=()):
        self.match, self.mem, self.hop1, self.hop2 = list(match), list(mem), list(hop1), list(hop2)
        self.calls = []

    async def fetch(self, sql, *a):
        self.calls.append((sql, a))
        if sql is R._MATCH_SQL:
            return self.match
        if sql is R._MEMORY_SEEDS_SQL:
            return self.mem
        if a[3]:          # skip list given -> hop 2
            return self.hop2
        return self.hop1


def _env(monkeypatch, conn, **cfg):
    vals = {"kg.extract_era_start": ERA.isoformat(), "system.timezone": "UTC"}
    vals.update(cfg)

    async def gc(key, default=None):
        return vals.get(key, default)

    @asynccontextmanager
    async def gconn(timeout=None):
        yield conn
    monkeypatch.setattr(R, "get_config", gc)
    monkeypatch.setattr(R, "get_connection", gconn)


async def test_recall_kg_two_hops(monkeypatch):
    conn = FakeConn(
        match=[{"id": 1, "norm_name": "riyogarta pratikto"}],
        mem=[{"source_id": 77, "subject_id": 3, "object_id": 4}],
        hop1=[_rel(10, 1, 9, _at(5), sub="Riyogarta Pratikto", obj="Paket C")],
        hop2=[_rel(20, 9, 8, _at(6), sub="Paket C", obj="PKBM")],
    )
    _env(monkeypatch, conn)
    lines = await R.recall_kg("Riyogarta Pratikto", [77, "x", 77])
    assert lines == [
        "- Riyogarta Pratikto [person] --studies--> Paket C [concept] (sumber: pesan #1, 18 Nov 2026)",
        "- Paket C [person] --studies--> PKBM [concept] (sumber: pesan #1, 18 Nov 2026)",
    ]
    hop1_call = conn.calls[2]
    assert hop1_call[1][0] == [1, 3, 4] and hop1_call[1][1] == ERA
    assert conn.calls[1][1][0] == [77]
    assert conn.calls[3][1][0] == [9] and conn.calls[3][1][3] == [10]


async def test_recall_kg_one_hop_and_off(monkeypatch):
    conn = FakeConn(match=[{"id": 1, "norm_name": "paket"}], hop1=[_rel(10, 1, 9, _at(5))])
    _env(monkeypatch, conn, **{"kg.graph_hops": 1})
    assert len(await R.recall_kg("paket", [])) == 1
    assert len(conn.calls) == 2

    conn2 = FakeConn(match=[{"id": 1, "norm_name": "paket"}])
    _env(monkeypatch, conn2, **{"kg.graph_recall_limit": 0})
    assert await R.recall_kg("paket", []) == [] and conn2.calls == []

    conn3 = FakeConn(match=[{"id": 1, "norm_name": "paket"}])
    _env(monkeypatch, conn3, **{"kg.extract_era_start": None})
    assert await R.recall_kg("paket", []) == [] and conn3.calls == []


async def test_recall_kg_public_passes_categories_and_session(monkeypatch):
    import syne.security as S
    conn = FakeConn(match=[{"id": 1, "norm_name": "paket"}], hop1=[_rel(10, 1, 9, _at(5))])
    _env(monkeypatch, conn, **{"kg.graph_hops": 1})

    async def cats():
        return ["fact"]
    monkeypatch.setattr(S, "_load_public_categories", cats)
    await R.recall_kg("paket", [], session_id=42, requester_access_level="public")
    sql, a = conn.calls[1]
    assert "g.session_id = $6" in sql and a[4] == ["fact"] and a[5] == 42


async def test_recall_kg_failure_is_quiet(monkeypatch):
    class Boom(FakeConn):
        async def fetch(self, sql, *a):
            raise RuntimeError("db down")
    _env(monkeypatch, Boom())
    assert await R.recall_kg("paket", [1]) == []


# ── extraction: technical names are not entities ──────────────────────────

@pytest.mark.parametrize("name", ["3560851", "c45f3c0", "#57757", "memori #57757",
                                  "pesan 96492", "/reasoning", "/models",
                                  "syne/memory/kg_extract.py", "~/.log-syne/syne.log",
                                  "https://github.com/x", "schema.sql"])
def test_technical_names_dropped(name):
    assert K.technical_name(name)


@pytest.mark.parametrize("name", ["Riyogarta Pratikto", "Paket C", "gemini-3.8-flash",
                                  "Telegram", "18 Nov 2026", "2026", "Fiqih Islam Wa Adillatuhu",
                                  "decade", "café"])
def test_real_names_kept(name):
    assert not K.technical_name(name)


def test_parse_extraction_drops_technical_entities():
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    s["kg.extract_predicates"] = dict(K.DEFAULT_PREDICATES)
    raw = json.dumps({
        "entities": [{"name": "3560851", "type": "item"}, {"name": "REPO DEV", "type": "concept"},
                     {"name": "Riyo", "type": "person"}, {"name": "Paket C", "type": "concept"}],
        "relations": [{"subject": "3560851", "predicate": "part_of", "object": "REPO DEV"},
                      {"subject": "Riyo", "predicate": "studies", "object": "Paket C"}]})
    out = K.parse_extraction(raw, s)
    assert out["relations"] == [{"subject": "Riyo", "predicate": "studies", "object": "Paket C"}]
    assert "commit hash" in K.build_prompt("x", speaker="Riyo", context="", s=s)


# ── migration / schema / wiring ───────────────────────────────────────────

def test_migration_54_registered():
    assert (54, M._m54_kg_recall_config, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 54


async def test_migration_54_seeds():
    calls = []

    class C:
        async def execute(self, sql, *a):
            calls.append(a)
    await M._m54_kg_recall_config(C())
    assert {c[0] for c in calls} == set(R.DEFAULTS)


def test_schema_sql_seeds():
    with open(_SCHEMA) as f:
        schema = f.read()
    for key, (value, desc) in R.DEFAULTS.items():
        assert f"('{key}', '{json.dumps(value)}', '{desc}')" in schema, key


def test_conversation_injects_kg_block():
    import syne.conversation as C
    src = inspect.getsource(C)
    assert "from .memory.kg_recall import recall_kg" in src
    assert "# Knowledge Graph (chat & memories)" in src
    assert src.index("_recall_kg(") < src.index("# Knowledge Graph (chat & memories)")


def test_version():
    import syne
    assert syne.__version__ == "1.23.01"
