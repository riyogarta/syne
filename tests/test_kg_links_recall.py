"""v1.22.65: KG rule config in DB, per-reference aliases, chunk-link recall."""

import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

from syne.memory import graph as G
from syne.memory import kg_build as B
from syne.memory import kg_refs as R


def _q(mid, surah, name, a1, a2):
    return {"id": mid, "category": "alquran",
            "content": f"Al-Quran - {name} (x) - y\nSurat ke-{surah}, Madaniyah, Ayat {a1}-{a2}\n\n..."}


QROWS = [_q(1, 3, "Ali 'Imran", 1, 200), _q(2, 4, "An-Nisa'", 1, 176),
         _q(3, 18, "Al-Kahf", 61, 90), _q(4, 31, "Luqman", 1, 34),
         _q(5, 32, "As-Sajdah", 1, 30), _q(6, 49, "Al-Hujurat", 1, 18)]


# ── config + aliases ─────────────────────────────────────────────────────

def test_default_config_keys():
    cfg = B.default_config()
    assert set(cfg) == {"kg.ref_sources", "kg.qs_pattern", "kg.part_pattern",
                        "kg.max_mention_range", "kg.chunk_link_limit"}
    assert cfg["kg.ref_sources"][0]["alquran"]["aliases"] == R.DEFAULT_SURAH_ALIASES
    for v, d in cfg.values():
        json.dumps(v)
        assert d


def test_new_aliases_link():
    i = R.build_surah_index((r["content"] for r in QROWS), R.DEFAULT_SURAH_ALIASES)
    for name, s in [("Alu 'Imran", 3), ("Qn-Nisa", 4), ("Lukman", 31),
                    ("As Sajadah", 32), ("Al Hujarat", 49)]:
        assert i.resolve(name) == s, name


def test_aliases_come_from_reference_entry():
    rows = QROWS + [{"id": 9, "category": "bukhari", "content": "H\n\nQS Gua 65 dan QS Al-Kahfi 66"}]
    own = {"alquran": {"kind": "quran", "aliases": {"Gua": 18}}, "bukhari": {"mentions": True}}
    plan = B.plan_links(rows, own)
    objs = {o[0] for (_s, p, o) in plan.relations if p == "cites"}
    assert objs == {"qs 18:65"}          # config list replaces the code list
    fallback = {"alquran": {"kind": "quran"}, "bukhari": {"mentions": True}}
    objs = {o[0] for (_s, p, o) in B.plan_links(rows, fallback).relations if p == "cites"}
    assert objs == {"qs 18:66"}          # no 'aliases' -> code list


async def test_migration_43_seeds_without_overwrite():
    from syne.db import migrations as M
    assert M.MIGRATIONS[-1][0] == 43 and M.MIGRATIONS[-1][1] is M._m43_seed_kg_rule_config
    assert M.CURRENT_SCHEMA_VERSION == 43
    conn = MagicMock()
    conn.execute = AsyncMock()
    await M._m43_seed_kg_rule_config(conn)
    calls = conn.execute.call_args_list
    assert len(calls) == 5
    for c in calls:
        assert "ON CONFLICT (key) DO NOTHING" in c.args[0]
        assert json.loads(c.args[2]) == B.default_config()[c.args[1]][0]


def test_schema_sql_seeds_same_values():
    sql = (pathlib.Path(__file__).parent.parent / "syne/db/schema.sql").read_text()
    for k, (v, _d) in B.default_config().items():
        lit = "'" + json.dumps(v, ensure_ascii=False).replace("'", "''") + "'"
        assert f"('{k}', {lit}," in sql, k


# ── recall_chunk_links ───────────────────────────────────────────────────

def _rel(rid, s, sn, st, p, o, on, ot):
    return {"rid": rid, "subject_id": s, "object_id": o, "predicate": p,
            "subject": sn, "s_type": st, "object": on, "o_type": ot}


def _ctx(seeds, hop1, hop2):
    cap = []
    conn = MagicMock()

    async def fetch(sql, *params):
        cap.append((sql, params))
        if "description = ANY" in sql:
            return seeds
        if "WITH ORDINALITY" in sql:
            return hop2
        return hop1

    conn.fetch = AsyncMock(side_effect=fetch)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, cap


SEEDS = [{"id": 100, "description": "memory:10"}]
HOP1 = [
    _rel(5, 100, "Tafsir 2:27 (bagian 1/2) [#10]", "chunk", "next_part", 101, "Tafsir 2:27 (bagian 2/2) [#11]", "chunk"),
    _rel(6, 100, "Tafsir 2:27 (bagian 1/2) [#10]", "chunk", "explains", 900, "QS 2:27", "ayah"),
]
HOP2 = [
    _rel(7, 200, "Al-Quran - Al-Baqarah [#1]", "chunk", "contains", 900, "QS 2:27", "ayah"),
    _rel(8, 300, "Shahih Bukhari [#20]", "chunk", "cites", 900, "QS 2:27", "ayah"),
]


async def _links(ctx, ids, **kw):
    with patch("syne.memory.graph.get_connection", return_value=ctx):
        return await G.recall_chunk_links(ids, **kw)


async def test_chunk_links_order_and_limit():
    ctx, cap = _ctx(SEEDS, HOP1, HOP2)
    out = await _links(ctx, [10, 10, "x"], limit=10)
    assert out == [
        "- Tafsir 2:27 (bagian 1/2) [#10] [chunk] --explains--> QS 2:27 [ayah]",
        "- Al-Quran - Al-Baqarah [#1] [chunk] --contains--> QS 2:27 [ayah]",
        "- Shahih Bukhari [#20] [chunk] --cites--> QS 2:27 [ayah]",
        "- Tafsir 2:27 (bagian 1/2) [#10] [chunk] --next_part--> Tafsir 2:27 (bagian 2/2) [#11] [chunk]",
    ]
    assert cap[0][1][0] == ["memory:10"]
    assert cap[2][1][0] == [900] and cap[2][1][2] == [100]
    ctx, _ = _ctx(SEEDS, HOP1, HOP2)
    assert len(await _links(ctx, [10], limit=2)) == 2


async def test_chunk_links_no_seed_or_off():
    ctx, cap = _ctx([], HOP1, HOP2)
    assert await _links(ctx, [10]) == [] and len(cap) == 1
    ctx, cap = _ctx(SEEDS, HOP1, HOP2)
    assert await _links(ctx, [10], limit=0) == [] and cap == []
    assert await _links(ctx, []) == []


async def test_chunk_links_public_filter():
    ctx, cap = _ctx(SEEDS, HOP1, HOP2)
    with patch("syne.security._load_public_categories", AsyncMock(return_value=["islam"])):
        await _links(ctx, [10], requester_access_level="public")
    assert "JOIN memory m" in cap[1][0] and "ANY($3)" in cap[1][0] and cap[1][1][-1] == ["islam"]
    assert "JOIN memory m" in cap[2][0] and "ANY($4)" in cap[2][0] and cap[2][1][-1] == ["islam"]
    ctx, cap = _ctx(SEEDS, HOP1, HOP2)
    with patch("syne.security._load_public_categories", AsyncMock(return_value=[])):
        assert await _links(ctx, [10], requester_access_level="public") == []
    assert cap == []


def test_conversation_wires_chunk_links():
    src = (pathlib.Path(__file__).parent.parent / "syne/conversation.py").read_text()
    assert "recall_chunk_links" in src and "kg.chunk_link_limit" in src
