"""v1.23.02: any reference can be a KG link target (rules in kg.ref_sources)."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syne.memory import graph as G
from syne.memory import kg_build as B
from syne.memory import kg_extract as K
from syne.memory import kg_refs as R

POLIS = {
    "node_type": "pasal",
    "node_format": "Polis {unit} Pasal {item}",
    "node_regex": r"Polis (\d{1,3}) Pasal (\d{1,3})",
    "head": r"^Polis (?P<name>.+?) \(no\. (?P<unit>\d+)\)\nPasal (?P<a1>\d+)(?:-(?P<a2>\d+))?",
    "mention": r"\b(?i:polis)\s+(?P<name>[A-Za-z][A-Za-z ]{0,30}?)\s+(?i:pasal)\s+"
               r"(?P<a1>\d{1,3})(?:\s*-\s*(?P<a2>\d{1,3}))?(?!\d)",
    "aliases": {"Rawat Inap": 7},
}

SOURCES = {
    "alquran": {"kind": "quran"},
    "polis": {"target": POLIS},
    "catatan": {"mentions": True},
}


def _q(mid, surah, name, a1, a2):
    return {"id": mid, "category": "alquran",
            "content": f"Al-Quran - {name} (x) - y\nSurat ke-{surah}, Madaniyah, Ayat {a1}-{a2}\n\n..."}


ROWS = [
    _q(1, 2, "Al-Baqarah", 1, 30),
    {"id": 2, "category": "polis", "content": "Polis Kesehatan Keluarga (no. 7)\nPasal 1-20\n\n..."},
    {"id": 3, "category": "catatan",
     "content": "Klaim\n\nlihat polis kesehatan keluarga pasal 12, polis rawat inap pasal 3-4, "
                "QS. Al-Baqarah: 25, polis mobil pasal 2"},
]


def _rels(plan):
    return {(s[0], p, o[0]) for (s, p, o) in plan.relations}


def test_target_config_quran_defaults_and_legacy():
    t = R.target_config({"kind": "quran", "aliases": {"Gua": 18}})
    assert t["node_type"] == "ayah" and t["mention"] == R.DEFAULT_QS_PATTERN
    assert t["aliases"] == {"Gua": 18}
    assert R.target_config({"kind": "quran"}, "X")["mention"] == "X"
    own = R.target_config({"kind": "quran", "target": {"mention": "Y"}}, "X")
    assert own["mention"] == "Y"
    assert R.target_config({"mentions": True}) is None
    assert R.target_config({"target": {"node_type": "x"}}) is None          # incomplete
    assert R.target_config({"target": {**POLIS, "mention": "("}}) is None   # bad regex
    assert R.target_node_types(SOURCES) == ["ayah", "pasal"]


def test_plan_links_non_quran_target():
    plan = B.plan_links(ROWS, SOURCES)
    rels = _rels(plan)
    note = "klaim [#3]"
    assert ("polis kesehatan keluarga (no. 7) [#2]", "contains", "polis 7 pasal 12") in rels
    assert plan.per_predicate["contains"] == 30 + 20
    assert (note, "cites", "polis 7 pasal 12") in rels
    assert (note, "cites", "polis 7 pasal 3") in rels and (note, "cites", "polis 7 pasal 4") in rels
    assert (note, "cites", "qs 2:25") in rels                     # Qur'an unchanged
    assert plan.rejects[("catatan", "unknown_surah")] == 1        # "polis mobil"
    assert plan.entities[("polis 7 pasal 12", "pasal")] == \
        ("Polis 7 Pasal 12", "pasal", "Kesehatan Keluarga 12")


def test_plain_normalizer():
    i = R.build_target_index([ROWS[1]["content"]], R.target_config(SOURCES["polis"]))
    assert i.resolve("KESEHATAN-keluarga") == 7 and i.resolve("Kesehatan") is None
    assert R.parse_single_ayah("polis kesehatan keluarga pasal 12", i) == (7, 12)
    assert R.parse_single_ayah("polis kesehatan keluarga pasal 21", i) is None


# ── extraction joins the target node ─────────────────────────────────────

class _Conn:
    def __init__(self):
        self.lookups = []

    async def fetch(self, sql, *a):
        if "WHERE entity_type = $1" in sql:
            if a[0] == "pasal":
                return [{"name": "Polis 7 Pasal 20", "description": "Kesehatan Keluarga 20"}]
            return [{"name": "QS 2:286", "description": "Al-Baqarah 286"}]
        if "name = ANY" in sql:
            return [{"id": 5000 + int(n.rsplit(" ", 1)[-1]), "name": n} for n in a[0]]
        return []

    async def fetchval(self, sql, *a):
        if "WHERE name = $1 AND entity_type = $2" in sql:
            self.lookups.append((a[0], a[1]))
            return 4242
        return 77

    async def fetchrow(self, sql, *a):
        return None

    async def execute(self, sql, *a):
        pass


@pytest.fixture
def _cfg(monkeypatch):
    monkeypatch.setitem(K._AYAH_INDEX, "index", None)
    monkeypatch.setitem(K._AYAH_INDEX, "at", 0.0)
    store = {"kg.ref_sources": SOURCES}

    async def gc(key, default=None):
        return store.get(key, default)
    monkeypatch.setattr(K, "get_config", gc)
    return store


async def test_resolve_entity_joins_non_quran_node(_cfg):
    c = _Conn()
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    assert await K.resolve_entity(c, "Polis Kesehatan Keluarga pasal 12", "concept", None, s) \
        == (4242, "ayah")
    assert await K.resolve_entity(c, "QS 2:255", "concept", None, s) == (4242, "ayah")
    assert c.lookups == [("Polis 7 Pasal 12", "pasal"), ("QS 2:255", "ayah")]
    assert "pasal" in K.reference_types()


async def test_resolve_span_non_quran(_cfg):
    c = _Conn()
    ids = await K.resolve_ref_span(c, "Polis Kesehatan Keluarga pasal 3-5")
    assert ids == [5003, 5004, 5005]


# ── chunk-link recall follows any target node ────────────────────────────

def _rel(rid, s, sn, st, p, o, on, ot):
    return {"rid": rid, "subject_id": s, "object_id": o, "predicate": p,
            "subject": sn, "s_type": st, "object": on, "o_type": ot}


async def test_chunk_links_hop2_through_pasal():
    hop1 = [_rel(1, 100, "Klaim [#3]", "chunk", "cites", 900, "Polis 7 Pasal 12", "pasal")]
    hop2 = [_rel(2, 200, "Polis Kesehatan Keluarga [#2]", "chunk", "contains",
                 900, "Polis 7 Pasal 12", "pasal")]
    cap = []
    conn = MagicMock()

    async def fetch(sql, *params):
        cap.append(sql)
        if "description = ANY" in sql:
            return [{"id": 100, "description": "memory:3"}]
        if "WITH ORDINALITY" in sql:
            return hop2
        return hop1
    conn.fetch = AsyncMock(side_effect=fetch)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    async def gc(key, default=None):
        return SOURCES if key == "kg.ref_sources" else default
    with patch("syne.memory.graph.get_connection", return_value=ctx), \
         patch("syne.memory.graph.get_config", gc):
        out = await G.recall_chunk_links([3], limit=10)
    assert any("WITH ORDINALITY" in s for s in cap)
    assert out[1] == "- Polis Kesehatan Keluarga [#2] [chunk] --contains--> Polis 7 Pasal 12 [pasal]"


async def test_target_types_fallback_and_sanitized():
    async def gc(key, default=None):
        return {"x": {"target": {**POLIS, "node_type": "bad'); drop"}}}
    with patch("syne.memory.graph.get_config", gc):
        assert await G._target_types() == ["ayah"]


# ── migration 55 ─────────────────────────────────────────────────────────

async def test_migration_55_moves_qs_pattern_and_aliases():
    from syne.db import migrations as M
    assert (55, M._m55_kg_link_targets, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 55
    old_src = {"alquran": {"kind": "quran", "aliases": {"Gua": 18}},
               "bukhari": {"mentions": True}}
    vals = {"kg.qs_pattern": json.dumps("OWNER-EDIT"), "kg.ref_sources": json.dumps(old_src)}
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=lambda sql: next(
        v for k, v in vals.items() if f"'{k}'" in sql))
    conn.execute = AsyncMock()
    await M._m55_kg_link_targets(conn)
    calls = conn.execute.call_args_list
    upd = next(c for c in calls if "SET value" in c.args[0])
    new = json.loads(upd.args[1])
    t = new["alquran"]["target"]
    assert t["mention"] == "OWNER-EDIT" and t["aliases"] == {"Gua": 18}
    assert "aliases" not in new["alquran"] and new["bukhari"] == {"mentions": True}
    assert R.target_config(new["alquran"]) == {**R.quran_target("OWNER-EDIT", {"Gua": 18})}
    assert "DELETE FROM config WHERE key = 'kg.qs_pattern'" in calls[-1].args[0]


async def test_migration_55_idempotent():
    from syne.db import migrations as M
    src = {"alquran": {"kind": "quran", "target": R.quran_target()}}
    conn = MagicMock()
    conn.fetchval = AsyncMock(side_effect=lambda sql: None if "qs_pattern" in sql
                              else json.dumps(src))
    conn.execute = AsyncMock()
    await M._m55_kg_link_targets(conn)
    assert not any("SET value" in c.args[0] for c in conn.execute.call_args_list)


def test_schema_has_no_qs_pattern():
    import pathlib
    sql = (pathlib.Path(__file__).parent.parent / "syne/db/schema.sql").read_text()
    assert "('kg.qs_pattern'" not in sql
