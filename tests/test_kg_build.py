"""Rule-built KG, step 2: link planning (syne/memory/kg_build.py)."""

from unittest.mock import AsyncMock, MagicMock

from syne.memory import kg_build as B


def _q(mid, surah, name, a1, a2):
    return {"id": mid, "category": "alquran",
            "content": f"Al-Quran - {name} (x) - y\nSurat ke-{surah}, Madaniyah, Ayat {a1}-{a2}\n\n..."}


ROWS = [
    _q(1, 2, "Al-Baqarah", 1, 30),
    _q(2, 18, "Al-Kahf", 61, 90),
    {"id": 10, "category": "ibnukatsir",
     "content": "Tafsir Ibnu Katsir — Al Baqarah Ayat 27 (bagian 1/2)\n\nlihat (QS. Al-Kahfi: 71-73)"},
    {"id": 11, "category": "ibnukatsir",
     "content": "Tafsir Ibnu Katsir — Al Baqarah Ayat 27 (bagian 2/2)\n\nQS Al-Baqarah 27 lagi"},
    {"id": 20, "category": "bukhari",
     "content": "Shahih Bukhari — Hadits 1\n\n(QS Bukan Surah 5) dan QS. Al-Baqarah: 25"},
    {"id": 30, "category": "pribadi", "content": "QS Al-Baqarah 1"},
]

SOURCES = {
    "alquran": {"kind": "quran"},
    "ibnukatsir": B.DEFAULT_REF_SOURCES["ibnukatsir"],
    "bukhari": {"mentions": True},
}


def _rels(plan):
    return {(s[0], p, o[0]) for (s, p, o) in plan.relations}


def test_plan_links():
    plan = B.plan_links(ROWS, SOURCES)
    rels = _rels(plan)
    c10 = "tafsir ibnu katsir — al baqarah ayat 27 (bagian 1/2) [#10]"
    c11 = "tafsir ibnu katsir — al baqarah ayat 27 (bagian 2/2) [#11]"
    c20 = "shahih bukhari — hadits 1 [#20]"
    # Qur'an chunks contain their verses
    assert ("al-quran - al-baqarah (x) - y [#1]", "contains", "qs 2:27") in rels
    assert plan.per_predicate["contains"] == 30 + 30
    # tafsir title -> verse
    assert (c10, "explains", "qs 2:27") in rels
    # mention in text, range split per verse
    for a in (71, 72, 73):
        assert (c10, "cites", f"qs 18:{a}") in rels
    # a mention equal to the title verse is not repeated as 'cites'
    assert (c11, "cites", "qs 2:27") not in rels
    # parts chained
    assert (c10, "next_part", c11) in rels
    # hadith mention linked, unknown surah rejected
    assert (c20, "cites", "qs 2:25") in rels
    assert plan.rejects[("bukhari", "unknown_surah")] == 1
    # categories outside ref sources are ignored
    assert not any("[#30]" in s for (s, _p, _o) in rels)
    assert plan.chunks == 5


def test_chunk_and_ayah_entities():
    plan = B.plan_links(ROWS, SOURCES)
    assert plan.entities[("qs 2:27", "ayah")] == ("QS 2:27", "ayah", "Al-Baqarah 27")
    name, etype, desc = plan.entities[("shahih bukhari — hadits 1 [#20]", "chunk")]
    assert etype == "chunk" and desc == "memory:20"


def test_range_cap():
    rows = ROWS[:2] + [{"id": 40, "category": "bukhari",
                        "content": "H\n\nQS. Al-Baqarah: 1-30"}]
    plan = B.plan_links(rows, SOURCES, max_range=10)
    assert plan.rejects[("bukhari", "range_too_long")] == 1
    assert not any(p == "cites" for (_s, p, _o) in plan.relations)


def test_aliases_from_config():
    rows = ROWS[:2] + [{"id": 41, "category": "bukhari", "content": "H\n\nQS Gua 65"}]
    assert B.plan_links(rows, SOURCES).per_predicate["cites"] == 0
    assert B.plan_links(rows, SOURCES, aliases={"Gua": 18}).per_predicate["cites"] == 1


def test_idempotent_plan():
    a = B.plan_links(ROWS, SOURCES)
    b = B.plan_links(ROWS, SOURCES)
    assert a.relations == b.relations and a.entities == b.entities


async def test_write_plan_sql():
    plan = B.plan_links(ROWS[:1], SOURCES)
    conn = MagicMock()
    ent_rows = [{"id": i + 1, "lname": k[0], "entity_type": k[1]}
                for i, k in enumerate(plan.entities)]
    rel_rows = [{"id": 100 + i, "source_memory_id": 1} for i in range(len(plan.relations))]
    conn.fetch = AsyncMock(side_effect=[ent_rows, rel_rows])
    conn.execute = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    await B.write_plan(conn, plan)
    sqls = [c.args[0] for c in conn.fetch.call_args_list] + \
           [c.args[0] for c in conn.execute.call_args_list]
    assert "INSERT INTO kg_entities" in sqls[0]
    assert "INSERT INTO kg_relations" in sqls[1]
    assert "'rule'" in sqls[2] and "kg_relation_sources" in sqls[2]
    assert len(conn.execute.call_args_list[0].args[1]) == 30


def test_unlinked_chunks_get_no_entity():
    rows = ROWS[:1] + [{"id": 50, "category": "bukhari", "content": "Hadits tanpa rujukan\n\nisi"}]
    plan = B.plan_links(rows, SOURCES)
    assert not any("[#50]" in k[0] for k in plan.entities)
    assert plan.chunks == 2
    used = {k for rel in plan.relations for k in (rel[0], rel[2])}
    assert set(plan.entities) == used
