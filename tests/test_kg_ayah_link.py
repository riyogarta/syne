"""Extracted Qur'an verse names join the 'ayah' entity (Riyo, 18 Nov 2026)."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from syne.memory import kg_extract as K
from syne.memory import kg_recall as R
from syne.memory import kg_refs as Q

AYAH_ROWS = [
    {"name": "QS 54:46", "description": "Al-Qamar 46"},
    {"name": "QS 54:55", "description": "Al-Qamar 55"},
    {"name": "QS 2:255", "description": "Al-Baqarah 255"},
    {"name": "QS 2:286", "description": "Al-Baqarah 286"},
    {"name": "QS 18:110", "description": "Al-Kahf 110"},
    {"name": "QS 94:8", "description": "Ash-Sharh 8"},
]


@pytest.fixture(autouse=True)
def _fresh_index(monkeypatch):
    monkeypatch.setitem(K._AYAH_INDEX, "index", None)
    monkeypatch.setitem(K._AYAH_INDEX, "at", 0.0)

    async def gc(key, default=None):
        return default
    monkeypatch.setattr(K, "get_config", gc)
    monkeypatch.setattr(R, "get_config", gc)


def _index():
    return Q.surah_index_from({54: "Al-Qamar", 2: "Al-Baqarah", 18: "Al-Kahf", 94: "Ash-Sharh"},
                              {54: 55, 2: 286, 18: 110, 94: 8}, Q.DEFAULT_SURAH_ALIASES)


# ── parser ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,ref", [
    ("QS 54:46", (54, 46)),
    ("QS. 2:255", (2, 255)),
    ("QS. Al-Qamar: 46", (54, 46)),
    ("qs al-qamar 46", (54, 46)),
    ("Al-Baqarah ayat 255", (2, 255)),
    ("Al Baqarah 255", (2, 255)),
    ("Surah Al-Kahfi 10", (18, 10)),
    ("Al-Insyirah 5", (94, 5)),
    ("QS. Al-Baqarah (2): 255", (2, 255)),
    ("QS. Al-Baqarah 2:255", (2, 255)),
])
def test_single_ayah_parsed(name, ref):
    assert Q.parse_single_ayah(name, _index()) == ref


@pytest.mark.parametrize("name", [
    "QS 2:255-257", "QS 54:99", "QS 115:1", "Windows 11", "Al-Baqarah", "Surah 5",
    "Tafsir Ibnu Katsir — Al-Qamar Ayat 46", "2026", "18 Nov 2026", "v1.22.101",
    "gemini-3.8-flash", "Paket C 3", "QS Al-Baqarah 2:300", "QS. Al-Qamar (2): 46",
])
def test_not_a_single_ayah(name):
    assert Q.parse_single_ayah(name, _index()) is None


def test_build_surah_index_unchanged():
    q = ["Al-Quran - Al-Qamar (القمر) - Bulan\nSurat ke-54, Makkiyah, Ayat 1-55"]
    i = Q.build_surah_index(q, {"Bulan": 54})
    j = Q.surah_index_from({54: "Al-Qamar"}, {54: 55}, {"Bulan": 54})
    assert (i.names, i.ayah_counts, i.keys) == (j.names, j.ayah_counts, j.keys)


# ── resolve_entity ────────────────────────────────────────────────────────

class _AConn:
    def __init__(self, ayah_id=132676):
        self.ayah_id, self.fetches, self.lookups = ayah_id, 0, []

    async def fetch(self, sql, *a):
        self.fetches += 1
        return AYAH_ROWS

    async def fetchval(self, sql, *a):
        if "entity_type = 'ayah'" in sql:
            self.lookups.append(a[0])
            return self.ayah_id
        return 77

    async def fetchrow(self, sql, *a):
        return None

    async def execute(self, sql, *a):
        pass


async def test_verse_name_joins_ayah_entity():
    c = _AConn()
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    assert await K.resolve_entity(c, "QS. Al-Qamar: 46", "concept", None, s) == (132676, "ayah")
    assert await K.resolve_entity(c, "QS 2:255", "item", None, s) == (132676, "ayah")
    assert c.lookups == ["QS 54:46", "QS 2:255"]
    assert c.fetches == 1                       # index cached


async def test_person_and_missing_ayah_are_new():
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    c = _AConn()
    assert await K.resolve_entity(c, "Yusuf 12", "person", None, s) == (77, "new")
    assert c.fetches == 0 and c.lookups == []
    c2 = _AConn(ayah_id=None)
    assert await K.resolve_entity(c2, "QS 2:255", "concept", None, s) == (77, "new")


async def test_ayah_index_failure_falls_back():
    class Boom(_AConn):
        async def fetch(self, sql, *a):
            raise RuntimeError("db down")
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    assert await K.resolve_entity(Boom(), "QS 2:255", "concept", None, s) == (77, "new")


# ── kg_recall: "QS ..." in the message seeds the ayah ─────────────────────

ERA = datetime(2026, 11, 18, 4, 45, tzinfo=timezone.utc)


async def test_recall_seeds_mentioned_ayah(monkeypatch):
    calls = []

    class C:
        async def fetch(self, sql, *a):
            calls.append((sql, a))
            if "description FROM kg_entities" in sql:
                return AYAH_ROWS
            if sql is R._AYAH_SQL:
                return [{"id": 132676, "name": "QS 54:46"}]
            if sql is R._MATCH_SQL:
                return []
            return [{"rid": 1, "subject_id": 314515, "object_id": 132676, "predicate": "mentions",
                     "eid": 132676, "src_type": "message", "src_id": 9, "src_at": ERA,
                     "subject": "Riyo", "s_type": "person", "object": "QS 54:46", "o_type": "ayah"}]

    vals = {"kg.extract_era_start": ERA.isoformat(), "kg.graph_hops": 1}

    async def gc(key, default=None):
        return vals.get(key, default)

    @asynccontextmanager
    async def gconn(timeout=None):
        yield C()
    monkeypatch.setattr(R, "get_config", gc)
    monkeypatch.setattr(R, "get_connection", gconn)
    lines = await R.recall_kg("apa kata QS Al-Qamar: 46 dan QS 2:1-286", [])
    ayah_call = next(a for sql, a in calls if sql is R._AYAH_SQL)
    assert ayah_call[0] == ["QS 54:46"]          # long range seeds nothing
    expand = [a for sql, a in calls if "unnest($1::int[])" in sql]
    assert expand[0][0] == [132676]
    assert lines and "QS 54:46 [ayah]" in lines[0]
