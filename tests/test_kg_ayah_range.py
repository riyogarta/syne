"""Extracted verse ranges: one link per ayah (decision 14 Nov 2026)."""
import pytest

from syne.memory import kg_extract as K
from syne.memory import kg_refs as Q

AYAH_ROWS = [
    {"name": "QS 2:286", "description": "Al-Baqarah 286"},
    {"name": "QS 18:110", "description": "Al-Kahf 110"},
    {"name": "QS 54:55", "description": "Al-Qamar 55"},
]
CFG: dict = {}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setitem(K._AYAH_INDEX, "index", None)
    monkeypatch.setitem(K._AYAH_INDEX, "at", 0.0)
    CFG.clear()

    async def gc(key, default=None):
        return CFG.get(key, default)
    monkeypatch.setattr(K, "get_config", gc)


def _index():
    return Q.surah_index_from({2: "Al-Baqarah", 18: "Al-Kahf", 54: "Al-Qamar"},
                              {2: 286, 18: 110, 54: 55}, Q.DEFAULT_SURAH_ALIASES)


def test_parse_ayah_ref_range():
    i = _index()
    r = Q.parse_ayah_ref("QS 18:71-73", i)
    assert (r.surah, r.start, r.end) == (18, 71, 73)
    r = Q.parse_ayah_ref("QS. Al-Baqarah: 255-257", i)
    assert (r.surah, r.start, r.end) == (2, 255, 257)
    assert Q.parse_ayah_ref("QS 2:255-300", i) is None
    assert Q.parse_ayah_ref("QS 2:257-255", i) is None
    assert Q.parse_single_ayah("QS 18:71-73", i) is None
    assert Q.parse_single_ayah("QS 18:71", i) == (18, 71)


class WConn:
    def __init__(self):
        self.rels, self.next_entity = [], 500

    async def fetch(self, sql, *a):
        if "description FROM kg_entities" in sql:
            return AYAH_ROWS
        if "name = ANY" in sql:
            return [{"id": 1000 + int(n.split(":")[1]), "name": n} for n in a[0]]
        return []

    async def fetchval(self, sql, *a):
        if "INSERT INTO kg_relations" in sql:
            self.rels.append((a[0], a[2]))
            return len(self.rels)
        if "INSERT INTO kg_entities" in sql:
            self.next_entity += 1
            return self.next_entity
        return None

    async def fetchrow(self, sql, *a):
        return None

    async def execute(self, sql, *a):
        pass


def _s():
    return {k: v[0] for k, v in K.DEFAULTS.items()}


def _x(name):
    return {"entities": [{"name": "Riyo", "type": "person"}, {"name": name, "type": "concept"}],
            "relations": [{"subject": "Riyo", "predicate": "mentions", "object": name}]}


async def test_write_graph_splits_range_per_ayah():
    c = WConn()
    n_e, n_r = await K.write_graph(c, _x("QS 18:71-73"), "message", 5, None, _s())
    assert c.rels == [(501, 1071), (501, 1072), (501, 1073)]
    assert (n_e, n_r) == (4, 3)


async def test_write_graph_single_ayah():
    c = WConn()
    n_e, n_r = await K.write_graph(c, _x("QS. Al-Qamar: 46"), "message", 5, None, _s())
    assert c.rels == [(501, 1046)] and (n_e, n_r) == (2, 1)


async def test_write_graph_range_over_limit_is_plain_entity():
    CFG["kg.max_mention_range"] = 2
    c = WConn()
    await K.write_graph(c, _x("QS 18:71-73"), "message", 5, None, _s())
    assert c.rels == [(501, 502)]


async def test_write_graph_not_a_verse():
    c = WConn()
    await K.write_graph(c, _x("Windows 11"), "message", 5, None, _s())
    assert c.rels == [(501, 502)]
