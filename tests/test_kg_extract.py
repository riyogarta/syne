"""KG ingestion path (Riyo, 18 Nov 2026) - KG-RAG stage 2."""
import json
import os
from datetime import timezone

import pytest

import syne.db.migrations as M
from syne.cli.helpers import _split_sql_statements
from syne.db.kg_extract_ddl import KG_EXTRACT_SCHEMA_SQL
from syne.memory import kg_extract as K

_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "syne", "db", "schema.sql",
)


def _s():
    s = {k: v[0] for k, v in K.DEFAULTS.items()}
    s["kg.extract_predicates"] = dict(K.DEFAULT_PREDICATES)
    return s


# ── normalisation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [
    ("Imam Malik bin Anas", "Malik ibnu Anas"),
    ("Asy-Syafi'i", "Syafi'i"),
    ("Ibnu Taimiyyah", "Ibn Taimiyah"),
    ("Dzulqarnain", "Zulqarnain"),
])
def test_normalize_joins_spellings(a, b):
    assert K.normalize_name(a) == K.normalize_name(b)


def test_normalize_keeps_different_names_apart():
    assert K.normalize_name("Agha") != K.normalize_name("Yuli")
    assert K.normalize_name("  Paket C. ") == "paket c"


# ── chunk selection (decision 9) ──────────────────────────────────────────

def test_cheap_entities():
    assert K.cheap_entities("Anakku lulus Paket C") == ["Paket", "C"]
    assert K.cheap_entities("ya, yang kedua saja") == []
    assert K.cheap_entities("Oke sudah update") == []
    assert K.cheap_entities("pakai v2 dong") == ["v2"]
    assert K.cheap_entities("pakai BM25 dong") == ["BM25"]


class _Conn:
    def __init__(self, hit=None):
        self.hit, self.calls = hit, []

    async def fetchval(self, sql, *a):
        self.calls.append(a)
        return self.hit


async def test_select_chunk():
    c = _Conn()
    assert await K.select_chunk(c, " ", 10) == (False, "empty")
    assert (await K.select_chunk(c, " ".join(["kata"] * 10), 10))[1] == "long"
    assert await K.select_chunk(c, "Anakku lulus Paket C", 10) == (True, "entity")
    assert await K.select_chunk(c, "ya, yang kedua saja", 10) == (False, "short_no_entity")
    assert await K.select_chunk(_Conn(hit=1), "anakku lulus paket c", 10) == (True, "known_entity")


# ── extraction parsing ────────────────────────────────────────────────────

def test_parse_extraction_filters():
    raw = """```json
    {"entities":[{"name":"Agha","type":"person"},{"name":"Paket C","type":"degree"},
                 {"name":"Unused","type":"person"}],
     "relations":[{"subject":"Agha","predicate":"studies","object":"Paket C"},
                  {"subject":"Agha","predicate":"loves_deeply","object":"Paket C"},
                  {"subject":"Agha","predicate":"studies","object":"Nobody"},
                  {"subject":"agha","predicate":"studies","object":"paket c"}]}
    ```"""
    out = K.parse_extraction(raw, _s())
    assert out["relations"] == [{"subject": "Agha", "predicate": "studies", "object": "Paket C"}]
    assert {e["name"] for e in out["entities"]} == {"Agha", "Paket C"}   # orphan dropped
    assert {e["name"]: e["type"] for e in out["entities"]}["Paket C"] == "concept"


def test_parse_extraction_empty_and_garbage():
    assert K.parse_extraction('{"entities":[],"relations":[]}', _s()) == {"entities": [], "relations": []}
    with pytest.raises(ValueError):
        K.parse_extraction("no json here", _s())


def test_prompt_has_fixed_predicates_and_speaker():
    p = K.build_prompt("Anakku lulus Paket C", speaker="Riyo (the user)", context="", s=_s())
    assert "studies:" in p and "related_to:" in p
    assert "SPEAKER: Riyo (the user)" in p and "(none)" in p
    assert p.rstrip().endswith("Anakku lulus Paket C")


def test_speaker_of():
    msg = ('Conversation info (untrusted metadata):\n```json\n{\n  "conversation_label": "Riyogarta Pratikto",\n'
           '  "group_subject": "YBA"\n}\n```\n\nSender (untrusted metadata):\n```json\n{\n  "name": "Yuli",\n'
           '  "id": "1"\n}\n```\n\nAnakku lulus Paket C')
    who, body = K.speaker_of(msg, "user")
    assert who == "Yuli (the user)" and body == "Anakku lulus Paket C"
    dm = 'Conversation info (untrusted metadata):\n```json\n{\n  "conversation_label": "Riyogarta Pratikto"\n}\n```\n\nhalo'
    assert K.speaker_of(dm, "user")[0] == "Riyogarta Pratikto (the user)"
    assert K.speaker_of("jawaban", "assistant")[0].startswith("Molt")


def test_parse_era():
    assert K.parse_era(None) is None and K.parse_era("bukan tanggal") is None
    dt = K.parse_era("2026-11-18T04:10:00")
    assert dt.tzinfo == timezone.utc


async def test_settings_clamped(monkeypatch):
    import syne.db.models as models
    store = {"kg.extract_batch": 999, "kg.extract_enabled": "false",
             "kg.resolve_min_similarity": "x", "kg.extract_era_start": "2026-11-18T04:00:00+00:00"}

    async def gc(key, default=None):
        return store.get(key, default)
    monkeypatch.setattr(models, "get_config", gc)
    monkeypatch.setattr(K, "get_config", gc)
    s = await K.load_settings()
    assert s["kg.extract_batch"] == 50 and s["kg.extract_enabled"] is False
    assert s["kg.resolve_min_similarity"] == 0.92
    assert s["kg.extract_era_start"].year == 2026


# ── entity resolution ─────────────────────────────────────────────────────

class _RConn:
    def __init__(self, norm_row=None, emb_row=None):
        self.norm_row, self.emb_row, self.executed, self.inserted = norm_row, emb_row, [], None

    async def fetchrow(self, sql, *a):
        return self.norm_row if "norm_name = $1" in sql else self.emb_row

    async def execute(self, sql, *a):
        self.executed.append((sql, a))

    async def fetchval(self, sql, *a):
        self.inserted = a
        return 77


async def _emb(_):
    return [0.1, 0.2]


async def test_resolve_by_norm():
    assert await K.resolve_entity(_RConn(norm_row={"id": 5}), "Imam Malik", "person", _emb, _s()) == (5, "norm")


async def test_resolve_by_embedding_needs_spelling_too():
    s = _s()
    c = _RConn(emb_row={"id": 9, "sim": 0.95, "trgm": 0.6})
    assert await K.resolve_entity(c, "Agha Gamitra", "person", _emb, s) == (9, "embedding")
    assert c.executed and "array_append" in c.executed[0][0]
    c2 = _RConn(emb_row={"id": 9, "sim": 0.95, "trgm": 0.1})        # sounds alike, spelled apart
    assert await K.resolve_entity(c2, "Agus", "person", _emb, s) == (77, "new")
    assert c2.inserted[2] == "agus"


# ── migration / schema ────────────────────────────────────────────────────

def test_migration_53_registered():
    assert (53, M._m53_kg_extract, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 53


async def test_migration_53_runs_ddl_and_seeds():
    calls = []

    class C:
        async def execute(self, sql, *a):
            calls.append((sql, a))
    await M._m53_kg_extract(C())
    sqls = " ".join(c[0] for c in calls)
    assert "kg_extract_log" in sqls and "norm_name" in sqls and "kg.extract_era_start" in sqls
    assert {c[1][0] for c in calls if c[1]} == set(K.DEFAULTS)


def test_schema_sql_carries_ddl_and_seeds():
    with open(_SCHEMA) as f:
        schema = f.read()
    assert KG_EXTRACT_SCHEMA_SQL in schema
    for key, (value, desc) in K.DEFAULTS.items():
        assert f"('{key}', '{json.dumps(value)}', '{desc}')" in schema, key


def test_ddl_splits_into_statements():
    stmts = [s.strip() for s in _split_sql_statements(KG_EXTRACT_SCHEMA_SQL)]
    assert sum(1 for s in stmts if s.endswith(";")) == 4
