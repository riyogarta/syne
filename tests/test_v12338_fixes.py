"""v1.23.38: KG person entities anchored to account ids; review E (every
word of a multi-word name), F (recall in groups), H (directory decoded twice),
bug 1 (owner accounts without the access flag, unlinked accounts notice),
A (HNSW above 2000 dimensions), B (relax_columns drop/add, no count under lock)."""
import inspect
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import syne.db.migrations as M
from syne.memory import embed_dims as D
from syne.memory import kg_extract as K
from syne.memory import kg_recall as R
from tests.test_v12336_fixes import _C, RIYO, AGHA as AGHA_TG
from tests.test_v12337_fixes import YULI, AGHA

ROOT = Path(__file__).resolve().parents[1]


def _u(platform, pid, name, access="owner", display=None):
    return {"platform": platform, "platform_id": pid, "access_level": access,
            "name": name, "display_name": display, "aliases": "{}"}


OWNER_TG = _u("telegram", "3478983", "Riyogarta", display="Riyogarta Pratikto")
OWNER_WA = _u("whatsapp", "628111681624", "Pak Riyo")
CLI = _u("cli", "cli:syne", "syne")


# ── H: family.directory stored as a JSON string ───────────────────────────

class _Twice(_C):
    async def fetchval(self, q, *a):
        if "family.directory" in q:
            return json.dumps(json.dumps(self.directory))
        return await super().fetchval(q, *a)


def test_as_json_decodes_twice():
    d = [{"names": ["Riyo"]}]
    assert K._as_json(json.dumps(json.dumps(d)), []) == d
    assert K._as_json(json.dumps("plain text"), []) == []
    assert K._as_json("not json", []) == []
    assert K._as_json(None, {}) == {}
    assert K._as_json(d, []) is d


async def test_directory_stored_as_a_string():
    g = await K.person_groups(_Twice([RIYO, YULI]), fresh=True)
    assert g["bunli"].canon == "Yuli" and g["yahyo"].canon == "Riyo"


# ── E: a multi-word name needs every word ─────────────────────────────────

async def test_multi_word_name_needs_every_word():
    G = "chat:telegram:-100"
    known = await K.known_people(_C([RIYO, AGHA_TG], [OWNER_TG],
                                    groups={("telegram", "-100"): ["3478983", "41005565"]}))
    assert K.person_scope("Riyogarta Hadi", "person", G, known) == G
    assert K.person_scope("Hadi Riyogarta", "person", G, known) == G
    assert K.person_scope("Riyo Gamitra", "person", G, known) == G      # two people
    for n in ("Riyogarta", "Riyogarta Pratikto", "Pak Riyogarta Pratikto", "Riyo Yahyo",
              "Agha Gamitra"):
        assert K.person_scope(n, "person", G, known) is None, n


# ── bug 1: owner rows when no entry has access "owner" ────────────────────

async def test_owner_rows_join_entry_without_access_flag():
    plain = {k: v for k, v in RIYO.items() if k != "access"}
    g = await K.person_groups(_C([plain, YULI], [OWNER_TG, OWNER_WA, CLI]), fresh=True)
    assert g["riyo"].canon == "Riyo"
    assert {("telegram", "3478983"), ("whatsapp", "628111681624")} <= g["riyo"].accounts
    assert "riyogarta pratikto" in g["riyo"]
    # no account of theirs in the directory: joined by a shared name
    g2 = await K.person_groups(_C([plain, YULI], [OWNER_WA]), fresh=True)
    assert ("whatsapp", "628111681624") in g2["riyo"].accounts


async def test_unlinked_accounts_listed():
    yuli_wa = _u("whatsapp", "628111681524", "Yuli WA", access="family")
    rows = await K.unlinked_accounts(_C([RIYO, YULI], [OWNER_TG, OWNER_WA, CLI, yuli_wa]))
    assert rows == [("whatsapp", "628111681624", "Pak Riyo")]
    t = K.unlinked_text(rows)
    assert "628111681624" in t and '"accounts"' in t


async def test_unlinked_notice_sent_once(monkeypatch):
    import syne.communication.registry as REG
    conn = _C([RIYO], [OWNER_TG, OWNER_WA])

    @asynccontextmanager
    async def gconn(timeout=None):
        yield conn
    store: dict = {}

    async def gc(k, d=None):
        return store.get(k, d)

    async def sc(k, v, desc=None):
        store[k] = v
    sent: list = []

    class Ch:
        async def notify_owner(self, text):
            sent.append(text)
            return True
    monkeypatch.setattr(K, "get_connection", gconn)
    monkeypatch.setattr(K, "get_config", gc)
    monkeypatch.setattr(K, "set_config", sc)
    monkeypatch.setattr(REG, "get_channel", lambda name: Ch())
    monkeypatch.setattr(K, "_unlinked_retry_at", 0.0)
    assert await K.maybe_notify_unlinked() is True
    monkeypatch.setattr(K, "_unlinked_retry_at", 0.0)
    assert await K.maybe_notify_unlinked() is False          # same set: told once
    assert await K.maybe_notify_unlinked() is False          # not due yet
    assert len(sent) == 1 and store[K.UNLINKED_KEY]


# ── speaker by account ───────────────────────────────────────────────────

async def test_speaker_known_by_account():
    g = await K.person_groups(_C([RIYO, YULI]), fresh=True)
    it = {"role": "user", "sender_platform": "whatsapp",
          "sender_account": "628111681524@s.whatsapp.net"}
    assert K.speaker_person(it, g).canon == "Yuli"
    assert K.speaker_person({**it, "role": "assistant"}, g) is None
    assert K.speaker_person({**it, "sender_account": "62800"}, g) is None
    assert K.speaker_person({**it, "sender_account": None}, g) is None
    src = inspect.getsource(K.process_one)
    assert "anchor.canon" in src and "speaker=anchor" in src
    assert "sender_account" in inspect.getsource(K.pending)


# ── F: recall in a group skips family who are not members ────────────────

class _RC:
    def __init__(self, grp=True, members=("3478983",)):
        self.grp, self.members, self.calls = grp, members, []

    async def fetchrow(self, q, *a):
        return {"platform": "telegram", "platform_chat_id": "-300", "grp": self.grp}

    async def fetchval(self, q, *a):
        return json.dumps({"members": {m: {} for m in self.members}})

    async def fetch(self, q, *a):
        self.calls.append(a)
        return [{"entity_id": 99}]


async def test_recall_skips_family_outside_the_group():
    pg = await K.person_groups(_C([RIYO, AGHA_TG]), fresh=True)
    c = _RC()
    people, ids = await R._outsiders(c, 7, pg)
    assert [p.canon for p in people] == ["Agha"] and ids == [99]
    assert c.calls == [(["telegram"], ["41005565"])]
    assert R._key_in("gamitra", pg, people) == ()
    assert R._key_in("yahyo", pg, people) is pg["riyo"]
    assert await R._outsiders(_RC(grp=False), 7, pg) == ([], [])
    assert await R._outsiders(_RC(), None, pg) == ([], [])
    assert "NOT (e.id = ANY($5::int[]))" in R._MATCH_SQL


# ── migrations, schema, A, B ─────────────────────────────────────────────

def test_migrations_and_schema():
    assert (67, M._m67_kg_entity_accounts, "transactional") in M.MIGRATIONS
    assert (68, M._m68_hnsw_dims_guard, "transactional") in M.MIGRATIONS
    sql = (ROOT / "syne/db/schema.sql").read_text()
    for stmt in M.KG_ENTITY_ACCOUNTS_DDL:
        assert stmt in sql
    assert sql.count("IF dim > 2000 THEN RETURN; END IF;") == 2
    for f in M.HNSW_GUARD_SQL:
        assert "IF dim > 2000 THEN RETURN; END IF;" in f and f in sql


def test_repair_knows_the_2000_limit():
    src = (ROOT / "syne/cli/cmd_repair.py").read_text()
    seg = src[src.index("# -- 8. Vector Indexes"):src.index("# -- Zombie memories")]
    assert "HNSW not available above 2000" in seg
    assert seg.index("> 2000") < seg.index('issues.append(f"HNSW vector index missing')


def test_relax_and_clear_drop_add():
    rel = inspect.getsource(D.relax_columns)
    assert "count(*)" not in rel and "EXISTS" in rel and "_rebuild_empty_column" in rel
    assert "USING embedding::" not in rel
    reb = inspect.getsource(D._rebuild_empty_column)
    assert "DROP COLUMN embedding" in reb and "ADD COLUMN embedding" in reb
    assert "clear_history_vectors(conn, nd, n_hist)" in inspect.getsource(D.apply_switch)
    t = D.clear_confirm_text({"blocked": [("messages", 1024, 5)], "relax": []}, "m")
    assert "VACUUM FULL messages" in t and "<b>5</b>" in t


# ── real Postgres ────────────────────────────────────────────────────────

_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


async def _schema(c, vector=False):
    if vector and not await c.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'"):
        pytest.skip("pgvector not installed in the test database")
    s = "v38_" + uuid.uuid4().hex[:10]
    await c.execute(f"CREATE SCHEMA {s}; SET search_path TO {s}, public")
    return s


@pg
async def test_person_entities_anchored_by_account():
    import asyncpg
    from tests.test_v12332_fixes import _BASE
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    try:
        await c.execute(_BASE)
        await c.execute("CREATE UNIQUE INDEX ON kg_entities (LOWER(name), entity_type, coalesce(scope, ''))")
        await c.execute("INSERT INTO config VALUES ('family.directory', $1::jsonb)",
                        json.dumps([RIYO, AGHA]))

        async def ent(name):
            return await c.fetchval(
                "INSERT INTO kg_entities (name, entity_type, norm_name) "
                "VALUES ($1, 'person', $2) RETURNING id", name, K.person_norm(name))

        agha = await ent("Agha")
        await M._m67_kg_entity_accounts(c)
        rows = await c.fetch("SELECT platform, account_id FROM kg_entity_accounts "
                             "WHERE entity_id = $1 ORDER BY 1", agha)
        assert [tuple(r) for r in rows] == [("discord", "652096670653218826"),
                                            ("telegram", "41005565")]
        await M._m67_kg_entity_accounts(c)                  # idempotent
        assert await c.fetchval("SELECT count(*) FROM kg_entity_accounts") == 2

        g = await K.person_groups(c, fresh=True)
        st = {"_person_groups": g}
        # the entity no longer carries a name of the person: found by account
        await c.execute("UPDATE kg_entities SET name = 'Gamitra X', norm_name = 'gamitra x' "
                        "WHERE id = $1", agha)
        assert await K.resolve_entity(c, "Gamitra", "person", None, st, None) == (agha, "person")
        # a person with no entity yet: created under the directory name, linked
        riyo, how = await K.resolve_entity(c, "Yahyo", "person", None, st, None)
        assert how == "person"
        assert await c.fetchval("SELECT entity_id FROM kg_entity_accounts WHERE "
                                "platform = 'telegram' AND account_id = '3478983'") == riyo
        assert await K.resolve_entity(c, "Riyogarta", "person", None, st, None) == (riyo, "person")

        # one person's accounts on two entities: merged into the oldest
        dup = await ent("Gamitra Dup")
        await c.execute("UPDATE kg_entity_accounts SET entity_id = $1 WHERE platform = 'discord'", dup)
        await c.execute("INSERT INTO kg_relations (subject_id, predicate, object_id) "
                        "VALUES ($1, 'knows', $2)", dup, riyo)
        assert await K.account_entity(c, g["agha"].accounts) == agha
        assert await c.fetchval("SELECT count(*) FROM kg_entities WHERE id = $1", dup) == 0
        assert await c.fetchval("SELECT count(*) FROM kg_entity_accounts WHERE entity_id = $1", agha) == 2
        assert await c.fetchval("SELECT subject_id FROM kg_relations WHERE predicate = 'knows'") == agha

        # a name never picks an entity anchored to another person
        await c.execute("UPDATE kg_entity_accounts SET account_id = '999' WHERE entity_id = $1", riyo)
        assert await K.anchored_person(c, g["riyo"]) is None
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_pending_reads_sender_account():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c)
    try:
        await c.execute("""
CREATE TABLE memory (id SERIAL PRIMARY KEY, content TEXT, category TEXT,
    updated_at TIMESTAMPTZ DEFAULT now(), created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE users (id SERIAL PRIMARY KEY, platform TEXT, platform_id TEXT, name TEXT);
CREATE TABLE sessions (id SERIAL PRIMARY KEY, user_id INT, platform TEXT, platform_chat_id TEXT);
CREATE TABLE groups (platform TEXT, platform_group_id TEXT);
CREATE TABLE messages (id SERIAL PRIMARY KEY, session_id INT, role TEXT, content TEXT,
    metadata JSONB DEFAULT '{}', created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE kg_extract_log (source_type TEXT, source_id INT, status TEXT, attempts INT,
    updated_at TIMESTAMPTZ);
""")
        uid = await c.fetchval("INSERT INTO users (platform, platform_id, name) "
                               "VALUES ('telegram', '3478983', 'Riyo') RETURNING id")
        dm = await c.fetchval("INSERT INTO sessions (user_id, platform, platform_chat_id) "
                              "VALUES ($1, 'telegram', '3478983') RETURNING id", uid)
        grp = await c.fetchval("INSERT INTO sessions (user_id, platform, platform_chat_id) "
                               "VALUES ($1, 'telegram', '-100') RETURNING id", uid)
        await c.execute("INSERT INTO messages (session_id, role, content, metadata) VALUES "
                        "($1, 'user', 'halo dm', '{}'), "
                        "($2, 'user', 'halo grup', '{\"sender_id\": \"41005565\"}'), "
                        "($2, 'assistant', 'jawab', '{}')", dm, grp)
        st = {"kg.extract_batch": 10, "kg.extract_max_attempts": 3,
              "kg.extract_skip_categories": [], "kg.extract_skip_scheduled": True}
        rows = await K.pending(c, {"memory": 0, "messages": 0},
                               datetime.now(timezone.utc) - timedelta(days=1), st)
        got = [(r["content"], r["sender_platform"], r["sender_account"], r["chat_scope"])
               for r in rows]
        assert got == [("halo dm", "telegram", "3478983", None),
                       ("halo grup", "telegram", "41005565", "chat:telegram:-100"),
                       ("jawab", "telegram", None, "chat:telegram:-100")]
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_hnsw_functions_skip_above_2000_dims():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c, vector=True)
    try:
        for t in ("memory", "messages"):
            await c.execute(f"CREATE TABLE {t} (id SERIAL PRIMARY KEY, embedding vector)")
        for f in M.HNSW_GUARD_SQL:
            await c.execute(f)
        big = "[" + ",".join(["0.1"] * 2001) + "]"
        await c.execute(f"INSERT INTO memory (embedding) VALUES ('{big}')")
        await c.execute("SELECT ensure_memory_hnsw_index()")        # no error
        has = ("SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname = $1 "
               "AND indexname = $2)")
        assert not await c.fetchval(has, s, "idx_memory_embedding_hnsw")
        assert await c.fetchval(D._COL_SQL, "memory") == "vector"
        await c.execute("INSERT INTO messages (embedding) VALUES ('[1,2,3]')")
        await c.execute("SELECT ensure_messages_hnsw_index()")
        assert await c.fetchval(has, s, "idx_messages_embedding_hnsw")
        assert await c.fetchval(D._COL_SQL, "messages") == "vector(3)"
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


@pg
async def test_relax_keeps_rows_and_pending_index():
    import asyncpg
    c = await asyncpg.connect(_DB)
    s = await _schema(c, vector=True)
    try:
        await c.execute("CREATE TABLE memory (id SERIAL PRIMARY KEY, embedding vector(4))")
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, role TEXT, "
                        "content TEXT, embedding vector(4))")
        await c.execute("CREATE INDEX idx_messages_embedding_pending ON messages (id) "
                        "WHERE role IN ('user', 'assistant') AND embedding IS NULL")
        await c.execute("INSERT INTO messages (role, content) VALUES ('user', 'a'), ('assistant', 'b')")
        assert await D.relax_columns(c, ["messages"], 3) == ["messages"]
        assert await c.fetchval(D._COL_SQL, "messages") == "vector(3)"
        assert [r["content"] for r in await c.fetch("SELECT content FROM messages ORDER BY id")] == ["a", "b"]
        for idx in ("idx_messages_embedding_pending", D.HNSW_INDEX["messages"]):
            assert await c.fetchval("SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                                    "AND indexname = $2", s, idx) == 1, idx
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()
