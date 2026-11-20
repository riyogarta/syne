"""v1.23.37: directory entries with several accounts; review B (no table
rewrite when history vectors are cleared); review D (fixed API key used
without a restart)."""
import inspect
import json
import os

import pytest

from syne.memory import kg_extract as K
from syne.memory import embed_dims as D
from tests.test_v12336_fixes import _C, RIYO


YULI = {"names": ["Yuli", "Yuliazmi", "Bunli"], "access": "family", "chat_id": "1544010",
        "platform": "telegram",
        "accounts": [{"platform": "whatsapp", "chat_id": "628111681524@s.whatsapp.net"}]}
AGHA = {"names": ["Agha", "Gamitra"], "access": "family", "chat_id": "41005565",
        "platform": "telegram",
        "accounts": [{"platform": "discord", "chat_id": "652096670653218826"}, "bad", {}]}


def test_entry_accounts():
    assert K.entry_accounts(YULI) == [("telegram", "1544010"), ("whatsapp", "628111681524")]
    assert K.entry_accounts(AGHA) == [("telegram", "41005565"),
                                      ("discord", "652096670653218826")]
    assert K.entry_accounts({"names": ["X"]}) == []
    assert K.entry_accounts({"chat_id": "628:51@s.whatsapp.net", "platform": "whatsapp"}) == \
        [("whatsapp", "628")]


async def test_member_on_another_platform_is_the_family_person():
    groups = {("whatsapp", "120@g.us"): ["628111681524"],
              ("discord", "1120"): ["652096670653218826"],
              ("whatsapp", "999@g.us"): ["628000"]}
    known = await K.known_people(_C([RIYO, YULI, AGHA], groups=groups))
    assert K.person_scope("Bunli", "person", "chat:whatsapp:120@g.us", known) is None
    assert K.person_scope("Gamitra", "person", "chat:discord:1120", known) is None
    assert K.person_scope("Bunli", "person", "chat:whatsapp:999@g.us", known) == \
        "chat:whatsapp:999@g.us"
    g = await K.person_groups(_C([RIYO, YULI, AGHA]), fresh=True)
    assert ("whatsapp", "628111681524") in g["bunli"].accounts


async def test_linked_user_row_on_extra_account_joins_the_entry():
    users = [{"platform": "whatsapp", "platform_id": "628111681524", "access_level": "family",
              "name": "Bu Yuli WA", "display_name": None, "aliases": "{}"}]
    g = await K.person_groups(_C([RIYO, YULI], users), fresh=True)
    assert "yuli wa" in g["yuli"]


def test_boot_lists_extra_accounts():
    import syne.boot as boot
    src = inspect.getsource(boot)
    assert 'person.get("accounts")' in src and '"@s.whatsapp.net"' in src
    assert 'if ap in ("telegram", "whatsapp")' in src


# ── B ─────────────────────────────────────────────────────────────────────

def test_clear_does_not_rewrite():
    # v1.23.38: the drop/add moved to _rebuild_empty_column (shared with relax_columns)
    src = inspect.getsource(D.clear_history_vectors) + inspect.getsource(D._rebuild_empty_column)
    assert "DROP COLUMN embedding" in src and "ADD COLUMN embedding" in src
    assert "UPDATE messages SET embedding = NULL" not in src
    assert "USING embedding::" not in src


_DB = os.environ.get("SYNE_TEST_DATABASE_URL")
pg = pytest.mark.skipif(not _DB, reason="SYNE_TEST_DATABASE_URL not set")


@pg
async def test_clear_keeps_rows_and_rebuilds_pending_index():
    import asyncpg
    from tests.test_v12318_fixes import _schema
    c = await asyncpg.connect(_DB)
    s = await _schema(c, vector=True)
    try:
        await c.execute("CREATE TABLE messages (id SERIAL PRIMARY KEY, session_id INT, role TEXT, "
                        "content TEXT, embedding vector(4), status TEXT DEFAULT 'x')")
        await c.execute("CREATE INDEX idx_messages_embedding_pending ON messages (session_id, id) "
                        "WHERE role IN ('user', 'assistant') AND embedding IS NULL")
        await c.execute("INSERT INTO messages (session_id, role, content, embedding) VALUES "
                        "(1, 'user', 'a', '[1,2,3,4]'), (1, 'assistant', 'b', '[4,3,2,1]'), "
                        "(1, 'user', 'c', NULL)")
        assert await D.clear_history_vectors(c, 3) == 2
        rows = await c.fetch("SELECT content, status, embedding FROM messages ORDER BY id")
        assert [r["content"] for r in rows] == ["a", "b", "c"]
        assert all(r["embedding"] is None and r["status"] == "x" for r in rows)
        assert await c.fetchval(D._COL_SQL, "messages") == "vector(3)"
        idx = {r["indexname"] for r in await c.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = $1", s)}
        assert {"idx_messages_embedding_pending", D.HNSW_INDEX["messages"]} <= idx
        await c.execute("INSERT INTO messages (session_id, role, content, embedding) "
                        "VALUES (1, 'user', 'd', '[1,2,3]')")
        with pytest.raises(Exception):
            await c.execute("INSERT INTO messages (session_id, role, content, embedding) "
                            "VALUES (1, 'user', 'e', '[1,2,3,4]')")
    finally:
        await c.execute(f"DROP SCHEMA {s} CASCADE")
        await c.close()


# ── D ─────────────────────────────────────────────────────────────────────

async def test_fingerprint_follows_entry_and_key(monkeypatch):
    cfg = {"provider.models": [{"key": "m", "driver": "anthropic", "credential_key": "credential.k"}],
           "credential.k": "old"}

    async def get(key, default=None):
        return cfg.get(key, default)
    monkeypatch.setattr(K, "get_config", get)
    a = await K.judge_fingerprint("m")
    cfg["credential.k"] = "new"
    b = await K.judge_fingerprint("m")
    assert a != b and "new" not in b and "old" not in a
    assert await K.judge_fingerprint("") is None
    assert await K.judge_fingerprint("gone") == "missing"
    assert K.judge_cache_key("m", None, a) != K.judge_cache_key("m", None, b)


def test_worker_wiring():
    src = inspect.getsource(K)
    assert "fp = await gated(judge_fingerprint(" in src
    i = src.index('if kind == "auth":\n                                # v1.23.37')
    assert "judge, judge_key = None, None" in src[i:i + 300]
