"""delete_user (v1.23.11): DM sessions deleted, shared sessions detached,
group rows anonymised, memories kept without owner, schedules and group
member entries removed - all in one transaction."""
from syne.db import models as M


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.in_tx = True

    async def __aexit__(self, *a):
        self.conn.in_tx = False
        return False


class _Conn:
    def __init__(self, found=True, dm=(11,), group_rows=(501, 502)):
        self.found = found
        self.dm = list(dm)
        self.group_rows = list(group_rows)
        self.sql = []
        self.in_tx = False

    async def fetchrow(self, q, *a):
        if not self.found:
            return None
        return {"id": 7, "name": "Budi", "display_name": None, "access_level": "public"}

    async def fetch(self, q, *a):
        if "FROM sessions s WHERE s.user_id" in q:
            self.dm_args = a
            return [{"id": i} for i in self.dm]
        if "FROM messages m JOIN sessions" in q:
            return [{"id": i} for i in self.group_rows]
        return []

    async def fetchval(self, q, *a):
        return 3

    async def execute(self, q, *a):
        self.sql.append((" ".join(q.split()), a, self.in_tx))
        return "OK"

    def transaction(self):
        return _Tx(self)


class _CM:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *a):
        return False


def _sqls(conn):
    return [s for s, _a, _tx in conn.sql]


async def test_delete_user_keeps_memories(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    res = await M.delete_user("telegram", "123")
    assert res and res["name"] == "Budi"
    sqls = _sqls(conn)
    assert not any(s.startswith("DELETE FROM memory") for s in sqls)
    assert "UPDATE memory SET user_id = NULL WHERE user_id = $1" in sqls
    assert all(tx for _s, _a, tx in conn.sql)


async def test_only_dm_sessions_deleted_shared_detached(monkeypatch):
    conn = _Conn(dm=(11, 12))
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user("telegram", "123")
    sqls = _sqls(conn)
    assert not any(s == "DELETE FROM sessions WHERE user_id = $1" for s in sqls)
    dels = [(s, a) for s, a, _t in conn.sql if s.startswith("DELETE FROM sessions")]
    assert dels == [("DELETE FROM sessions WHERE id = ANY($1::int[])", ([11, 12],))]
    detach = sqls.index("UPDATE sessions SET user_id = NULL WHERE user_id = $1")
    sub = next(i for i, s in enumerate(sqls) if s.startswith("UPDATE subagent_runs"))
    assert sub < sqls.index("DELETE FROM sessions WHERE id = ANY($1::int[])") < detach


async def test_group_rows_anonymised_and_pruned(monkeypatch):
    conn = _Conn(group_rows=(501, 502))
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user("telegram", "123")
    anon = [(s, a) for s, a, _t in conn.sql if s.startswith("UPDATE messages SET content")]
    assert anon and anon[0][1] == ([501, 502],)
    assert "embedding = NULL" in anon[0][0]
    prune = [(s, a) for s, a, _t in conn.sql if "kg_prune_sources" in s]
    assert prune == [("SELECT kg_prune_sources('message', $1::int[])", ([501, 502],))]


async def test_schedules_and_group_entries_removed(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user("telegram", "123")
    sched = [(s, a) for s, a, _t in conn.sql if s.startswith("DELETE FROM scheduled_tasks")]
    assert sched == [("DELETE FROM scheduled_tasks WHERE created_by = $1", (123,))]
    grp = [a for s, a, _t in conn.sql if s.startswith("UPDATE groups SET settings")]
    assert grp == [("telegram", "123")]
    assert _sqls(conn)[-1] == "DELETE FROM users WHERE id = $1"


async def test_whatsapp_id_skips_schedules_when_not_numeric(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user("whatsapp", "abc")
    assert not any(s.startswith("DELETE FROM scheduled_tasks") for s in _sqls(conn))


async def test_no_group_rows_no_prune(monkeypatch):
    conn = _Conn(group_rows=())
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user("telegram", "123")
    assert not any("kg_prune_sources" in s for s in _sqls(conn))


async def test_delete_user_unknown(monkeypatch):
    conn = _Conn(found=False)
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    assert await M.delete_user("telegram", "999") is None
    assert conn.sql == []


async def test_preview_changes_nothing(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    p = await M.delete_user_preview("telegram", "123")
    assert p["dm_session_ids"] == [11] and p["group_message_ids"] == [501, 502]
    assert conn.sql == []


def test_manage_user_delete_always_gated():
    from syne.consent import is_always_gate
    assert is_always_gate("manage_user", {"action": "delete", "user_id": "1"})
    assert not is_always_gate("manage_user", {"action": "list"})
    assert is_always_gate("memory_update", {})
    assert not is_always_gate("exec", {"command": "ls"})


def test_cli_user_delete_registered():
    from syne.cli import cli
    assert "user" in cli.commands
    assert "delete" in cli.commands["user"].commands


def test_dm_recognised_by_group_predicate_not_chat_id():
    from syne.tools.history import group_session_sql
    dm, rows = M._delete_user_sql()
    g = group_session_sql("s")
    assert dm.endswith(f"AND NOT {g}") and "platform_chat_id = ANY" not in dm
    assert rows.endswith(f"AND {g}") and "NOT" not in rows.split("sender_id")[1]


async def test_dm_query_takes_only_user_id(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    await M.delete_user_preview("discord", "652096670653218826")
    assert conn.dm_args == (7,)
