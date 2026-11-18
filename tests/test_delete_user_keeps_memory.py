"""delete_user keeps the user's memories (v1.23.06)."""
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
    def __init__(self, found=True):
        self.found = found
        self.sql = []
        self.in_tx = False

    async def fetchrow(self, q, *a):
        return {"id": 7} if self.found else None

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


async def test_delete_user_keeps_memories(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    assert await M.delete_user("telegram", "123") is True
    sqls = [s for s, _a, _tx in conn.sql]
    assert not any(s.startswith("DELETE FROM memory") for s in sqls)
    assert sqls[0] == "UPDATE memory SET user_id = NULL WHERE user_id = $1"
    assert sqls[1:] == ["DELETE FROM sessions WHERE user_id = $1",
                        "DELETE FROM users WHERE id = $1"]
    assert all(a == (7,) and tx for _s, a, tx in conn.sql)


async def test_delete_user_unknown(monkeypatch):
    conn = _Conn(found=False)
    monkeypatch.setattr(M, "get_connection", lambda: _CM(conn))
    assert await M.delete_user("telegram", "999") is False
    assert conn.sql == []
