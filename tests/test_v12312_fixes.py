"""v1.23.12: history scope per turn, KG prune of sources without write access."""
from types import SimpleNamespace

import syne.db.migrations as M
from syne.tools import history as H


def test_scope_uses_this_turns_inbound():
    turn = SimpleNamespace(platform="telegram", chat_id="-100", sender_id="3478983")
    later = SimpleNamespace(platform="telegram", chat_id="-100", sender_id="1544010")
    conv = SimpleNamespace(inbound=later, _inb=lambda: turn, user={"platform": "telegram"},
                           is_group=True, chat_id="-100", session_id=1)
    sc = H.build_history_scope(conv, "owner")
    assert sc["sender_pid"] == "3478983" and sc["access"] == "owner"


def test_migration_59_registered():
    assert (59, M._m59_prune_kg_without_write_access, "transactional") in M.MIGRATIONS
    assert M.CURRENT_SCHEMA_VERSION >= 59


class _C:
    def __init__(self, ids):
        self.ids, self.calls = ids, []

    async def fetchval(self, sql, *a):
        self.calls.append(sql)
        return self.ids

    async def execute(self, sql, *a):
        self.calls.append((sql, a))


async def test_migration_59_prunes_non_owner_family_sources():
    c = _C([4, 9])
    await M._m59_prune_kg_without_write_access(c)
    assert "NOT IN ('owner', 'family')" in c.calls[0]
    assert c.calls[1] == ("SELECT kg_prune_sources('message', $1::int[])", ([4, 9],))


async def test_migration_59_nothing_to_prune():
    c = _C(None)
    await M._m59_prune_kg_without_write_access(c)
    assert len(c.calls) == 1
