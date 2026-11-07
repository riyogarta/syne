"""Visibility scope of history_search / history_expand (Riyo, 7 Nov 2026).

public       -> the chat they ask from
owner/family -> that chat + their own DMs on the same platform
owner in DM  -> may widen to any chat / platform explicitly
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from syne.tools import history as H
from syne.tools.registry import ToolRegistry
from syne.security import TOOL_PERMISSIONS, filter_tools_for_group, get_permission_digit


def scope(access, chat_id, is_group, sender, platform="telegram"):
    return {"access": access, "platform": platform, "chat_id": chat_id,
            "is_group": is_group, "sender_pid": sender, "session_id": 1}


OWNER_DM = scope("owner", "3478983", False, "3478983")
OWNER_GROUP = scope("owner", "-1001185869359", True, "3478983")
FAMILY_DM = scope("family", "1544010", False, "1544010")
FAMILY_GROUP = scope("family", "-1001185869359", True, "1544010")
PUBLIC_GROUP = scope("public", "-1003879924586", True, "55281842")


def _conn_ctx(fetch):
    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=fetch)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


async def _search(sc, **kw):
    prov = MagicMock()
    prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1]))
    H.set_embedding_provider(prov)
    cap = {}

    async def fetch(sql, *params):
        cap["sql"], cap["params"] = sql, params
        return []

    with patch("syne.tools.history.get_connection", return_value=_conn_ctx(fetch)):
        res = await H.history_search_handler(query="x", _scope=sc, **kw)
    return res, cap


class TestScopeSql:

    def test_public_is_current_chat_only(self):
        sql, params = H.scope_sql(PUBLIC_GROUP, 2)
        assert params == ["telegram", "-1003879924586"]
        assert "FROM users u" not in sql

    def test_family_adds_own_dm_by_session_owner(self):
        sql, params = H.scope_sql(FAMILY_GROUP, 5)
        assert params == ["telegram", "-1001185869359", "1544010"]
        assert "u.platform_id = $7" in sql
        assert "NOT " + H.group_session_sql("s") in sql

    def test_group_predicate_covers_every_platform(self):
        g = H.group_session_sql("s")
        for marker in ("LIKE '-%'", "LIKE '%@g.us'", "LIKE 'scheduled:-%'", "FROM groups g"):
            assert marker in g


class TestSearchScope:

    async def test_no_scope_is_refused(self):
        res, cap = await _search(None)
        assert res.startswith("Error:") and not cap

    async def test_family_dm_is_scoped(self):
        res, cap = await _search(FAMILY_DM)
        assert "FROM users u" in cap["sql"]
        assert "1544010" in cap["params"]

    async def test_public_is_scoped_to_current_chat(self):
        res, cap = await _search(PUBLIC_GROUP)
        assert "FROM users u" not in cap["sql"]
        assert "-1003879924586" in cap["params"]

    async def test_owner_dm_default_is_still_scoped(self):
        res, cap = await _search(OWNER_DM)
        assert "FROM users u" in cap["sql"]

    async def test_owner_dm_all_chats_is_unrestricted(self):
        res, cap = await _search(OWNER_DM, all_chats=True)
        assert "platform_chat_id = $" not in cap["sql"]
        assert len(cap["params"]) == 3  # vector, limit, offset

    async def test_owner_dm_can_pick_another_platform(self):
        res, cap = await _search(OWNER_DM, platform="discord")
        assert "discord" in cap["params"]
        assert "FROM users u" not in cap["sql"]

    async def test_owner_in_group_cannot_widen(self):
        res, cap = await _search(OWNER_GROUP, platform="discord")
        assert res.startswith("Error:") and not cap

    async def test_family_cannot_widen(self):
        for kw in ({"all_chats": True}, {"chat_id": "3478983"}, {"platform": "whatsapp"}):
            res, cap = await _search(FAMILY_DM, **kw)
            assert res.startswith("Error:") and not cap

    async def test_public_cannot_widen(self):
        res, cap = await _search(PUBLIC_GROUP, chat_id="3478983")
        assert res.startswith("Error:") and not cap


class TestExpandScope:

    async def test_no_scope_is_refused(self):
        res = await H.history_expand_handler(anchor_ids=[5])
        assert res.startswith("Error:")

    async def test_non_owner_anchor_lookup_is_scoped(self):
        seen = {}

        async def fetch(sql, *params, timeout=None):
            seen["sql"], seen["params"] = sql, params
            return []  # anchor outside scope -> not returned

        with patch("syne.tools.history.get_connection", return_value=_conn_ctx(fetch)):
            res = await H.history_expand_handler(anchor_ids=[5], _scope=FAMILY_GROUP)
        assert "JOIN sessions s" in seen["sql"]
        assert seen["params"][1:] == ("telegram", "-1001185869359", "1544010")
        assert 5 in __import__("json").loads(res)["missing_anchor_ids"]


class TestBuildScope:

    def test_none_conv_denies(self):
        assert H.build_history_scope(None, "owner") is None

    def test_group_uses_typing_sender(self):
        conv = SimpleNamespace(
            inbound=SimpleNamespace(platform="whatsapp", chat_id="120363431995914143@g.us",
                                    sender_id="628111681524"),
            user={"platform": "whatsapp", "platform_id": "628111681624"},
            is_group=True, chat_id="120363431995914143@g.us", session_id=3,
        )
        sc = H.build_history_scope(conv, "family")
        assert sc["sender_pid"] == "628111681524"
        assert sc["is_group"] is True and sc["platform"] == "whatsapp"

    def test_dm_uses_session_user(self):
        conv = SimpleNamespace(
            inbound=SimpleNamespace(platform="discord", chat_id="1567489965582057503",
                                    sender_id=None),
            user={"platform": "discord", "platform_id": "652096670653218826"},
            is_group=False, chat_id="1567489965582057503", session_id=4,
        )
        sc = H.build_history_scope(conv, "family")
        assert sc["sender_pid"] == "652096670653218826"
        assert sc["chat_id"] == "1567489965582057503"


class TestRegistryInjection:

    def _reg(self, seen):
        reg = ToolRegistry()

        async def handler(query, _scope=None, **kw):
            seen["scope"] = _scope
            return "ok"

        reg.register(name="history_search", description="d",
                     parameters={"type": "object", "properties": {}},
                     handler=handler, permission=0o444)
        return reg

    async def test_model_supplied_scope_is_overwritten(self):
        seen = {}
        conv = SimpleNamespace(
            inbound=SimpleNamespace(platform="telegram", chat_id="-1001185869359",
                                    sender_id="1544010"),
            user={"platform": "telegram", "platform_id": "3478983"},
            is_group=True, chat_id="-1001185869359", session_id=9,
        )
        forged = {"access": "owner", "is_group": False, "platform": "telegram",
                  "chat_id": "3478983", "sender_pid": "3478983"}
        await self._reg(seen).execute(
            "history_search", {"query": "x", "_scope": forged}, "family", conv=conv,
        )
        assert seen["scope"]["access"] == "family"
        assert seen["scope"]["is_group"] is True
        assert seen["scope"]["sender_pid"] == "1544010"

    async def test_no_conv_no_scope(self):
        seen = {}
        await self._reg(seen).execute("history_search", {"query": "x"}, "owner")
        assert seen["scope"] is None

    async def test_explicit_scope_wins(self):
        seen = {}
        await self._reg(seen).execute(
            "history_search", {"query": "x"}, "owner", history_scope=OWNER_DM,
        )
        assert seen["scope"] == OWNER_DM


class TestPermissions:

    def test_spawn_subagent_owner_only(self):
        assert get_permission_digit(TOOL_PERMISSIONS["spawn_subagent"], "family") == 0
        assert get_permission_digit(TOOL_PERMISSIONS["spawn_subagent"], "public") == 0

    def test_group_keeps_history_drops_subagent(self):
        tools = [{"type": "function", "function": {"name": n}}
                 for n in ("history_search", "history_expand", "spawn_subagent")]
        names = [t["function"]["name"] for t in filter_tools_for_group(tools)]
        assert names == ["history_search", "history_expand"]
