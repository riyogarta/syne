"""history.auto_recall = history_search + history_expand (Riyo, 12 Nov 2026).

auto_recall calls the tool handlers themselves. These lock the two internal
knobs it needs and that the model can never set.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from syne.tools import history as H
from syne.tools.registry import ToolRegistry

OWNER_DM = {"access": "owner", "platform": "telegram", "chat_id": "3478983",
            "is_group": False, "sender_pid": "3478983", "session_id": 7}


def _conn_ctx(fetch):
    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=fetch)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


async def _run(**kw):
    cap = {}

    async def fetch(sql, *params):
        cap["sql"], cap["params"] = sql, params
        return []

    with patch("syne.tools.history.get_connection", return_value=_conn_ctx(fetch)):
        await H.history_search_handler(query="x", _scope=OWNER_DM, **kw)
    return cap


async def test_exclude_session_adds_predicate():
    prov = MagicMock()
    prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1]))
    H.set_embedding_provider(prov)
    cap = await _run(_exclude_session=7)
    assert "m.session_id <> $" in cap["sql"]
    assert 7 in cap["params"]


async def test_without_exclude_no_predicate():
    prov = MagicMock()
    prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1]))
    H.set_embedding_provider(prov)
    cap = await _run()
    assert "m.session_id <>" not in cap["sql"]


async def test_query_vector_skips_embedding():
    prov = MagicMock()
    prov.embed = AsyncMock(side_effect=AssertionError("must not embed"))
    H.set_embedding_provider(prov)
    cap = await _run(_query_vector=[0.5, 0.25])
    assert cap["params"][0] == str([0.5, 0.25])
    prov.embed.assert_not_called()


async def test_registry_strips_internal_knobs():
    seen = {}
    reg = ToolRegistry()

    async def handler(query, _scope=None, **kw):
        seen.update(kw)
        return "ok"

    reg.register(name="history_search", description="d",
                 parameters={"type": "object", "properties": {}},
                 handler=handler, permission=0o444)
    conv = SimpleNamespace(
        inbound=SimpleNamespace(platform="telegram", chat_id="3478983", sender_id="3478983"),
        user={"platform": "telegram", "platform_id": "3478983"},
        is_group=False, chat_id="3478983", session_id=7,
    )
    await reg.execute("history_search",
                      {"query": "x", "_exclude_session": 1, "_query_vector": [1.0]},
                      "owner", conv=conv)
    assert "_exclude_session" not in seen
    assert "_query_vector" not in seen
