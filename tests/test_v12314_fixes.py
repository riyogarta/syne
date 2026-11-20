"""v1.23.14: overlapping embed timeouts count once, a refused turn holds its
history embed, the embedding switch checks memory and messages and switches
empty typed columns, KG names compare only with vectors of their dimension."""
import asyncio
import inspect
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from syne.conversation import Conversation
from syne.memory import embed_dims as D
from syne.memory import kg_extract as K


@pytest.fixture
def eh(monkeypatch):
    from syne import embed_health as E
    for k, v in (("_since", None), ("_error", ""), ("_driver", ""), ("_incident", 0),
                 ("_notified_incident", 0), ("_groups_told", set()), ("_soft_fails", 0),
                 ("_soft_fail_at", 0.0)):
        monkeypatch.setattr(E, k, v)
    monkeypatch.setattr(E, "recall_timeout", AsyncMock(return_value=0.2))
    monkeypatch.setattr(E, "active_embedding_entry", AsyncMock(return_value={"driver": "ollama"}))
    monkeypatch.setattr(E, "schedule_owner_notice", lambda: None)
    return E


class _Slow:
    async def embed_fast(self, text, timeout=10.0):
        await asyncio.sleep(timeout + 0.05)
        raise asyncio.TimeoutError()


async def test_two_concurrent_timeouts_are_not_down(eh):
    from syne.llm.provider import EmbeddingUnavailableError
    res = await asyncio.gather(eh.quick_embed(_Slow(), "a b"), eh.quick_embed(_Slow(), "c d"),
                               return_exceptions=True)
    assert all(isinstance(r, EmbeddingUnavailableError) for r in res)
    assert not eh.is_flagged()


async def test_sequential_timeouts_still_mark_down(eh):
    from syne.llm.provider import EmbeddingUnavailableError
    for _ in range(2):
        with pytest.raises(EmbeddingUnavailableError):
            await eh.quick_embed(_Slow(), "a b")
    assert eh.is_flagged()


async def test_old_soft_failure_expires(eh, monkeypatch):
    from syne.llm.provider import EmbeddingUnavailableError
    with pytest.raises(EmbeddingUnavailableError):
        await eh.quick_embed(_Slow(), "a b")
    monkeypatch.setattr(eh, "_soft_fail_at", time.monotonic() - eh.SOFT_FAIL_WINDOW - 1)
    with pytest.raises(EmbeddingUnavailableError):
        await eh.quick_embed(_Slow(), "a b")
    assert not eh.is_flagged()


# ── refused turn holds its history embed ─────────────────────────────────

def _conv():
    c = object.__new__(Conversation)
    c._deferred_embed = []
    c._recall_busy = False
    c.seen = []

    async def fake(mid, content):
        c.seen.append(mid)
    c._embed_message_row = fake
    return c


async def test_hold_then_next_turn_flushes_both():
    c = _conv()
    c._deferred_embed.append((1, "panjang"))
    c._hold_deferred_embed(delay=10)
    await asyncio.sleep(0)
    assert c.seen == [] and c._deferred_embed == [(1, "panjang")]
    c._deferred_embed.append((2, "retry"))
    c._flush_deferred_embed()
    await asyncio.sleep(0)
    assert c.seen == [1, 2]


async def test_hold_runs_late_when_no_turn(monkeypatch):
    from syne import embed_health as E
    monkeypatch.setattr(E, "_since", None)
    c = _conv()
    c._deferred_embed.append((5, "x"))
    c._hold_deferred_embed(delay=0.01)
    await asyncio.sleep(0.05)
    assert c.seen == [5] and c._deferred_embed == []


async def test_hold_waits_while_down(monkeypatch):
    from syne import embed_health as E
    monkeypatch.setattr(E, "_since", 1.0)
    c = _conv()
    c._deferred_embed.append((5, "x"))
    c._hold_deferred_embed(delay=0.01)
    await asyncio.sleep(0.05)
    assert c.seen == [] and c._deferred_embed == [(5, "x")]


def test_chat_inner_holds_on_refusal():
    src = inspect.getsource(Conversation._chat_inner)
    i = src.index("self.build_context(")
    seg = src[i:i + 600]
    assert "except EmbeddingUnavailableError:" in seg and "self._hold_deferred_embed()" in seg


# ── embedding switch ─────────────────────────────────────────────────────

class _DimConn:
    def __init__(self, cols):
        self.cols = cols   # table -> (column dims or None, vector dims or None, count)

    async def fetchval(self, q, *a):
        if "format_type" in q:
            col = self.cols[a[0]][0]
            return f"vector({col})" if col else "vector"
        t = next(t for t in self.cols if f"FROM {t} " in q)
        if "count(*)" in q:
            return self.cols[t][2]
        return self.cols[t][1]


async def test_empty_typed_columns_switch():
    c = _DimConn({"memory": (1024, None, 0), "messages": (1024, None, 0)})
    plan = await D.plan_switch(c, 768)
    assert plan == {"blocked": [], "relax": ["memory", "messages"]}
    assert D.refusal_for_plan(plan, 768) is None


async def test_history_vectors_block_switch():
    c = _DimConn({"memory": (None, None, 0), "messages": (1024, 1024, 50)})
    plan = await D.plan_switch(c, 768)
    assert plan["blocked"] == [("messages", 1024, 50)]
    assert "chat history" in D.refusal_for_plan(plan, 768)


async def test_memory_vectors_block_with_old_message():
    c = _DimConn({"memory": (1024, 1024, 9), "messages": (1024, 1024, 5)})
    plan = await D.plan_switch(c, 768)
    assert "Memories stored: <b>9</b>" in D.refusal_for_plan(plan, 768)


async def test_same_dims_nothing_to_do():
    c = _DimConn({"memory": (1024, 1024, 9), "messages": (1024, 1024, 5)})
    assert await D.plan_switch(c, 1024) == {"blocked": [], "relax": []}


def test_kg_entities_not_checked():
    assert "kg_entities" not in D.TABLES


def test_kg_resolve_compares_same_dimension_only():
    src = inspect.getsource(K.resolve_entity)
    assert "vector_dims(embedding) = $4" in src and "len(vec)" in src


def test_apply_embedding_relaxes_columns():
    from syne.communication import telegram as T
    src = inspect.getsource(T)
    assert "await apply_switch(conn, dims," in src
    assert "relax_columns(conn, plan[\"relax\"], nd)" in inspect.getsource(D.apply_switch)
    assert "stored_embedding_dims" not in src
