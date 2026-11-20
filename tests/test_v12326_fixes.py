"""v1.23.26: held history embeds re-check before every row; history embed
tasks are tracked by the restore gate (review v1.23.16-19 #1 partial)."""
import asyncio

from syne import embed_health as E
from syne import maintenance as MT
from syne.conversation import Conversation


def _conv(on_embed=None):
    c = object.__new__(Conversation)
    c._deferred_embed = []
    c._recall_busy = False
    c.seen = []

    async def fake(mid, content):
        c.seen.append(mid)
        if on_embed:
            on_embed(mid)
        await asyncio.sleep(0)
    c._embed_message_row = fake
    return c


async def test_recall_starting_mid_batch_stops_the_rest(monkeypatch):
    monkeypatch.setattr(E, "_since", None)
    monkeypatch.setattr(E, "_recall_inflight", 0)

    def recall_starts(mid):
        if mid == 1:
            E._recall_inflight = 1  # another chat begins a recall
    c = _conv(recall_starts)
    c._deferred_embed += [(1, "a"), (2, "b"), (3, "c")]
    c._hold_deferred_embed(delay=0.01)
    await asyncio.sleep(0.015)
    assert c.seen == [1] and c._deferred_embed == [(2, "b"), (3, "c")]
    # the recall ends: the next round embeds the rest
    E._recall_inflight = 0
    await asyncio.sleep(0.03)
    assert c.seen == [1, 2, 3] and c._deferred_embed == []


async def test_held_embed_gives_up_after_rounds(monkeypatch):
    monkeypatch.setattr(E, "_since", 1.0)
    c = _conv()
    c._deferred_embed.append((5, "x"))
    monkeypatch.setattr(MT, "_turns", set())
    c._hold_deferred_embed(delay=0.005)
    await asyncio.sleep(0.005 * Conversation.LATE_ROUNDS + 0.03)
    assert c.seen == [] and c._deferred_embed == [(5, "x")] and not MT._turns


async def test_flushed_embeds_are_tracked_and_cancelled_by_restore(monkeypatch):
    monkeypatch.setattr(MT, "_turns", set())
    c = object.__new__(Conversation)
    started = asyncio.Event()

    async def slow(mid, content):
        started.set()
        await asyncio.sleep(10)
    c._embed_message_row = slow
    c._deferred_embed = [(7, "x")]
    c._flush_deferred_embed()
    assert len(MT._turns) == 1
    await started.wait()
    n = await MT.begin("test", timeout=1)
    MT.end()
    await asyncio.sleep(0)
    assert n == 1 and not MT._turns


async def test_no_embed_starts_during_restore(monkeypatch):
    monkeypatch.setattr(MT, "_turns", set())
    monkeypatch.setattr(MT, "_reason", "restore")
    c = _conv()
    c._spawn_embed(9, "x")
    await asyncio.sleep(0)
    assert c.seen == [] and not MT._turns
