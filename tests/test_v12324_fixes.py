"""v1.23.24: a hung Ollama reaches DOWN even while history embeds run
(review v1.23.16-19 bug 4)."""
import asyncio
import time

import pytest

from syne import embed_health as E


@pytest.fixture
def eh(monkeypatch):
    for k, v in (("_since", None), ("_error", ""), ("_driver", ""), ("_incident", 0),
                 ("_notified_incident", 0), ("_groups_told", set()), ("_soft_fails", 0),
                 ("_soft_fail_at", 0.0), ("_bg_inflight", 0), ("_bg_last_end", 0.0),
                 ("_bg_started", {}), ("_excused", 0), ("_excused_at", 0.0)):
        monkeypatch.setattr(E, k, v)
    notices = []

    async def rt():
        return 0.02

    async def entry():
        return {"driver": "ollama"}
    monkeypatch.setattr(E, "recall_timeout", rt)
    monkeypatch.setattr(E, "active_embedding_entry", entry)
    monkeypatch.setattr(E, "schedule_owner_notice", lambda: notices.append(1))
    E.notices = notices
    return E


class _Timeout:
    async def embed_fast(self, text, timeout=None):
        await asyncio.sleep(0.005)
        raise asyncio.TimeoutError()


class _Ok:
    async def embed_fast(self, text, timeout=None):
        return type("R", (), {"vector": [0.1]})()


async def _refuse(n):
    for _ in range(n):
        with pytest.raises(E.EmbeddingUnavailableError):
            await E.quick_embed(_Timeout(), "q")


async def test_hung_background_embed_does_not_excuse(eh, monkeypatch):
    async with E.background_embed():
        # the running history embed is older than the grace: hung, not busy
        tok = next(iter(E._bg_started))
        E._bg_started[tok] = time.monotonic() - E.BG_BUSY_GRACE - 1
        await _refuse(2)
    assert E.is_flagged() and E.notices == [1]


async def test_young_background_embed_excuses_at_most_max(eh):
    async with E.background_embed():
        await _refuse(E.BG_EXCUSE_MAX)
        assert not E.is_flagged() and E._soft_fails == 0
        await _refuse(2)
    assert E.is_flagged() and E.notices == [1]


async def test_failed_background_embed_end_does_not_excuse(eh):
    async def bg_fails():
        with pytest.raises(RuntimeError):
            async with E.background_embed():
                await asyncio.sleep(0.002)
                raise RuntimeError("ollama timeout")
    for _ in range(2):
        t = asyncio.create_task(bg_fails())
        with pytest.raises(E.EmbeddingUnavailableError):
            await E.quick_embed(_Timeout(), "q")
        await t
    assert E.is_flagged()


async def test_successful_background_end_still_excuses(eh):
    async def bg_ok():
        async with E.background_embed():
            await asyncio.sleep(0.002)
    t = asyncio.create_task(bg_ok())
    with pytest.raises(E.EmbeddingUnavailableError):
        await E.quick_embed(_Timeout(), "q")
    await t
    assert E._excused == 1 and E._soft_fails == 0 and not E.is_flagged()


async def test_success_resets_excuses(eh):
    async with E.background_embed():
        await _refuse(E.BG_EXCUSE_MAX)
        assert await E.quick_embed(_Ok(), "q") == [0.1]
        assert E._excused == 0
        await _refuse(E.BG_EXCUSE_MAX)
    assert not E.is_flagged()
