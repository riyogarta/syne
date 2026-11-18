"""1.22.95: background Ollama embeds yield to recall embeds in flight, so a
recall never queues behind a new long background request (one request at a
time on Ollama) and misses embedding.recall_timeout."""
import asyncio

import syne.llm.ollama as ol


async def test_background_waits_for_recall(monkeypatch):
    ev = asyncio.Event()
    monkeypatch.setattr(ol, "_fast_idle", ev)  # fresh, bound to this loop
    t = asyncio.create_task(ol._yield_to_recall())
    await asyncio.sleep(0.05)
    assert not t.done()
    ev.set()
    await asyncio.wait_for(t, 1)


async def test_yield_is_bounded(monkeypatch):
    monkeypatch.setattr(ol, "_fast_idle", asyncio.Event())  # never set
    monkeypatch.setattr(ol, "_FAST_YIELD_MAX", 0.05)
    await asyncio.wait_for(ol._yield_to_recall(), 1)


async def test_no_wait_when_idle(monkeypatch):
    ev = asyncio.Event()
    ev.set()
    monkeypatch.setattr(ol, "_fast_idle", ev)
    await asyncio.wait_for(ol._yield_to_recall(), 0.01)


async def test_embed_fast_marks_inflight(monkeypatch):
    ev = asyncio.Event()
    ev.set()
    monkeypatch.setattr(ol, "_fast_idle", ev)
    monkeypatch.setattr(ol, "_fast_inflight", 0)
    seen = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"embeddings": [[0.1, 0.2]]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            seen.append(ol._fast_idle.is_set())
            return _Resp()

    monkeypatch.setattr(ol.httpx, "AsyncClient", _Client)
    p = object.__new__(ol.OllamaProvider)
    p.embedding_model = "m"
    p.base_url = "http://x"
    r = await p.embed_fast("a b", timeout=1)
    assert r.vector == [0.1, 0.2]
    assert seen == [False]          # background embeds held while in flight
    assert ol._fast_idle.is_set()   # released afterwards
    assert ol._fast_inflight == 0
