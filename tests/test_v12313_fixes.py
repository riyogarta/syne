"""v1.23.13: scheduler survives /restore, KG worker behind the gate, linear
mention_grams, provider outage keeps chunks, judge follows the default
model, embeddings outside the write transaction, advancing queue floor."""
import asyncio
import inspect
import time

import pytest

from syne import maintenance as MT
from syne.memory import kg_extract as K
from syne.memory import kg_recall as R
from syne.scheduler import Scheduler


# ── #1 scheduler ──────────────────────────────────────────────────────────

async def test_scheduled_turn_cancel_does_not_kill_loop():
    started = asyncio.Event()

    async def on_exec(*a):
        me = asyncio.current_task()
        MT.track(me)
        try:
            started.set()
            await asyncio.sleep(10)
        finally:
            MT.untrack(me)

    sch = Scheduler(on_exec)
    runner = asyncio.create_task(sch._execute_isolated(1, "p", 1, "x", "private"))
    await started.wait()
    await MT.begin("test", timeout=2)
    try:
        assert await runner is False      # callback cancelled, caller alive
        assert not runner.cancelled()
    finally:
        MT.end()


async def test_scheduled_turn_error_propagates():
    async def boom(*a):
        raise ValueError("x")
    with pytest.raises(ValueError):
        await Scheduler(boom)._execute_isolated(1)


async def test_cancelling_loop_cancels_callback():
    hold = asyncio.Event()
    seen = {}

    async def on_exec(*a):
        try:
            hold.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
    t = asyncio.create_task(Scheduler(on_exec)._execute_isolated(1))
    await hold.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    await asyncio.sleep(0)
    assert seen.get("cancelled")


def test_scheduler_checks_gate_per_task():
    src = inspect.getsource(Scheduler._check_and_execute)
    assert "if _maint.active():" in src and "_execute_isolated" in src


# ── #2 KG worker gate ─────────────────────────────────────────────────────

def test_worker_tracks_each_chunk():
    src = inspect.getsource(K.run_worker)
    assert "_maint.track(task)" in src and "_maint.untrack(task)" in src
    assert src.index("if _maint.active():\n                        break") < src.index("_maint.track(task)")


# ── #3 mention_grams ──────────────────────────────────────────────────────

def test_mention_grams_linear_and_same_order():
    text = " ".join(f"kata{i}" for i in range(20000))
    t0 = time.monotonic()
    g = R.mention_grams(text)
    assert time.monotonic() - t0 < 5
    small = R.mention_grams("anak Riyogarta Pratikto sekolah di Jakarta")
    ref = K.ngrams("anak Riyogarta Pratikto sekolah di Jakarta")
    assert small == sorted(ref, key=lambda x: (-len(x.split()), ref.index(x)))


# ── #7 provider outage ────────────────────────────────────────────────────

class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


@pytest.mark.parametrize("err,down", [
    (asyncio.TimeoutError(), True),
    (_Status(429), True),
    (_Status(503), True),
    (_Status(400), False),
    (ConnectionError("refused"), True),
    (ValueError("bad json"), False),
])
def test_provider_down_classification(err, down):
    assert K.provider_down(err) is down


async def test_process_one_raises_provider_down(monkeypatch):
    async def judge(prompt, timeout):
        raise _Status(503)
    judge.label = "t"

    class _CM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(K, "get_connection", lambda: _CM())
    item = {"type": "memory", "id": 1, "content": "Budi tinggal di Bandung", "category": "fact"}
    s = {"kg.extract_max_chars": 6000, "kg.extract_timeout": 5}
    monkeypatch.setattr(K, "build_prompt", lambda *a, **k: "p")
    with pytest.raises(K.ProviderDown):
        await K.process_one(item, judge, None, s)


def test_error_retry_backs_off():
    assert "power(2, least(l.attempts - 1, 6))" in K._PENDING


# ── #8 judge follows the default model ────────────────────────────────────

def test_judge_key_follows_default_provider():
    a = type("P", (), {"chat_model": "m1"})()
    b = type("P", (), {"chat_model": "m2"})()
    assert K.judge_cache_key("", a) != K.judge_cache_key("", b)
    assert K.judge_cache_key("x", a) == K.judge_cache_key("x", b)


# ── #12 embeddings before the transaction ─────────────────────────────────

def test_write_graph_gets_cached_embedder():
    src = inspect.getsource(K.process_one)
    assert src.index("prepare_embeddings(") < src.index("conn.transaction()")
    assert "_cached if embed is not None else None" in src


async def test_prepare_embeddings_skips_known_names(monkeypatch):
    calls = []

    class _Conn:
        async def fetchval(self, q, *a):
            return 1 if a[0] == "bandung" else None

    class _CM:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(K, "get_connection", lambda: _CM())

    async def embed(n):
        calls.append(n)
        return [0.1]
    out = await K.prepare_embeddings(
        {"entities": [{"name": "Bandung", "type": "place"}, {"name": "Budi", "type": "person"}]},
        embed)
    assert calls == ["Budi"] and out == {"Budi": [0.1]}


# ── #13 queue floor ───────────────────────────────────────────────────────

class _FloorConn:
    def __init__(self, first, settled):
        self.first, self.settled = first, settled

    async def fetchval(self, q, *a):
        return self.first if "ORDER BY x.id LIMIT 1" in q else self.settled


_S = {"kg.extract_skip_scheduled": True, "kg.extract_max_attempts": 3}


async def test_floor_stops_below_first_open_row():
    assert await K.advance_floor(_FloorConn(120, 500), 100, None, _S) == 119


async def test_floor_stops_at_recent_rows():
    assert await K.advance_floor(_FloorConn(None, 300), 100, None, _S) == 300


async def test_floor_never_goes_back():
    assert await K.advance_floor(_FloorConn(50, 500), 100, None, _S) == 100


# ── #5 CLI ────────────────────────────────────────────────────────────────

def test_cli_bm25_rebuild_registered():
    from syne.cli import cli
    assert "rebuild" in cli.commands["bm25"].commands
