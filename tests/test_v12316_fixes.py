"""v1.23.16: fixes from the review of v1.23.13-15 (KG worker, embed health)."""
import asyncio
import inspect
import time

import pytest

from syne import embed_health as E
from syne import maintenance as MT
from syne.conversation import Conversation
from syne.memory import kg_extract as K


# ── #7 provider_down knows Syne's own driver errors ───────────────────────

@pytest.mark.parametrize("text,down", [
    ("Anthropic stream failed after 3 attempts: ConnectError: boom", True),
    ("OpenAI stream failed after 3 attempts: ReadTimeout", True),
    ("OpenAI responses stream failed after 3 attempts: RemoteProtocolError", True),
    ("Codex stream failed after 3 attempts: ReadError", True),
    ("Anthropic transient error after 3 attempts", True),
    ("Codex API failed after 3 attempts", True),
    ("Rate limited (429) after 3 attempts", True),
    ("Anthropic API error 400: bad request", False),
    ("Claude OAuth token expired or invalid. Run 'syne init' to re-authenticate.", False),
    ("Expecting value: line 1 column 1", False),
])
def test_provider_down_driver_runtime_errors(text, down):
    assert K.provider_down(RuntimeError(text)) is down


async def test_nap_wakes_when_extraction_is_turned_off(monkeypatch):
    monkeypatch.setattr(K.asyncio, "sleep", _fast_sleep)
    calls = []

    async def cfg(key, default=None):
        calls.append(key)
        return False
    monkeypatch.setattr(K, "get_config", cfg)
    t0 = time.monotonic()
    await K._nap(1800)
    assert calls == ["kg.extract_enabled"] and time.monotonic() - t0 < 1


async def test_nap_wakes_on_restore(monkeypatch):
    monkeypatch.setattr(K.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(MT, "_reason", "restore")
    await asyncio.wait_for(K._nap(1800), 1)


_real_sleep = asyncio.sleep


async def _fast_sleep(_s):
    await _real_sleep(0)


def test_outage_never_spends_attempts_while_model_is_down():
    src = inspect.getsource(K.run_worker)
    assert "spends_attempt(kind, first, strikes, last_ok, now)" in src
    assert "await _nap(down_wait)" in src
    assert "await asyncio.sleep(down_wait)" not in src


# ── bug 1: scoped entities get their vector ───────────────────────────────

async def test_prepare_embeddings_checks_the_same_scope(monkeypatch):
    seen = []

    class _Conn:
        async def fetchval(self, q, *a):
            seen.append(a)
            # A shared "budi" exists; nothing in the group scope.
            return 1 if a[3] is None else None

    class _CM:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(K, "get_connection", lambda: _CM())

    async def embed(name):
        return [0.1, 0.2]
    ext = {"entities": [{"name": "Budi", "type": "person"}]}
    out = await K.prepare_embeddings(ext, embed, "chat:telegram:-100", frozenset())
    assert out == {"Budi": [0.1, 0.2]}
    assert seen[0][3] == "chat:telegram:-100"
    # No chat scope (memory / DM): the shared one is found, nothing embedded.
    seen.clear()
    assert await K.prepare_embeddings(ext, embed, None, frozenset()) == {}
    assert seen[0][3] is None


def test_process_one_passes_scope_to_prepare_embeddings():
    src = inspect.getsource(K.process_one)
    assert 'prepare_embeddings(extracted, embed, item.get("chat_scope")' in src


# ── bug 6: only groups scope people ───────────────────────────────────────

def test_pending_scopes_group_sessions_only():
    src = inspect.getsource(K.pending)
    assert 'group_session_sql("se")' in src and "END AS chat_scope" in src


# ── bug 7: known_people ───────────────────────────────────────────────────

async def test_known_people_tolerates_bad_directory(monkeypatch):
    class _Conn:
        async def fetch(self, q, *a):
            if "FROM groups" in q:
                return [{"platform": "telegram", "platform_group_id": "-1",
                         "settings": {"members": {"7": {}}}}]
            return [{"platform": "telegram", "platform_id": "7", "access_level": "owner",
                     "name": "Riyogarta Pratikto", "display_name": None,
                     "aliases": '{"default": "Yahyo"}'}]

        async def fetchval(self, q, *a):
            if "family.directory" in q:
                return ["oops", {"names": "Bunli"}, {"names": ["Agha", 7]}, {"names": None}]
            return "Molt"

    known = await K.known_people(_Conn())
    assert {"riyogarta pratikto", "riyogarta", "pratikto", "yahyo",
            "bunli", "agha", "molt"} <= known
    assert "b" not in known and "u" not in known
    assert K.person_scope("Riyogarta", "person", "chat:telegram:-1", known) is None
    assert K.person_scope("Budi", "person", "chat:telegram:-1", known) == "chat:telegram:-1"


# ── bug 5: worker lock connection released on cancellation ───────────────

async def test_take_worker_lock_releases_on_cancel(monkeypatch):
    released = []
    gate = asyncio.Event()

    class _Conn:
        async def fetchval(self, q, *a):
            gate.set()
            await asyncio.sleep(10)

    class _Pool:
        async def acquire(self):
            return _Conn()

        async def release(self, c):
            released.append(c)
    import syne.db.connection as DC
    monkeypatch.setattr(DC, "get_pool", lambda: _Pool())
    t = asyncio.create_task(K.take_worker_lock())
    await gate.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert len(released) == 1


# ── bug 2: held history embed vs recalls of other chats ───────────────────

def _conv():
    c = object.__new__(Conversation)
    c._deferred_embed = []
    c._recall_busy = False
    c.seen = []

    async def fake(mid, content):
        c.seen.append(mid)
    c._embed_message_row = fake
    return c


async def test_held_embed_waits_while_another_chat_recalls(monkeypatch):
    monkeypatch.setattr(E, "_since", None)
    monkeypatch.setattr(E, "_recall_inflight", 1)
    c = _conv()
    c._deferred_embed.append((5, "x"))
    c._hold_deferred_embed(delay=0.01)
    await asyncio.sleep(0.05)
    assert c.seen == [] and c._deferred_embed == [(5, "x")]


async def test_held_embed_is_cancelled_by_restore(monkeypatch):
    monkeypatch.setattr(E, "_since", None)
    monkeypatch.setattr(MT, "_turns", set())
    c = _conv()
    c._deferred_embed.append((5, "x"))
    c._hold_deferred_embed(delay=5)
    await asyncio.sleep(0)
    assert len(MT._turns) == 1
    n = await MT.begin("test", timeout=1)
    MT.end()
    assert n == 1 and c.seen == [] and not MT._turns


async def test_recall_timeout_behind_background_embed_is_not_counted(monkeypatch):
    monkeypatch.setattr(E, "_since", None)
    monkeypatch.setattr(E, "_soft_fails", 0)
    monkeypatch.setattr(E, "_soft_fail_at", 0.0)
    monkeypatch.setattr(E, "_bg_inflight", 0)
    monkeypatch.setattr(E, "_bg_last_end", 0.0)
    monkeypatch.setattr(E, "_bg_started", {})
    monkeypatch.setattr(E, "_excused", 0)
    monkeypatch.setattr(E, "schedule_owner_notice", lambda: None)

    async def rt():
        return 0.05
    monkeypatch.setattr(E, "recall_timeout", rt)

    class _P:
        async def embed_fast(self, text, timeout=None):
            await asyncio.sleep(0.01)
            raise asyncio.TimeoutError()

    # Two timeouts in a row, both while a long history embed runs: busy,
    # never DOWN, no owner DM.
    async with E.background_embed():
        for _ in range(2):
            with pytest.raises(E.EmbeddingUnavailableError):
                await E.quick_embed(_P(), "q")
    assert E._soft_fails == 0 and E._since is None
    # A recall that started before the background embed ended: same.
    async def bg_then_end():
        async with E.background_embed():
            await asyncio.sleep(0.005)
    t = asyncio.create_task(bg_then_end())
    with pytest.raises(E.EmbeddingUnavailableError):
        await E.quick_embed(_P(), "q")
    await t
    assert E._soft_fails == 0 and E._since is None
    assert E._recall_inflight == 0
    # No background embed at all: two real timeouts still mean DOWN.
    monkeypatch.setattr(E, "_bg_last_end", 0.0)
    for _ in range(2):
        with pytest.raises(E.EmbeddingUnavailableError):
            await E.quick_embed(_P(), "q")
    assert E._since is not None


async def test_quick_embed_counts_recalls_in_flight(monkeypatch):
    monkeypatch.setattr(E, "_since", None)
    seen = []

    async def rt():
        return 1.0
    monkeypatch.setattr(E, "recall_timeout", rt)

    class _P:
        async def embed_fast(self, text, timeout=None):
            seen.append(E.recall_in_flight())
            return type("R", (), {"vector": [0.1]})()
    assert await E.quick_embed(_P(), "q") == [0.1]
    assert seen == [True] and not E.recall_in_flight()
