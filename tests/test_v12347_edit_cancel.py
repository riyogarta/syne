"""v1.23.47: edited Telegram messages are re-read; turns wait in order.

- An edited text message is processed again. The turn still working on (or
  queued with) the old version is cancelled; other turns are not touched.
- /cancel stops the running turn of the chat, not the newest waiting one.
- The session lock waits for a long turn as long as it shows activity (an LLM
  call or tool in flight, or one finished recently) and replaces the lock only
  after _LOCK_IDLE_S without any.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as N

import pytest
from telegram import Chat, Message, Update, User

import syne.conversation as conv_mod
from syne.communication.telegram import TelegramChannel


# ── helpers ─────────────────────────────────────────────────────────────

def _channel():
    ch = TelegramChannel.__new__(TelegramChannel)
    ch._seen_inbound = {}
    ch._active_tasks = {}
    ch._auth_state = {}
    ch._remote_node = {}
    ch._xai_device_tasks = {}
    return ch


def _msg(mid=7, text="halo", chat_type="private", edit_ts=None):
    return Message(
        message_id=mid,
        date=datetime(2026, 11, 26, tzinfo=timezone.utc),
        chat=Chat(id=1 if chat_type == "private" else -100, type=chat_type),
        from_user=User(id=1, first_name="Riyo", is_bot=False),
        text=text,
        edit_date=datetime.fromtimestamp(edit_ts, timezone.utc) if edit_ts else None,
    )


async def _forever():
    await asyncio.Event().wait()


def _now():
    return conv_mod._ts_time.monotonic()


def _conv():
    c = conv_mod.Conversation.__new__(conv_mod.Conversation)
    c._lock = asyncio.Lock()
    c.session_id = 1
    c._processing = False
    c._inflight = 0
    c._last_activity = _now()
    return c


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(conv_mod, "_LOCK_POLL_S", 0.01)
    monkeypatch.setattr(conv_mod, "_LOCK_IDLE_S", 0.1)
    monkeypatch.setattr(conv_mod, "_LOCK_BUSY_MAX_S", 0.5)


# ── edited messages ─────────────────────────────────────────────────────

def test_edited_update_is_rewrapped_as_message():
    m = _msg(edit_ts=1_800_000_000)
    plain = TelegramChannel._edited_as_message(Update(update_id=3, edited_message=m))
    assert plain.message is m
    assert plain.edited_message is None
    assert plain.effective_chat.id == 1


def test_each_edit_is_new_but_redelivery_is_duplicate():
    ch = _channel()
    assert not ch._is_duplicate_inbound(1, 7)
    assert ch._is_duplicate_inbound(1, 7)
    k1 = TelegramChannel._edit_key(_msg(edit_ts=1_800_000_000))
    k2 = TelegramChannel._edit_key(_msg(edit_ts=1_800_000_060))
    assert not ch._is_duplicate_inbound(1, 7, k1)
    assert ch._is_duplicate_inbound(1, 7, k1)
    assert not ch._is_duplicate_inbound(1, 7, k2)


async def test_edit_cancels_only_its_own_turn():
    ch = _channel()
    a = asyncio.create_task(_forever())
    b = asyncio.create_task(_forever())
    ch._active_tasks[1] = {10: a, 11: b}
    assert ch._cancel_turn_for_edit(1, 11)
    await asyncio.sleep(0)
    assert b.cancelled()
    assert not a.done()
    assert not ch._cancel_turn_for_edit(1, 99)
    a.cancel()


async def test_group_edit_without_mention_cancels_nothing():
    ch = _channel()
    running = asyncio.create_task(_forever())
    ch._active_tasks[-100] = {7: running}

    async def not_for_us(update, context, text):
        return None

    ch._process_group_message = not_for_us
    up = Update(update_id=4, edited_message=_msg(
        chat_type="supergroup", text="edit tanpa mention", edit_ts=1_800_000_000))
    await ch._handle_message(up, None)
    await asyncio.sleep(0)
    assert not running.done()
    running.cancel()


def test_drop_turn_keeps_newer_entry():
    ch = _channel()
    old, new = object(), object()
    ch._active_tasks[1] = {10: new}
    ch._drop_turn(1, 10, old)
    assert ch._active_tasks[1][10] is new
    ch._drop_turn(1, 10, new)
    assert 1 not in ch._active_tasks


# ── /cancel ─────────────────────────────────────────────────────────────

async def test_cancel_stops_running_turn_not_waiting_one():
    ch = _channel()
    ch.agent = N(subagents=None, conversations=None)
    a = asyncio.create_task(_forever())   # running (arrived first)
    b = asyncio.create_task(_forever())   # waiting behind a
    ch._active_tasks[1] = {10: a, 11: b}
    replies = []

    async def reply_text(t):
        replies.append(t)

    upd = N(effective_user=N(id=1), effective_chat=N(id=1),
            message=N(reply_text=reply_text))
    await ch._cmd_cancel(upd, None)
    await asyncio.sleep(0)
    assert a.cancelled()
    assert not b.done()
    assert replies == ["✋ Cancelled."]
    b.cancel()


# ── session lock ────────────────────────────────────────────────────────

async def test_waiter_waits_while_holder_is_active(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    w = asyncio.create_task(c._acquire_turn_lock())
    for _ in range(30):                      # 0.3 s = 3x the idle limit
        c._last_activity = _now()
        await asyncio.sleep(0.01)
    assert not w.done()
    held.release()
    assert await asyncio.wait_for(w, 1) is held


async def test_waiter_waits_while_call_in_flight(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    w = asyncio.create_task(c._acquire_turn_lock())
    with c._busy():
        await asyncio.sleep(0.3)             # longer than idle, shorter than busy max
        assert not w.done()
    held.release()
    assert await asyncio.wait_for(w, 1) is held


async def test_idle_holder_is_declared_stuck(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    got = await asyncio.wait_for(asyncio.create_task(c._acquire_turn_lock()), 2)
    assert got is not held
    assert c._lock is got and got.locked()


async def test_in_flight_holder_stuck_after_busy_max(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    c._inflight = 1
    w = asyncio.create_task(c._acquire_turn_lock())
    await asyncio.sleep(0.3)
    assert not w.done()
    got = await asyncio.wait_for(w, 2)
    assert got is not held


async def test_waiters_run_in_arrival_order(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    order = []

    async def turn(name):
        lock = await c._acquire_turn_lock()
        c._last_activity = _now()
        order.append(name)
        await asyncio.sleep(0.02)
        lock.release()

    b = asyncio.create_task(turn("B"))
    await asyncio.sleep(0.03)
    c._last_activity = _now()
    d = asyncio.create_task(turn("C"))
    await asyncio.sleep(0.03)
    c._last_activity = _now()
    held.release()
    await asyncio.wait_for(asyncio.gather(b, d), 2)
    assert order == ["B", "C"]


async def test_cancelled_waiter_leaves_queue(fast):
    c = _conv()
    held = await c._acquire_turn_lock()
    b = asyncio.create_task(c._acquire_turn_lock())
    d = asyncio.create_task(c._acquire_turn_lock())
    await asyncio.sleep(0.03)
    c._last_activity = _now()
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b
    held.release()
    got = await asyncio.wait_for(d, 1)
    assert got is held and held.locked()
    got.release()
    assert not held.locked()
