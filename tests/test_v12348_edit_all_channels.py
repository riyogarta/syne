"""v1.23.48: edits are processed again on every channel.

- Telegram: an edited caption, or a replaced photo/document, is processed
  again like an edited text (v1.23.47).
- WhatsApp: a MESSAGE_EDIT protocol message is read (wrapped or already
  unwrapped by whatsmeow) and processed as the new version of key.ID.
- Discord: on_raw_message_edit processes a real edit (edited_at set, text
  or attachments changed) through _handle_message with an edit key.
- turns.TurnTracker cancels only the turn of the edited message.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as N

import pytest
from telegram import Chat, Message, PhotoSize, Document, Update, User

from syne.communication.telegram import TelegramChannel
from syne.communication.turns import TurnTracker, edit_marker


async def _forever():
    await asyncio.Event().wait()


# ── TurnTracker ─────────────────────────────────────────────────────────

async def test_tracker_cancels_only_edited_turn():
    t = TurnTracker()
    a = asyncio.create_task(_forever())
    b = asyncio.create_task(_forever())
    t._turns["c"] = {"m1": a, "m2": b}
    assert t.cancel("c", "m2")
    await asyncio.sleep(0)
    assert b.cancelled() and not a.done()
    assert not t.cancel("c", "nope")
    a.cancel()


async def test_tracker_register_and_drop():
    t = TurnTracker()
    me = t.register("c", 5)
    assert t.tasks("c") == [me]
    t.drop("c", 5, object())          # someone else's task: kept
    assert t.tasks("c") == [me]
    t.drop("c", 5, me)
    assert t.tasks("c") == []


def test_edit_marker_mentions_message():
    assert "message 42" in edit_marker(42)


# ── Telegram photo / document edits ─────────────────────────────────────

def _channel():
    ch = TelegramChannel.__new__(TelegramChannel)
    ch._seen_inbound = {}
    ch._active_tasks = {}
    return ch


def _edited(kind, caption):
    extra = {}
    if kind == "photo":
        extra["photo"] = (PhotoSize("f", "u", 10, 10),)
    else:
        extra["document"] = Document("f", "u", file_name="a.pdf")
    m = Message(
        message_id=9,
        date=datetime(2026, 11, 26, tzinfo=timezone.utc),
        chat=Chat(id=-100, type="supergroup"),
        from_user=User(id=1, first_name="Riyo", is_bot=False),
        caption=caption,
        edit_date=datetime.fromtimestamp(1_800_000_000, timezone.utc),
        **extra,
    )
    return Update(update_id=5, edited_message=m)


@pytest.mark.parametrize("kind", ["photo", "document"])
async def test_group_media_edit_not_for_us_cancels_nothing(kind):
    ch = _channel()
    running = asyncio.create_task(_forever())
    ch._active_tasks[-100] = {9: running}
    seen = []

    async def not_for_us(update, context, text):
        seen.append(text)
        return None

    ch._process_group_message = not_for_us
    handler = ch._handle_photo if kind == "photo" else ch._handle_document
    await handler(_edited(kind, "caption baru tanpa mention"), None)
    await asyncio.sleep(0)
    assert seen == ["caption baru tanpa mention"]   # the edit was read
    assert not running.done()
    running.cancel()


async def test_media_edit_redelivery_is_duplicate():
    ch = _channel()
    calls = []

    async def route(update, context, text):
        calls.append(text)
        return None

    ch._process_group_message = route
    up = _edited("photo", "x")
    await ch._handle_photo(up, None)
    await ch._handle_photo(up, None)
    assert calls == ["x"]


# ── WhatsApp edits ──────────────────────────────────────────────────────

neonize = pytest.importorskip("neonize")
from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (  # noqa: E402
    FutureProofMessage, ImageMessage, Message as WAMsg, ProtocolMessage,
)
from neonize.proto.waCommon.WACommon_pb2 import MessageKey  # noqa: E402
from syne.communication.whatsapp import WhatsAppChannel  # noqa: E402


def _pm(new, orig="ORIG1", ts=1234):
    return ProtocolMessage(
        type=ProtocolMessage.MESSAGE_EDIT, key=MessageKey(ID=orig),
        editedMessage=new, timestampMS=ts,
    )


def test_wa_edit_unwrapped():
    new = WAMsg(conversation="Molt versi baru")
    got = WhatsAppChannel._extract_edit(WAMsg(protocolMessage=_pm(new)))
    assert got[0] == "ORIG1" and got[2] == 1234
    assert got[1].conversation == "Molt versi baru"


def test_wa_edit_wrapped_in_future_proof():
    new = WAMsg(imageMessage=ImageMessage(caption="Molt caption baru"))
    wrapped = WAMsg(editedMessage=FutureProofMessage(message=WAMsg(protocolMessage=_pm(new))))
    got = WhatsAppChannel._extract_edit(wrapped)
    assert got[0] == "ORIG1"
    assert got[1].imageMessage.caption == "Molt caption baru"


def test_wa_other_protocol_messages_are_not_edits():
    revoke = ProtocolMessage(type=ProtocolMessage.REVOKE, key=MessageKey(ID="X"))
    assert WhatsAppChannel._extract_edit(WAMsg(protocolMessage=revoke)) is None
    assert WhatsAppChannel._extract_edit(WAMsg(conversation="biasa")) is None


# ── Discord edits ───────────────────────────────────────────────────────

pytest.importorskip("discord")
from syne.communication.discord import DiscordChannel  # noqa: E402


def _dc():
    ch = DiscordChannel.__new__(DiscordChannel)
    ch.calls = []

    async def fake(message, edit_key=None):
        ch.calls.append((message.id, edit_key))

    ch._handle_message = fake
    return ch


def _dmsg(content, atts=(), edited=True):
    return N(id=77, content=content, attachments=[N(id=i) for i in atts],
             edited_at=datetime.fromtimestamp(1_800_000_000, timezone.utc) if edited else None)


async def test_dc_real_edit_is_processed():
    ch = _dc()
    await ch._handle_edit(N(message=_dmsg("baru"), cached_message=_dmsg("lama")))
    assert ch.calls == [(77, 1_800_000_000_000)]


async def test_dc_replaced_attachment_is_processed():
    ch = _dc()
    await ch._handle_edit(N(message=_dmsg("x", [2]), cached_message=_dmsg("x", [1])))
    assert len(ch.calls) == 1


async def test_dc_non_edit_updates_ignored():
    ch = _dc()
    await ch._handle_edit(N(message=_dmsg("x", edited=False), cached_message=None))
    await ch._handle_edit(N(message=_dmsg("sama"), cached_message=_dmsg("sama")))
    assert ch.calls == []


def test_dc_edit_dedup_key_separate_from_message():
    ch = DiscordChannel.__new__(DiscordChannel)
    ch._seen, ch._seen_order = set(), []
    assert not ch._is_duplicate(77)
    assert not ch._is_duplicate((77, 1))
    assert ch._is_duplicate((77, 1))
    assert not ch._is_duplicate((77, 2))
