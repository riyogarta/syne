"""1.22.81 hardening: MEDIA: trust, API key redaction, guard scoping,
reaction gate, blocked Delete, WhatsApp photo size cap."""

import inspect
from unittest.mock import AsyncMock

import pytest

from syne.communication.errors import classify_error, redact_secrets
from syne.communication.outbound import strip_media_lines


# ── MEDIA: lines from model text ─────────────────────────────────────

def test_strip_media_lines_removes_whole_lines_with_spaces():
    text = "Halo\n\nMEDIA: /home/syne/.env\nMEDIA: /tmp/a b c.pdf\nselesai"
    out = strip_media_lines(text)
    assert "MEDIA" not in out
    assert "a b c" not in out
    assert out.startswith("Halo") and out.endswith("selesai")


def test_strip_media_lines_keeps_inline_mention():
    text = "Baris yang memuat kata MEDIA: di tengah kalimat tetap ada."
    assert strip_media_lines(text) == text


def test_strip_media_lines_empty():
    assert strip_media_lines("") == ""
    assert strip_media_lines(None) is None


def test_only_file_tools_are_trusted_for_media():
    from syne import conversation as C
    assert C._MEDIA_TOOLS == frozenset({"send_file", "memory_get_file"})
    src = inspect.getsource(C)
    assert "if name in _MEDIA_TOOLS and (" in src
    # Final reply always strips model-written MEDIA: lines.
    assert "final_response = strip_media_lines(final_response)" in src
    assert "re.sub(r'\\n*MEDIA:\\s*\\S*'" not in src


async def test_send_message_strips_media_lines_from_text(monkeypatch):
    from syne.tools import send_message as SM

    sent = {}

    class _TG:
        async def _send_response_with_media(self, chat_id, text, reply_to_message_id=None):
            sent["text"] = text
            return None

    monkeypatch.setattr(SM, "_telegram_channel", _TG())
    res = await SM.send_message_handler(
        chat_id="12345", message="hai\n\nMEDIA: /home/syne/.env", _access_level="owner",
    )
    assert "Error" not in res
    assert sent["text"] == "hai"


async def test_send_message_media_only_text_is_empty_error():
    from syne.tools import send_message as SM
    res = await SM.send_message_handler(
        chat_id="12345", message="MEDIA: /etc/passwd", _access_level="owner",
    )
    assert res == "Error: message cannot be empty"


# ── API key never in errors / URLs ───────────────────────────────────

KEY = "AIza" + "A" * 35


def test_redact_query_key():
    s = f"Client error '400' for url 'https://x.googleapis.com/v1/m:embed?key={KEY}&alt=sse'"
    out = redact_secrets(s)
    assert KEY not in out and "?key=***" in out and "&alt=sse" in out


def test_redact_bare_google_key():
    assert KEY not in redact_secrets(f"oops {KEY} oops")


def test_classify_error_redacts_for_owner():
    msg = classify_error(RuntimeError(f"url ...?key={KEY}"), access_level="owner")
    assert KEY not in msg


def test_google_provider_sends_key_in_header():
    from syne.llm import google as G
    src = inspect.getsource(G)
    assert 'params={"key"' not in src
    assert '"key": self.api_key' not in src
    assert src.count('"x-goog-api-key": self.api_key') == 3


# ── Memory guard scoped to the speaker ───────────────────────────────

async def test_check_draft_passes_user_id(monkeypatch):
    from syne import memory_guard as MG

    fc = AsyncMock(return_value=[])
    monkeypatch.setattr(MG, "find_candidates", fc)

    class _V:
        vector = [0.1, 0.2]

    await MG.check_draft(
        draft="ini draft yang cukup panjang untuk diperiksa oleh guard memori",
        user_message="x", embed=AsyncMock(return_value=_V()),
        judge=AsyncMock(return_value="CLEAN"), exclude_ids=[],
        access_level="owner",
        cfg={"categories": ["fact"], "limit": 5, "min_similarity": 0.5, "timeout": 5},
        user_id=42,
    )
    assert fc.call_args.kwargs["user_id"] == 42


def test_find_candidates_sql_filters_user():
    from syne import memory_guard as MG
    src = inspect.getsource(MG.find_candidates)
    assert "($5::int IS NULL OR user_id = $5::int OR user_id IS NULL)" in src


def test_conversation_passes_user_id_to_guard():
    from syne import conversation as C
    assert 'user_id=(getattr(self, "user", None) or {}).get("id"),' in inspect.getsource(C)


# ── Telegram reactions pass the message gates ────────────────────────

class _Chat:
    def __init__(self, type_, id_=1):
        self.type, self.id = type_, id_


class _User:
    id = 777


def _tg():
    from syne.communication.telegram import TelegramChannel
    return TelegramChannel.__new__(TelegramChannel)


@pytest.mark.parametrize("row,policy,expected", [
    (None, "approval", False),
    ({"access_level": "blocked", "preferences": {}}, "approval", False),
    ({"access_level": "public", "preferences": {"pending_approval": True}}, "approval", False),
    ({"access_level": "public", "preferences": {"pending_approval": "revoked"}}, "approval", False),
    ({"access_level": "public", "preferences": {"pending_approval": True}}, "open", True),
    ({"access_level": "public", "preferences": {}}, "approval", True),
    ({"access_level": "family", "preferences": {}}, "approval", True),
])
async def test_reaction_gate_dm(monkeypatch, row, policy, expected):
    from syne.communication import telegram as T
    monkeypatch.setattr(T, "get_user", AsyncMock(return_value=row))
    monkeypatch.setattr(T, "get_config", AsyncMock(return_value=policy))
    assert await _tg()._reaction_sender_allowed(_Chat("private"), _User()) is expected


@pytest.mark.parametrize("group,policy,user_row,expected", [
    (None, "allowlist", {"id": 1}, False),
    ({"enabled": False}, "allowlist", {"id": 1}, False),
    ({"enabled": True, "allow_from": "registered"}, "allowlist", None, False),
    ({"enabled": True, "allow_from": "registered"}, "allowlist", {"id": 1}, True),
    ({"enabled": True, "allow_from": "all"}, "allowlist", None, True),
])
async def test_reaction_gate_group(monkeypatch, group, policy, user_row, expected):
    from syne.communication import telegram as T
    monkeypatch.setattr(T, "get_group", AsyncMock(return_value=group))
    monkeypatch.setattr(T, "get_config", AsyncMock(return_value=policy))
    monkeypatch.setattr(T, "get_user", AsyncMock(return_value=user_row))
    assert await _tg()._reaction_sender_allowed(_Chat("group", -5), _User()) is expected


async def test_reaction_gate_no_user():
    assert await _tg()._reaction_sender_allowed(_Chat("private"), None) is False


def test_reaction_handler_calls_gate():
    from syne.communication.telegram import TelegramChannel
    src = inspect.getsource(TelegramChannel._handle_reaction_update)
    assert "await self._reaction_sender_allowed(chat, user)" in src


# ── Delete keeps a block ─────────────────────────────────────────────

def test_delete_refuses_blocked_on_all_member_menus():
    from syne.communication import telegram as T
    src = inspect.getsource(T)
    assert src.count("User is blocked — unblock first to revoke") == 2  # telegram + discord (v1.23.40)
    assert "Blocked — unblock first to revoke" in src                  # whatsapp (v1.23.41)


# ── WhatsApp photo size cap ──────────────────────────────────────────

def test_whatsapp_image_cap():
    from syne.communication import whatsapp as W
    assert W._IMAGE_MAX_BYTES == 25 * 1024 * 1024
    src = inspect.getsource(W)
    assert "declared = int(ev.Message.imageMessage.fileLength or 0)" in src
    assert "declared = int(q.imageMessage.fileLength or 0)" in src


async def test_quoted_image_too_large_not_downloaded():
    from syne.communication import whatsapp as W

    class _IM:
        fileLength = W._IMAGE_MAX_BYTES + 1
        mimetype = "image/jpeg"

    class _Q:
        imageMessage = _IM()

    ch = W.WhatsAppChannel.__new__(W.WhatsAppChannel)
    ch.client = type("C", (), {"download_any": AsyncMock(return_value=b"x")})()
    assert await ch._fetch_quoted_image(_Q()) is None
    ch.client.download_any.assert_not_called()
