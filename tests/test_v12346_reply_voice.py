"""v1.23.46: a voice note in a replied-to Telegram message is transcribed.

In groups a voice note only reaches Molt when it replies to Molt, so replying
to someone else's voice note with a mention is the way to ask about it.
Before this fix the reply carried only the label "[Voice message]".
"""
import asyncio
from types import SimpleNamespace as N

import syne.tools.voice as voice_mod
from syne.communication.telegram import TelegramChannel


class _File:
    def __init__(self, data):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class _Voice:
    def __init__(self, data=b"OggS-fake", size=None):
        self._data = data
        self.file_size = len(data) if size is None else size
        self.file_unique_id = "abc123"
        self.mime_type = "audio/ogg"

    async def get_file(self):
        return _File(self._data)


def _update(voice=None, audio=None):
    reply = N(photo=None, document=None, voice=voice, audio=audio)
    return N(message=N(reply_to_message=reply))


def _run(update):
    return asyncio.run(TelegramChannel._download_reply_media(N(), update))


def test_replied_voice_is_transcribed(monkeypatch):
    async def fake(data, filename, language=""):
        assert data == b"OggS-fake"
        assert filename == "voice_abc123.ogg"
        return True, " halo Molt "

    monkeypatch.setattr(voice_mod, "transcribe_audio", fake)
    r = _run(_update(voice=_Voice()))
    assert r["transcription"] == "halo Molt"
    assert r["audio"]["from_reply"] is True
    assert r["audio"]["mime_type"] == "audio/ogg"
    assert "audio_error" not in r


def test_replied_audio_file_is_transcribed(monkeypatch):
    async def fake(data, filename, language=""):
        return True, "isi audio"

    monkeypatch.setattr(voice_mod, "transcribe_audio", fake)
    r = _run(_update(audio=_Voice()))
    assert r["transcription"] == "isi audio"


def test_failed_transcription_reports_reason(monkeypatch):
    async def fake(data, filename, language=""):
        return False, "Groq API error 401"

    monkeypatch.setattr(voice_mod, "transcribe_audio", fake)
    r = _run(_update(voice=_Voice()))
    assert "transcription" not in r
    assert r["audio_error"] == "Groq API error 401"


def test_oversized_voice_is_not_downloaded(monkeypatch):
    async def fake(data, filename, language=""):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr(voice_mod, "transcribe_audio", fake)
    r = _run(_update(voice=_Voice(size=21 * 1024 * 1024)))
    assert r["audio_error"] == "file is larger than 20 MB"
    assert "audio" not in r
