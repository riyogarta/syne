"""Discord inbound attachments and replied-to attachments (Riyo, 16 Nov 2026)."""

import asyncio
import base64
import os
from types import SimpleNamespace

import pytest

discord = pytest.importorskip("discord")

from syne.communication.discord import DiscordChannel, _ATT_MAX_BYTES  # noqa: E402


class Att:
    def __init__(self, filename, content_type=None, data=b"%PDF x", size=None,
                 exc=None, id=111):
        self.filename, self.content_type, self.data = filename, content_type, data
        self.size = len(data) if size is None else size
        self.exc, self.id, self.reads = exc, id, 0

    async def read(self):
        self.reads += 1
        if self.exc:
            raise self.exc
        return self.data


def _ch(tmp_path):
    ch = DiscordChannel.__new__(DiscordChannel)
    ch.agent = SimpleNamespace(workspace_uploads=str(tmp_path / "up"))
    return ch


def _msg(attachments=(), reference=None, channel=None):
    return SimpleNamespace(id=999, attachments=list(attachments),
                           reference=reference, channel=channel)


def run(c):
    return asyncio.run(c)


def test_pdf_attachment(tmp_path):
    media, notes = run(_ch(tmp_path)._collect_media(
        _msg([Att("Nomor 5.pdf", "application/pdf")])))
    doc = media["document"]
    assert base64.b64decode(doc["base64"]) == b"%PDF x"
    assert doc["mime_type"] == "application/pdf" and doc["filename"] == "Nomor 5.pdf"
    assert os.path.isfile(doc["path"]) and media["_temp_upload_paths"] == [doc["path"]]
    assert "image" not in media
    assert notes[0].startswith("[User sent a file: Nomor 5.pdf (application/pdf, 6 bytes)")


def test_single_image(tmp_path):
    media, notes = run(_ch(tmp_path)._collect_media(
        _msg([Att("a.png", "image/png", b"\x89PNG")])))
    assert base64.b64decode(media["image"]["base64"]) == b"\x89PNG"
    assert "document" not in media
    assert notes[0].startswith("[User sent a photo: a.png")


def test_image_by_extension_without_type(tmp_path):
    media, _ = run(_ch(tmp_path)._collect_media(_msg([Att("x.JPG", None, b"j")])))
    assert media["image"]["mime_type"] == "image/jpeg"


def test_multiple_images_and_docs(tmp_path):
    atts = [Att("a.png", "image/png", b"1", id=1), Att("b.jpg", "image/jpeg", b"2", id=2),
            Att("c.pdf", "application/pdf", id=3), Att("d.docx", None, b"PK", id=4)]
    media, notes = run(_ch(tmp_path)._collect_media(_msg(atts)))
    assert [i["filename"] for i in media["images"]] == ["a.png", "b.jpg"]
    assert media["document"]["filename"] == "c.pdf"
    assert len(media["_temp_upload_paths"]) == 4 and len(notes) == 4


def test_charset_in_content_type(tmp_path):
    media, _ = run(_ch(tmp_path)._collect_media(
        _msg([Att("t.txt", "text/plain; charset=utf-8", b"hi")])))
    assert media["document"]["mime_type"] == "text/plain"
    assert "base64" not in media["document"]


def test_office_by_extension(tmp_path):
    media, _ = run(_ch(tmp_path)._collect_media(
        _msg([Att("a.xlsx", "application/octet-stream", b"PK")])))
    assert "base64" in media["document"]


def test_name_sanitised(tmp_path):
    media, _ = run(_ch(tmp_path)._collect_media(
        _msg([Att("../../x.pdf", "application/pdf", id="1/../2")])))
    p = media["document"]["path"]
    assert os.path.dirname(p) == str(tmp_path / "up") and "/" not in os.path.basename(p)


def test_too_large_not_read(tmp_path):
    a = Att("big.pdf", "application/pdf", size=_ATT_MAX_BYTES + 1)
    media, notes = run(_ch(tmp_path)._collect_media(_msg([a])))
    assert media == {} and a.reads == 0 and "not downloaded" in notes[0]


def test_download_error_note(tmp_path):
    media, notes = run(_ch(tmp_path)._collect_media(
        _msg([Att("a.pdf", "application/pdf", exc=RuntimeError("x"))])))
    assert media == {} and notes == ["[File could not be downloaded: a.pdf]"]


def _resolved(content="", attachments=(), embeds=()):
    m = discord.Message.__new__(discord.Message)
    m.content, m.attachments, m.embeds = content, list(attachments), list(embeds)
    return m


def test_reply_to_attachment(tmp_path):
    ref = SimpleNamespace(resolved=_resolved("", [Att("p.pdf", "application/pdf")]))
    media, notes = run(_ch(tmp_path)._collect_media(_msg(reference=ref)))
    assert media["document"]["from_reply"] is True
    assert "_reply_" in os.path.basename(media["document"]["path"])
    assert notes[0].startswith("[Attached: file p.pdf from replied message")


def test_own_attachments_win_over_reply(tmp_path):
    ref = SimpleNamespace(resolved=_resolved("", [Att("p.pdf", "application/pdf")]))
    media, _ = run(_ch(tmp_path)._collect_media(
        _msg([Att("own.png", "image/png", b"i")], reference=ref)))
    assert "document" not in media and media["image"]["filename"] == "own.png"


def test_reply_empty_gets_note(tmp_path):
    ref = SimpleNamespace(resolved=_resolved(""))
    media, notes = run(_ch(tmp_path)._collect_media(_msg(reference=ref)))
    assert media == {} and notes[0].startswith("[Replied message arrived empty")


def test_reply_text_only_no_note(tmp_path):
    ref = SimpleNamespace(resolved=_resolved("halo"))
    assert run(_ch(tmp_path)._collect_media(_msg(reference=ref))) == ({}, [])


def test_reply_unresolved_is_fetched(tmp_path):
    fetched = _resolved("", [Att("q.png", "image/png", b"q")])

    async def fetch_message(mid):
        assert mid == 42
        return fetched

    ref = SimpleNamespace(resolved=None, message_id=42)
    chan = SimpleNamespace(fetch_message=fetch_message)
    media, _ = run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan)))
    assert media["image"]["from_reply"] is True


def test_reply_fetch_failure_is_quiet(tmp_path):
    async def fetch_message(mid):
        raise RuntimeError("403")

    ref = SimpleNamespace(resolved=None, message_id=42)
    chan = SimpleNamespace(fetch_message=fetch_message)
    assert run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan))) == ({}, [])


def test_deleted_reference_ignored(tmp_path):
    ref = SimpleNamespace(resolved=object(), message_id=42)
    assert run(_ch(tmp_path)._collect_media(_msg(reference=ref))) == ({}, [])


def test_no_attachments_no_reply(tmp_path):
    assert run(_ch(tmp_path)._collect_media(_msg())) == ({}, [])


# ── empty resolved copy is refetched over REST (Riyo, 16 Nov 2026) ──────

def _resolved_id(content="", attachments=(), mid=77):
    m = _resolved(content, attachments)
    m.id = mid
    return m


def test_empty_resolved_is_refetched(tmp_path):
    full = _resolved_id("", [Att("p.pdf", "application/pdf")])
    calls = []

    async def fetch_message(mid):
        calls.append(mid)
        return full

    ref = SimpleNamespace(resolved=_resolved_id(""), message_id=77)
    chan = SimpleNamespace(fetch_message=fetch_message)
    media, notes = run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan)))
    assert calls == [77]
    assert media["document"]["from_reply"] is True
    assert notes[0].startswith("[Attached: file p.pdf from replied message")


def test_empty_after_refetch_keeps_note(tmp_path):
    async def fetch_message(mid):
        return _resolved_id("")

    ref = SimpleNamespace(resolved=_resolved_id(""), message_id=77)
    chan = SimpleNamespace(fetch_message=fetch_message)
    media, notes = run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan)))
    assert media == {} and notes[0].startswith("[Replied message arrived empty, also when")


def test_refetch_failure_falls_back_to_note(tmp_path):
    async def fetch_message(mid):
        raise RuntimeError("403")

    ref = SimpleNamespace(resolved=_resolved_id(""), message_id=77)
    chan = SimpleNamespace(fetch_message=fetch_message)
    media, notes = run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan)))
    assert media == {} and notes[0].startswith("[Replied message arrived empty")


def test_nonempty_resolved_not_refetched(tmp_path):
    async def fetch_message(mid):
        raise AssertionError("must not fetch")

    ref = SimpleNamespace(resolved=_resolved_id("", [Att("a.png", "image/png", b"i")]),
                          message_id=77)
    chan = SimpleNamespace(fetch_message=fetch_message)
    media, _ = run(_ch(tmp_path)._collect_media(_msg(reference=ref, channel=chan)))
    assert media["image"]["from_reply"] is True
