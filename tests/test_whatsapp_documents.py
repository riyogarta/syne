"""WhatsApp inbound documents (PDF / Office) and replied-to media (Riyo, 16 Nov 2026)."""

import asyncio
import base64
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("neonize")

from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (  # noqa: E402
    ContextInfo,
    DocumentMessage,
    ExtendedTextMessage,
    FutureProofMessage,
    ImageMessage,
    Message,
)

from syne.communication.whatsapp import WhatsAppChannel, _DOC_MAX_BYTES  # noqa: E402


def _ev(msg):
    return SimpleNamespace(Message=msg)


def _wrapped(doc):
    return Message(documentWithCaptionMessage=FutureProofMessage(
        message=Message(documentMessage=doc)))


def _reply(quoted, text="Molt baca ini", stanza="ABC"):
    ctx = ContextInfo(stanzaID=stanza, quotedMessage=quoted)
    return Message(extendedTextMessage=ExtendedTextMessage(text=text, contextInfo=ctx))


class FakeClient:
    def __init__(self, data=b"%PDF-1.4 x", exc=None):
        self.data, self.exc, self.calls = data, exc, []

    async def download_any(self, message):
        self.calls.append(message)
        if self.exc:
            raise self.exc
        return self.data


def _channel(tmp_path, client):
    ch = WhatsAppChannel.__new__(WhatsAppChannel)
    ch.client = client
    ch.agent = SimpleNamespace(workspace_uploads=str(tmp_path / "uploads"))
    return ch


# ── recognising documents ───────────────────────────────────────────────

def test_plain_document_kind_and_caption():
    m = Message(documentMessage=DocumentMessage(fileName="a.pdf", caption="Molt baca"))
    assert WhatsAppChannel._extract_content(_ev(m)) == ("", "Molt baca", "document")


def test_caption_wrapped_document_is_recognised():
    m = _wrapped(DocumentMessage(fileName="a.pdf", caption="Molt baca ini"))
    assert WhatsAppChannel._extract_content(_ev(m)) == ("", "Molt baca ini", "document")
    assert WhatsAppChannel._document_of(m).fileName == "a.pdf"


def test_document_of_none_for_text():
    assert WhatsAppChannel._document_of(Message(conversation="hai")) is None


# ── replies ─────────────────────────────────────────────────────────────

def test_reply_to_document_quote_and_message():
    q = Message(documentMessage=DocumentMessage(fileName="p.pdf", caption="cap"))
    m = _reply(q)
    assert WhatsAppChannel._extract_quote(_ev(m)) == ("ABC", "cap")
    got = WhatsAppChannel._quoted_message(_ev(m))
    assert WhatsAppChannel._document_of(got).fileName == "p.pdf"


def test_reply_to_wrapped_document_caption():
    q = _wrapped(DocumentMessage(fileName="p.pdf", caption="cap2"))
    assert WhatsAppChannel._extract_quote(_ev(_reply(q))) == ("ABC", "cap2")


def test_wrapped_document_carries_its_own_quote():
    ctx = ContextInfo(stanzaID="XYZ", quotedMessage=Message(conversation="hai"))
    m = _wrapped(DocumentMessage(fileName="a.pdf", caption="Molt", contextInfo=ctx))
    assert WhatsAppChannel._extract_quote(_ev(m)) == ("XYZ", "hai")


def test_reply_to_text_unchanged():
    m = _reply(Message(conversation="halo"))
    assert WhatsAppChannel._extract_quote(_ev(m)) == ("ABC", "halo")


def test_no_quote():
    m = Message(conversation="x")
    assert WhatsAppChannel._extract_quote(_ev(m)) == ("", "")
    assert WhatsAppChannel._quoted_message(_ev(m)) is None


def test_context_without_stanza_is_not_a_reply():
    ctx = ContextInfo(quotedMessage=Message(conversation="x"))
    m = Message(extendedTextMessage=ExtendedTextMessage(text="t", contextInfo=ctx))
    assert WhatsAppChannel._extract_quote(_ev(m)) == ("", "")


# ── downloading ─────────────────────────────────────────────────────────

def test_fetch_pdf(tmp_path):
    client = FakeClient()
    ch = _channel(tmp_path, client)
    doc = DocumentMessage(fileName="Nomor 5.pdf", mimetype="application/pdf", fileLength=10)
    meta, note = asyncio.run(ch._fetch_document(doc, "MSG1", from_reply=False))
    with open(meta["path"], "rb") as f:
        assert f.read() == b"%PDF-1.4 x"
    assert meta["base64"] == base64.b64encode(b"%PDF-1.4 x").decode()
    assert meta["mime_type"] == "application/pdf" and meta["size"] == 10
    assert meta["filename"] == "Nomor 5.pdf"
    assert "from_reply" not in meta
    assert note.startswith("[User sent a file: Nomor 5.pdf (application/pdf, 10 bytes)")
    assert client.calls[0].documentMessage.fileName == "Nomor 5.pdf"


def test_fetch_reply_sanitises_names(tmp_path):
    ch = _channel(tmp_path, FakeClient(data=b"x"))
    doc = DocumentMessage(fileName="../../etc/passwd", mimetype="text/plain")
    meta, note = asyncio.run(ch._fetch_document(doc, "ID/../9", from_reply=True))
    assert os.path.dirname(meta["path"]) == str(tmp_path / "uploads")
    assert "/" not in os.path.basename(meta["path"])
    assert meta["from_reply"] is True
    assert "base64" not in meta
    assert note.startswith("[Attached: ../../etc/passwd from replied message, saved at: ")


def test_office_matched_by_extension(tmp_path):
    ch = _channel(tmp_path, FakeClient(b"PK"))
    doc = DocumentMessage(fileName="a.DOCX", mimetype="application/octet-stream")
    meta, _ = asyncio.run(ch._fetch_document(doc, "M", from_reply=False))
    assert "base64" in meta


def test_image_file_gets_base64(tmp_path):
    ch = _channel(tmp_path, FakeClient(b"\x89PNG"))
    doc = DocumentMessage(fileName="scan.png", mimetype="image/png")
    meta, _ = asyncio.run(ch._fetch_document(doc, "M", from_reply=False))
    assert base64.b64decode(meta["base64"]) == b"\x89PNG"


def test_declared_too_large_is_not_downloaded(tmp_path):
    client = FakeClient()
    doc = DocumentMessage(fileName="big.pdf", fileLength=_DOC_MAX_BYTES + 1)
    meta, note = asyncio.run(_channel(tmp_path, client)._fetch_document(doc, "M", False))
    assert meta is None and "not downloaded" in note and client.calls == []


def test_download_error_reports_note(tmp_path):
    ch = _channel(tmp_path, FakeClient(exc=RuntimeError("boom")))
    meta, note = asyncio.run(ch._fetch_document(DocumentMessage(fileName="a.pdf"), "M", False))
    assert meta is None and note == "[File could not be downloaded: a.pdf]"


def test_empty_download_reports_note(tmp_path):
    ch = _channel(tmp_path, FakeClient(data=b""))
    meta, note = asyncio.run(ch._fetch_document(DocumentMessage(fileName="a.pdf"), "M", True))
    assert meta is None and "from replied message could not be downloaded" in note


def test_quoted_image(tmp_path):
    client = FakeClient(b"\xff\xd8img")
    q = Message(imageMessage=ImageMessage(mimetype="image/png"))
    meta = asyncio.run(_channel(tmp_path, client)._fetch_quoted_image(q))
    assert meta["mime_type"] == "image/png" and meta["from_reply"] is True
    assert base64.b64decode(meta["base64"]) == b"\xff\xd8img"


def test_quoted_image_failure_is_none(tmp_path):
    ch = _channel(tmp_path, FakeClient(exc=RuntimeError("x")))
    q = Message(imageMessage=ImageMessage())
    assert asyncio.run(ch._fetch_quoted_image(q)) is None
