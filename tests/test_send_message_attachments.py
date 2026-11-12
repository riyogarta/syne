"""send_message attachments: memory:<id> and workspace/ paths become MEDIA: trailers."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import syne.tools.send_message as sm


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "outputs").mkdir(parents=True)
    import syne.tools.file_ops as fo
    monkeypatch.setattr(fo, "_workspace_dir", str(ws))
    return ws


@pytest.fixture
def tg(monkeypatch):
    ch = MagicMock()
    sent = MagicMock()
    sent.message_id = 42
    ch._send_response_with_media = AsyncMock(return_value=sent)
    monkeypatch.setattr(sm, "_telegram_channel", ch)
    return ch


def _sent_text(tg):
    return tg._send_response_with_media.call_args.kwargs["text"]


async def _send(**kw):
    kw.setdefault("chat_id", "-1001")
    kw.setdefault("message", "Halo")
    kw.setdefault("_scheduled", True)
    kw.setdefault("_access_level", "owner")
    return await sm.send_message_handler(**kw)


async def test_workspace_path_becomes_media_trailer(workspace, tg):
    img = workspace / "outputs" / "a.png"
    img.write_bytes(b"x")
    out = await _send(attachments=["outputs/a.png"])
    assert "message_id: 42" in out and "1 attachment" in out
    assert _sent_text(tg) == f"Halo\n\nMEDIA: {os.path.realpath(img)}"


async def test_memory_attachment_written_to_temp(workspace, tg, monkeypatch):
    monkeypatch.setattr(sm, "_fetch_memory_blob", AsyncMock(return_value={
        "category": "event", "filename": "long weekend.jpg", "content": b"img",
    }))
    out = await _send(attachments=["memory:57621"])
    expected = os.path.join(os.path.realpath(workspace), "temp", "mem57621_long_weekend.jpg")
    assert "1 attachment" in out
    assert open(expected, "rb").read() == b"img"
    assert _sent_text(tg) == f"Halo\n\nMEDIA: {expected}"


async def test_multiple_attachments_and_json_string(workspace, tg):
    for n in ("a.png", "b.pdf"):
        (workspace / "outputs" / n).write_bytes(b"x")
    out = await _send(attachments='["outputs/a.png", "outputs/b.pdf"]')
    assert "2 attachment" in out
    text = _sent_text(tg)
    assert text.count("MEDIA: ") == 2 and text.startswith("Halo\n\n")


async def test_empty_message_allowed_with_attachment(workspace, tg):
    (workspace / "outputs" / "a.png").write_bytes(b"x")
    await _send(message="", attachments="outputs/a.png")
    assert _sent_text(tg).startswith("MEDIA: ")


async def test_empty_message_without_attachment_rejected(workspace, tg):
    out = await _send(message="  ")
    assert out.startswith("Error") and not tg._send_response_with_media.called


async def test_path_outside_workspace_rejected(workspace, tg, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("s")
    out = await _send(attachments=[str(outside)])
    assert "outside workspace" in out and not tg._send_response_with_media.called
    out = await _send(attachments=["../secret.txt"])
    assert "outside workspace" in out


async def test_missing_file_rejected(workspace, tg):
    out = await _send(attachments=["outputs/nope.png"])
    assert "not found" in out and not tg._send_response_with_media.called


async def test_memory_without_file_rejected(workspace, tg, monkeypatch):
    monkeypatch.setattr(sm, "_fetch_memory_blob", AsyncMock(return_value={
        "category": "event", "filename": None, "content": None,
    }))
    out = await _send(attachments=["memory:5"])
    assert "no attached file" in out and not tg._send_response_with_media.called


async def test_memory_permission_denied(workspace, tg, monkeypatch):
    monkeypatch.setattr(sm, "_fetch_memory_blob", AsyncMock(return_value={
        "category": "health", "filename": "x.jpg", "content": b"x",
    }))
    import syne.security as sec
    monkeypatch.setattr(sec, "check_rule_760", lambda cat, acc: (False, "denied"))
    out = await _send(attachments=["memory:5"], _access_level="public")
    assert "denied" in out and not tg._send_response_with_media.called


async def test_telegram_caption_limit(workspace, tg):
    (workspace / "outputs" / "a.png").write_bytes(b"x")
    out = await _send(message="x" * 1025, attachments=["outputs/a.png"])
    assert "1024" in out and not tg._send_response_with_media.called


async def test_group_guard_runs_before_resolving(workspace, tg, monkeypatch):
    fetch = AsyncMock()
    monkeypatch.setattr(sm, "_fetch_memory_blob", fetch)
    out = await _send(attachments=["memory:5"], _scheduled=False)
    assert "Cannot send to groups" in out and not fetch.called


async def test_no_attachments_unchanged(workspace, tg):
    out = await _send()
    assert _sent_text(tg) == "Halo" and "attachment" not in out
