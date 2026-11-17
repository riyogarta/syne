"""Regression tests for the 1.22.89 fixes (report items 7-10, 12-20)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── 7: idle cleanup counts group activity ───────────────────────────────

async def test_activity_rows_include_group_seen():
    from syne import members_cleanup as MC
    conn = SimpleNamespace(fetch=AsyncMock(return_value=[]))
    await MC._activity_rows(conn, ["telegram"])
    sql = conn.fetch.call_args.args[0]
    assert "g.settings->'members'" in sql
    assert "mem.value->>'seen'" in sql
    assert "mem.key = u.platform_id" in sql


# ── 8: orphan chunk prune parses the memory id correctly ────────────────

async def test_prune_orphan_chunks_skips_memory_prefix():
    from syne.memory import kg_build as B
    conn = SimpleNamespace(execute=AsyncMock(return_value="DELETE 3"))
    assert await B.prune_orphan_chunks(conn) == 3
    sql = conn.execute.call_args.args[0]
    assert "from 8)::int" in sql and "from 7)" not in sql
    assert len("memory:") + 1 == 8


# ── 9: refreshed OAuth tokens are saved on the main loop ────────────────

async def _saved_keys(monkeypatch, provider):
    import syne.db.models as DBM
    calls = []

    async def fake_set(key, value, *a, **k):
        calls.append((key, value))

    monkeypatch.setattr(DBM, "set_config", fake_set)
    provider._save_tokens_to_db("A", "R", 123.0)
    for _ in range(5):
        await asyncio.sleep(0)
    return calls


async def test_codex_tokens_saved_on_running_loop(monkeypatch):
    from syne.llm.codex import CodexProvider
    calls = await _saved_keys(monkeypatch, CodexProvider(access_token="x"))
    assert calls == [
        ("credential.codex_access_token", "A"),
        ("credential.codex_refresh_token", "R"),
        ("credential.codex_expires_at", 123.0),
    ]


async def test_xai_tokens_use_xai_keys(monkeypatch):
    from syne.llm.xai import XaiProvider
    from syne.auth.xai_oauth import CRED_ACCESS, CRED_REFRESH, CRED_EXPIRES
    calls = await _saved_keys(monkeypatch, XaiProvider(access_token="x"))
    assert [k for k, _ in calls] == [CRED_ACCESS, CRED_REFRESH, CRED_EXPIRES]


async def test_token_save_error_is_logged_not_swallowed(monkeypatch, caplog):
    import syne.db.models as DBM
    from syne.llm.codex import CodexProvider
    monkeypatch.setattr(DBM, "set_config", AsyncMock(side_effect=RuntimeError("db down")))
    CodexProvider(access_token="x")._save_tokens_to_db("A", "R", 1.0)
    for _ in range(5):
        await asyncio.sleep(0)
    assert "db down" in caplog.text


def test_token_save_outside_loop_does_not_start_thread(caplog):
    import threading
    from syne.llm.codex import CodexProvider
    before = threading.active_count()
    CodexProvider(access_token="x")._save_tokens_to_db("A", "R", 1.0)
    assert threading.active_count() == before
    assert "not saved to DB" in caplog.text


# ── 10: OAuth login reloads the running provider ────────────────────────

async def test_reload_after_oauth_never_raises():
    from syne.communication.telegram import TelegramChannel
    agent = SimpleNamespace(reload_provider=AsyncMock(side_effect=RuntimeError("x")))
    await TelegramChannel._reload_after_oauth(SimpleNamespace(agent=agent))
    agent.reload_provider.assert_awaited_once_with(raise_on_unavailable=False)


# ── 12: a memory-guard rewrite is judged by the rule checker again ──────

def test_guard_rewrite_is_rechecked():
    import inspect
    from syne.conversation import Conversation
    src = inspect.getsource(Conversation)
    guard = src.index("self._memory_guard_and_maybe_rewrite(")
    recheck = src.index("response is not _pre_guard_response")
    assert guard < recheck
    tail = src[recheck:recheck + 1500]
    assert "self._rule_check_and_maybe_regenerate(" in tail
    assert "Rule checker unavailable" in tail


# ── 13: reload_provider can skip re-raising an unavailable default ──────

async def _reload(monkeypatch, raise_flag):
    import syne.agent as A
    from syne.agent import SyneAgent
    from syne.llm.unavailable import UnavailableProvider

    async def gc(key, default=None):
        return default

    monkeypatch.setattr(A, "get_config", gc)
    fake = SimpleNamespace(
        _init_provider=AsyncMock(return_value=UnavailableProvider(RuntimeError("boom"), "k")),
        conversations=None, subagents=None,
    )
    await SyneAgent.reload_provider(fake, raise_on_unavailable=raise_flag)
    return fake


async def test_reload_provider_raises_by_default(monkeypatch):
    with pytest.raises(RuntimeError, match="boom"):
        await _reload(monkeypatch, True)


async def test_reload_provider_quiet_still_applies(monkeypatch):
    from syne.llm.unavailable import UnavailableProvider
    fake = await _reload(monkeypatch, False)
    assert isinstance(fake.provider, UnavailableProvider)


# ── 14: a broken embedding config does not break chat ───────────────────

async def test_hybrid_survives_embedding_failure(monkeypatch):
    import syne.llm.drivers as D
    chat = SimpleNamespace(name="chat", chat=AsyncMock(return_value="OK"))
    monkeypatch.setattr(D, "create_provider", AsyncMock(return_value=chat))
    monkeypatch.setattr(
        D, "create_active_embedding_provider",
        AsyncMock(side_effect=RuntimeError("embedding gone")),
    )
    prov = await D.create_hybrid_provider({"key": "m", "driver": "x"})
    assert await prov.chat([]) == "OK"
    with pytest.raises(RuntimeError, match="embedding gone"):
        await prov.embed("hi")


# ── 15: nothing runs while /restore replaces the database ───────────────

async def test_restore_gate_blocks_updates_while_running():
    from telegram.ext import ApplicationHandlerStop
    from syne.communication.telegram import TelegramChannel
    q = SimpleNamespace(answer=AsyncMock())
    upd = SimpleNamespace(callback_query=q, effective_message=None, effective_chat=None)
    with pytest.raises(ApplicationHandlerStop):
        await TelegramChannel._restore_gate(SimpleNamespace(_restore_running=True), upd, None)
    q.answer.assert_awaited_once()


async def test_restore_gate_passes_when_idle():
    from syne.communication.telegram import TelegramChannel
    upd = SimpleNamespace(callback_query=None, effective_message=None, effective_chat=None)
    assert await TelegramChannel._restore_gate(SimpleNamespace(_restore_running=False), upd, None) is None


async def test_second_restore_confirmation_is_refused(monkeypatch, tmp_path):
    import syne.cli.cmd_backup as CB
    from syne.communication.telegram import TelegramChannel
    (tmp_path / "b.sql").write_text("x")
    monkeypatch.setattr(CB, "_get_backup_dir", lambda: str(tmp_path))
    run = MagicMock(return_value=(True, ""))
    monkeypatch.setattr(CB, "run_restore", run)
    q = SimpleNamespace(answer=AsyncMock(), edit_message_text=AsyncMock())
    await TelegramChannel._handle_restore_callback(
        SimpleNamespace(_restore_running=True), q, "confirmed:b.sql",
    )
    run.assert_not_called()
    q.answer.assert_awaited_once()


async def test_failed_restore_releases_the_gate(monkeypatch, tmp_path):
    import syne.cli.cmd_backup as CB
    from syne.communication.telegram import TelegramChannel
    (tmp_path / "b.sql").write_text("x")
    monkeypatch.setattr(CB, "_get_backup_dir", lambda: str(tmp_path))
    monkeypatch.setattr(CB, "run_restore", MagicMock(return_value=(False, "bad dump")))
    me = SimpleNamespace(_restore_running=False)
    q = SimpleNamespace(answer=AsyncMock(), edit_message_text=AsyncMock())
    await TelegramChannel._handle_restore_callback(me, q, "confirmed:b.sql")
    assert me._restore_running is False
    assert "bad dump" in q.edit_message_text.call_args.args[0]


# ── 16: Office 97-2003 files get a clear refusal ────────────────────────

@pytest.mark.parametrize("name,mime,new", [
    ("a.doc", "", ".docx"),
    ("a.xls", "", ".xlsx"),
    ("a.ppt", "", ".pptx"),
    ("x", "application/vnd.ms-excel", ".xlsx"),
    ("x", "application/msword", ".docx"),
])
async def test_legacy_office_refused(name, mime, new):
    from syne.abilities.office import OfficeAbility
    with pytest.raises(ValueError, match=f"Save it as \\{new}"):
        await OfficeAbility().pre_process(
            "document", {"filename": name, "mime_type": mime, "base64": "eA=="}, "",
        )


async def test_modern_office_still_claimed():
    from syne.abilities.office import OfficeAbility
    with pytest.raises(ValueError, match="has no data"):
        await OfficeAbility().pre_process("document", {"filename": "a.docx"}, "")


async def test_non_office_falls_through():
    from syne.abilities.office import OfficeAbility
    assert await OfficeAbility().pre_process("document", {"filename": "a.pdf"}, "") is None


# ── 17: photo failure is tracked, not parsed from the text ──────────────

async def test_album_description_mentioning_failure_words_is_not_an_error():
    from syne.abilities.image_analysis import ImageAnalysisAbility
    ab = ImageAnalysisAbility()
    ab.execute = AsyncMock(return_value={
        "success": True, "result": "Tulisan di papan: 'Foto gagal dibaca' [Analisis gagal: tidak]",
    })
    out = await ab.pre_process("images", [{"base64": "eA=="}, {"base64": "eA=="}], "")
    assert "Foto 2/2" in out


async def test_album_real_failure_still_raises():
    from syne.abilities.image_analysis import ImageAnalysisAbility
    ab = ImageAnalysisAbility()
    ab.execute = AsyncMock(side_effect=[
        {"success": True, "result": "ok"},
        {"success": False, "error": "quota"},
    ])
    with pytest.raises(RuntimeError, match="quota"):
        await ab.pre_process("images", [{"base64": "eA=="}, {"base64": "eA=="}], "")


# ── 18: PDF read from a saved path (WhatsApp / Discord) ─────────────────

async def test_pdf_read_from_path(tmp_path):
    import fitz
    from syne.abilities.pdf import PdfAbility
    p = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Laporan keuangan kuartal tiga tahun ini lengkap.")
    doc.save(str(p))
    doc.close()
    out = await PdfAbility().pre_process(
        "document", {"filename": "doc.pdf", "mime_type": "application/pdf", "path": str(p)}, "",
    )
    assert "Laporan keuangan" in out


async def test_pdf_without_data_raises():
    from syne.abilities.pdf import PdfAbility
    with pytest.raises(ValueError, match="has no data"):
        await PdfAbility().pre_process("document", {"filename": "a.pdf"}, "")


# ── 19: upload file names are made safe ─────────────────────────────────

def test_safe_upload_name_strips_controls_and_paths():
    from syne.communication.outbound import safe_upload_name
    out = safe_upload_name("../a\x00b\nc/d\\e.pdf")
    assert "\x00" not in out and "\n" not in out
    assert "/" not in out and "\\" not in out and ".." not in out
    assert out.endswith(".pdf")


def test_safe_upload_name_caps_length_keeps_extension():
    from syne.communication.outbound import safe_upload_name
    out = safe_upload_name("é" * 400 + ".xlsx")
    assert len(out.encode("utf-8")) <= 120
    assert out.endswith(".xlsx")
    out.encode("utf-8").decode("utf-8")  # no broken multibyte char


def test_safe_upload_name_never_empty():
    from syne.communication.outbound import safe_upload_name
    assert safe_upload_name("") == "file"
    assert safe_upload_name("\x00\x01") == "file"
    assert safe_upload_name("normal.docx") == "normal.docx"


# ── 20: /guard skips model keys too long for callback_data ──────────────

async def test_guard_menu_skips_oversized_keys(monkeypatch):
    import syne.communication.telegram as T
    import syne.memory_guard as MG
    from syne.communication.telegram import TelegramChannel

    long_key = "k" * 60
    models = [{"key": "ok", "label": "OK"}, {"key": long_key, "label": "Long"}]

    async def gc(key, default=None):
        return {"provider.models": models, "provider.active_model": "ok"}.get(key, default)

    monkeypatch.setattr(T, "get_config", gc)
    monkeypatch.setattr(T, "_checker_default_model_label", AsyncMock(return_value="Def"))
    monkeypatch.setattr(MG, "load_config", AsyncMock(return_value={"enabled": True, "model": ""}))

    fake = SimpleNamespace(_sort_models_for_display=TelegramChannel._sort_models_for_display)
    msg = SimpleNamespace(reply_text=AsyncMock())
    upd = SimpleNamespace(message=msg)
    await TelegramChannel._guard_menu_main(fake, upd)
    markup = msg.reply_text.call_args.kwargs["reply_markup"]
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "guard:set:ok" in datas
    assert all(len(d.encode()) <= 64 for d in datas)
    assert f"guard:set:{long_key}" not in datas
