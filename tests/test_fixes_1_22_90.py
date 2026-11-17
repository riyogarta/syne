"""Regression tests for the 1.22.90 follow-ups (review of 1.22.89)."""

import pytest


# ── #16: a .csv sent with an Excel mime is not an Office 97-2003 file ────

@pytest.mark.parametrize("name", ["data.csv", "DATA.CSV", "report.txt"])
async def test_csv_with_excel_mime_not_refused(name):
    from syne.abilities.office import OfficeAbility
    out = await OfficeAbility().pre_process(
        "document",
        {"filename": name, "mime_type": "application/vnd.ms-excel", "base64": "eA=="},
        "",
    )
    assert out is None  # falls through to the next handler


async def test_legacy_mime_without_extension_still_refused():
    from syne.abilities.office import OfficeAbility
    with pytest.raises(ValueError, match="Save it as \\.xlsx"):
        await OfficeAbility().pre_process(
            "document",
            {"filename": "noext", "mime_type": "application/vnd.ms-excel", "base64": "eA=="},
            "",
        )


async def test_legacy_extension_still_refused_whatever_the_mime():
    from syne.abilities.office import OfficeAbility
    with pytest.raises(ValueError, match="Save it as \\.docx"):
        await OfficeAbility().pre_process(
            "document",
            {"filename": "a.doc", "mime_type": "application/octet-stream", "base64": "eA=="},
            "",
        )


# ── #18: PDF size is checked on disk before the file is read ────────────

async def test_pdf_too_large_on_disk_is_not_read(tmp_path, monkeypatch):
    import builtins
    from syne.abilities.pdf import PdfAbility
    p = tmp_path / "big.pdf"
    p.write_bytes(b"%PDF-1.4 " + b"0" * 2048)
    ab = PdfAbility()
    monkeypatch.setattr(ab, "_MAX_PDF_BYTES", 1024)
    real_open = builtins.open

    def guarded_open(f, *a, **k):
        if str(f) == str(p):
            raise AssertionError("file was opened before the size check")
        return real_open(f, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    out = await ab.pre_process(
        "document", {"filename": "big.pdf", "mime_type": "application/pdf", "path": str(p)}, "",
    )
    assert out.startswith("[PDF too large")


# ── #14: embedding down refuses the turn (no fail-open) ─────────────────

import asyncio as _asyncio
from types import SimpleNamespace as _NS
from unittest.mock import AsyncMock as _AM


@pytest.fixture
def eh(monkeypatch):
    from syne import embed_health as E
    monkeypatch.setattr(E, "_since", None)
    monkeypatch.setattr(E, "_error", "")
    monkeypatch.setattr(E, "_driver", "")
    monkeypatch.setattr(E, "_incident", 0)
    monkeypatch.setattr(E, "_notified_incident", 0)
    monkeypatch.setattr(E, "_groups_told", set())
    monkeypatch.setattr(E, "recall_timeout", _AM(return_value=0.2))
    monkeypatch.setattr(E, "active_embedding_entry", _AM(return_value={"driver": "ollama"}))
    monkeypatch.setattr(E, "schedule_owner_notice", lambda: None)
    return E


class _DeadEmbed:
    calls = 0

    async def embed_fast(self, text, timeout=10.0):
        type(self).calls += 1
        raise ConnectionError("connection refused")


class _SlowEmbed:
    async def embed_fast(self, text, timeout=10.0):
        await _asyncio.sleep(timeout + 1)


class _OkEmbed:
    async def embed_fast(self, text, timeout=10.0):
        return _NS(vector=[0.1, 0.2])


async def test_quick_embed_failure_marks_down_and_raises(eh):
    from syne.llm.provider import EmbeddingUnavailableError
    with pytest.raises(EmbeddingUnavailableError, match="refused"):
        await eh.quick_embed(_DeadEmbed(), "dua kata")
    assert eh.is_flagged() and eh.within_ttl()


async def test_quick_embed_is_bounded_by_timeout(eh):
    from syne.llm.provider import EmbeddingUnavailableError
    from syne.llm.provider import LLMProvider
    slow = _NS(embed=None)

    class P(LLMProvider):
        name = "p"
        supports_vision = False
        async def chat(self, *a, **k): ...
        async def embed(self, text, model=None):
            await _asyncio.sleep(5)
        async def embed_batch(self, texts, model=None): ...

    t0 = _asyncio.get_running_loop().time()
    with pytest.raises(EmbeddingUnavailableError, match="no answer within"):
        await eh.quick_embed(P(), "dua kata")
    assert _asyncio.get_running_loop().time() - t0 < 2


async def test_gate_refuses_within_ttl_without_probing(eh):
    from syne.llm.provider import EmbeddingUnavailableError
    with pytest.raises(EmbeddingUnavailableError):
        await eh.quick_embed(_DeadEmbed(), "x y")
    _DeadEmbed.calls = 0
    with pytest.raises(EmbeddingUnavailableError):
        await eh.gate(_DeadEmbed())
    assert _DeadEmbed.calls == 0  # refused at once


async def test_gate_probes_after_ttl_and_recovers(eh, monkeypatch):
    from syne.llm.provider import EmbeddingUnavailableError
    with pytest.raises(EmbeddingUnavailableError):
        await eh.quick_embed(_DeadEmbed(), "x y")
    monkeypatch.setattr(eh, "_checked", eh._checked - eh.DOWN_TTL - 1)
    await eh.gate(_OkEmbed())
    assert not eh.is_flagged()


async def test_gate_healthy_costs_nothing(eh):
    await eh.gate(None)  # provider never touched


def test_notice_text_and_group_once(eh, monkeypatch):
    monkeypatch.setattr(eh, "_driver", "ollama")
    assert "Embedding service (Ollama) is not responding" in eh.user_notice()
    assert "not processed" in eh.user_notice()
    monkeypatch.setattr(eh, "_driver", "together")
    assert "(Ollama)" not in eh.user_notice()
    assert eh.should_reply(False, "telegram", 1) and eh.should_reply(False, "telegram", 1)
    assert eh.should_reply(True, "telegram", -5) is True
    assert eh.should_reply(True, "telegram", -5) is False


def test_classify_error_maps_embedding_error_for_everyone():
    from syne.communication.errors import classify_error
    from syne.llm.provider import EmbeddingUnavailableError
    for lvl in ("owner", "public"):
        assert "not processed" in classify_error(EmbeddingUnavailableError("x"), access_level=lvl)


async def test_handle_message_refuses_turn_and_skips_llm(eh, monkeypatch):
    from syne.conversation import ConversationManager
    from syne.llm.provider import EmbeddingUnavailableError
    async def _chat(msg, message_metadata=None):
        # the real chat() reaches recall, whose quick_embed marks the state down
        await eh.quick_embed(_DeadEmbed(), msg)
    conv = _NS(memory=_NS(provider=_DeadEmbed()), provider=None,
               chat=_AM(side_effect=_chat),
               load_history=_AM(), stream_callbacks=None)
    mgr = ConversationManager.__new__(ConversationManager)
    mgr._stream_callbacks = None
    mgr.get_or_create_session = _AM(return_value=conv)
    inbound = _NS(is_group=False)
    out = await mgr.handle_message("telegram", "1", {"id": 1}, "halo apa kabar",
                                   message_metadata={"inbound": inbound})
    assert "not processed" in out
    # second message within TTL: refused by the gate, chat() not called again
    conv.chat.reset_mock()
    out = await mgr.handle_message("telegram", "1", {"id": 1}, "ok",
                                   message_metadata={"inbound": inbound})
    assert "not processed" in out
    conv.chat.assert_not_called()
    # a group gets the notice once, then silence
    g = _NS(is_group=True)
    assert "not processed" in await mgr.handle_message("telegram", "-9", {"id": 1}, "a b", message_metadata={"inbound": g})
    assert await mgr.handle_message("telegram", "-9", {"id": 1}, "a b", message_metadata={"inbound": g}) is None


async def test_build_context_recall_uses_quick_embed():
    import inspect
    from syne.conversation import Conversation
    src = inspect.getsource(Conversation.build_context)
    assert "_eh.quick_embed(" in src
    assert "Query embed failed" not in src  # no more silent None


async def test_chat_drops_saved_user_message_on_refusal():
    import inspect
    from syne.conversation import Conversation
    src = inspect.getsource(Conversation.chat)
    i = src.index("except EmbeddingUnavailableError")
    assert "_delete_last_user_message()" in src[i:i + 400]


# ── /restartollama ──────────────────────────────────────────────────────

class _Proc:
    def __init__(self, rc=0, err=b""):
        self.returncode = rc
        self._err = err

    async def communicate(self):
        return b"", self._err


class _Resp:
    def __init__(self, code=200, data=None):
        self.status_code = code
        self._d = data or {}

    def json(self):
        return self._d


def _fake_httpx(monkeypatch, models):
    import httpx

    class C:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            if url.endswith("/api/version"):
                return _Resp(200, {"version": "0.16.2"})
            return _Resp(200, {"models": [{"name": m} for m in models]})
    monkeypatch.setattr(httpx, "AsyncClient", C)


@pytest.fixture
def ro(eh, monkeypatch):
    monkeypatch.setattr(eh, "active_embedding_entry", _AM(return_value={
        "driver": "ollama", "model_id": "qwen3-embedding:0.6b", "base_url": "http://localhost:11434"}))
    calls = []

    async def fake_exec(*args, **kw):
        calls.append(args)
        return _Proc()
    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fake_exec)
    from syne.llm import ollama as O
    monkeypatch.setattr(O.OllamaProvider, "embed_fast", _AM(return_value=_NS(vector=[1.0])))
    pull = _AM(return_value="")
    monkeypatch.setattr(eh, "_pull", pull)
    return _NS(calls=calls, pull=pull)


async def test_restart_ollama_model_present_no_pull(eh, ro, monkeypatch):
    _fake_httpx(monkeypatch, ["qwen3-embedding:0.6b", "bge-m3:latest"])
    out = await eh.restart_ollama()
    assert out == "✅ Ollama restarted, embedding OK."
    assert ro.calls[0] == ("sudo", "-n", "/usr/bin/systemctl", "restart", "ollama")
    ro.pull.assert_not_called()


async def test_restart_ollama_model_missing_pulls(eh, ro, monkeypatch):
    _fake_httpx(monkeypatch, ["bge-m3:latest"])
    out = await eh.restart_ollama()
    ro.pull.assert_awaited_once()
    assert "model pulled" in out and out.startswith("✅")


async def test_restart_ollama_sudo_failure_shows_setup(eh, ro, monkeypatch):
    async def fail_exec(*a, **k):
        return _Proc(1, b"sudo: a password is required")
    monkeypatch.setattr(_asyncio, "create_subprocess_exec", fail_exec)
    out = await eh.restart_ollama()
    assert "password is required" in out and "sudoers.d/syne-ollama" in out


async def test_restart_ollama_clears_down_state(eh, ro, monkeypatch):
    from syne.llm.provider import EmbeddingUnavailableError
    with pytest.raises(EmbeddingUnavailableError):
        await eh.quick_embed(_DeadEmbed(), "x y")
    _fake_httpx(monkeypatch, ["qwen3-embedding:0.6b"])
    await eh.restart_ollama()
    assert not eh.is_flagged()


async def test_restart_ollama_non_ollama_driver(eh, monkeypatch):
    monkeypatch.setattr(eh, "active_embedding_entry", _AM(return_value={"driver": "together"}))
    assert "does not use Ollama" in await eh.restart_ollama()


# ── #10: OAuth login rebuilds overrides on the same driver ──────────────

async def test_refresh_oauth_overrides(monkeypatch):
    from syne import agent as A
    default = object()
    convs = {
        "a": _NS(_model_key="m-codex", provider="old-codex"),
        "b": _NS(_model_key="m-claude", provider="old-claude"),
        "c": _NS(_model_key=None, provider=default),
        "d": _NS(_model_key="m-key", provider="old-key"),
    }
    models = [
        {"key": "m-codex", "driver": "codex", "auth": "oauth"},
        {"key": "m-claude", "driver": "anthropic", "auth": "oauth"},
        {"key": "m-key", "driver": "codex", "auth": "api_key"},
    ]
    monkeypatch.setattr(A, "get_config", _AM(return_value=models))
    ag = A.SyneAgent.__new__(A.SyneAgent)
    ag.conversations = _NS(_active=convs, provider=default)
    ag.create_provider_for_model = _AM(side_effect=lambda k: f"new-{k}")
    assert await ag.refresh_oauth_overrides({"codex"}) == 1
    assert convs["a"].provider == "new-m-codex"
    assert convs["b"].provider == "old-claude"
    assert convs["d"].provider == "old-key"
    assert convs["c"].provider is default


async def test_reload_after_oauth_refreshes_overrides():
    from syne.communication.telegram import TelegramChannel
    agent = _NS(reload_provider=_AM(), refresh_oauth_overrides=_AM(return_value=0))
    await TelegramChannel._reload_after_oauth(_NS(agent=agent), {"xai"})
    agent.refresh_oauth_overrides.assert_awaited_once_with({"xai"})


# ── #12: re-check uses the tools the guard rewrite ran ──────────────────

def test_guard_recheck_counts_guard_tools():
    import inspect
    from syne.conversation import Conversation
    src = inspect.getsource(Conversation)
    assert "tools_ran=tools_ran_this_turn or self._guard_ran_tools" in src
    g = inspect.getsource(Conversation._memory_guard_and_maybe_rewrite)
    i = g.index("if new_resp.tool_calls:")
    assert "self._guard_ran_tools = True" in g[i:i + 120]
