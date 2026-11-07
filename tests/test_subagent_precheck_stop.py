"""Sub-agent: feasibility pre-check, hard stop on consent block, short notices."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import syne.shell_exec as shell_exec
import syne.subagent as sa
from syne.shell_exec import Outcome, ShellResult
from syne.subagent import (
    SubAgentIncomplete,
    SubAgentManager,
    _normalize_planned,
    format_subagent_notice,
)


# ── notices ──────────────────────────────────────────────────────────

def test_notice_completed_has_no_run_id_or_body():
    assert format_subagent_notice("completed", "ignored") == "✅ Sub-agent selesai."


def test_notice_incomplete_carries_reason():
    msg = format_subagent_notice("incomplete", "perintah `python3 x.py` butuh izin owner")
    assert msg == "⚠️ Sub-agent berhenti: perintah `python3 x.py` butuh izin owner"


def test_notice_failed_and_long_reason_truncated():
    msg = format_subagent_notice("failed", "x" * 1000)
    assert msg.startswith("❌ Sub-agent gagal: ")
    assert len(msg) < 400


def test_notice_cancelled():
    assert format_subagent_notice("cancelled") == "⚠️ Sub-agent dibatalkan."


# ── planned_commands normalization ───────────────────────────────────

def test_normalize_planned_accepts_list_json_and_lines():
    assert _normalize_planned(None) == []
    assert _normalize_planned([]) == []
    assert _normalize_planned(["ls", " ", "git status"]) == ["ls", "git status"]
    assert _normalize_planned(json.dumps(["ls", "pwd"])) == ["ls", "pwd"]
    assert _normalize_planned("ls\npwd\n") == ["ls", "pwd"]


# ── pre-check ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_precheck_reports_consent_and_denied(monkeypatch):
    calls = []

    async def fake_run_shell(cmd, **kw):
        calls.append((cmd, kw))
        if cmd.startswith("python3"):
            return ShellResult(Outcome.NEEDS_CONSENT, reason="interpreter")
        if cmd.startswith("rm"):
            return ShellResult(Outcome.DENIED, reason="destructive")
        return ShellResult(Outcome.ALLOWED)

    monkeypatch.setattr(shell_exec, "run_shell", fake_run_shell)
    mgr = SubAgentManager(provider=None, system_prompt="")
    blocked = await mgr.precheck_commands(["ls", "python3 x.py", "rm -rf /"])

    assert [c for c, _ in blocked] == ["python3 x.py", "rm -rf /"]
    assert "needs owner consent" in blocked[0][1]
    assert "hard-denied" in blocked[1][1]
    # same settings as the real sub-agent run, but nothing executed
    for _, kw in calls:
        assert kw["source"] == "subagent"
        assert kw["check_only"] is True
        assert kw["consent_enabled"] is True
        assert kw["approved"] is False


@pytest.mark.asyncio
async def test_precheck_empty_plan_is_ok():
    mgr = SubAgentManager(provider=None, system_prompt="")
    assert await mgr.precheck_commands([]) == []


# ── hard stop on consent block ───────────────────────────────────────

class _FakeConn:
    async def execute(self, *a, **kw):
        return None


@asynccontextmanager
async def _fake_conn():
    yield _FakeConn()


class _FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.3):
        self.calls += 1
        if tools is None:
            return SimpleNamespace(tool_calls=None, content="summary", input_tokens=0, output_tokens=0)
        # Always wants to keep calling shell — a runaway agent.
        return SimpleNamespace(
            tool_calls=[{"id": f"t{self.calls}", "function": {
                "name": "shell", "arguments": json.dumps({"command": "python3 x.py"})}}],
            content="", input_tokens=0, output_tokens=0,
        )


@pytest.mark.asyncio
async def test_consent_block_stops_immediately_with_real_reason(monkeypatch):
    async def fake_get_config(key, default=None):
        if key == "subagents.round_delay":
            return 0
        return default

    monkeypatch.setattr(sa, "get_config", fake_get_config)
    monkeypatch.setattr(sa, "get_connection", _fake_conn)

    llm = _FakeLLM()
    mgr = SubAgentManager(provider=llm, system_prompt="")
    # No real ToolRegistry in tests -> give the loop a non-empty schema list so
    # the LLM is offered tools (otherwise tools=None and it never calls shell).
    monkeypatch.setattr(mgr, "_get_tool_schemas", lambda al: [
        {"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}
    ])

    async def blocked_tool(name, args, access_level):
        raise SubAgentIncomplete("blocked msg", reason="perintah `python3 x.py` butuh izin owner")

    monkeypatch.setattr(mgr, "_execute_tool", blocked_tool)

    with pytest.raises(SubAgentIncomplete) as ei:
        await mgr._execute_task("run1234567", "task", None, None, llm)

    assert ei.value.reason == "perintah `python3 x.py` butuh izin owner"
    assert "max rounds" not in ei.value.reason
    # initial call + one forced summary — no extra rounds burned
    assert llm.calls == 2
