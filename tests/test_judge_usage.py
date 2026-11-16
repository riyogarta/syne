"""Judge token usage in turn_stats (Riyo, 16 Nov 2026).

Rule checker and memory guard judge calls are made outside
Conversation._llm_call. record_judge_usage collects their provider-reported
tokens; turn_stats keeps them in judge_calls / judge_usage, apart from the
main model's calls and usage.
"""

from types import SimpleNamespace

import pytest

from syne import rule_checker as RC
from syne.llm.provider import ChatResponse


class _FakeProvider:
    chat_model = "judge-model"

    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    async def chat(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._resp


def _resp():
    return ChatResponse(
        content="CLEAN", model="judge-model",
        input_tokens=1200, output_tokens=3,
        cache_read_tokens=800, cache_write_tokens=50,
    )


async def test_provider_judge_usage_recorded_inside_block():
    sink = []
    with RC.record_judge_usage(sink):
        raw = await RC._check_via_provider("prompt", _FakeProvider(_resp()), timeout=5)
    assert raw == "CLEAN"
    assert len(sink) == 1
    u = sink[0]
    assert u["source"] == "provider"
    assert u["model"] == "judge-model"
    assert (u["in"], u["out"], u["cache_read"], u["cache_write"]) == (1200, 3, 800, 50)
    assert u["lat"] >= 0


async def test_no_block_records_nothing_and_still_returns():
    raw = await RC._check_via_provider("prompt", _FakeProvider(_resp()), timeout=5)
    assert raw == "CLEAN"
    assert RC._judge_usage_sink.get() is None


async def test_failed_judge_call_records_nothing():
    sink = []
    with RC.record_judge_usage(sink):
        with pytest.raises(RuntimeError):
            await RC._check_via_provider(
                "prompt", _FakeProvider(exc=RuntimeError("down")), timeout=5,
            )
    assert sink == []


async def test_sink_is_reset_after_block():
    sink = []
    with RC.record_judge_usage(sink):
        pass
    await RC._check_via_provider("prompt", _FakeProvider(_resp()), timeout=5)
    assert sink == []


async def test_ollama_judge_usage_from_eval_counts(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {"content": "CLEAN"},
                "prompt_eval_count": 900, "eval_count": 4,
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(RC.httpx, "AsyncClient", _Client)
    sink = []
    with RC.record_judge_usage(sink):
        raw = await RC._check_via_ollama("prompt", model="gemma3:4b", timeout=5)
    assert raw == "CLEAN"
    assert len(sink) == 1
    assert sink[0]["source"] == "ollama"
    assert sink[0]["model"] == "gemma3:4b"
    assert (sink[0]["in"], sink[0]["out"]) == (900, 4)
    assert (sink[0]["cache_read"], sink[0]["cache_write"]) == (0, 0)


async def test_check_response_records_through_judge_callable():
    """A /checker-picked judge goes through _check_via_provider too."""
    prov = _FakeProvider(_resp())

    async def judge(prompt, timeout):
        return await RC._check_via_provider(prompt, prov, timeout=timeout)

    sink = []
    with RC.record_judge_usage(sink):
        res = await RC.check_response(
            draft="Jawaban.", user_message="Tanya.",
            hard_rules=[{"code": "X1", "name": "n", "description": "d"}],
            evaluator_driver="provider", judge=judge, timeout=5,
        )
    assert res.state == RC.VerdictState.CLEAN
    assert len(sink) == 1 and sink[0]["in"] == 1200


def _ts_fake():
    ts = {
        "t0": 0.0, "mode": "log", "budget": {"x": 0, "r": 0, "m": 0},
        "base_len": 0, "x_messages": 0, "r_parts": {},
        "m_used": 0, "m_results": 0, "m_skipped": 0,
        "calls": [{
            "phase": "initial", "in": 5000, "out": 200,
            "cache_read": 4000, "cache_write": 0,
            "est": {"s": 1, "x": 2, "r": 3, "m": 0, "w": 0}, "lat": 1.0,
        }],
        "judge_calls": [],
        "lat": {"recall": 0.1, "llm": 1.0, "tool": 0.0, "judge": 0.0},
        "trimmed": {"x_messages": 0, "r_items": 0},
        "violations": [],
    }
    return SimpleNamespace(_ts=ts, session_id=1)


def test_turn_stats_keeps_judge_apart_from_main_usage():
    from syne.conversation import Conversation
    fake = _ts_fake()
    u = {"source": "provider", "model": "m", "in": 1200, "out": 3,
         "cache_read": 800, "cache_write": 50, "lat": 0.5}
    Conversation._ts_note_judge(fake, "rule_check", 1, [dict(u)])
    Conversation._ts_note_judge(fake, "memory_guard_judge", 1, [dict(u)])
    Conversation._ts_note_judge(fake, "rule_check", 2, [])  # failed call: nothing
    stats = Conversation._ts_finish(fake)
    assert stats["v"] == 2
    # main model usage untouched
    assert stats["usage"] == {"in": 5000, "out": 200, "cache_read": 4000, "cache_write": 0}
    assert [c["phase"] for c in stats["calls"]] == ["initial"]
    # judges recorded apart
    assert [(c["phase"], c["attempt"]) for c in stats["judge_calls"]] == [
        ("rule_check", 1), ("memory_guard_judge", 1),
    ]
    assert stats["judge_usage"] == {
        "calls": 2, "in": 2400, "out": 6, "cache_read": 1600, "cache_write": 100,
    }
    assert stats["latency"]["judge"] == 1.0


def test_note_judge_without_open_turn_is_silent():
    from syne.conversation import Conversation
    fake = SimpleNamespace(_ts=None)
    Conversation._ts_note_judge(fake, "rule_check", 1, [{"in": 1}])
