"""OpenAI driver: /v1/responses for reasoning models, effort+tools fallback,
and one-line error humanization."""

import json

import pytest

import syne.llm.openai as oa
from syne.communication.errors import classify_error, _humanize
from syne.llm.provider import ChatMessage, LLMBadRequestError


# ── error humanization ───────────────────────────────────────────────

RAW = (
    'OpenAI 400: {\n  "error": {\n    "message": "Function tools with reasoning_effort '
    'are not supported for gpt-6-luna in /v1/chat/completions.",\n    "type": '
    '"invalid_request_error",\n    "param": "reasoning_effort",\n    "code": null\n  }\n}'
)


def test_humanize_collapses_json_to_one_line():
    out = _humanize(RAW)
    assert "\n" not in out and "{" not in out
    assert out == (
        "OpenAI 400: Function tools with reasoning_effort are not supported for "
        "gpt-6-luna in /v1/chat/completions. (param: reasoning_effort)"
    )


def test_humanize_truncated_json_uses_regex():
    out = _humanize(RAW[:200])
    assert out.startswith("OpenAI 400: Function tools with reasoning_effort")


def test_humanize_leaves_plain_text_alone():
    assert _humanize("boom") == "boom"
    assert _humanize("OpenAI 400: {'error': 'bad param'}") == "OpenAI 400: {'error': 'bad param'}"


def test_owner_error_is_single_line():
    msg = classify_error(LLMBadRequestError(RAW), model="openai+ollama", access_level="owner")
    assert msg.startswith("⚠️ [openai+ollama] LLMBadRequestError: OpenAI 400: Function tools")
    assert "\n" not in msg


# ── responses input / tools formatting ───────────────────────────────

def test_to_responses_input_pairs_calls_and_drops_orphans():
    msgs = [
        ChatMessage(role="system", content="A"),
        ChatMessage(role="system", content="B"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="", metadata={"tool_calls": [
            {"name": "shell", "args": {"command": "ls"}, "id": "call_1"},
            {"name": "orphan", "args": {}, "id": "call_x"},
        ]}),
        ChatMessage(role="tool", content="out", metadata={"tool_name": "shell", "tool_call_id": "call_1"}),
        ChatMessage(role="tool", content="lost", metadata={"tool_name": "z", "tool_call_id": "call_gone"}),
    ]
    instr, items = oa._to_responses_input(msgs)
    assert instr == "A\n\nB"
    assert items[0] == {"role": "user", "content": "hi"}
    calls = [i for i in items if i.get("type") == "function_call"]
    outs = [i for i in items if i.get("type") == "function_call_output"]
    assert [c["call_id"] for c in calls] == ["call_1"]
    assert json.loads(calls[0]["arguments"]) == {"command": "ls"}
    assert [o["call_id"] for o in outs] == ["call_1"]


def test_format_responses_tools_flattens_function():
    tools = [{"type": "function", "function": {"name": "shell", "description": "d",
                                                "parameters": {"type": "object"}}}]
    assert oa._format_responses_tools(tools) == [{
        "type": "function", "name": "shell", "description": "d",
        "parameters": {"type": "object"}, "strict": False,
    }]


def test_responses_body_keeps_reasoning_with_tools():
    p = oa.OpenAIProvider(api_key="k", chat_model="gpt-6-luna")
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    body = p._build_responses_body([ChatMessage(role="user", content="x")],
                                   "gpt-6-luna", None, tools, None)
    assert body["reasoning"] == {"effort": "high"}
    assert body["tools"][0]["name"] == "t"
    assert "reasoning_effort" not in body and "temperature" not in body


# ── stream accumulator ───────────────────────────────────────────────

def test_stream_accumulates_text_tool_calls_usage():
    s = oa._ResponsesStream()
    for ev in [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.reasoning_summary_text.delta", "delta": "think"},
        {"type": "response.output_item.added", "output_index": 1,
         "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "shell"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"command":'},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '"ls"}'},
        {"type": "response.completed", "response": {"model": "gpt-6-luna", "status": "completed",
                                                     "usage": {"input_tokens": 10, "output_tokens": 3}}},
    ]:
        s.feed(ev)
    assert s.content == "Hello"
    assert s.thinking == "think"
    assert s.tool_calls() == [{"name": "shell", "args": {"command": "ls"}, "id": "call_1"}]
    assert s.model == "gpt-6-luna" and s.usage["input_tokens"] == 10


def test_stream_failed_event_raises():
    s = oa._ResponsesStream()
    with pytest.raises(oa._ResponsesStreamError) as ei:
        s.feed({"type": "response.failed", "response": {"error": {"code": "server_error", "message": "x"}}})
    assert ei.value.code == "server_error"


# ── routing ──────────────────────────────────────────────────────────

def test_responses_only_on_official_endpoint():
    assert oa.OpenAIProvider(api_key="k")._use_responses_api("gpt-6-luna")
    groq = oa.OpenAIProvider(api_key="k", base_url="https://api.groq.com/openai/v1")
    assert not groq._use_responses_api("gpt-6-luna")


@pytest.mark.asyncio
async def test_falls_back_to_chat_completions_when_responses_unavailable(monkeypatch):
    p = oa.OpenAIProvider(api_key="k", chat_model="gpt-6-test-fallback")

    async def unavailable(*a, **kw):
        raise oa._ResponsesUnavailable("HTTP 404")

    sentinel = RuntimeError("reached chat/completions")

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise sentinel

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(p, "_chat_responses", unavailable)
    monkeypatch.setattr(oa.httpx, "AsyncClient", _Client)
    with pytest.raises(RuntimeError) as ei:
        await p.chat([ChatMessage(role="user", content="x")])
    assert ei.value is sentinel
    assert p._model_key("gpt-6-test-fallback") in oa._RESPONSES_UNSUPPORTED
    oa._RESPONSES_UNSUPPORTED.discard(p._model_key("gpt-6-test-fallback"))


def test_effort_tools_rejection_detector():
    assert oa._is_effort_tools_rejection(RAW)
    assert not oa._is_effort_tools_rejection('{"error":{"message":"bad temperature"}}')
