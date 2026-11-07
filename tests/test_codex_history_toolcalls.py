"""Codex/xAI input: tool calls stored by other drivers (Riyo, 7 Nov 2026)."""

import json

from syne.llm.codex import CodexProvider
from syne.llm.provider import ChatMessage


def _items(msgs):
    _, items = CodexProvider(access_token="t")._format_input(msgs)
    return items


def _calls(items):
    return [i for i in items if i.get("type") == "function_call"]


def _outputs(items):
    return [i for i in items if i.get("type") == "function_call_output"]


def test_vertex_shape_keeps_name_and_args():
    items = _items([
        ChatMessage("user", "hi"),
        ChatMessage("assistant", "", metadata={"tool_calls": [{
            "id": "call_8654607402812060376",
            "function": {"name": "db_query", "arguments": json.dumps({"sql": "SELECT 1"})},
            "thoughtSignature": "abc",
        }]}),
        ChatMessage("tool", "ok", metadata={"tool_call_id": "call_8654607402812060376",
                                            "tool_name": "db_query"}),
    ])
    fc = _calls(items)
    assert fc[0]["name"] == "db_query"
    assert json.loads(fc[0]["arguments"]) == {"sql": "SELECT 1"}
    assert _outputs(items)[0]["call_id"] == fc[0]["call_id"]


def test_flat_shape_unchanged():
    items = _items([
        ChatMessage("assistant", "", metadata={"tool_calls": [{
            "id": "toolu_01", "name": "shell", "args": {"command": "ls"},
        }]}),
        ChatMessage("tool", "out", metadata={"tool_call_id": "toolu_01", "tool_name": "shell"}),
    ])
    fc = _calls(items)
    assert fc[0]["name"] == "shell"
    assert json.loads(fc[0]["arguments"]) == {"command": "ls"}
    assert fc[0]["call_id"] == "fc_toolu_01"
    assert len(_outputs(items)) == 1


def test_nameless_call_and_its_output_are_dropped():
    items = _items([
        ChatMessage("assistant", "", metadata={"tool_calls": [{"id": "x1", "args": {}}]}),
        ChatMessage("tool", "out", metadata={"tool_call_id": "x1", "tool_name": "unknown"}),
    ])
    assert _calls(items) == []
    assert _outputs(items) == []


def test_no_null_name_ever_sent():
    items = _items([
        ChatMessage("assistant", "", metadata={"tool_calls": [
            {"id": "a", "function": {"name": "web_search", "arguments": {"query": "q"}}},
            {"id": "b"},
            {"id": None, "name": "check_auth"},
        ]}),
    ])
    names = [c["name"] for c in _calls(items)]
    assert names == ["web_search", "check_auth"]
    assert all(isinstance(c["arguments"], str) for c in _calls(items))
