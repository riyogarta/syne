"""Gemini/Vertex tool schemas: enum values must be strings.

Vertex rejected every request carrying history_search because `level` had
"enum": [1, 2] (400 INVALID_ARGUMENT, enum[0] TYPE_STRING).
"""
import pathlib
import re

from syne.llm.gemini_common import _convert_tools_to_gemini, _stringify_enums
from syne.tools import history as H


def _enums(node):
    if isinstance(node, dict):
        e = node.get("enum")
        if isinstance(e, list):
            yield e
        for v in node.values():
            yield from _enums(v)
    elif isinstance(node, list):
        for v in node:
            yield from _enums(v)


def test_integer_enum_becomes_string():
    src = {"type": "object", "properties": {"n": {"type": "integer", "enum": [1, 2]}}}
    out = _stringify_enums(src)
    assert out["properties"]["n"] == {"type": "string", "enum": ["1", "2"]}
    # input untouched
    assert src["properties"]["n"] == {"type": "integer", "enum": [1, 2]}


def test_string_enum_untouched():
    src = {"type": "string", "enum": ["a", "b"], "description": "x"}
    assert _stringify_enums(src) == src


def test_property_named_enum_is_not_confused():
    src = {"type": "object", "properties": {"enum": {"type": "string"}}}
    assert _stringify_enums(src) == src


def test_vertex_declarations_have_only_string_enums():
    tools = [{"type": "function", "function": H.HISTORY_SEARCH_TOOL},
             {"type": "function", "function": {"name": "t", "description": "d",
              "parameters": {"type": "object", "properties": {
                  "x": {"type": "number", "enum": [0.5, 1]}}}}}]
    decls = _convert_tools_to_gemini(tools, use_parameters=True)[0]["functionDeclarations"]
    for d in decls:
        for e in _enums(d["parameters"]):
            assert all(isinstance(v, str) for v in e), (d["name"], e)


def test_no_numeric_enum_in_source():
    root = pathlib.Path(__file__).resolve().parent.parent / "syne"
    pat = re.compile(r"[\"']enum[\"']\s*:\s*\[\s*-?\d")
    hits = [str(f) for f in root.rglob("*.py") if pat.search(f.read_text(errors="ignore"))]
    assert not hits, hits
