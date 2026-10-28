"""Vertex endpoint detection/selection + Anthropic adaptive-model parsing."""
from unittest.mock import patch

import pytest

from syne.llm.anthropic import _is_adaptive_model
from syne.llm import vertex as vx


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-4-20250514", False),
    ("claude-opus-4-20250514", False),
    ("claude-opus-4-1-20250805", False),
    ("claude-opus-4-6", True),
    ("claude-opus-4-9-20260401", True),
    ("claude-sonnet-5-20260701", True),
    ("claude-opus-5-5", True),
    ("claude-sonnet-5-5", True),
])
def test_adaptive_model_ignores_snapshot_date(model, expected):
    assert _is_adaptive_model(model) is expected


ENTRIES = [
    {"model_id": "gemini-2.5-pro", "region": "us-west1"},
    {"model_id": "gemini-4-argon", "region": "global", "project_id": "p-1"},
]


def test_pick_exact_model_wins():
    assert vx.pick_vertex_entry(ENTRIES, "gemini-4-argon")["region"] == "global"
    assert vx.pick_vertex_entry(ENTRIES, "gemini-2.5-pro", prefer_global=True)["region"] == "us-west1"


def test_pick_prefers_global_for_default_model():
    assert vx.pick_vertex_entry(ENTRIES, "gemini-3.8-flash", prefer_global=True)["project_id"] == "p-1"


def test_pick_legacy_first_entry():
    assert vx.pick_vertex_entry(ENTRIES, "gemini-2.5-flash")["region"] == "us-west1"
    assert vx.pick_vertex_entry([], "x") is None


class _Resp:
    def __init__(self, code):
        self.status_code = code


def _fake_client(ok):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            return _Resp(200 if ok(url) else 404)

    return _Client


async def test_detect_races_global_when_project_given():
    with patch.object(vx.httpx, "AsyncClient", _fake_client(lambda u: "/locations/global/" in u)):
        assert await vx.detect_vertex_region("k", "gemini-4-argon", project_id="p-1") == "global"


async def test_detect_skips_global_without_project():
    with patch.object(vx.httpx, "AsyncClient", _fake_client(lambda u: "/locations/global/" in u)):
        assert await vx.detect_vertex_region("k", "gemini-4-argon") == "us-central1"
        assert await vx.detect_vertex_region("k", "gemini-4-argon", fallback="") == ""


async def test_detect_regional_still_works():
    with patch.object(vx.httpx, "AsyncClient", _fake_client(lambda u: u.startswith("https://us-west1-"))):
        assert await vx.detect_vertex_region("k", "gemini-2.5-pro", project_id="p-1") == "us-west1"
