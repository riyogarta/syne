"""1.23.51: image_gen takes an optional size preset; default stays 1024x1024
and the Together request carries the preset's width/height."""

import pytest

from syne.abilities import image_gen as G


def test_presets_multiple_of_8_and_about_1mp():
    for name, (w, h) in G.SIZE_PRESETS.items():
        assert w % 8 == 0 and h % 8 == 0, name
        assert 0.95e6 <= w * h <= 1.1e6, name


def test_default_and_unknown_fall_back_to_square():
    assert G.resolve_size("") == ("square", 1024, 1024)
    assert G.resolve_size(None) == ("square", 1024, 1024)
    assert G.resolve_size("huge") == ("square", 1024, 1024)
    assert G.resolve_size("Photo_Portrait") == ("photo_portrait", 832, 1248)


def test_schema_lists_presets():
    props = G.ImageGenAbility().get_schema()["function"]["parameters"]["properties"]
    assert props["size"]["enum"] == list(G.SIZE_PRESETS)
    assert "size" not in G.ImageGenAbility().get_schema()["function"]["parameters"]["required"]


class _Resp:
    status_code = 200

    def json(self):
        import base64
        return {"data": [{"b64_json": base64.b64encode(b"png").decode()}]}


class _Client:
    sent = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, headers=None, json=None):
        _Client.sent.append(json)
        return _Resp()


@pytest.mark.parametrize("size,wh", [("", (1024, 1024)), ("story_portrait", (768, 1344))])
async def test_together_request_uses_size(monkeypatch, size, wh):
    _Client.sent = []
    monkeypatch.setattr(G.httpx, "AsyncClient", _Client)
    ab = G.ImageGenAbility()

    async def fake_resolve(key):
        return {"driver": "openai_compat", "model_id": "m", "base_url": "https://api.together.xyz/v1"}, "k"

    monkeypatch.setattr(ab, "_resolve_model_entry", fake_resolve)
    monkeypatch.setattr(ab, "get_output_dir", lambda: "/tmp")
    res = await ab.execute({"prompt": "x", "size": size}, {"config": {"model_key": "a"}, "session_id": 1})
    assert res["success"] is True
    assert (_Client.sent[0]["width"], _Client.sent[0]["height"]) == wh


async def test_non_together_notes_square(monkeypatch):
    ab = G.ImageGenAbility()

    async def fake_resolve(key):
        return {"driver": "openai_compat", "model_id": "gpt-image-2", "base_url": "https://api.openai.com/v1"}, "k"

    async def fake_openai(prompt, api_key, model):
        return b"png"

    monkeypatch.setattr(ab, "_resolve_model_entry", fake_resolve)
    monkeypatch.setattr(ab, "_gen_openai", fake_openai)
    monkeypatch.setattr(ab, "get_output_dir", lambda: "/tmp")
    res = await ab.execute({"prompt": "x", "size": "photo_portrait"}, {"config": {"model_key": "a"}, "session_id": 1})
    assert res["success"] is True
    assert "not supported for this provider yet" in res["result"]
