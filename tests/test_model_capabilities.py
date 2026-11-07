"""Per-model capability overrides: provider.models[].capabilities.

Missing/null keys keep each driver's name-based auto-detection; set keys win.
"""

from syne.llm import drivers
from syne.llm.anthropic import AnthropicProvider, _BETA_INTERLEAVED, _build_beta_header
from syne.llm.codex import CodexProvider
from syne.llm.gemini_common import _build_thinking_config, _uses_thinking_level
from syne.llm.google import GoogleProvider
from syne.llm.openai import OpenAIProvider
from syne.llm.provider import clean_capabilities, validate_capabilities
from syne.llm.vertex import VertexProvider


def _openai(model, base_url="https://api.openai.com/v1", caps=None):
    p = OpenAIProvider(api_key="k", chat_model=model, base_url=base_url)
    if caps:
        p.capabilities = caps
    return p


def _claude(model, caps=None):
    p = AnthropicProvider(chat_model=model)
    if caps:
        p.capabilities = caps
    return p


# ── schema ──────────────────────────────────────────────

def test_validate_accepts_known_keys_and_null():
    caps = {"thinking_mode": "adaptive", "api_mode": None,
            "reserved_output_tokens": 8000, "vision": False}
    assert validate_capabilities(caps) == []


def test_validate_rejects_bad_values():
    errors = validate_capabilities({
        "thinking_mode": "turbo", "api_mode": "grpc",
        "reserved_output_tokens": True, "vision": "yes", "colour": 1,
    })
    assert len(errors) == 5


def test_validate_rejects_non_object():
    assert validate_capabilities([1, 2]) != []


def test_clean_drops_null_and_invalid():
    clean, errors = clean_capabilities({"vision": False, "api_mode": None, "bogus": 1})
    assert clean == {"vision": False}
    assert len(errors) == 1
    assert clean_capabilities(None) == ({}, [])


async def test_create_provider_applies_capabilities(monkeypatch):
    async def fake_raw(entry):
        return OpenAIProvider(api_key="k", chat_model=entry["model_id"])

    monkeypatch.setattr(drivers, "_create_provider_raw", fake_raw)
    p = await drivers.create_provider({
        "key": "x", "driver": "openai_compat", "model_id": "x",
        "capabilities": {"vision": False, "api_mode": None, "bogus": 1},
    })
    assert p.capabilities == {"vision": False}
    p2 = await drivers.create_provider({"key": "y", "driver": "openai_compat", "model_id": "y"})
    assert p2.capabilities == {}


# ── OpenAI ──────────────────────────────────────────────

def test_openai_auto_detection_unchanged():
    assert _openai("gpt-6-capstest")._is_reasoning("gpt-6-capstest") is True
    assert _openai("gpt-7-capstest")._is_reasoning("gpt-7-capstest") is True
    assert _openai("gpt-4o")._is_reasoning("gpt-4o") is False
    assert _openai("gpt-6-capstest-a")._use_responses_api("gpt-6-capstest-a") is True
    groq = _openai("caps-llama", base_url="https://api.groq.com/openai/v1")
    assert groq._use_responses_api("caps-llama") is False


def test_openai_thinking_mode_override():
    assert _openai("acme-think", caps={"thinking_mode": "effort"})._is_reasoning("acme-think") is True
    p = _openai("gpt-6-capstest", caps={"thinking_mode": "none"})
    assert p._is_reasoning("gpt-6-capstest") is False


def test_openai_override_does_not_leak_to_other_model():
    p = _openai("gpt-6-capstest", caps={"thinking_mode": "none"})
    assert p._is_reasoning("gpt-7-capstest") is True


def test_openai_api_mode_override():
    groq = _openai("caps-m", base_url="https://api.groq.com/openai/v1", caps={"api_mode": "responses"})
    assert groq._use_responses_api("caps-m") is True
    official = _openai("gpt-6-capstest-b", caps={"api_mode": "chat"})
    assert official._use_responses_api("gpt-6-capstest-b") is False


def test_openai_reserved_and_vision_override():
    assert _openai("gpt-6-capstest").reserved_output_tokens == 32_000
    p = _openai("gpt-6-capstest", caps={"reserved_output_tokens": 1234, "vision": False})
    assert p.reserved_output_tokens == 1234
    assert p.supports_vision is False


# ── Anthropic ───────────────────────────────────────────

def test_anthropic_auto_detection_unchanged():
    assert _claude("claude-opus-5-5")._adaptive_for("claude-opus-5-5") is True
    assert _claude("claude-sonnet-4-20250514")._adaptive_for("claude-sonnet-4-20250514") is False


def test_anthropic_thinking_mode_override():
    assert _claude("acme-claude", caps={"thinking_mode": "adaptive"})._adaptive_for("acme-claude") is True
    p = _claude("claude-opus-5-5", caps={"thinking_mode": "budget"})
    assert p._adaptive_for("claude-opus-5-5") is False
    assert p.context_window == 200_000


def test_anthropic_beta_header_follows_override():
    assert _BETA_INTERLEAVED not in _build_beta_header(False, "acme", adaptive=True)
    assert _BETA_INTERLEAVED in _build_beta_header(False, "claude-opus-5-5", adaptive=False)
    assert _BETA_INTERLEAVED not in _build_beta_header(False, "claude-opus-5-5")


def test_anthropic_reserved_and_vision_override():
    p = _claude("claude-opus-5-5", caps={"reserved_output_tokens": 5000, "vision": False})
    assert p.reserved_output_tokens == 5000
    assert p.supports_vision is False


# ── Gemini (Vertex + Google) ────────────────────────────

def test_gemini_auto_detection_covers_new_majors():
    assert _uses_thinking_level("gemini-4-argon") is True
    assert _uses_thinking_level("gemini-5-pro") is True
    assert _uses_thinking_level("gemini-3.8-flash") is True
    assert _uses_thinking_level("gemini-2.5-pro") is False


def test_gemini_thinking_mode_override():
    assert _uses_thinking_level("gemini-4-argon", "budget") is False
    assert _uses_thinking_level("acme-gem", "level") is True
    assert "thinkingLevel" in _build_thinking_config("acme-gem", None, "level")
    assert "thinkingBudget" in _build_thinking_config("gemini-4-argon", 4096, "budget")
    assert _build_thinking_config("gemini-4-argon", None, "none") is None
    assert "thinkingLevel" in _build_thinking_config("gemini-4-argon", None)


def test_vertex_google_vision_and_reserved_override():
    for p in (VertexProvider(api_key="k"), GoogleProvider(api_key="k")):
        assert p.supports_vision is True
        assert p.reserved_output_tokens == 4096
        p.capabilities = {"vision": False, "reserved_output_tokens": 7000}
        assert p.supports_vision is False
        assert p.reserved_output_tokens == 7000


# ── Codex ───────────────────────────────────────────────

def test_codex_vision_and_reserved_override():
    p = CodexProvider(access_token="t")
    assert p.supports_vision is True
    assert p.reserved_output_tokens == 4096
    p.capabilities = {"vision": False, "reserved_output_tokens": 9000}
    assert p.supports_vision is False
    assert p.reserved_output_tokens == 9000
