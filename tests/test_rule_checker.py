"""Tests for the hard-rule compliance checker parser.

The LLM call itself is best exercised end-to-end; these tests pin the
verdict-line PARSER so a small-model output format we've already accepted
never silently regresses into ERROR (or CLEAN) after a refactor.
"""

import pytest

from syne.rule_checker import (
    CheckResult,
    VerdictState,
    _parse_verdict_line,
    _strip_think,
    check_response,
)


class TestParseVerdictLine:

    def test_clean_exact(self):
        r = _parse_verdict_line("CLEAN")
        assert r.state == VerdictState.CLEAN

    def test_clean_with_surrounding_whitespace(self):
        r = _parse_verdict_line("   CLEAN   \n")
        assert r.state == VerdictState.CLEAN

    def test_clean_lowercase_accepted(self):
        # We uppercase before compare; lowercase is fine.
        r = _parse_verdict_line("clean")
        assert r.state == VerdictState.CLEAN

    def test_violated_full_shape(self):
        r = _parse_verdict_line("VIOLATED|DATE_VERIFY|asserted date without get_current_time")
        assert r.state == VerdictState.VIOLATED
        assert r.violated == ["DATE_VERIFY"]
        assert "asserted date" in r.reason

    def test_violated_multiple_codes(self):
        r = _parse_verdict_line("VIOLATED|DATE_VERIFY,TONE_STRICT|two issues here")
        assert r.state == VerdictState.VIOLATED
        assert r.violated == ["DATE_VERIFY", "TONE_STRICT"]

    def test_violated_with_think_block_prefix(self):
        raw = "<think>hmm let me see</think>\nVIOLATED|DATE_VERIFY|wrong date"
        r = _parse_verdict_line(raw)
        assert r.state == VerdictState.VIOLATED
        assert "DATE_VERIFY" in r.violated

    def test_junk_without_verdict_becomes_error(self):
        r = _parse_verdict_line("looks fine to me")
        assert r.state == VerdictState.ERROR
        assert r.reason  # non-empty

    def test_empty_becomes_error(self):
        r = _parse_verdict_line("")
        assert r.state == VerdictState.ERROR

    def test_whitespace_only_becomes_error(self):
        r = _parse_verdict_line("   \n\n  ")
        assert r.state == VerdictState.ERROR

    def test_violated_word_only_with_uppercase_codes_fallback(self):
        # Model writes "VIOLATED" then lists codes on next lines without pipe.
        raw = "VIOLATED\nRule DATE_VERIFY was broken."
        r = _parse_verdict_line(raw)
        assert r.state == VerdictState.VIOLATED
        assert "DATE_VERIFY" in r.violated

    def test_clean_preferred_over_junk(self):
        # If any line is CLEAN, verdict is CLEAN.
        raw = "some preamble\nCLEAN\nother noise"
        r = _parse_verdict_line(raw)
        assert r.state == VerdictState.CLEAN


class TestStripThink:

    def test_no_think_passthrough(self):
        assert _strip_think("hello") == "hello"

    def test_removes_think_block(self):
        assert _strip_think("<think>a</think>result") == "result"

    def test_removes_multiline_think(self):
        raw = "<think>line1\nline2</think>\nCLEAN"
        assert _strip_think(raw).strip() == "CLEAN"


class TestCheckResponse:

    async def test_no_hard_rules_returns_clean(self):
        r = await check_response(
            draft="anything",
            user_message="hi",
            hard_rules=[],
        )
        assert r.state == VerdictState.CLEAN

    async def test_empty_draft_returns_clean(self):
        r = await check_response(
            draft="",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
        )
        assert r.state == VerdictState.CLEAN

    async def test_whitespace_draft_returns_clean(self):
        r = await check_response(
            draft="   \n\n",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
        )
        assert r.state == VerdictState.CLEAN

    async def test_provider_driver_without_provider_returns_error(self):
        r = await check_response(
            draft="something",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
            evaluator_driver="provider",
            provider=None,
        )
        assert r.state == VerdictState.ERROR

    async def test_unknown_codes_from_checker_becomes_error(self, monkeypatch):
        """Model hallucinates a rule code we don't know about — must not
        pass through as VIOLATED (that would block or nag the user with a
        phantom rule). Downgrade to ERROR so caller fails-open."""
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            return "VIOLATED|MADEUP_RULE|this rule does not exist"
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)
        r = await check_response(
            draft="hello",
            user_message="hi",
            hard_rules=[{"code": "DATE_VERIFY", "name": "n", "description": "d"}],
        )
        assert r.state == VerdictState.ERROR

    async def test_known_code_survives_filter(self, monkeypatch):
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            return "VIOLATED|DATE_VERIFY|wrong date"
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)
        r = await check_response(
            draft="today is monday",
            user_message="what day",
            hard_rules=[{"code": "DATE_VERIFY", "name": "n", "description": "d"}],
        )
        assert r.state == VerdictState.VIOLATED
        assert r.violated == ["DATE_VERIFY"]

    async def test_ollama_exception_without_provider_returns_error(self, monkeypatch):
        """Ollama fails and no provider available for fallback → ERROR."""
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            raise RuntimeError("ollama down")
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)
        r = await check_response(
            draft="hi",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
            provider=None,
        )
        assert r.state == VerdictState.ERROR
        assert "checker call failed" in r.reason

    async def test_ollama_failure_falls_back_to_provider(self, monkeypatch):
        """Ollama fails but a provider is available → auto-fallback and use provider verdict."""
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            raise RuntimeError("ollama not installed")
        called_provider = {"count": 0}
        async def fake_provider_check(prompt, provider):
            called_provider["count"] += 1
            return "CLEAN"
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)
        monkeypatch.setattr("syne.rule_checker._check_via_provider", fake_provider_check)

        # Any non-None provider will do — the mock replaces _check_via_provider.
        dummy_provider = object()
        r = await check_response(
            draft="hello",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
            evaluator_driver="ollama",
            provider=dummy_provider,
        )
        assert r.state == VerdictState.CLEAN
        assert called_provider["count"] == 1

    async def test_both_drivers_failure_returns_error(self, monkeypatch):
        """Ollama fails, provider fallback also fails → ERROR, and reason
        mentions BOTH drivers so the owner can see which paths blew up."""
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            raise RuntimeError("ollama refused")
        async def fake_provider_check(prompt, provider):
            raise TimeoutError("provider timed out")
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)
        monkeypatch.setattr("syne.rule_checker._check_via_provider", fake_provider_check)

        r = await check_response(
            draft="hello",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
            evaluator_driver="ollama",
            provider=object(),
        )
        assert r.state == VerdictState.ERROR
        assert "both drivers failed" in r.reason
        assert "RuntimeError" in r.reason
        assert "TimeoutError" in r.reason

    async def test_provider_driver_failure_does_not_fallback_to_ollama(self, monkeypatch):
        """When user configured 'provider' driver and it fails, we do NOT
        try Ollama as fallback — the reverse fallback would just paper
        over a real problem (main LLM down) with a very fragile local
        model that has no guarantee of being pulled."""
        async def fake_provider_check(prompt, provider):
            raise RuntimeError("provider down")
        ollama_called = {"count": 0}
        async def fake_ollama(prompt, model, base_url="http://localhost:11434", timeout=30.0):
            ollama_called["count"] += 1
            return "CLEAN"
        monkeypatch.setattr("syne.rule_checker._check_via_provider", fake_provider_check)
        monkeypatch.setattr("syne.rule_checker._check_via_ollama", fake_ollama)

        r = await check_response(
            draft="hi",
            user_message="hi",
            hard_rules=[{"code": "X", "name": "n", "description": "d"}],
            evaluator_driver="provider",
            provider=object(),
        )
        assert r.state == VerdictState.ERROR
        assert ollama_called["count"] == 0
