"""v1.23.25: a partial outage and a 401 no longer cost KG chunks their
attempts (review v1.23.16-19 #7 partial)."""
import asyncio
import inspect

import httpx
import pytest

from syne.llm.provider import LLMAuthError
from syne.memory import kg_extract as K


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


@pytest.mark.parametrize("err,kind", [
    (asyncio.TimeoutError(), "timeout"),
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.ConnectTimeout("no route"), "down"),
    (RuntimeError("OpenAI stream failed after 3 attempts: ReadTimeout"), "timeout"),
    (RuntimeError("Anthropic stream failed after 3 attempts: ConnectError"), "down"),
    (_Status(429), "down"),
    (_Status(503), "down"),
    (ConnectionError("refused"), "down"),
    (_Status(401), "auth"),
    (_Status(403), "auth"),
    (LLMAuthError("Auth error (401)"), "auth"),
    (RuntimeError("Claude OAuth token expired or invalid. Run 'syne init' to re-authenticate."), "auth"),
    (RuntimeError("Anthropic API error: authentication_error: invalid x-api-key"), "auth"),
    (_Status(400), None),
    (ValueError("bad json"), None),
])
def test_down_kind(err, kind):
    assert K.down_kind(err) == kind


def test_partial_outage_never_blames_the_chunk():
    # 429 / 5xx while other chunks get answers: no attempt, however often
    for strikes in range(1, K.PROVIDER_DOWN_STRIKES):
        assert not K.spends_attempt("down", first=10.0, strikes=strikes, last_ok=50.0, now=60.0)


def test_timeout_blames_the_chunk_only_when_it_repeats_while_others_answer():
    assert not K.spends_attempt("timeout", 10.0, 1, last_ok=50.0, now=60.0)
    assert K.spends_attempt("timeout", 10.0, K.CHUNK_FAULT_STRIKES, last_ok=50.0, now=60.0)
    # nobody answered since the chunk first failed: plain outage
    assert not K.spends_attempt("timeout", 10.0, 5, last_ok=5.0, now=60.0)


def test_a_day_of_failures_still_gives_up_except_auth():
    day = K.PROVIDER_DOWN_GIVE_UP
    assert K.spends_attempt("down", 0.0, K.PROVIDER_DOWN_STRIKES, 0.0, day + 1)
    assert not K.spends_attempt("auth", 0.0, 99, last_ok=0.0, now=day * 9)


async def test_process_one_raises_auth_as_provider_down(monkeypatch):
    async def judge(prompt, timeout):
        raise LLMAuthError("Auth error (401)")
    judge.label = "t"

    class _CM:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(K, "get_connection", lambda: _CM())

    async def _noop(item):
        return None
    monkeypatch.setattr(K, "_count_chunk", _noop)
    monkeypatch.setattr(K, "build_prompt", lambda *a, **k: "p")
    item = {"type": "memory", "id": 1, "content": "Budi tinggal di Bandung", "category": "fact"}
    s = {"kg.extract_max_chars": 6000, "kg.extract_timeout": 5}
    with pytest.raises(K.ProviderDown) as ei:
        await K.process_one(item, judge, None, s)
    assert ei.value.kind == "auth"


def test_worker_pauses_on_auth_and_logs_error_once():
    src = inspect.getsource(K.run_worker)
    seg = src[src.index("except ProviderDown as e:"):src.index("except Exception as e:", src.index("except ProviderDown as e:"))]
    assert "spends_attempt(kind, first, strikes, last_ok, now)" in seg
    # v1.23.30: auth failures are tracked as suspects too
    assert 'if kind != "auth":' not in seg and "logger.error(msg)" in seg
    assert "await _nap(down_wait)" in seg
