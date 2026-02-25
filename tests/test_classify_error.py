"""Tests for _classify_error() in Telegram channel."""

import asyncio

import httpx
import pytest

from syne.communication.telegram import _classify_error
from syne.llm.provider import (
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMRateLimitError,
)


# ── Typed LLM exceptions ────────────────────────────────────

class TestLLMExceptions:
    def test_rate_limit(self):
        assert "Rate limited" in _classify_error(LLMRateLimitError("429"))

    def test_auth_error(self):
        assert "Authentication" in _classify_error(LLMAuthError("invalid key"))

    def test_bad_request(self):
        msg = _classify_error(LLMBadRequestError("context too long"))
        assert "/clear" in msg

    def test_empty_response(self):
        assert "empty response" in _classify_error(LLMEmptyResponseError())


# ── httpx.HTTPStatusError ────────────────────────────────────

def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"{status_code} error", request=request, response=response
    )


class TestHTTPStatusError:
    def test_429(self):
        assert "Rate limited" in _classify_error(_make_http_error(429))

    def test_401(self):
        assert "Authentication" in _classify_error(_make_http_error(401))

    def test_403(self):
        assert "Authentication" in _classify_error(_make_http_error(403))

    def test_400(self):
        assert "/clear" in _classify_error(_make_http_error(400))

    def test_500(self):
        assert "server issues" in _classify_error(_make_http_error(500))

    def test_503(self):
        assert "server issues" in _classify_error(_make_http_error(503))

    def test_unknown_status(self):
        msg = _classify_error(_make_http_error(418))
        assert "HTTP 418" in msg


# ── RuntimeError (string matching) ──────────────────────────

class TestRuntimeError:
    def test_429_in_message(self):
        assert "Rate limited" in _classify_error(RuntimeError("Error 429: too many requests"))

    def test_rate_in_message(self):
        assert "Rate limited" in _classify_error(RuntimeError("Rate limit exceeded"))

    def test_401_in_message(self):
        assert "Authentication" in _classify_error(RuntimeError("401 Unauthorized"))

    def test_auth_in_message(self):
        assert "Authentication" in _classify_error(RuntimeError("Authentication failed"))

    def test_400_in_message(self):
        assert "/clear" in _classify_error(RuntimeError("400 Bad Request"))

    def test_overloaded(self):
        assert "overloaded" in _classify_error(RuntimeError("API is overloaded"))

    def test_529(self):
        assert "overloaded" in _classify_error(RuntimeError("529 overloaded"))

    def test_generic_runtime(self):
        # RuntimeError with no known pattern → fallback
        msg = _classify_error(RuntimeError("something unexpected"))
        assert "RuntimeError" in msg


# ── Database errors ──────────────────────────────────────────

class TestDatabaseErrors:
    def test_asyncpg_interface_error(self):
        import asyncpg
        msg = _classify_error(asyncpg.InterfaceError("connection pool exhausted"))
        assert "connection pool" in msg.lower()

    def test_asyncpg_postgres_error(self):
        import asyncpg
        msg = _classify_error(asyncpg.PostgresError("relation does not exist"))
        assert "Database error" in msg


# ── Network / timeout errors ────────────────────────────────

class TestNetworkErrors:
    def test_connect_error(self):
        assert "Cannot connect" in _classify_error(
            httpx.ConnectError("Connection refused")
        )

    def test_read_timeout(self):
        assert "timed out" in _classify_error(httpx.ReadTimeout("read timed out"))

    def test_connect_timeout(self):
        assert "timed out" in _classify_error(httpx.ConnectTimeout("connect timed out"))

    def test_pool_timeout(self):
        assert "timed out" in _classify_error(httpx.PoolTimeout("pool timed out"))

    def test_asyncio_timeout(self):
        assert "timed out" in _classify_error(asyncio.TimeoutError())


# ── Misc exception types ────────────────────────────────────

class TestMiscErrors:
    def test_key_error(self):
        assert "Unexpected response format" in _classify_error(KeyError("choices"))

    def test_index_error(self):
        assert "Unexpected response format" in _classify_error(IndexError("list index out of range"))

    def test_not_implemented(self):
        assert "not supported" in _classify_error(NotImplementedError("streaming"))


# ── Fallback ─────────────────────────────────────────────────

class TestFallback:
    def test_unknown_exception_includes_type_name(self):
        msg = _classify_error(ValueError("bad value"))
        assert "ValueError" in msg
        assert "Something went wrong" in msg

    def test_custom_exception(self):
        class MyCustomError(Exception):
            pass
        msg = _classify_error(MyCustomError("oops"))
        assert "MyCustomError" in msg
