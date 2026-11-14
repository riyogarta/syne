"""Tests for syne/memory/evaluator.py — evaluator model helper.

Memory auto-capture was removed in 1.22.59; only check_model_available
(used by /evaluator and the rule checker setup) remains.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import syne.memory.evaluator as E


def _client_returning(models):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"models": models})
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


async def test_model_available():
    with patch.object(E.httpx, "AsyncClient", return_value=_client_returning([{"name": "qwen3:0.6b"}])):
        assert await E.check_model_available("qwen3:0.6b") is True


async def test_model_missing():
    with patch.object(E.httpx, "AsyncClient", return_value=_client_returning([{"name": "llama3:8b"}])):
        assert await E.check_model_available("qwen3:0.6b") is False


async def test_ollama_unreachable():
    with patch.object(E.httpx, "AsyncClient", side_effect=RuntimeError("down")):
        assert await E.check_model_available("qwen3:0.6b") is False


def test_auto_capture_api_removed():
    for name in ("evaluate_and_store", "evaluate_message", "evaluate_message_ollama", "_is_explicit_remember"):
        assert not hasattr(E, name), name
