"""Judge model picked from /models, shared by the memory guard and the rule checker.

Riyo, 15 Nov 2026. /guard (memory.guard_model) and /checker
(security.rule_checker_model) both let the owner pick which model judges a
draft. The key is "" for the default model (provider.active_model, held by
the ConversationManager, never a per-chat override) or a key from /models
(provider.models). An Ollama entry is called directly on its base_url
(/api/chat); every other driver is built with llm.drivers.create_provider.
A key that is missing or cannot be built returns None: the caller decides
what that means, and another model is never substituted.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("syne.judge_model")

_OLLAMA_DEFAULT_URL = "http://localhost:11434"

Judge = Callable[[str, float], Awaitable[str]]


async def build_judge(
    model_key: str,
    *,
    default_provider,
    get_config: Callable[..., Awaitable],
    who: str = "Judge",
) -> Optional[Judge]:
    """Judge callable `(prompt, timeout) -> raw text`, or None when unusable.

    "" -> default_provider. Ollama entry -> direct /api/chat on its base_url.
    Any other /models entry -> llm.drivers.create_provider(entry). The
    returned callable carries a `label` attribute for logging. `who` only
    prefixes the log lines ("Memory guard", "Rule checker").
    """
    from . import rule_checker as RC

    if not model_key:
        if default_provider is None:
            logger.warning(f"{who}: no default provider")
            return None

        async def _default(prompt: str, timeout: float) -> str:
            return await RC._check_via_provider(prompt, default_provider, timeout=timeout)

        _default.label = f"default:{getattr(default_provider, 'chat_model', '?')}"
        return _default

    try:
        models = await get_config("provider.models", []) or []
    except Exception as e:
        logger.warning(f"{who}: cannot read provider.models: {type(e).__name__}: {e}")
        return None
    entry = next(
        (m for m in models if isinstance(m, dict) and m.get("key") == model_key), None
    )
    if entry is None:
        logger.warning(f"{who}: model '{model_key}' not found in /models")
        return None

    if entry.get("driver") == "ollama":
        base_url = entry.get("base_url") or _OLLAMA_DEFAULT_URL
        model_id = entry.get("model_id") or ""
        if not model_id:
            logger.warning(f"{who}: Ollama entry '{model_key}' has no model_id")
            return None

        async def _ollama(prompt: str, timeout: float) -> str:
            return await asyncio.wait_for(
                RC._check_via_ollama(prompt, model=model_id, base_url=base_url, timeout=timeout),
                timeout=timeout,
            )

        _ollama.label = f"ollama:{model_id}"
        return _ollama

    from .llm.drivers import create_provider
    try:
        provider = await create_provider(entry)
    except Exception as e:
        logger.warning(f"{who}: cannot build model '{model_key}': {type(e).__name__}: {e}")
        return None

    async def _picked(prompt: str, timeout: float) -> str:
        return await RC._check_via_provider(prompt, provider, timeout=timeout)

    _picked.label = f"{entry.get('driver')}:{entry.get('model_id')}"
    return _picked
