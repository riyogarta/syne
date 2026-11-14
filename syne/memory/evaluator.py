"""Evaluator model helpers.

The evaluator model (memory.evaluator_driver / memory.evaluator_model,
managed via /evaluator) is used by the rule checker. Memory auto-capture,
which used to live here, was removed in 1.22.59: the owner's own words
stay in the message history and are reached through history_search /
history_expand instead of a lossy small-model paraphrase.
"""

import logging

import httpx

logger = logging.getLogger("syne.memory.evaluator")


async def check_model_available(
    model: str = "qwen3:0.6b",
    base_url: str = "http://localhost:11434",
) -> bool:
    """Check if an Ollama model is available locally."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        model_base = model.split(":")[0]
        for m in data.get("models", []):
            if model_base in m.get("name", ""):
                return True
        return False
    except Exception:
        return False
