"""Google Gemini provider via the Gemini API (API key).

Google has closed all OAuth access paths for Gemini, so the former Cloud Code
Assist (OAuth) mode has been removed.

Message formatting, tool schemas, thinking config and error classification are
shared with VertexProvider via gemini_common.
"""

import asyncio
import json
import logging
import random
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse, LLMRateLimitError, LLMContextWindowError, StreamCallbacks
from .gemini_common import (
    _convert_tools_to_gemini,
    _build_thinking_config,
    format_messages_for_gemini,
    _classify_google_error,
)

logger = logging.getLogger("syne.llm.google")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants — match OpenClaw exactly
# ═══════════════════════════════════════════════════════════════════════════════

# Standard Gemini API endpoint (used with API keys)
_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

# Retry — import from global module
from .retry import (
    MAX_RETRIES as _MAX_RETRIES,
    current_max_retries,
    BASE_DELAY_MS as _BASE_DELAY_MS,
    MAX_RETRY_DELAY_MS as _MAX_RETRY_DELAY_MS,
)

# Limit concurrent Google LLM calls to avoid quota burst
_chat_semaphore = asyncio.Semaphore(2)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider class
# ═══════════════════════════════════════════════════════════════════════════════

class GoogleProvider(LLMProvider):
    """Google Gemini via the standard Gemini API (API key)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_model: str = "gemini-2.5-pro",
        embedding_model: str = "text-embedding-004",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self._base_url = base_url or _GEMINI_API

        if not api_key:
            raise ValueError("api_key is required (Gemini OAuth is no longer available)")

    @property
    def name(self) -> str:
        return "google"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def context_window(self) -> int:
        return 1_000_000  # Gemini 2.5 Pro default

    def _format_messages(self, messages: list[ChatMessage], model: str = "") -> tuple[Optional[str], list[dict]]:
        """Convert ChatMessages to Gemini format. Delegates to gemini_common."""
        return format_messages_for_gemini(messages, model)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stream_callbacks: Optional[StreamCallbacks] = None,
    ) -> ChatResponse:
        model = model or self.chat_model

        async with _chat_semaphore:
            return await self._chat_api(messages, model, temperature, max_tokens, tools, thinking_budget, top_p, top_k, stream_callbacks=stream_callbacks)

    async def _chat_api(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream_callbacks: Optional[StreamCallbacks] = None,
    ) -> ChatResponse:
        """Chat via standard Gemini API (API key)."""
        system_text, contents = self._format_messages(messages, model)

        body: dict = {
            "contents": contents,
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        gen_config: dict = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            gen_config["topP"] = top_p
        if top_k is not None:
            gen_config["topK"] = top_k
        # #8: Thinking config — always on, model-appropriate defaults
        gen_config["thinkingConfig"] = _build_thinking_config(model, thinking_budget)

        # #13: Only set if non-empty
        if gen_config:
            body["generationConfig"] = gen_config

        # Tools — standard API uses `parameters` (OpenAPI schema)
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools, use_parameters=True)
            if gemini_tools:
                body["tools"] = gemini_tools

        url = f"{self._base_url}/models/{model}:generateContent"

        data = None
        current_delay = _BASE_DELAY_MS
        # Guard callers (rule checker) cap this via retry_budget() —
        # unbounded retries on 429 would outlive their own timeout.
        max_retries = current_max_retries()
        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=body, params={"key": self.api_key})
                if resp.status_code == 200:
                    data = resp.json()
                    break

                error_text = resp.text[:500]
                logger.error(f"Gemini API error {resp.status_code}: url={resp.request.url} body={error_text}")

                classification = _classify_google_error(resp.status_code, error_text, resp.headers)

                if classification.is_terminal:
                    raise LLMRateLimitError(f"Terminal quota error: {classification.reason}")

                if attempt < max_retries and classification.is_retryable:
                    delay_ms = max(current_delay, classification.delay_ms)
                    if classification.delay_ms > 0:
                        jitter = random.uniform(1.0, 1.2)
                    else:
                        jitter = random.uniform(0.7, 1.3)
                    delay_ms = int(delay_ms * jitter)
                    logger.warning(
                        f"Gemini API {resp.status_code} ({classification.reason}), retrying in {delay_ms}ms "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue

                if resp.status_code == 429:
                    raise LLMRateLimitError(f"Rate limited (429) after {max_retries + 1} attempts.")
                if resp.status_code == 400:
                    _lower = error_text.lower()
                    if "exceeds the maximum" in _lower or "token count" in _lower or "too many tokens" in _lower:
                        raise LLMContextWindowError(f"Input tokens exceed limit: {error_text[:300]}")
                resp.raise_for_status()

        if data is None:
            raise LLMRateLimitError(f"Rate limited after {max_retries + 1} attempts.")

        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])

        text_parts = []
        thinking_parts = []
        tool_calls = []

        for part in parts:
            if part.get("thought") and "text" in part:
                thinking_parts.append(part["text"])
            elif "text" in part:
                # #10: Skip empty
                if part["text"].strip():
                    text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        # Non-streaming: fire callbacks with full content at once
        if stream_callbacks:
            if thinking_parts and stream_callbacks.on_thinking:
                stream_callbacks.on_thinking("".join(thinking_parts))
            if text_parts and stream_callbacks.on_text:
                stream_callbacks.on_text("".join(text_parts))

        usage = data.get("usageMetadata", {})
        logger.info(
            "DIAG-CACHE provider=google-api model=%s in=%s cached=%s out=%s",
            model, usage.get("promptTokenCount", 0),
            usage.get("cachedContentTokenCount", 0), usage.get("candidatesTokenCount", 0),
        )

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            tool_calls=tool_calls if tool_calls else None,
            thinking="".join(thinking_parts) if thinking_parts else None,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Embedding via the Gemini API.

        Includes retry with exponential backoff for transient errors (429, 5xx).
        """
        model = model or self.embedding_model
        url = f"{self._base_url}/models/{model}:embedContent"

        body = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
        }

        current_delay = _BASE_DELAY_MS
        for attempt in range(_MAX_RETRIES + 1):
            headers = {}
            params = {"key": self.api_key}

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(url, json=body, headers=headers, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                vector = data["embedding"]["values"]
                return EmbeddingResponse(vector=vector, model=model, dimensions=len(vector))

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                error_text = e.response.text[:300] if hasattr(e.response, 'text') else str(e)
                classification = _classify_google_error(status, error_text, e.response.headers)
                if classification.is_terminal:
                    raise LLMRateLimitError(f"Terminal quota error: {classification.reason}") from e
                if attempt < _MAX_RETRIES and classification.is_retryable:
                    delay_ms = max(current_delay, classification.delay_ms)
                    if classification.delay_ms > 0:
                        jitter = random.uniform(1.0, 1.2)
                    else:
                        jitter = random.uniform(0.7, 1.3)
                    delay_ms = int(delay_ms * jitter)
                    logger.warning(f"Embed {status} ({classification.reason}), retrying in {delay_ms}ms (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                if status == 429:
                    raise LLMRateLimitError(f"Embed rate limited (429) after {attempt + 1} attempts.") from e
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"Embed network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                raise

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Batch embedding with retry + exponential backoff."""
        model = model or self.embedding_model
        url = f"{self._base_url}/models/{model}:batchEmbedContents"

        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]

        current_delay = _BASE_DELAY_MS
        for attempt in range(_MAX_RETRIES + 1):
            headers = {}
            params = {"key": self.api_key}

            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        url, json={"requests": requests}, headers=headers, params=params
                    )
                    resp.raise_for_status()
                    data = resp.json()

                return [
                    EmbeddingResponse(vector=emb["values"], model=model, dimensions=len(emb["values"]))
                    for emb in data["embeddings"]
                ]

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                error_text = e.response.text[:300] if hasattr(e.response, 'text') else str(e)
                classification = _classify_google_error(status, error_text, e.response.headers)
                if classification.is_terminal:
                    raise LLMRateLimitError(f"Terminal quota error: {classification.reason}") from e
                if attempt < _MAX_RETRIES and classification.is_retryable:
                    delay_ms = max(current_delay, classification.delay_ms)
                    if classification.delay_ms > 0:
                        jitter = random.uniform(1.0, 1.2)
                    else:
                        jitter = random.uniform(0.7, 1.3)
                    delay_ms = int(delay_ms * jitter)
                    logger.warning(f"Batch embed {status} ({classification.reason}), retrying in {delay_ms}ms (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                if status == 429:
                    raise LLMRateLimitError(f"Batch embed rate limited (429) after {attempt + 1} attempts.") from e
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"Batch embed network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                raise
