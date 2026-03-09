"""Google Vertex AI provider — native Vertex endpoint (no CCA envelope).

Uses per-model region from model_entry["region"], with auto-detection if not set.
SSE streaming via :streamGenerateContent?alt=sse.
Retry logic aligned with Gemini CLI: error classification, jitter, 10 attempts.
"""

import asyncio
import json
import logging
import random
import re
import time
from typing import Optional

import httpx

from .provider import (
    LLMProvider,
    ChatMessage,
    ChatResponse,
    EmbeddingResponse,
    LLMRateLimitError,
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    StreamCallbacks,
)
from .gemini_common import (
    _sanitize_surrogates,
    _is_valid_thought_signature,
    _convert_tools_to_gemini,
    _build_thinking_config,
    format_messages_for_gemini,
    _classify_google_error,
)

logger = logging.getLogger("syne.llm.vertex")

# Retry configuration — aligned with Gemini CLI
_MAX_RETRIES = 9  # 10 total attempts
_BASE_DELAY_MS = 5_000
_MAX_RETRY_DELAY_MS = 30_000
_MAX_EMPTY_STREAM_RETRIES = 1
_EMPTY_STREAM_BASE_DELAY_MS = 500
_STREAM_OVERALL_TIMEOUT = 300  # 5min hard cap

# Limit concurrent Vertex LLM calls
_chat_semaphore = asyncio.Semaphore(2)

# Common Vertex AI regions for auto-detection
_VERTEX_REGIONS = [
    "us-central1",
    "us-east4",
    "us-west1",
    "europe-west1",
    "europe-west4",
    "asia-northeast1",
    "asia-southeast1",
    "me-central1",
]


def _vertex_endpoint(region: str, model: str) -> str:
    """Build Vertex AI endpoint URL."""
    return f"https://{region}-aiplatform.googleapis.com/v1/publishers/google/models/{model}"


async def detect_vertex_region(api_key: str, model: str = "gemini-2.5-flash") -> str:
    """Auto-detect the best Vertex AI region by racing all regions.

    Sends a lightweight request to each region concurrently and returns
    the first region that responds with HTTP 200 (actual success).
    Falls back to us-central1 if none succeed.
    """
    async def _test_region(region: str) -> str:
        url = f"{_vertex_endpoint(region, model)}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    url,
                    json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                    params={"key": api_key},
                )
                if resp.status_code == 200:
                    return region
        except Exception:
            pass
        return ""

    # Race all regions — first 200 wins
    tasks = {asyncio.create_task(_test_region(r)): r for r in _VERTEX_REGIONS}
    try:
        while tasks:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=15)
            if not done:
                break
            for task in done:
                result = task.result()
                if result:
                    # Cancel remaining tasks
                    for t in tasks:
                        if t not in done:
                            t.cancel()
                    logger.info(f"Vertex AI auto-detected region: {result}")
                    return result
                del tasks[task]
    except Exception as e:
        logger.warning(f"Vertex AI region detection error: {e}")

    logger.warning("Vertex AI region auto-detection failed, using us-central1")
    return "us-central1"


class VertexProvider(LLMProvider):
    """Google Vertex AI provider — direct Vertex endpoint with API key.

    Region is per-model (stored in model_entry["region"]).
    Uses standard Gemini API format, no CCA envelope.
    """

    def __init__(
        self,
        api_key: str,
        region: str = "us-central1",
        chat_model: str = "gemini-2.5-pro",
        embedding_model: str = "text-embedding-004",
    ):
        self.api_key = api_key
        self.region = region
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def context_window(self) -> int:
        return 1_000_000

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
            return await self._chat_streaming(
                messages, model, temperature, max_tokens, tools,
                thinking_budget, top_p, top_k,
                stream_callbacks=stream_callbacks,
            )

    async def _chat_streaming(
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
        """Chat via Vertex AI streaming endpoint with Gemini CLI-style retry."""
        system_text, contents = format_messages_for_gemini(messages, model)

        body: dict = {"contents": contents}

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

        gen_config["thinkingConfig"] = _build_thinking_config(model, thinking_budget)

        if gen_config:
            body["generationConfig"] = gen_config

        # Vertex AI uses `parameters` (OpenAPI schema) for tool schemas
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools, use_parameters=True)
            if gemini_tools:
                body["tools"] = gemini_tools

        url = f"{_vertex_endpoint(self.region, model)}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        params = {"key": self.api_key}

        body_json = json.dumps(body)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        input_tokens = 0
        output_tokens = 0

        def _reset_output():
            nonlocal input_tokens, output_tokens
            text_parts.clear()
            thinking_parts.clear()
            tool_calls.clear()
            input_tokens = 0
            output_tokens = 0

        async def _do_streaming_request() -> bool:
            """Execute one streaming request, parse SSE, return True if content received."""
            nonlocal input_tokens, output_tokens
            has_content = False
            stream_deadline = time.monotonic() + _STREAM_OVERALL_TIMEOUT

            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream(
                    "POST", url, content=body_json, headers=headers, params=params,
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if time.monotonic() > stream_deadline:
                            logger.warning(f"Vertex stream exceeded {_STREAM_OVERALL_TIMEOUT}s — aborting")
                            break
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Vertex returns candidates directly (no response envelope)
                        for candidate in chunk.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    has_content = True
                                    if part.get("thought"):
                                        thinking_parts.append(part["text"])
                                        if stream_callbacks and stream_callbacks.on_thinking:
                                            stream_callbacks.on_thinking(part["text"])
                                    else:
                                        if part["text"].strip():
                                            text_parts.append(part["text"])
                                            if stream_callbacks and stream_callbacks.on_text:
                                                stream_callbacks.on_text(part["text"])
                                elif "functionCall" in part:
                                    has_content = True
                                    fc = part["functionCall"]
                                    tc_id = fc.get("id") or f"{fc.get('name', 'unknown')}_{time.time_ns()}"
                                    tc_entry: dict = {
                                        "id": tc_id,
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": json.dumps(fc.get("args", {})),
                                        },
                                    }
                                    if part.get("thoughtSignature"):
                                        tc_entry["thoughtSignature"] = part["thoughtSignature"]
                                    tool_calls.append(tc_entry)

                        usage = chunk.get("usageMetadata", {})
                        if usage.get("promptTokenCount"):
                            input_tokens = usage["promptTokenCount"]
                        if usage.get("candidatesTokenCount"):
                            output_tokens = usage["candidatesTokenCount"]

            return has_content

        # ─── Retry loop ───
        streamed = False
        current_delay = _BASE_DELAY_MS

        for attempt in range(_MAX_RETRIES + 1):
            try:
                streamed = await _do_streaming_request()
                break

            except httpx.HTTPStatusError as e:
                try:
                    await e.response.aread()
                    error_text = e.response.text
                except Exception:
                    error_text = str(e)
                status = e.response.status_code

                classification = _classify_google_error(status, error_text, e.response.headers)

                if classification.is_terminal:
                    raise LLMRateLimitError(
                        f"Terminal quota error: {classification.reason}"
                    ) from e

                if status == 400:
                    logger.error(f"Vertex 400 Bad Request: {error_text[:2000]}")
                    raise LLMBadRequestError(f"Bad request (400): {error_text[:500]}") from e

                if status in (401, 403):
                    raise LLMAuthError(f"Authentication failed ({status}): {error_text[:200]}") from e

                if attempt < _MAX_RETRIES and classification.is_retryable:
                    delay_ms = max(current_delay, classification.delay_ms)
                    if classification.delay_ms > 0:
                        jitter = random.uniform(1.0, 1.2)
                    else:
                        jitter = random.uniform(0.7, 1.3)
                    delay_ms = int(delay_ms * jitter)
                    logger.warning(
                        f"Vertex {status} ({classification.reason}), retrying in {delay_ms}ms "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})"
                    )
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    _reset_output()
                    continue

                if status == 429:
                    raise LLMRateLimitError(f"Rate limited (429) after {_MAX_RETRIES + 1} attempts.") from e
                raise

            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError):
                raise

            except Exception as e:
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"Vertex network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    _reset_output()
                    continue
                raise

        # ─── Empty stream retry ───
        if not streamed:
            for empty_attempt in range(1, _MAX_EMPTY_STREAM_RETRIES + 1):
                backoff_ms = _EMPTY_STREAM_BASE_DELAY_MS * (2 ** (empty_attempt - 1))
                logger.warning(f"Vertex empty stream, retry {empty_attempt}/{_MAX_EMPTY_STREAM_RETRIES} in {backoff_ms}ms")
                await asyncio.sleep(backoff_ms / 1000)
                _reset_output()
                try:
                    streamed = await _do_streaming_request()
                    if streamed:
                        break
                except Exception as e:
                    logger.warning(f"Vertex empty stream retry failed: {e}")

        if not streamed:
            raise LLMEmptyResponseError(f"Vertex AI returned an empty response for {model}")

        content = "".join(text_parts)
        thinking = "".join(thinking_parts) if thinking_parts else None

        return ChatResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls if tool_calls else None,
            thinking=thinking,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Embedding via Vertex AI :predict endpoint with retry."""
        model = model or self.embedding_model
        url = f"{_vertex_endpoint(self.region, model)}:predict"

        body = {
            "instances": [{"content": text}],
        }

        current_delay = _BASE_DELAY_MS
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        url, json=body, params={"key": self.api_key},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                # Vertex AI returns predictions[].embeddings.values
                prediction = data["predictions"][0]
                values = prediction["embeddings"]["values"]
                return EmbeddingResponse(vector=values, model=model, dimensions=len(values))

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
                    logger.warning(f"Vertex embed {status} ({classification.reason}), retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                if status == 429:
                    raise LLMRateLimitError(f"Embed rate limited (429) after {attempt + 1} attempts.") from e
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"Vertex embed network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                raise

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Batch embedding via Vertex AI :predict with multiple instances."""
        model = model or self.embedding_model
        url = f"{_vertex_endpoint(self.region, model)}:predict"

        body = {
            "instances": [{"content": t} for t in texts],
        }

        current_delay = _BASE_DELAY_MS
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        url, json=body, params={"key": self.api_key},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                return [
                    EmbeddingResponse(
                        vector=pred["embeddings"]["values"],
                        model=model,
                        dimensions=len(pred["embeddings"]["values"]),
                    )
                    for pred in data["predictions"]
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
                    logger.warning(f"Vertex batch embed {status} ({classification.reason}), retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                if status == 429:
                    raise LLMRateLimitError(f"Batch embed rate limited (429) after {attempt + 1} attempts.") from e
                raise

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"Vertex batch embed network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue
                raise
