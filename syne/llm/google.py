"""Google Gemini provider via Cloud Code Assist (OAuth) or Gemini API (key).

Aligned with OpenClaw's google-gemini-cli.js + google-shared.js implementation.
All 14 differences resolved:
  1. parametersJsonSchema (not parameters) for tool schemas
  2. Server-parsed retry delays (extractRetryDelay)
  3. Full retryable error detection (429,500,502,503,504 + regex)
  4. Endpoint fallback (daily-cloudcode-pa + cloudcode-pa)
  5. Empty stream retry (MAX_EMPTY_STREAM_RETRIES=2)
  6. Unicode surrogate sanitization
  7. sessionId in request
  8. Thinking config (includeThoughts + thinkingLevel for Gemini 3)
  9. functionResponse format ({output: value} / {error: value})
 10. Empty text blocks skipped
 11. thoughtSignature preserved & replayed
 12. transformMessages (skip errored/aborted, synthetic orphan results)
 13. generationConfig conditional (only if non-empty)
 14. Temperature only if defined (not always 0.7)
"""

import asyncio
import json
import logging
import re
import time
import random
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse, LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMContextWindowError, LLMEmptyResponseError, StreamCallbacks
from ..auth.google_oauth import GoogleCredentials
from .gemini_common import (
    _sanitize_surrogates,
    _extract_retry_delay,
    _is_retryable_error,
    _is_valid_thought_signature,
    _clean_schema_for_gemini,
    _convert_tools_to_gemini,
    _transform_messages,
    _sanitize_turn_ordering,
    _build_thinking_config,
    format_messages_for_gemini,
    _classify_google_error,
)

logger = logging.getLogger("syne.llm.google")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants — match OpenClaw exactly
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
_ENDPOINT_FALLBACKS = [_ANTIGRAVITY_DAILY_ENDPOINT, _DEFAULT_ENDPOINT]

_CCA_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}

# Standard Gemini API endpoint (used with API keys)
_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

# Retry configuration — aligned with Gemini CLI
_MAX_RETRIES = 9  # 10 total attempts
_BASE_DELAY_MS = 5_000
_MAX_EMPTY_STREAM_RETRIES = 1
_EMPTY_STREAM_BASE_DELAY_MS = 500
_MAX_RETRY_DELAY_MS = 30_000
_STREAM_OVERALL_TIMEOUT = 300  # 5min hard cap on entire SSE stream

# CCA rate limiting is handled server-side. No client-side throttle —
# if the server returns 429, the retry loop handles it with backoff.

# Limit concurrent Google LLM calls to avoid quota burst
_chat_semaphore = asyncio.Semaphore(2)

# Tool call ID counter
_tool_call_counter = 0




# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers imported from gemini_common — re-exported for backward compat
# _sanitize_surrogates, _extract_retry_delay, _is_retryable_error,
# _is_valid_thought_signature, _clean_schema_for_gemini, _convert_tools_to_gemini,
# _transform_messages, _sanitize_turn_ordering, _build_thinking_config,
# format_messages_for_gemini, _classify_google_error
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Provider class
# ═══════════════════════════════════════════════════════════════════════════════

class GoogleProvider(LLMProvider):
    """Google Gemini via Cloud Code Assist OAuth (free) or Gemini API key (paid).

    OAuth mode uses the Cloud Code Assist wrapper API.
    API key mode uses the standard Gemini API directly.
    """

    def __init__(
        self,
        credentials: Optional[GoogleCredentials] = None,
        api_key: Optional[str] = None,
        chat_model: str = "gemini-2.5-pro",
        embedding_model: str = "text-embedding-004",
        base_url: Optional[str] = None,
    ):
        self.credentials = credentials
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self._base_url = base_url or _GEMINI_API

        if not credentials and not api_key:
            raise ValueError("Either credentials (OAuth) or api_key is required")

        self._use_cca = credentials is not None

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
            if self._use_cca:
                return await self._chat_cca(messages, model, temperature, max_tokens, tools, thinking_budget, top_p, top_k, stream_callbacks=stream_callbacks)
            else:
                return await self._chat_api(messages, model, temperature, max_tokens, tools, thinking_budget, top_p, top_k, stream_callbacks=stream_callbacks)

    async def _chat_cca(
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
        """Chat via Cloud Code Assist (OAuth, free)."""
        token = await self.credentials.get_token()
        system_text, contents = self._format_messages(messages, model)

        # Build inner request
        inner: dict = {"contents": contents}

        # #7: sessionId in request
        session_id = f"syne-{int(time.time())}-{random.randint(1000, 9999)}"
        inner["sessionId"] = session_id

        if system_text:
            inner["systemInstruction"] = {"parts": [{"text": system_text}]}

        # #13 & #14: generationConfig conditional — only include if non-empty
        # #14: Temperature only if explicitly set (not always 0.7)
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

        # #13: Only set generationConfig if non-empty
        if gen_config:
            inner["generationConfig"] = gen_config

        # #1: Add tools — use parametersJsonSchema (not parameters)
        if tools:
            # Claude models on CCA need legacy `parameters` field
            use_params = model.startswith("claude-")
            gemini_tools = _convert_tools_to_gemini(tools, use_parameters=use_params)
            if gemini_tools:
                inner["tools"] = gemini_tools

        # Wrap in CCA envelope
        request_id = f"syne-{int(time.time())}-{random.randint(1000, 9999)}"
        body = {
            "project": self.credentials.project_id,
            "model": model,
            "request": inner,
            "userAgent": "syne-agent",
            "requestId": request_id,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **_CCA_HEADERS,
        }

        return await self._chat_cca_streaming(body, headers, model, stream_callbacks=stream_callbacks)

    async def _chat_cca_streaming(self, body: dict, headers: dict, model: str, stream_callbacks: Optional[StreamCallbacks] = None) -> ChatResponse:
        """Chat via CCA streaming endpoint with full OpenClaw-style retry logic.

        Implements:
        - #2: Server-parsed retry delays
        - #3: Full retryable error detection
        - #4: Endpoint fallback
        - #5: Empty stream retry
        """
        headers["Accept"] = "text/event-stream"
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

        async def _do_streaming_request(url: str) -> bool:
            """Execute one streaming request, parse SSE, return True if content received."""
            nonlocal input_tokens, output_tokens
            has_content = False

            # Overall stream timeout — prevents keep-alive lines from holding
            # the connection open forever (httpx read timeout only checks for
            # ANY data, including empty keep-alive lines).
            stream_deadline = time.monotonic() + _STREAM_OVERALL_TIMEOUT

            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", url, content=body_json, headers=headers) as resp:
                    # Check status BEFORE consuming stream.
                    # On error, read body for retry delay info, then raise.
                    if resp.status_code >= 400:
                        await resp.aread()
                        resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if time.monotonic() > stream_deadline:
                            logger.warning(f"CCA stream exceeded {_STREAM_OVERALL_TIMEOUT}s overall — aborting")
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

                        # Unwrap response envelope
                        response_data = chunk.get("response")
                        if not response_data:
                            continue

                        # Log first chunk
                        if not has_content:
                            logger.debug(f"CCA first chunk: {data_str[:500]}")

                        for candidate in response_data.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    has_content = True
                                    if part.get("thought"):
                                        thinking_parts.append(part["text"])
                                        if stream_callbacks and stream_callbacks.on_thinking:
                                            stream_callbacks.on_thinking(part["text"])
                                    else:
                                        # #10: Skip empty text blocks
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
                                    # #11: Preserve thoughtSignature
                                    if part.get("thoughtSignature"):
                                        tc_entry["thoughtSignature"] = part["thoughtSignature"]
                                    tool_calls.append(tc_entry)

                        # Extract usage
                        usage = response_data.get("usageMetadata", {})
                        if usage.get("promptTokenCount"):
                            input_tokens = usage["promptTokenCount"]
                        if usage.get("candidatesTokenCount"):
                            output_tokens = usage["candidatesTokenCount"]

            return has_content

        # ─── Retry loop with endpoint fallback (#3, #4) ───
        last_error: Optional[Exception] = None
        request_url: Optional[str] = None
        streamed = False
        _auth_refresh_attempted = False
        current_delay = _BASE_DELAY_MS

        for attempt in range(_MAX_RETRIES + 1):
            try:
                # #4: Cycle through endpoints on retry
                endpoint = _ENDPOINT_FALLBACKS[min(attempt, len(_ENDPOINT_FALLBACKS) - 1)]
                request_url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"

                streamed = await _do_streaming_request(request_url)
                break  # Success

            except httpx.HTTPStatusError as e:
                try:
                    await e.response.aread()
                    error_text = e.response.text
                except Exception:
                    error_text = str(e)
                status = e.response.status_code

                # Classify the error
                classification = _classify_google_error(status, error_text, e.response.headers)

                # Terminal errors — raise immediately
                if classification.is_terminal:
                    raise LLMRateLimitError(
                        f"Terminal quota error: {classification.reason}"
                    ) from e

                # Auth errors — try token refresh once
                if status in (401, 403):
                    if not _auth_refresh_attempted and self.credentials:
                        _auth_refresh_attempted = True
                        logger.info(f"CCA {status} — attempting token refresh...")
                        try:
                            self.credentials.expires_at = 0
                            new_token = await self.credentials.get_token()
                            headers["Authorization"] = f"Bearer {new_token}"
                            logger.info("CCA token refreshed, retrying...")
                            _reset_output()
                            continue
                        except Exception as refresh_err:
                            logger.error(f"CCA token refresh failed: {refresh_err}")
                    raise LLMAuthError(f"Authentication failed ({status}): {error_text[:200]}") from e

                if status == 400:
                    logger.error(f"Gemini 400 Bad Request (full): {error_text[:2000]}")
                    combined = (error_text + " " + str(e)).lower()
                    if "exceeds the maximum" in combined or "token count" in combined or "too many tokens" in combined:
                        raise LLMContextWindowError(f"Input tokens exceed limit: {error_text[:300]}") from e
                    raise LLMBadRequestError(f"Bad request (400): {error_text[:500]}") from e

                # Retryable — backoff with jitter
                if attempt < _MAX_RETRIES and classification.is_retryable:
                    # On 429: try next endpoint immediately if different
                    next_endpoint = _ENDPOINT_FALLBACKS[min(attempt + 1, len(_ENDPOINT_FALLBACKS) - 1)]
                    if status == 429 and next_endpoint != endpoint:
                        delay_ms = random.randint(1500, 2500)
                        logger.warning(
                            f"CCA 429 on {endpoint.split('//')[1].split('/')[0]}, "
                            f"switching to {next_endpoint.split('//')[1].split('/')[0]} "
                            f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})"
                        )
                    else:
                        # Use max of backoff delay and server-requested delay
                        delay_ms = max(current_delay, classification.delay_ms)
                        # Jitter: +0-20% for quota w/ server delay, ±30% for others
                        if classification.delay_ms > 0:
                            jitter = random.uniform(1.0, 1.2)
                        else:
                            jitter = random.uniform(0.7, 1.3)
                        delay_ms = int(delay_ms * jitter)
                        logger.warning(
                            f"CCA {status} ({classification.reason}), retrying in {delay_ms}ms "
                            f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}, endpoint: {endpoint})"
                        )

                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    _reset_output()
                    continue

                # Max retries exceeded
                if status == 429:
                    raise LLMRateLimitError(f"Rate limited (429) after {_MAX_RETRIES + 1} attempts.") from e
                raise

            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMContextWindowError):
                raise

            except Exception as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay_ms = int(current_delay * random.uniform(0.7, 1.3))
                    logger.warning(f"CCA network error: {e}, retrying in {delay_ms}ms")
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    _reset_output()
                    continue
                raise

        # ─── Empty stream retry (#5) ───
        if not streamed and request_url:
            for empty_attempt in range(1, _MAX_EMPTY_STREAM_RETRIES + 1):
                backoff_ms = _EMPTY_STREAM_BASE_DELAY_MS * (2 ** (empty_attempt - 1))
                logger.warning(f"CCA empty stream, retry {empty_attempt}/{_MAX_EMPTY_STREAM_RETRIES} in {backoff_ms}ms")
                await asyncio.sleep(backoff_ms / 1000)
                _reset_output()

                try:
                    streamed = await _do_streaming_request(request_url)
                    if streamed:
                        break
                except Exception as e:
                    logger.warning(f"CCA empty stream retry failed: {e}")

        if not streamed:
            raise LLMEmptyResponseError(f"Cloud Code Assist API returned an empty response for {model}")

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
        """Chat via standard Gemini API (API key, paid)."""
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
        for attempt in range(_MAX_RETRIES + 1):
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

                if attempt < _MAX_RETRIES and classification.is_retryable:
                    delay_ms = max(current_delay, classification.delay_ms)
                    if classification.delay_ms > 0:
                        jitter = random.uniform(1.0, 1.2)
                    else:
                        jitter = random.uniform(0.7, 1.3)
                    delay_ms = int(delay_ms * jitter)
                    logger.warning(
                        f"Gemini API {resp.status_code} ({classification.reason}), retrying in {delay_ms}ms "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})"
                    )
                    await asyncio.sleep(delay_ms / 1000)
                    current_delay = min(_MAX_RETRY_DELAY_MS, current_delay * 2)
                    continue

                if resp.status_code == 429:
                    raise LLMRateLimitError(f"Rate limited (429) after {_MAX_RETRIES + 1} attempts.")
                if resp.status_code == 400:
                    _lower = error_text.lower()
                    if "exceeds the maximum" in _lower or "token count" in _lower or "too many tokens" in _lower:
                        raise LLMContextWindowError(f"Input tokens exceed limit: {error_text[:300]}")
                resp.raise_for_status()

        if data is None:
            raise LLMRateLimitError(f"Rate limited after {_MAX_RETRIES + 1} attempts.")

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
        """Embedding — always uses Gemini API (CCA doesn't have embed endpoint).

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
            if self._use_cca:
                token = await self.credentials.get_token()
                headers = {"Authorization": f"Bearer {token}"}
                params = {}
            else:
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
            if self._use_cca:
                token = await self.credentials.get_token()
                headers = {"Authorization": f"Bearer {token}"}
                params = {}
            else:
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
