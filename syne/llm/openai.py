"""OpenAI-compatible provider (OpenAI, Groq, Together, etc.)."""

import asyncio
import json
import logging
import time
import httpx
from typing import Optional
from .provider import (
    LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse,
    LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMContextWindowError, StreamCallbacks,
)
from .openai_common import (
    _MAX_RETRIES, _BASE_DELAY_MS, _STREAM_IDLE_TIMEOUT,
    _backoff_delay, _classify_openai_error,
)

logger = logging.getLogger("syne.llm.openai")

_TOTAL_ATTEMPTS = _MAX_RETRIES + 1  # 5


def _merge_consecutive(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages (can happen after orphan removal)."""
    if not messages:
        return messages
    result = [messages[0]]
    for msg in messages[1:]:
        prev = result[-1]
        if msg["role"] == prev["role"] and msg["role"] in ("user", "system"):
            # Merge content
            prev_content = prev.get("content", "") or ""
            msg_content = msg.get("content", "") or ""
            prev["content"] = (prev_content + "\n" + msg_content).strip()
        else:
            result.append(msg)
    return result


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider.

    Works with any OpenAI-compatible endpoint:
    - OpenAI: https://api.openai.com/v1
    - Groq:   https://api.groq.com/openai/v1
    - Together: https://api.together.xyz/v1
    """

    def __init__(
        self,
        api_key: str,
        chat_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
    ):
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip("/")
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def supports_vision(self) -> bool:
        # Groq Llama models don't support vision well
        return self._provider_name == "openai"

    @property
    def context_window(self) -> int:
        m = self.chat_model.lower()
        if "gpt-5" in m:
            return 400_000
        return 128_000  # GPT-4o default

    @property
    def reserved_output_tokens(self) -> int:
        m = self.chat_model.lower()
        if "gpt-5" in m:
            return 32_000  # GPT-5.x can output up to 128K; reserve conservatively
        return 4096

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _thinking_to_reasoning_effort(thinking_budget: Optional[int]) -> str:
        """Map thinking_budget to OpenAI reasoning_effort level.

        Budget thresholds (in tokens) → effort:
            None      → "high" (default — reduces hallucination)
            0         → "low"  (minimal reasoning, never fully off)
            1–2048    → "low"
            2049–8192 → "medium"
            8193–16K  → "high"
            >16K      → "xhigh"
        """
        if thinking_budget is None:
            return "high"
        if thinking_budget == 0:
            return "low"
        if thinking_budget <= 2048:
            return "low"
        if thinking_budget <= 8192:
            return "medium"
        if thinking_budget <= 16384:
            return "high"
        return "xhigh"

    def _format_messages(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Convert ChatMessages to OpenAI format, handling tool calls/results.

        Tracks active tool_call IDs to skip orphan tool results (e.g. after compaction).
        """
        formatted = []
        active_tool_ids: set[str] = set()

        for msg in messages:
            if msg.role == "system":
                formatted.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                active_tool_ids = set()
                formatted.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                active_tool_ids = set()
                entry: dict = {"role": "assistant"}
                # Check if this assistant message had tool calls
                if msg.metadata and msg.metadata.get("tool_calls"):
                    entry["content"] = msg.content or None
                    # Convert normalized tool_calls back to OpenAI API format
                    api_tcs = []
                    for tc in msg.metadata["tool_calls"]:
                        if "function" in tc:
                            # Already in API format
                            tc_id = tc.get("id", "")
                            api_tcs.append(tc)
                        else:
                            # Normalized format → convert back
                            args = tc.get("args", {})
                            tc_id = tc.get("id") or f"call_{tc.get('name', 'unknown')}"
                            api_tcs.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc.get("name", ""),
                                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args or "{}"),
                                },
                            })
                        if tc_id:
                            active_tool_ids.add(tc_id)
                    entry["tool_calls"] = api_tcs
                else:
                    entry["content"] = msg.content
                formatted.append(entry)
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                tool_call_id = msg.metadata.get("tool_call_id", f"call_{tool_name}") if msg.metadata else f"call_{tool_name}"
                # Skip orphan tool results — no matching tool call in preceding assistant message
                if active_tool_ids and tool_call_id not in active_tool_ids:
                    logger.debug(f"Skipping orphan tool result: {tool_call_id}")
                    continue
                formatted.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": tool_call_id,
                })

        return _merge_consecutive(formatted)

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
        is_reasoning = "gpt-5" in model.lower() or "o3" in model.lower() or "o1" in model.lower()

        body: dict = {
            "model": model,
            "messages": self._format_messages(messages, tools),
        }

        # Reasoning models (GPT-5.x, o3, o1) ignore temperature; use reasoning_effort instead.
        # Default to "high" — reduces hallucination significantly vs "none"/"low".
        if is_reasoning:
            effort = self._thinking_to_reasoning_effort(thinking_budget)
            body["reasoning_effort"] = effort
        else:
            body["temperature"] = temperature

        # max_tokens — value comes from model params JSON
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if tools:
            body["tools"] = tools

        logger.debug(f"Request: model={model}, messages={len(messages)}, tools={len(tools) if tools else 0}")

        stream_body = {**body, "stream": True, "stream_options": {"include_usage": True}}

        async with httpx.AsyncClient(timeout=180) as client:
          for attempt in range(_TOTAL_ATTEMPTS):
            # Accumulated state
            content_parts: list[str] = []
            # tool_calls keyed by index: {index: {"id": ..., "name": ..., "arguments": ...}}
            tc_accum: dict[int, dict] = {}
            thinking_text: str | None = None
            resp_model = model
            usage: dict = {}
            last_attempt = attempt >= _MAX_RETRIES

            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=stream_body,
                    headers=self._get_headers(),
                    timeout=httpx.Timeout(30.0 if attempt > 0 else 120.0, connect=10.0),
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        error_text = resp.text[:500]
                        classification = _classify_openai_error(resp.status_code, resp.text, resp.headers)

                        if classification.is_terminal:
                            if "context_length" in (classification.reason or ""):
                                raise LLMContextWindowError(classification.reason)
                            raise LLMRateLimitError(classification.reason)

                        if resp.status_code in (401, 403):
                            raise LLMAuthError(classification.reason or f"Auth error ({resp.status_code})")

                        if not classification.is_retryable or last_attempt:
                            if resp.status_code == 400:
                                raise LLMBadRequestError(f"OpenAI 400: {error_text}")
                            raise RuntimeError(f"OpenAI {resp.status_code}: {error_text}")

                        # Retryable — backoff with server delay
                        backoff = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                        server_delay = classification.delay_ms / 1000.0 if classification.delay_ms else 0
                        delay = max(backoff, server_delay)
                        logger.warning(
                            f"OpenAI {resp.status_code}, retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{_TOTAL_ATTEMPTS}): {error_text[:200]}"
                        )
                        await asyncio.sleep(delay)
                        continue

                    # ── Parse SSE stream with idle timeout ──
                    last_data_time = time.monotonic()
                    async for line in resp.aiter_lines():
                        now = time.monotonic()
                        if now - last_data_time > _STREAM_IDLE_TIMEOUT:
                            logger.warning(f"OpenAI stream idle for {_STREAM_IDLE_TIMEOUT}s, aborting")
                            raise httpx.ReadTimeout(f"Stream idle timeout ({_STREAM_IDLE_TIMEOUT}s)")
                        last_data_time = now

                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        resp_model = chunk.get("model", resp_model)

                        # Usage comes in the final chunk with stream_options
                        if chunk.get("usage"):
                            usage = chunk["usage"]

                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Content
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                            if stream_callbacks and stream_callbacks.on_text:
                                stream_callbacks.on_text(delta["content"])

                        # Reasoning / thinking
                        if delta.get("reasoning_summary"):
                            chunk = delta["reasoning_summary"]
                            thinking_text = (thinking_text or "") + chunk
                            if stream_callbacks and stream_callbacks.on_thinking:
                                stream_callbacks.on_thinking(chunk)
                        elif delta.get("reasoning"):
                            chunk = delta["reasoning"]
                            thinking_text = (thinking_text or "") + chunk
                            if stream_callbacks and stream_callbacks.on_thinking:
                                stream_callbacks.on_thinking(chunk)

                        # Tool calls — accumulate by index
                        if delta.get("tool_calls"):
                            for tc_delta in delta["tool_calls"]:
                                idx = tc_delta.get("index", 0)
                                if idx not in tc_accum:
                                    tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc_delta.get("id"):
                                    tc_accum[idx]["id"] = tc_delta["id"]
                                fn = tc_delta.get("function", {})
                                if fn.get("name"):
                                    tc_accum[idx]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    tc_accum[idx]["arguments"] += fn["arguments"]

            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError):
                raise
            except httpx.ReadTimeout:
                if not last_attempt:
                    delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                    logger.warning(f"OpenAI stream read timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI stream timed out after {_TOTAL_ATTEMPTS} attempts")

            # If we got here without continue, we have a valid response
            break
          else:
            raise RuntimeError(f"OpenAI API failed after {_TOTAL_ATTEMPTS} attempts")

        # Parse accumulated tool calls
        tool_calls = None
        if tc_accum:
            tool_calls = []
            for idx in sorted(tc_accum):
                tc = tc_accum[idx]
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": tc["name"],
                    "args": args,
                    "id": tc["id"],
                })

        return ChatResponse(
            content="".join(content_parts),
            model=resp_model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            tool_calls=tool_calls,
            thinking=thinking_text,
        )

    async def _embed_with_retry(
        self,
        body: dict,
        timeout: int,
    ) -> dict:
        """Shared embed request with retry logic."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(_TOTAL_ATTEMPTS):
                last_attempt = attempt >= _MAX_RETRIES
                try:
                    resp = await client.post(
                        f"{self.base_url}/embeddings",
                        json=body,
                        headers=self._get_headers(),
                    )
                    if resp.status_code >= 400:
                        error_text = resp.text[:500]
                        classification = _classify_openai_error(resp.status_code, resp.text, resp.headers)

                        if classification.is_terminal:
                            raise LLMRateLimitError(classification.reason)
                        if resp.status_code in (401, 403):
                            raise LLMAuthError(classification.reason)
                        if not classification.is_retryable or last_attempt:
                            raise RuntimeError(f"OpenAI embed {resp.status_code}: {error_text}")

                        backoff = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                        server_delay = classification.delay_ms / 1000.0 if classification.delay_ms else 0
                        delay = max(backoff, server_delay)
                        logger.warning(f"OpenAI embed {resp.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                        await asyncio.sleep(delay)
                        continue

                    return resp.json()
                except (LLMRateLimitError, LLMAuthError):
                    raise
                except httpx.ReadTimeout:
                    if not last_attempt:
                        delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                        logger.warning(f"OpenAI embed timeout, retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"OpenAI embed timed out after {_TOTAL_ATTEMPTS} attempts")

        raise RuntimeError(f"OpenAI embed failed after {_TOTAL_ATTEMPTS} attempts")

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        model = model or self.embedding_model
        body = {"model": model, "input": text}
        data = await self._embed_with_retry(body, timeout=30)

        vector = data["data"][0]["embedding"]
        usage = data.get("usage", {})

        return EmbeddingResponse(
            vector=vector,
            model=model,
            dimensions=len(vector),
            input_tokens=usage.get("total_tokens", 0),
        )

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        model = model or self.embedding_model
        body = {"model": model, "input": texts}
        data = await self._embed_with_retry(body, timeout=60)

        usage = data.get("usage", {})
        results = []
        for item in data["data"]:
            results.append(EmbeddingResponse(
                vector=item["embedding"],
                model=model,
                dimensions=len(item["embedding"]),
                input_tokens=usage.get("total_tokens", 0) // len(texts),
            ))

        return results
