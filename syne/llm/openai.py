"""OpenAI-compatible provider (OpenAI, Groq, Together, etc.)."""

import asyncio
import json
import re
import logging
import time
import httpx
from typing import Optional
from .provider import (
    hoist_volatile,
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



def _gpt_major(model: str) -> int:
    """Major version of a gpt-N[.M]-... model id (0 if not a GPT id).

    GPT-5 and later are reasoning models: temperature is not accepted,
    reasoning_effort is.
    """
    m = re.search(r"gpt-(\d+)", model or "")
    return int(m.group(1)) if m else 0

# ── Responses API / reasoning_effort compatibility ──────────────────────
# Some reasoning models (e.g. gpt-6-luna) reject reasoning_effort together
# with function tools on /v1/chat/completions. On the official OpenAI
# endpoint, reasoning models are therefore driven through /v1/responses,
# where reasoning and tools work together. Fallbacks are learned at runtime
# per (base_url, model), so no model names are hardcoded:
#   * /responses answers 404/405 -> use chat/completions for that model.
#   * chat/completions rejects reasoning_effort + tools -> retry once with
#     reasoning_effort="none" and remember that for the model.
_RESPONSES_UNSUPPORTED: set[str] = set()
_EFFORT_NONE_WITH_TOOLS: set[str] = set()

_RESPONSES_RETRYABLE_CODES = {
    "rate_limit_exceeded", "server_is_overloaded", "server_error", "slow_down",
}


def _is_effort_tools_rejection(error_text: str) -> bool:
    """True if a 400 says reasoning_effort cannot be combined with tools."""
    t = (error_text or "").lower()
    return "reasoning_effort" in t and "tools" in t and "not supported" in t


class _ResponsesUnavailable(Exception):
    """The endpoint has no usable /responses route (404/405)."""


class _ResponsesStreamError(Exception):
    """An error event inside a /responses SSE stream."""

    def __init__(self, code: str, message: str):
        self.code = code or ""
        self.message = message or ""
        super().__init__(f"[{self.code}] {self.message}")


def _format_responses_tools(tools: list[dict]) -> list[dict]:
    """Chat-completions tool schemas -> Responses API function tools.

    strict is set to False explicitly: strict mode requires every property
    to be required and additionalProperties=false, which our schemas are not.
    """
    out = []
    for tool in tools or []:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                "strict": False,
            })
        else:
            out.append(tool)
    return out


def _to_responses_input(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    """ChatMessages -> (instructions, input items) for /v1/responses.

    System messages are concatenated into instructions (never overwritten).
    Tool-call ids use the same fallbacks as the chat-completions formatter
    (call_<name>) so calls and outputs still pair up. Unpaired function calls
    or outputs (e.g. after compaction) are dropped, since the API rejects
    them.
    """
    instructions = ""
    items: list[dict] = []
    for msg in messages:
        meta = msg.metadata or {}
        if msg.role == "system":
            if msg.content:
                instructions = f"{instructions}\n\n{msg.content}" if instructions else msg.content
        elif msg.role == "user":
            imgs: list = []
            if isinstance(meta.get("images"), list):
                imgs.extend([x for x in meta["images"] if isinstance(x, dict)])
            if meta.get("image"):
                imgs.append(meta["image"])
            if imgs:
                parts: list = [{"type": "input_text", "text": msg.content or ""}]
                for img in imgs:
                    mime = img.get("mime_type", "image/jpeg")
                    b64 = img.get("base64", "")
                    parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})
                items.append({"role": "user", "content": parts})
            else:
                items.append({"role": "user", "content": msg.content or ""})
        elif msg.role == "assistant":
            if msg.content:
                items.append({"role": "assistant", "content": msg.content})
            for tc in meta.get("tool_calls") or []:
                if "function" in tc:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                else:
                    name = tc.get("name", "")
                    args = tc.get("args", tc.get("arguments", {}))
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                call_id = tc.get("id") or tc.get("call_id") or f"call_{name or 'unknown'}"
                items.append({
                    "type": "function_call", "call_id": call_id,
                    "name": name, "arguments": args or "{}",
                })
        elif msg.role == "tool":
            name = meta.get("tool_name", "unknown")
            call_id = meta.get("tool_call_id") or f"call_{name}"
            items.append({
                "type": "function_call_output", "call_id": call_id,
                "output": msg.content or "",
            })

    call_ids = {i["call_id"] for i in items if i.get("type") == "function_call"}
    out_ids = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    items = [
        i for i in items
        if (i.get("type") != "function_call" or i["call_id"] in out_ids)
        and (i.get("type") != "function_call_output" or i["call_id"] in call_ids)
    ]
    return instructions, items


class _ResponsesStream:
    """Accumulates /v1/responses SSE events into a ChatResponse payload."""

    def __init__(self, stream_callbacks: Optional[StreamCallbacks] = None):
        self._cb = stream_callbacks
        self.text_parts: list[str] = []
        self.thinking = ""
        self._calls: dict[str, dict] = {}
        self._order: list[str] = []
        self.model: Optional[str] = None
        self.usage: dict = {}
        self.status = ""

    @staticmethod
    def _key(event: dict, item: Optional[dict] = None) -> str:
        return ((item or {}).get("id") or event.get("item_id")
                or f"#{event.get('output_index', 0)}")

    def _call(self, key: str) -> dict:
        if key not in self._calls:
            self._calls[key] = {"id": "", "name": "", "arguments": ""}
            self._order.append(key)
        return self._calls[key]

    def feed(self, event: dict) -> None:
        et = event.get("type", "")
        if et == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                self.text_parts.append(delta)
                if self._cb and self._cb.on_text:
                    self._cb.on_text(delta)
        elif et == "response.reasoning_summary_text.delta":
            delta = event.get("delta", "")
            if delta:
                self.thinking += delta
                if self._cb and self._cb.on_thinking:
                    self._cb.on_thinking(delta)
        elif et in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                c = self._call(self._key(event, item))
                c["id"] = item.get("call_id") or c["id"]
                c["name"] = item.get("name") or c["name"]
                if item.get("arguments"):
                    c["arguments"] = item["arguments"]
            elif item.get("type") == "reasoning" and et.endswith(".done") and not self.thinking:
                summaries = item.get("summary") or []
                self.thinking = "\n\n".join(s.get("text", "") for s in summaries if isinstance(s, dict))
        elif et == "response.function_call_arguments.delta":
            self._call(self._key(event))["arguments"] += event.get("delta", "")
        elif et == "response.function_call_arguments.done":
            c = self._call(self._key(event))
            c["arguments"] = event.get("arguments", c["arguments"])
        elif et in ("response.completed", "response.incomplete"):
            r = event.get("response") or {}
            self.model = r.get("model") or self.model
            self.usage = r.get("usage") or self.usage
            self.status = r.get("status", "")
            if self.status == "incomplete":
                logger.warning(f"OpenAI responses INCOMPLETE: {r.get('incomplete_details')}")
        elif et == "response.failed":
            err = (event.get("response") or {}).get("error") or {}
            raise _ResponsesStreamError(err.get("code", ""), err.get("message", "") or str(err))
        elif et == "error":
            raise _ResponsesStreamError(event.get("code", ""), event.get("message", ""))

    @property
    def content(self) -> str:
        return "".join(self.text_parts)

    def tool_calls(self) -> Optional[list[dict]]:
        out = []
        for key in self._order:
            c = self._calls[key]
            if not c["name"]:
                continue
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            out.append({"name": c["name"], "args": args, "id": c["id"] or f"call_{c['name']}"})
        return out or None


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
        chat_model: str = "gpt-6-luna",
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
        return bool(self.cap("vision", self._provider_name == "openai"))

    @property
    def context_window(self) -> int:
        m = self.chat_model.lower()
        if _gpt_major(m) >= 6:
            return 1_050_000
        if "gpt-5" in m:
            return 400_000
        return 128_000  # GPT-4o default

    @property
    def reserved_output_tokens(self) -> int:
        _override = self.cap("reserved_output_tokens")
        if _override:
            return int(_override)
        m = self.chat_model.lower()
        if _gpt_major(m) >= 5:
            return 32_000  # GPT-5.x/6.x can output up to 128K; reserve conservatively
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
                # Vision support — Chat Completions API accepts a list of content
                # parts mixing text and image_url. Supports "image" (single,
                # legacy) and "images" (list, from Telegram media-group albums).
                _imgs: list = []
                _meta = msg.metadata or {}
                if isinstance(_meta.get("images"), list):
                    _imgs.extend([x for x in _meta["images"] if isinstance(x, dict)])
                if _meta.get("image"):
                    _imgs.append(_meta["image"])
                if _imgs:
                    parts: list = [{"type": "text", "text": msg.content or ""}]
                    for img in _imgs:
                        mime = img.get("mime_type", "image/jpeg")
                        b64 = img.get("base64", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                    formatted.append({"role": "user", "content": parts})
                else:
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
        is_reasoning = self._is_reasoning(model)

        # Reasoning models on the official endpoint go through /v1/responses,
        # where reasoning_effort and function tools can be combined.
        if is_reasoning and self._use_responses_api(model):
            try:
                return await self._chat_responses(
                    messages, model=model, max_tokens=max_tokens, tools=tools,
                    thinking_budget=thinking_budget, stream_callbacks=stream_callbacks,
                )
            except _ResponsesUnavailable as e:
                logger.warning(
                    f"OpenAI /responses unavailable for {model} ({e}); "
                    "falling back to /chat/completions"
                )
                _RESPONSES_UNSUPPORTED.add(self._model_key(model))

        body: dict = {
            "model": model,
            "messages": self._format_messages(hoist_volatile(messages), tools),
        }

        # Reasoning models (GPT-5.x/6.x, o3, o1) ignore temperature; use reasoning_effort instead.
        # Default to "high" — reduces hallucination significantly vs "none"/"low".
        if is_reasoning:
            effort = self._thinking_to_reasoning_effort(thinking_budget)
            if tools and self._model_key(model) in _EFFORT_NONE_WITH_TOOLS:
                effort = "none"  # learned: this model rejects effort + tools here
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

                        # Model rejects reasoning_effort together with tools on
                        # this endpoint: retry once with "none" and remember it.
                        if (resp.status_code == 400 and tools
                                and body.get("reasoning_effort") not in (None, "none")
                                and _is_effort_tools_rejection(resp.text)):
                            _EFFORT_NONE_WITH_TOOLS.add(self._model_key(model))
                            logger.warning(
                                f"OpenAI: {model} rejects reasoning_effort with tools on "
                                "/chat/completions; retrying with reasoning_effort='none'"
                            )
                            body["reasoning_effort"] = "none"
                            stream_body["reasoning_effort"] = "none"
                            continue
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

            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMContextWindowError):
                raise
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                if not last_attempt:
                    delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                    logger.warning(f"OpenAI stream error ({type(exc).__name__}), retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI stream failed after {_TOTAL_ATTEMPTS} attempts: {type(exc).__name__}")

            # If we got here without continue, we have a valid response
            break
          else:
            raise RuntimeError(f"OpenAI API failed after {_TOTAL_ATTEMPTS} attempts")

        _ptd = usage.get("prompt_tokens_details") or {}
        logger.info(
            "DIAG-CACHE provider=%s model=%s in=%s cached=%s out=%s",
            self._provider_name, resp_model, usage.get("prompt_tokens", 0),
            _ptd.get("cached_tokens", 0), usage.get("completion_tokens", 0),
        )

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

    def _is_reasoning(self, model: str) -> bool:
        """Reasoning model? capabilities.thinking_mode wins over the name:
        "none" = plain model (temperature), any other value = reasoning."""
        mode = self.cap_for(model, "thinking_mode")
        if mode is not None:
            return mode != "none"
        m = (model or "").lower()
        return _gpt_major(m) >= 5 or "o3" in m or "o1" in m

    def _model_key(self, model: str) -> str:
        return f"{self.base_url}|{model}"

    def _use_responses_api(self, model: str) -> bool:
        """Use /v1/responses only on the official endpoint (OpenAI-compatible
        hosts such as Groq/Together may not implement it), unless it already
        proved unavailable for this model."""
        if self._model_key(model) in _RESPONSES_UNSUPPORTED:
            return False
        mode = self.cap_for(model, "api_mode")
        if mode == "chat":
            return False
        if mode == "responses":
            return True
        return "api.openai.com" in self.base_url

    def _build_responses_body(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: Optional[int],
        tools: Optional[list[dict]],
        thinking_budget: Optional[int],
    ) -> dict:
        instructions, items = _to_responses_input(hoist_volatile(messages))
        body: dict = {
            "model": model,
            "instructions": instructions or "You are a helpful assistant.",
            "input": items,
            "stream": True,
            "store": False,
            "reasoning": {"effort": self._thinking_to_reasoning_effort(thinking_budget)},
        }
        if max_tokens and max_tokens > 0:
            body["max_output_tokens"] = max_tokens
        if tools:
            body["tools"] = _format_responses_tools(tools)
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True
        return body

    async def _chat_responses(
        self,
        messages: list[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        stream_callbacks: Optional[StreamCallbacks] = None,
    ) -> ChatResponse:
        """Chat via /v1/responses (reasoning + tools). Raises
        _ResponsesUnavailable on 404/405 so the caller can fall back."""
        body = self._build_responses_body(messages, model, max_tokens, tools, thinking_budget)
        url = f"{self.base_url}/responses"
        logger.debug(
            f"Responses request: model={model}, items={len(body['input'])}, "
            f"tools={len(body.get('tools') or [])}, effort={body['reasoning']['effort']}"
        )

        acc = _ResponsesStream(stream_callbacks)
        async with httpx.AsyncClient(timeout=180) as client:
          for attempt in range(_TOTAL_ATTEMPTS):
            last_attempt = attempt >= _MAX_RETRIES
            acc = _ResponsesStream(stream_callbacks)
            try:
                async with client.stream(
                    "POST", url, json=body, headers=self._get_headers(),
                    timeout=httpx.Timeout(30.0 if attempt > 0 else 120.0, connect=10.0),
                ) as resp:
                    if resp.status_code >= 400:
                        await resp.aread()
                        error_text = resp.text[:500]
                        if resp.status_code in (404, 405):
                            raise _ResponsesUnavailable(f"HTTP {resp.status_code}")
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
                        backoff = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                        server_delay = classification.delay_ms / 1000.0 if classification.delay_ms else 0
                        delay = max(backoff, server_delay)
                        logger.warning(
                            f"OpenAI responses {resp.status_code}, retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{_TOTAL_ATTEMPTS}): {error_text[:200]}"
                        )
                        await asyncio.sleep(delay)
                        continue

                    last_data_time = time.monotonic()
                    async for line in resp.aiter_lines():
                        now = time.monotonic()
                        if now - last_data_time > _STREAM_IDLE_TIMEOUT:
                            logger.warning(f"OpenAI responses stream idle for {_STREAM_IDLE_TIMEOUT}s, aborting")
                            raise httpx.ReadTimeout(f"Stream idle timeout ({_STREAM_IDLE_TIMEOUT}s)")
                        last_data_time = now
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        acc.feed(event)
            except _ResponsesStreamError as e:
                if e.code == "context_length_exceeded":
                    raise LLMContextWindowError(e.message or "Context length exceeded")
                if e.code == "insufficient_quota":
                    raise LLMRateLimitError(e.message or "Insufficient quota")
                if e.code in _RESPONSES_RETRYABLE_CODES and not last_attempt:
                    delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                    logger.warning(f"OpenAI responses stream error [{e.code}], retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI responses failed: [{e.code}] {e.message[:300]}")
            except (LLMRateLimitError, LLMAuthError, LLMBadRequestError,
                    LLMContextWindowError, _ResponsesUnavailable):
                raise
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                if not last_attempt:
                    delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                    logger.warning(f"OpenAI responses stream error ({type(exc).__name__}), retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI responses stream failed after {_TOTAL_ATTEMPTS} attempts: {type(exc).__name__}")
            break
          else:
            raise RuntimeError(f"OpenAI responses API failed after {_TOTAL_ATTEMPTS} attempts")

        usage = acc.usage or {}
        _itd = usage.get("input_tokens_details") or {}
        logger.info(
            "DIAG-CACHE provider=%s(responses) model=%s in=%s cached=%s out=%s",
            self._provider_name, acc.model or model, usage.get("input_tokens", 0),
            _itd.get("cached_tokens", 0), usage.get("output_tokens", 0),
        )
        return ChatResponse(
            content=acc.content,
            model=acc.model or model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            tool_calls=acc.tool_calls(),
            thinking=acc.thinking or None,
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
                except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                    if not last_attempt:
                        delay = _backoff_delay(_BASE_DELAY_MS, attempt + 1)
                        logger.warning(f"OpenAI embed error ({type(exc).__name__}), retrying in {delay:.1f}s (attempt {attempt + 1}/{_TOTAL_ATTEMPTS})")
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"OpenAI embed failed after {_TOTAL_ATTEMPTS} attempts: {type(exc).__name__}")

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
