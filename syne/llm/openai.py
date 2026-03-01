"""OpenAI-compatible provider (OpenAI, Groq, Together, etc.)."""

import asyncio
import json
import logging
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.openai")


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
          for attempt in range(2):
            # Accumulated state
            content_parts: list[str] = []
            # tool_calls keyed by index: {index: {"id": ..., "name": ..., "arguments": ...}}
            tc_accum: dict[int, dict] = {}
            thinking_text: str | None = None
            resp_model = model
            usage: dict = {}

            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=stream_body,
                    headers=self._get_headers(),
                    timeout=httpx.Timeout(30.0 if attempt > 0 else 120.0, connect=10.0),
                ) as resp:
                    if resp.status_code == 429:
                        await resp.aread()
                        logger.error(f"Rate limited (429): {resp.text[:200]}")
                        retry_after = resp.headers.get("retry-after", "a moment")
                        raise httpx.HTTPStatusError(
                            f"Rate limited. Retry after {retry_after}.",
                            request=resp.request,
                            response=resp,
                        )

                    if 500 <= resp.status_code < 600 and attempt < 1:
                        await resp.aread()
                        logger.warning(f"OpenAI {resp.status_code}, retrying in 1s (attempt {attempt + 1}/2): {resp.text[:200]}")
                        await asyncio.sleep(1)
                        continue

                    if resp.status_code >= 400:
                        await resp.aread()
                        resp.raise_for_status()

                    async for line in resp.aiter_lines():
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

                        # Reasoning / thinking
                        if delta.get("reasoning_summary"):
                            thinking_text = (thinking_text or "") + delta["reasoning_summary"]
                        elif delta.get("reasoning"):
                            thinking_text = (thinking_text or "") + delta["reasoning"]

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

            except httpx.ReadTimeout:
                if attempt < 1:
                    logger.warning("OpenAI stream read timeout, retrying (attempt 1/2)")
                    await asyncio.sleep(1)
                    continue
                raise RuntimeError("OpenAI stream timed out after 2 attempts")

            # If we got here without continue, we have a valid response
            break
          else:
            raise RuntimeError("OpenAI API failed after 2 attempts")

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

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        model = model or self.embedding_model

        body = {
            "model": model,
            "input": text,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json=body,
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

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

        body = {
            "model": model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json=body,
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

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
