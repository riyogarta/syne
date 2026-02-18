"""OpenAI-compatible provider (OpenAI, Groq, Together, etc.)."""

import json
import logging
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.openai")


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

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_messages(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Convert ChatMessages to OpenAI format, handling tool calls/results."""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                formatted.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                entry: dict = {"role": "assistant"}
                # Check if this assistant message had tool calls
                if msg.metadata and msg.metadata.get("tool_calls"):
                    entry["content"] = msg.content or None
                    entry["tool_calls"] = msg.metadata["tool_calls"]
                else:
                    entry["content"] = msg.content
                formatted.append(entry)
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                tool_call_id = msg.metadata.get("tool_call_id", f"call_{tool_name}") if msg.metadata else f"call_{tool_name}"
                formatted.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": tool_call_id,
                })
        return formatted

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
    ) -> ChatResponse:
        model = model or self.chat_model

        body: dict = {
            "model": model,
            "messages": self._format_messages(messages, tools),
            "temperature": temperature,
        }

        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools

        logger.debug(f"Request: model={model}, messages={len(messages)}, tools={len(tools) if tools else 0}")

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._get_headers(),
            )
            
            if resp.status_code == 429:
                error_body = resp.text
                logger.error(f"Rate limited (429): {error_body[:200]}")
                # Try to extract retry-after
                retry_after = resp.headers.get("retry-after", "a moment")
                raise httpx.HTTPStatusError(
                    f"Rate limited. Retry after {retry_after}.",
                    request=resp.request,
                    response=resp,
                )
            
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        # Parse tool calls into Syne's format
        tool_calls = None
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "name": fn.get("name"),
                    "args": args,
                    "id": tc.get("id"),
                })

        return ChatResponse(
            content=message.get("content") or "",
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            tool_calls=tool_calls,
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
