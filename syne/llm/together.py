"""Together AI provider — primarily used for embeddings."""

import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse


class TogetherProvider(LLMProvider):
    """Together AI provider for chat and embedding."""

    def __init__(
        self,
        api_key: str,
        chat_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.base_url = "https://api.together.xyz/v1"

    @property
    def name(self) -> str:
        return "together"

    @property
    def supports_vision(self) -> bool:
        return False

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        model = model or self.chat_model

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict = {"model": model, "messages": msgs, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ChatResponse(
            content=choice["message"]["content"],
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        model = model or self.embedding_model

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": text},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        vector = data["data"][0]["embedding"]
        return EmbeddingResponse(
            vector=vector,
            model=model,
            dimensions=len(vector),
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        )

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        model = model or self.embedding_model

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": texts},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            EmbeddingResponse(
                vector=item["embedding"],
                model=model,
                dimensions=len(item["embedding"]),
            )
            for item in data["data"]
        ]
