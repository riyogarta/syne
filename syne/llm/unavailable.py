"""Placeholder for a chosen chat model that could not be loaded.

It never substitutes another model. Every chat call re-raises the ORIGINAL
exception from the failed load, unchanged, so the user sees exactly why.
Embedding does not depend on the chat model: it is delegated to the active
embedding provider when that one loaded, otherwise its own error is raised.
"""

from typing import Optional

from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse, StreamCallbacks


class UnavailableProvider(LLMProvider):
    def __init__(
        self,
        error: Exception,
        label: str = "",
        embed_provider: Optional[LLMProvider] = None,
        embed_error: Optional[Exception] = None,
    ):
        self.error = error
        self.model_id = label
        self._label = label
        self._embed_provider = embed_provider
        self._embed_error = embed_error

    @property
    def name(self) -> str:
        return f"unavailable:{self._label}" if self._label else "unavailable"

    @property
    def chat_model(self) -> str:
        return self._label

    @property
    def supports_vision(self) -> bool:
        return False

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
        raise self.error

    def _embedder(self) -> LLMProvider:
        if self._embed_provider is not None:
            return self._embed_provider
        if self._embed_error is not None:
            raise self._embed_error
        raise self.error

    async def embed(self, text: str, model: Optional[str] = None) -> EmbeddingResponse:
        return await self._embedder().embed(text, model)

    async def embed_batch(self, texts: list[str], model: Optional[str] = None) -> list[EmbeddingResponse]:
        return await self._embedder().embed_batch(texts, model)
