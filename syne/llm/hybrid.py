"""Hybrid provider — chat from one provider, embedding from another.

Default setup: Google CCA (free OAuth chat) + Together AI (cheap embedding).
"""

from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse


class HybridProvider(LLMProvider):
    """Delegates chat and embed to different underlying providers."""

    def __init__(
        self,
        chat_provider: LLMProvider,
        embed_provider: LLMProvider,
    ):
        self._chat = chat_provider
        self._embed = embed_provider

    @property
    def name(self) -> str:
        return f"{self._chat.name}+{self._embed.name}"

    @property
    def chat_model(self) -> str:
        return getattr(self._chat, "chat_model", "unknown")

    @property
    def supports_vision(self) -> bool:
        return self._chat.supports_vision

    @property
    def context_window(self) -> int:
        return self._chat.context_window

    @property
    def reserved_output_tokens(self) -> int:
        return self._chat.reserved_output_tokens

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
        return await self._chat.chat(
            messages, model, temperature, max_tokens, tools, thinking_budget,
            top_p=top_p, top_k=top_k,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        return await self._embed.embed(text, model)

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        return await self._embed.embed_batch(texts, model)
