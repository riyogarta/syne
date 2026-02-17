"""Provider-agnostic LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatMessage:
    role: str           # 'system', 'user', 'assistant', 'tool'
    content: str
    metadata: Optional[dict] = None


@dataclass
class ChatResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: Optional[list] = None
    thinking: Optional[str] = None  # Model's reasoning/thinking text (if returned)


@dataclass
class EmbeddingResponse:
    vector: list[float]
    model: str
    dimensions: int
    input_tokens: int = 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
    ) -> ChatResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate an embedding vector for text."""
        ...

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Generate embedding vectors for multiple texts."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports image input."""
        ...
