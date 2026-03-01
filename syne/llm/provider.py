"""Provider-agnostic LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ════════════════════════════════════════════════════════
# LLM Exception Hierarchy — classify errors by type,
# not by string matching.  telegram.py catches these.
# ════════════════════════════════════════════════════════

class LLMError(Exception):
    """Base class for all LLM provider errors."""
    pass

class LLMRateLimitError(LLMError):
    """429 — rate limited after all retries exhausted."""
    pass

class LLMAuthError(LLMError):
    """401/403 — authentication or authorization failure."""
    pass

class LLMBadRequestError(LLMError):
    """400 — bad request (malformed prompt, tool schema, etc.)."""
    pass

class LLMEmptyResponseError(LLMError):
    """LLM returned empty content after retries."""
    pass


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
class UsageAccumulator:
    """Accumulate token usage across multi-round tool call loops.

    Tracks total output tokens (genuinely new per round) and snapshots
    the last call's input tokens (avoids inflated sums from growing context).
    """
    total_input: int = 0
    total_output: int = 0
    last_input: int = 0
    rounds: int = 0

    def add(self, response: ChatResponse):
        """Add a response's token counts to the accumulator."""
        self.total_input += response.input_tokens
        self.total_output += response.output_tokens
        self.last_input = response.input_tokens
        self.rounds += 1

    def apply_to(self, response: ChatResponse) -> ChatResponse:
        """Return a new ChatResponse with accumulated token counts."""
        return ChatResponse(
            content=response.content,
            model=response.model,
            input_tokens=self.last_input,
            output_tokens=self.total_output,
            tool_calls=response.tool_calls,
            thinking=response.thinking,
        )


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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
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

    @property
    def context_window(self) -> int:
        """Maximum context window size in tokens for the default model.

        Override per-provider. Model registry in DB takes precedence if set.
        """
        return 200_000

    @property
    def reserved_output_tokens(self) -> int:
        """Tokens to reserve for output (response + thinking).

        ContextManager uses this to limit how much input context to keep.
        Override in providers that need more room (e.g. extended thinking).
        """
        return 4096
