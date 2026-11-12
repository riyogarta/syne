"""Provider-agnostic LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable


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

class LLMContextWindowError(LLMError):
    """Input tokens exceed the model's context window limit."""
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
    stop_reason: Optional[str] = None  # Provider stop reason (e.g. 'refusal', 'end_turn')
    # Prompt-cache usage as reported by the provider. 0 when the provider
    # does not report it (or has no prompt cache). cache_read = input tokens
    # served from cache; cache_write = input tokens written to cache
    # (Anthropic cache_creation_input_tokens; other providers report 0).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class StreamCallbacks:
    """Callbacks for real-time streaming of LLM output to the CLI."""
    on_text: Optional[Callable[[str], None]] = None       # text delta
    on_thinking: Optional[Callable[[str], None]] = None   # thinking delta


@dataclass
class UsageAccumulator:
    """Accumulate token usage across multi-round tool call loops.

    Tracks total output tokens (genuinely new per round) and snapshots
    the last call's input tokens (avoids inflated sums from growing context).
    Cache counters follow the same split: totals are summed per call,
    last_* snapshot the final call.
    """
    total_input: int = 0
    total_output: int = 0
    last_input: int = 0
    rounds: int = 0
    total_cache_read: int = 0
    total_cache_write: int = 0
    last_cache_read: int = 0
    last_cache_write: int = 0

    def add(self, response: ChatResponse):
        """Add a response's token counts to the accumulator."""
        self.total_input += response.input_tokens
        self.total_output += response.output_tokens
        self.last_input = response.input_tokens
        _cr = int(getattr(response, "cache_read_tokens", 0) or 0)
        _cw = int(getattr(response, "cache_write_tokens", 0) or 0)
        self.total_cache_read += _cr
        self.total_cache_write += _cw
        self.last_cache_read = _cr
        self.last_cache_write = _cw
        self.rounds += 1

    def apply_to(self, response: ChatResponse) -> ChatResponse:
        """Return a new ChatResponse with accumulated token counts.

        stop_reason is carried over: dropping it hid a final-round
        refusal from the caller's refusal check.
        """
        return ChatResponse(
            content=response.content,
            model=response.model,
            input_tokens=self.last_input,
            output_tokens=self.total_output,
            tool_calls=response.tool_calls,
            thinking=response.thinking,
            stop_reason=response.stop_reason,
            cache_read_tokens=self.last_cache_read,
            cache_write_tokens=self.last_cache_write,
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
        stream_callbacks: Optional[StreamCallbacks] = None,
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
        A positive ``capabilities.reserved_output_tokens`` wins.
        """
        return int(self.cap("reserved_output_tokens") or 4096)

    # ── Per-model capability overrides ──────────────────────────────
    # Set by drivers.create_provider() from provider.models[].capabilities.
    # Missing/None keys mean "auto-detect from the model name" (the
    # provider's built-in guess). Never mutated in place: drivers assign a
    # fresh dict per instance, so this class-level default stays empty.
    capabilities: dict = {}

    def cap(self, key: str, default=None):
        """Return a capability override, or ``default`` when not set."""
        val = (getattr(self, "capabilities", None) or {}).get(key)
        return default if val is None else val

    def cap_for(self, model: Optional[str], key: str, default=None):
        """Like cap(), but only for this entry's own model.

        A provider instance can be asked to run a model id other than the one
        its registry entry describes; overrides must not leak to that model.
        """
        own = getattr(self, "chat_model", None)
        if model and own and model != own:
            return default
        return self.cap(key, default)


def hoist_volatile(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Move volatile system context onto the last user message.

    Volatile system messages (metadata={"volatile": True}: wall clock,
    memory recall, knowledge graph) change every turn. Providers with
    automatic prefix caching (OpenAI, Codex, Gemini, Vertex) reuse a cached
    prefix only up to the first differing byte, so a volatile block in the
    system prompt or ahead of history keeps the history from ever being
    read from cache. Appending them to the last user message keeps
    everything before it byte-stable. They are rebuilt per request and never
    persisted, so when that message becomes history it matches what was
    cached up to its own text.

    Returns a new list; input messages are not mutated. If there is no user
    message, the input is returned unchanged.
    """
    volatile = [
        m.content for m in messages
        if m.role == "system" and m.content and (m.metadata or {}).get("volatile")
    ]
    if not volatile:
        return messages
    last_user = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"),
        None,
    )
    if last_user is None:
        return messages
    out: list[ChatMessage] = []
    for i, m in enumerate(messages):
        if m.role == "system" and (m.metadata or {}).get("volatile"):
            continue
        if i == last_user:
            base = m.content or ""
            text = "\n\n".join(volatile)
            m = ChatMessage(
                role=m.role,
                content=f"{base}\n\n{text}" if base else text,
                metadata=m.metadata,
            )
        out.append(m)
    return out


# ════════════════════════════════════════════════════════
# Capability overrides — schema shared by drivers and /models
# ════════════════════════════════════════════════════════

CAPABILITY_SCHEMA: dict = {
    "thinking_mode": ("adaptive", "budget", "level", "effort", "none"),
    "api_mode": ("responses", "chat"),
    "reserved_output_tokens": int,
    "vision": bool,
}


def _capability_error(key, value) -> Optional[str]:
    spec = CAPABILITY_SCHEMA.get(key)
    if spec is None:
        return f"{key}: unknown key (allowed: {', '.join(CAPABILITY_SCHEMA)})"
    if value is None:
        return None
    if isinstance(spec, tuple):
        if value not in spec:
            return f"{key}: must be one of {' | '.join(spec)}, or null"
    elif spec is int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return f"{key}: must be a positive integer, or null"
    elif spec is bool:
        if not isinstance(value, bool):
            return f"{key}: must be true or false, or null"
    return None


def validate_capabilities(caps) -> list[str]:
    """Return human-readable errors for a capabilities object (empty = valid)."""
    if not isinstance(caps, dict):
        return ["capabilities must be a JSON object"]
    return [e for e in (_capability_error(k, v) for k, v in caps.items()) if e]


def clean_capabilities(caps) -> tuple[dict, list[str]]:
    """Split a raw capabilities object into (valid non-null entries, errors)."""
    if not isinstance(caps, dict):
        return {}, ([] if caps is None else ["capabilities must be a JSON object"])
    clean: dict = {}
    errors: list[str] = []
    for k, v in caps.items():
        err = _capability_error(k, v)
        if err:
            errors.append(err)
        elif v is not None:
            clean[k] = v
    return clean, errors
