"""Context window management — keep token usage under control.

Follows OpenClaw's dynamic budget approach: system prompt and memory use
whatever space they need, history gets the rest. No fixed percentage splits.
"""

import logging
from typing import Optional
from .llm.provider import ChatMessage

logger = logging.getLogger("syne.context")

# Rough token estimation: ~4 chars per token for English (industry standard).
# Code/multilingual content is less efficient (~2-3 chars/token).
# Default 4.0 per OpenAI/Anthropic/Google documentation.
# Per-model override available via model_params["chars_per_token"].
DEFAULT_CHARS_PER_TOKEN = 4.0

# 20% safety buffer — chars/4 heuristic underestimates for multi-byte chars,
# code tokens, JSON, special tokens. Same approach as OpenClaw.
SAFETY_MARGIN = 1.2


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Rough token count estimation."""
    return int(len(text) / chars_per_token)


def estimate_messages_tokens(messages: list[ChatMessage], chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content, chars_per_token)
        total += 4  # overhead per message (role, formatting)
    return total


class ContextManager:
    """Manages context window to stay within model limits.

    Dynamic budget: system prompt and memory use what they need,
    history gets the remaining space. No fixed percentage allocations.
    """

    def __init__(
        self,
        max_context_tokens: int = 128000,
        reserved_output_tokens: int = 4096,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        # Legacy params — accepted but ignored for backward compatibility
        system_prompt_budget: float = 0.0,
        memory_budget: float = 0.0,
        history_budget: float = 0.0,
    ):
        self.max_context_tokens = max_context_tokens
        self.reserved_output = reserved_output_tokens
        self.chars_per_token = chars_per_token

        # Apply safety margin: effective capacity accounts for estimation inaccuracy
        self.available = int((max_context_tokens - reserved_output_tokens) / SAFETY_MARGIN)

    def trim_context(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Trim messages to fit within context window.

        Dynamic strategy (OpenClaw approach):
        1. System prompt always kept — uses whatever space it needs
        2. Memory context kept — uses whatever space it needs
        3. Current user message always kept
        4. History gets ALL remaining space, trimmed from oldest
        """
        if not messages:
            return messages

        total_tokens = estimate_messages_tokens(messages, self.chars_per_token)

        if total_tokens <= self.available:
            return messages  # Fits fine

        logger.info(f"Context too large ({total_tokens} tokens), trimming to {self.available}")

        # Separate message types
        system_msgs = []
        history_msgs = []
        current_msg = None

        for i, msg in enumerate(messages):
            if msg.role == "system":
                system_msgs.append(msg)
            elif i == len(messages) - 1:
                current_msg = msg
            else:
                history_msgs.append(msg)

        # System + current are non-negotiable — they use what they need
        fixed_tokens = estimate_messages_tokens(system_msgs, self.chars_per_token)
        if current_msg:
            fixed_tokens += estimate_tokens(current_msg.content, self.chars_per_token) + 4

        # History gets ALL remaining space
        history_allowed = self.available - fixed_tokens

        if history_allowed <= 0:
            # System prompt alone exceeds budget — keep it but warn
            logger.warning("System prompt exceeds available context")
            result = system_msgs
            if current_msg:
                result.append(current_msg)
            return result

        # Trim history from oldest (safety fallback — compaction should
        # have already run, but just in case)
        trimmed_history = []
        running_tokens = 0

        for msg in reversed(history_msgs):
            msg_tokens = estimate_tokens(msg.content, self.chars_per_token) + 4
            if running_tokens + msg_tokens > history_allowed:
                break
            trimmed_history.insert(0, msg)
            running_tokens += msg_tokens

        dropped = len(history_msgs) - len(trimmed_history)
        if dropped > 0:
            logger.warning(
                f"Emergency trim: dropped {dropped} old messages "
                f"(compaction should have prevented this)"
            )
            # Inject a notice so LLM knows context was lost
            notice = ChatMessage(
                role="system",
                content=(
                    f"⚠️ CONTEXT NOTICE: {dropped} older messages were dropped "
                    f"from this conversation due to context window limits. "
                    f"You may not have full history. If the user references "
                    f"something you don't see, acknowledge that earlier context "
                    f"was lost and ask them to clarify."
                ),
            )
            # Insert notice after system messages, before history
            system_msgs.append(notice)

        # Reassemble
        result = system_msgs + trimmed_history
        if current_msg:
            result.append(current_msg)

        return result

    def should_compact(self, messages: list[ChatMessage], threshold: float = 0.8) -> bool:
        """Check if context is getting too full and compaction should be triggered."""
        total = estimate_messages_tokens(messages, self.chars_per_token)
        return total >= (self.available * threshold)

    def get_usage(self, messages: list[ChatMessage]) -> dict:
        """Get context window usage stats."""
        total = estimate_messages_tokens(messages, self.chars_per_token)
        return {
            "used_tokens": total,
            "max_tokens": self.available,
            "usage_percent": round(total / self.available * 100, 1),
            "remaining_tokens": self.available - total,
            "message_count": len(messages),
        }
