"""Context window management — keep token usage under control."""

import logging
from typing import Optional
from .llm.provider import ChatMessage

logger = logging.getLogger("syne.context")

# Rough token estimation: 1 token ≈ 4 chars for English, ≈ 3 chars for CJK/Indonesian
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return int(len(text) / CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        total += 4  # overhead per message (role, formatting)
    return total


class ContextManager:
    """Manages context window to stay within model limits."""

    def __init__(
        self,
        max_context_tokens: int = 128000,
        reserved_output_tokens: int = 4096,
        system_prompt_budget: float = 0.15,    # 15% for system prompt
        memory_budget: float = 0.10,           # 10% for recalled memories
        history_budget: float = 0.65,          # 65% for conversation history
    ):
        self.max_context_tokens = max_context_tokens
        self.reserved_output = reserved_output_tokens
        self.available = max_context_tokens - reserved_output_tokens

        self.system_budget = int(self.available * system_prompt_budget)
        self.memory_budget_tokens = int(self.available * memory_budget)
        self.history_budget = int(self.available * history_budget)

    def trim_context(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Trim messages to fit within context window.
        
        Strategy:
        1. System prompt always kept (first message)
        2. Memory context kept (second system message if exists)
        3. History trimmed from oldest, keeping most recent
        4. Current user message always kept (last message)
        """
        if not messages:
            return messages

        total_tokens = estimate_messages_tokens(messages)

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

        # System + current are non-negotiable
        fixed_tokens = estimate_messages_tokens(system_msgs)
        if current_msg:
            fixed_tokens += estimate_tokens(current_msg.content) + 4

        # Budget remaining for history
        history_allowed = self.available - fixed_tokens

        if history_allowed <= 0:
            # Even system prompt is too big — truncate it
            logger.warning("System prompt exceeds budget, truncating")
            if system_msgs:
                system_msgs[0] = ChatMessage(
                    role="system",
                    content=system_msgs[0].content[:self.system_budget * 4],  # rough char limit
                )
            result = system_msgs
            if current_msg:
                result.append(current_msg)
            return result

        # Trim history from oldest
        trimmed_history = []
        running_tokens = 0

        for msg in reversed(history_msgs):
            msg_tokens = estimate_tokens(msg.content) + 4
            if running_tokens + msg_tokens > history_allowed:
                break
            trimmed_history.insert(0, msg)
            running_tokens += msg_tokens

        dropped = len(history_msgs) - len(trimmed_history)
        if dropped > 0:
            logger.info(f"Dropped {dropped} old messages to fit context window")

        # Reassemble
        result = system_msgs + trimmed_history
        if current_msg:
            result.append(current_msg)

        return result

    def should_compact(self, messages: list[ChatMessage], threshold: float = 0.8) -> bool:
        """Check if context is getting too full and compaction should be triggered."""
        total = estimate_messages_tokens(messages)
        return total >= (self.available * threshold)

    def get_usage(self, messages: list[ChatMessage]) -> dict:
        """Get context window usage stats."""
        total = estimate_messages_tokens(messages)
        return {
            "used_tokens": total,
            "max_tokens": self.available,
            "usage_percent": round(total / self.available * 100, 1),
            "remaining_tokens": self.available - total,
            "message_count": len(messages),
        }
