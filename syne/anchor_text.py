"""Text a message row is embedded from (history_search).

Channels prepend an untrusted context header to every user message: an
optional bracketed channel instruction (WhatsApp) followed by one or more
"<label>:\\n```json ... ```" blocks (Conversation info / Sender / Replied
message). The header is the same on every row, so embedding it pulls
unrelated messages together. The stored content keeps the header; only the
embedding input drops it.

Moved out of syne/ratification.py when ratification was removed
(Riyo, 13 Nov 2026).
"""

import re

_PREFIX_BLOCK = re.compile(r"\A\s*[^\n`]*:\n```json\n.*?\n```[ \t]*\n*", re.S)
# WhatsApp prepends a bracketed channel instruction before the JSON blocks.
_CHANNEL_INSTRUCTION = re.compile(r"\A\s*\[[A-Za-z ]*channel instruction:[^\]]*\][ \t]*\n*")


def split_context_prefix(text: str) -> tuple:
    """(prefix, body): peel the channel instruction and the untrusted JSON
    context blocks that channels prepend to every user message."""
    text = text or ""
    rest = text
    while True:
        m = _CHANNEL_INSTRUCTION.match(rest) or _PREFIX_BLOCK.match(rest)
        if not m:
            break
        rest = rest[m.end():]
    return text[: len(text) - len(rest)], rest.strip()


def strip_context_prefix(text: str) -> str:
    return split_context_prefix(text)[1]


def anchor_text(content: str) -> str:
    """Embedding input for a message row: the message without the channel
    header. Falls back to the full content when nothing else is left.
    Assistant rows carry no header, so they pass through unchanged."""
    return strip_context_prefix(content) or (content or "").strip()
