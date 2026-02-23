"""Response tag parsing — extract reply/react directives from LLM output.

These tags are channel-agnostic. The LLM embeds them in its response text,
and the communication layer strips and processes them before delivery.

Supported tags:
  [[reply_to_current]]       — reply to the triggering message
  [[reply_to:<message_id>]]  — reply to a specific message
  [[react:<emoji>]]          — react to the triggering message with emoji
"""

import re
from typing import Optional


def parse_reply_tag(text: str, incoming_message_id: Optional[int] = None) -> tuple[str, Optional[int]]:
    """Parse [[reply_to_current]] or [[reply_to:<id>]] tags from response text.

    Args:
        text: LLM response text
        incoming_message_id: The message ID that triggered this response

    Returns:
        Tuple of (cleaned_text, reply_to_message_id or None)
    """
    # Match [[reply_to_current]] or [[ reply_to_current ]]
    if re.search(r'\[\[\s*reply_to_current\s*\]\]', text):
        text = re.sub(r'\[\[\s*reply_to_current\s*\]\]', '', text).strip()
        return text, incoming_message_id

    # Match [[reply_to:<id>]] or [[ reply_to: <id> ]]
    m = re.search(r'\[\[\s*reply_to:\s*(\d+)\s*\]\]', text)
    if m:
        text = re.sub(r'\[\[\s*reply_to:\s*\d+\s*\]\]', '', text).strip()
        return text, int(m.group(1))

    return text, None


def parse_react_tags(text: str) -> tuple[str, list[str]]:
    """Parse [[react:<emoji>]] tags from response text.

    Args:
        text: LLM response text

    Returns:
        Tuple of (cleaned_text, list_of_emojis_to_react_with)
    """
    emojis = []
    for m in re.finditer(r'\[\[\s*react:\s*(.+?)\s*\]\]', text):
        emojis.append(m.group(1).strip())
    if emojis:
        text = re.sub(r'\[\[\s*react:\s*.+?\s*\]\]', '', text).strip()
    return text, emojis
