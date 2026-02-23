"""Outbound message processing — universal post-processing before delivery.

Handles:
- Server path stripping (security: never expose internal file paths)
- MEDIA: protocol extraction
- Message splitting for platform length limits
- Consecutive newline cleanup

These operations are channel-agnostic. Every channel applies them
before platform-specific formatting and delivery.
"""

import os
import re
from typing import Optional


# Regex: strip server file paths from outgoing messages
# Catches: "File: /home/syne/...", bare "/home/...", "Saved to: /tmp/..."
# (?<!\S) ensures we don't match URL paths like https://example.com/home/...
_PATH_STRIP_RE = re.compile(
    r'(?:(?:File|Path|Saved to|Output|Lokasi|Location)\s*:?\s*\n?\s*)?(?<!\S)/(?:home|tmp|var|opt|usr)/\S+',
    re.IGNORECASE,
)


def strip_server_paths(text: str) -> str:
    """Remove server file paths from outgoing text.

    Security measure: internal paths like /home/syne/workspace/outputs/...
    should never be visible to end users.
    """
    text = _PATH_STRIP_RE.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def extract_media(text: str) -> tuple[str, Optional[str]]:
    """Extract MEDIA: path from response text.

    The MEDIA: protocol is how tools signal file attachments.
    The channel layer sends the file using platform-specific APIs.

    Args:
        text: Response text (may contain MEDIA: path)

    Returns:
        Tuple of (text_without_media, media_path_or_None)
    """
    media_path = None

    if "\n\nMEDIA: " in text:
        parts = text.rsplit("\n\nMEDIA: ", 1)
        text = parts[0].strip()
        media_path = parts[1].strip()
    elif text.startswith("MEDIA: "):
        media_path = text[7:].strip()
        text = ""

    # Validate media path exists
    if media_path and not os.path.isfile(media_path):
        media_path = None

    return text, media_path


def split_message(text: str, max_length: int = 4096) -> list[str]:
    """Split a long message into chunks respecting platform length limits.

    Tries to split at newlines first, then spaces, then hard-cuts.

    Args:
        text: Message text to split
        max_length: Maximum length per chunk (default: 4096 for Telegram)

    Returns:
        List of message chunks
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try splitting at a newline
        split_at = remaining.rfind("\n", 0, max_length)
        if split_at == -1:
            # Try splitting at a space
            split_at = remaining.rfind(" ", 0, max_length)
        if split_at == -1:
            # Hard cut
            split_at = max_length

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()

    return chunks
