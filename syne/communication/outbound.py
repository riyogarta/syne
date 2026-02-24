"""Outbound message processing — universal post-processing before delivery.

Handles:
- Server path stripping (security: never expose internal file paths)
- Narration stripping (code-enforced, model-agnostic)
- MEDIA: protocol extraction
- Message splitting for platform length limits
- Consecutive newline cleanup

These operations are channel-agnostic. Every channel applies them
before platform-specific formatting and delivery.

Design principle: Anything that MUST happen regardless of LLM model
lives here as code, not in the system prompt.
"""

import os
import re
from typing import Optional


# ============================================================
# SERVER PATH STRIPPING (Security)
# ============================================================
# Never expose internal file paths to end users.
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


# ============================================================
# NARRATION STRIPPING (Model-Agnostic Behavior Enforcement)
# ============================================================
# Many LLMs narrate their thinking process ("Let me think...",
# "I'll check...", "Now I'm going to..."). This is a code-level
# filter that strips these phrases regardless of which model is used.
#
# Why code, not prompt: Different models interpret "don't narrate"
# differently. Some ignore it. Some follow it sometimes. Code is
# deterministic — it ALWAYS strips, no matter the model.

_NARRATION_PATTERNS = [
    # "Let me..." / "Hmm, let me..." patterns (EN)
    re.compile(r'^(?:Hmm,?\s*)?Let me\s+(?:think|check|search|see|look|try|find|read|analyze|investigate|diagnose|figure|verify|test).*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
    # "I'll/I'm going to..." patterns (EN)
    re.compile(r'^(?:I\'ll|I\'m going to|I will)\s+(?:check|search|look|try|find|read|analyze|investigate|diagnose|figure|verify|test|see|review).*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
    # "Now I'm..." / "Now let me..." (EN)
    re.compile(r'^Now (?:I\'m|let me|I\'ll|let\'s)\s+.*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
    # "First, I need to..." / "First, let me..." (EN)
    re.compile(r'^First,?\s+(?:I need to|let me|I\'ll)\s+.*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
    # Indonesian narration: "Oke/Baik, aku/saya akan..." — match to end of line
    re.compile(r'^(?:Oke|Baik|Ok|Hmm),?\s*(?:aku|saya)\s+(?:akan|perlu|coba|mau)\s+.*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
    # Indonesian: "Aku/Saya akan cek/coba cek/lihat/cari..."
    re.compile(r'^(?:Aku|Saya)\s+(?:akan|perlu|mau)\s+(?:coba\s+)?(?:cek|lihat|cari|baca|analisa|diagnosa|tes|verifikasi|periksa).*?[.!…]\s*\n?', re.IGNORECASE | re.MULTILINE),
]


def strip_narration(text: str) -> str:
    """Remove thinking-process narration from LLM output.

    Strips phrases like "Let me check...", "I'll search for...",
    "Now I'm going to..." that many models produce despite prompt
    instructions not to.

    Only strips from the BEGINNING of the response — narration
    mid-response is left alone as it might be intentional context.
    """
    if not text:
        return text

    # Only strip narration from the first ~500 chars (beginning of response)
    # to avoid removing legitimate content mid-message
    head = text[:500]
    tail = text[500:]

    for pattern in _NARRATION_PATTERNS:
        head = pattern.sub('', head)

    result = (head + tail).lstrip()

    # If stripping removed everything, return original
    return result if result.strip() else text


# ============================================================
# MEDIA EXTRACTION
# ============================================================
# The MEDIA: protocol is how tools signal file attachments.
# The channel layer sends the file using platform-specific APIs.

def extract_media(text: str) -> tuple[str, Optional[str]]:
    """Extract MEDIA: path from response text.

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


# ============================================================
# MESSAGE SPLITTING
# ============================================================

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


# ============================================================
# COMBINED POST-PROCESSING PIPELINE
# ============================================================

def process_outbound(text: str) -> str:
    """Apply all outbound post-processing in the correct order.

    This is the single entry point for all outbound text processing.
    Every channel should call this before platform-specific formatting.

    Pipeline order:
    1. Strip server paths (security)
    2. Strip narration (model-agnostic behavior)
    3. Clean up whitespace

    Args:
        text: Raw LLM response text

    Returns:
        Processed text ready for platform-specific formatting
    """
    if not text:
        return text

    text = strip_server_paths(text)
    text = strip_narration(text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text
