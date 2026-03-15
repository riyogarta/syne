"""Global retry constants and utilities for all LLM drivers.

Centralized retry logic — all drivers import from here instead of
defining their own constants and backoff functions.
"""

import random
import re
from typing import Optional

import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# Global retry constants — shared by ALL drivers
# ═══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES = 5         # 6 total attempts
TOTAL_ATTEMPTS = MAX_RETRIES + 1
BASE_DELAY_MS = 1_000   # 1 second initial delay
MAX_RETRY_DELAY_MS = 30_000  # 30 second cap
STREAM_IDLE_TIMEOUT = 300    # 5 minutes per-chunk timeout

# For empty stream retries (Google/Vertex specific)
EMPTY_STREAM_BASE_DELAY_MS = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Exponential backoff with jitter
# ═══════════════════════════════════════════════════════════════════════════════

def backoff_delay(base_ms: int = BASE_DELAY_MS, attempt: int = 1, max_ms: int = MAX_RETRY_DELAY_MS) -> float:
    """Exponential backoff with ±10% jitter.

    Formula: base × 2^(attempt-1) × jitter(0.9-1.1), capped at max_ms.
    Returns delay in seconds.
    """
    delay_ms = base_ms * (2 ** (attempt - 1))
    jitter = random.uniform(0.9, 1.1)
    delay_ms = min(delay_ms * jitter, max_ms)
    return delay_ms / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Retry-After header parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_retry_delay_header(resp: httpx.Response, default: float = 1.0) -> float:
    """Extract retry delay from response headers (retry-after).

    Falls back to default if header is missing or unparseable.
    Returns delay in seconds.
    """
    retry_after = resp.headers.get("retry-after", "")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-style retry delay parsing (body text + header)
# ═══════════════════════════════════════════════════════════════════════════════

_RETRY_DELAY_RE = re.compile(
    r'(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)'
)


def parse_openai_retry_delay(
    error_text: str,
    headers: Optional[httpx.Headers] = None,
) -> Optional[float]:
    """Parse retry delay from OpenAI error response. Returns seconds or None.

    Checks:
    1. Body text: "try again in Xs" / "try again in Xms"
    2. Retry-After header (seconds)
    """
    m = _RETRY_DELAY_RE.search(error_text)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "ms":
            return value / 1000.0
        return value

    if headers:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass

    return None
