"""Shared retry helpers for OpenAI-compatible providers (openai.py + codex.py).

Error classification and exponential backoff modeled after Codex CLI (../codex).
Reuses ErrorClassification/_ErrorClass from gemini_common.py.
"""

import json
import logging
import random
import re
from typing import Optional

import httpx

from .gemini_common import ErrorClassification, _ErrorClass

logger = logging.getLogger("syne.llm.openai_common")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (aligned with Codex CLI defaults)
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_RETRIES = 4  # 5 total attempts (Codex: request_max_retries=4)
_BASE_DELAY_MS = 1_000  # 1s (conservative vs Codex's 200ms)
_MAX_RETRY_DELAY_MS = 30_000
_STREAM_IDLE_TIMEOUT = 300  # 5min per-chunk (Codex DEFAULT_STREAM_IDLE_TIMEOUT_MS)


# ═══════════════════════════════════════════════════════════════════════════════
# Exponential backoff — from Codex retry.rs:38-47
# ═══════════════════════════════════════════════════════════════════════════════

def _backoff_delay(base_ms: int, attempt: int, max_ms: int = _MAX_RETRY_DELAY_MS) -> float:
    """Exponential backoff with ±10% jitter.

    Formula: base × 2^(attempt-1) × jitter(0.9-1.1), capped at max_ms.
    Returns delay in seconds.
    """
    delay_ms = base_ms * (2 ** (attempt - 1))
    jitter = random.uniform(0.9, 1.1)
    delay_ms = min(delay_ms * jitter, max_ms)
    return delay_ms / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Retry-After delay parsing — from Codex rate_limit_regex()
# ═══════════════════════════════════════════════════════════════════════════════

_RETRY_DELAY_RE = re.compile(
    r'(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)'
)


def _parse_openai_retry_delay(
    error_text: str,
    headers: Optional[httpx.Headers] = None,
) -> Optional[float]:
    """Parse retry delay from OpenAI error response. Returns seconds or None.

    Checks:
    1. Body text: "try again in Xs" / "try again in Xms"
    2. Retry-After header (seconds)
    """
    # Body text pattern (Codex rate_limit_regex)
    m = _RETRY_DELAY_RE.search(error_text)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "ms":
            return value / 1000.0
        return value  # seconds

    # Retry-After header
    if headers:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Error classification — modeled after Codex CLI error handling
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_openai_error(
    status: int,
    error_text: str,
    headers: Optional[httpx.Headers] = None,
) -> ErrorClassification:
    """Classify an OpenAI API error for retry decisions.

    Based on Codex CLI error classification:
    - context_length_exceeded → Terminal (is_context_window_error)
    - insufficient_quota → Terminal (is_quota_exceeded_error)
    - invalid_prompt → Terminal (is_invalid_prompt_error)
    - server_is_overloaded / slow_down → Retryable (is_server_overloaded_error)
    - rate_limit_exceeded → Retryable + parsed delay (try_parse_retry_after)
    - 5xx → Retryable (no delay)
    - 429 without code → Retryable + Retry-After header
    - 401/403 → Non-retryable (auth)
    """
    # Parse JSON error
    error_code = ""
    error_message = ""
    try:
        err_json = json.loads(error_text)
        err_obj = err_json.get("error", err_json)
        if isinstance(err_obj, dict):
            error_code = err_obj.get("code", "") or ""
            error_message = err_obj.get("message", "") or ""
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Terminal errors (Codex: raise immediately, no retry)
    if error_code == "context_length_exceeded":
        return ErrorClassification(
            _ErrorClass.TERMINAL, 0,
            error_message or "Context length exceeded"
        )
    if error_code == "insufficient_quota":
        return ErrorClassification(
            _ErrorClass.TERMINAL, 0,
            error_message or "Insufficient quota"
        )
    if error_code == "invalid_prompt":
        return ErrorClassification(
            _ErrorClass.TERMINAL, 0,
            error_message or "Invalid prompt"
        )

    # Auth errors
    if status in (401, 403):
        return ErrorClassification(
            _ErrorClass.NON_RETRYABLE, 0,
            error_message or f"Authentication error ({status})"
        )

    # 400 Bad Request (non-retryable unless it's a known retryable code)
    if status == 400:
        return ErrorClassification(
            _ErrorClass.NON_RETRYABLE, 0,
            error_message or "Bad request (400)"
        )

    # Retryable: server overloaded
    if error_code in ("server_is_overloaded", "slow_down"):
        delay = _parse_openai_retry_delay(error_text, headers)
        delay_ms = int(delay * 1000) if delay else 0
        return ErrorClassification(
            _ErrorClass.RETRYABLE, delay_ms,
            error_message or f"Server overloaded ({error_code})"
        )

    # Retryable: rate limit with parsed delay
    if error_code == "rate_limit_exceeded" or status == 429:
        delay = _parse_openai_retry_delay(error_text, headers)
        delay_ms = int(delay * 1000) if delay else 0
        return ErrorClassification(
            _ErrorClass.RETRYABLE, delay_ms,
            error_message or f"Rate limited ({status})"
        )

    # 5xx → Retryable (no delay — backoff handles it)
    if 500 <= status < 600:
        return ErrorClassification(
            _ErrorClass.RETRYABLE, 0,
            error_message or f"Server error ({status})"
        )

    # Fallback: non-retryable
    return ErrorClassification(
        _ErrorClass.NON_RETRYABLE, 0,
        error_message or f"Unknown error ({status})"
    )
