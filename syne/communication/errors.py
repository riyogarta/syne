"""Channel-agnostic error classification for user-facing messages."""

import asyncio
import httpx

from ..llm.provider import LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMEmptyResponseError


def classify_error(e: Exception) -> str:
    """Classify any exception into a user-friendly message.

    Works for all channels (Telegram, WhatsApp, CLI). Returns a short
    string suitable for sending directly to the user.
    """
    # 1-4: Typed LLM exceptions (from CCA streaming)
    if isinstance(e, LLMRateLimitError):
        return "Rate limited. Please wait a moment and try again."
    if isinstance(e, LLMAuthError):
        return "Authentication error. Owner may need to refresh credentials."
    if isinstance(e, LLMBadRequestError):
        return "LLM rejected the request. This may be a conversation format issue."
    if isinstance(e, LLMEmptyResponseError):
        return "LLM returned an empty response. Please try again."

    # 5: httpx HTTP status errors (from non-CCA drivers)
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return "Rate limited. Please wait a moment and try again."
        if code in (401, 403):
            return "Authentication error. Owner may need to refresh credentials."
        if code == 400:
            return "LLM rejected the request. This may be a conversation format issue."
        if 500 <= code < 600:
            return "LLM provider is having server issues. Please try again later."
        return f"LLM provider returned HTTP {code}. Please try again later."

    # 6: RuntimeError with known message patterns (from Anthropic/other drivers)
    if isinstance(e, RuntimeError):
        msg = str(e)
        if "429" in msg or "rate" in msg.lower():
            return "Rate limited. Please wait a moment and try again."
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            return "Authentication error. Owner may need to refresh credentials."
        if "400" in msg or "bad request" in msg.lower():
            return "LLM rejected the request. This may be a conversation format issue."
        if "overloaded" in msg.lower() or "529" in msg:
            return "LLM provider is overloaded. Please try again later."

    # 7-8: Database errors
    try:
        import asyncpg
        if isinstance(e, asyncpg.InterfaceError):
            return "Database connection pool exhausted. Please try again in a moment."
        if isinstance(e, asyncpg.PostgresError):
            return "Database error. Please try again later."
    except ImportError:
        pass

    # 9-10: Network / timeout errors
    if isinstance(e, httpx.ConnectError):
        return "Cannot connect to LLM provider. Please check connectivity and try again."
    if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout)):
        return "Request timed out. Please try again."

    # 11: asyncio timeout
    if isinstance(e, asyncio.TimeoutError):
        return "Request timed out. Please try again."

    # 12: Unexpected response shape
    if isinstance(e, (KeyError, IndexError)):
        return "Unexpected response format from LLM provider. Please try again."

    # 13: Not implemented
    if isinstance(e, NotImplementedError):
        return "This feature is not supported by the current LLM provider."

    # 14: Fallback — include type name for debugging
    type_name = type(e).__name__
    return f"Something went wrong ({type_name}). Check logs for details."
