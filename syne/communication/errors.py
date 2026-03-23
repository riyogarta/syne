"""Channel-agnostic error classification for user-facing messages."""

import asyncio
import httpx

from ..llm.provider import LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMContextWindowError, LLMEmptyResponseError


def classify_error(e: Exception, model: str = "") -> str:
    """Classify any exception into a user-friendly message.

    Works for all channels (Telegram, WhatsApp, CLI). Returns a short
    string suitable for sending directly to the user.
    Includes model name and error detail for debugging.
    """
    detail = str(e)[:200] if str(e) else ""
    tag = f"[{model}] " if model else ""

    # 1-4: Typed LLM exceptions (from CCA streaming)
    if isinstance(e, LLMRateLimitError):
        return f"{tag}Rate limited. Please wait a moment and try again."
    if isinstance(e, LLMAuthError):
        return f"{tag}Authentication error. Owner may need to refresh credentials."
    if isinstance(e, LLMContextWindowError):
        return f"{tag}Context too large. Use /compact to free up space, or start a new session."
    if isinstance(e, LLMBadRequestError):
        return f"{tag}LLM rejected: {detail}" if detail else f"{tag}LLM rejected the request."
    if isinstance(e, LLMEmptyResponseError):
        return f"{tag}LLM returned an empty response. Please try again."

    # 5: httpx HTTP status errors (from non-CCA drivers)
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return f"{tag}Rate limited. Please wait a moment and try again."
        if code in (401, 403):
            return f"{tag}Authentication error. Owner may need to refresh credentials."
        if code == 400:
            return f"{tag}LLM rejected (HTTP 400): {detail}" if detail else f"{tag}LLM rejected the request."
        if 500 <= code < 600:
            return f"{tag}LLM provider server error ({code}). Please try again later."
        return f"{tag}LLM provider returned HTTP {code}. Please try again later."

    # 6: RuntimeError with known message patterns (from Anthropic/other drivers)
    if isinstance(e, RuntimeError):
        msg = str(e)
        if "429" in msg or "rate" in msg.lower():
            return f"{tag}Rate limited. Please wait a moment and try again."
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            return f"{tag}Authentication error. Owner may need to refresh credentials."
        if "400" in msg or "bad request" in msg.lower():
            return f"{tag}LLM rejected: {msg[:200]}"
        if "overloaded" in msg.lower() or "529" in msg:
            return f"{tag}LLM provider is overloaded. Please try again later."

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
        return f"{tag}Cannot connect to LLM provider. Please check connectivity and try again."
    if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout)):
        return f"{tag}Request timed out. Please try again."
    if isinstance(e, httpx.RemoteProtocolError):
        return f"{tag}Connection lost. Please try again."

    # 11: asyncio timeout
    if isinstance(e, asyncio.TimeoutError):
        return f"{tag}Request timed out. Please try again."

    # 12: Unexpected response shape
    if isinstance(e, (KeyError, IndexError)):
        return f"{tag}Unexpected response format from LLM provider. Please try again."

    # 13: Not implemented
    if isinstance(e, NotImplementedError):
        return f"{tag}This feature is not supported by the current LLM provider."

    # 14: Fallback — include type name and detail
    type_name = type(e).__name__
    return f"{tag}Something went wrong ({type_name}): {detail}" if detail else f"{tag}Something went wrong ({type_name})."
