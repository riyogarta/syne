"""Web Fetch Tool — fetch a URL and extract readable content (SSRF-hardened)."""

import logging
import re
import httpx
from urllib.parse import urljoin

from ..db.models import get_config
from ..security import is_url_safe_async

logger = logging.getLogger("syne.tools.web_fetch")

# ---- Limits -------------------------------------------------------------
DEFAULT_MAX_BYTES = 2 * 1024 * 1024      # 2 MB body cap (streamed)
MAX_REDIRECTS = 5


def strip_html_tags(html: str) -> str:
    """Strip HTML tags and extract text content.

    Simple regex-based approach. For more complex cases, consider html2text.
    """
    # Remove script and style elements entirely
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

    # Replace common block elements with newlines
    html = re.sub(r'<(?:p|div|br|h[1-6]|li|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)

    # Remove all remaining tags
    html = re.sub(r'<[^>]+>', '', html)

    # Decode ALL HTML entities (named + numeric) via stdlib.
    from html import unescape as _html_unescape
    html = _html_unescape(html)

    # Clean up whitespace
    html = re.sub(r'\n\s*\n+', '\n\n', html)  # Multiple newlines to double
    html = re.sub(r' +', ' ', html)  # Multiple spaces to single

    return html.strip()


async def web_fetch_handler(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and extract readable text content.

    Security posture (SSRF-hardened):
    - Scheme allow-list (http/https only).
    - ``is_url_safe_async`` validates the URL AND resolves DNS, rejecting any
      hostname that resolves to a private/loopback/link-local/reserved IP
      (closes DNS-rebinding + obfuscated-IP bypasses). Re-checked on EVERY hop.
    - Auto-redirects DISABLED; each redirect Location is re-validated before
      being followed.
    - Response body is STREAMED and truncated at DEFAULT_MAX_BYTES during
      download (anti-DoS — a malicious server cannot OOM us with a huge body).

    Args:
        url: URL to fetch (must be http:// or https://)
        max_chars: Maximum characters to return (default 4000)

    Returns:
        Extracted text content or error message
    """
    if not url:
        return "Error: URL is required."

    # Validate URL (scheme first)
    if not url.startswith(('http://', 'https://')):
        return "Error: URL must start with http:// or https://"

    # SSRF protection: static check + DNS-resolve guard
    safe, reason = await is_url_safe_async(url)
    if not safe:
        return f"Error: URL blocked ({reason})"

    # Get timeout from config
    timeout_seconds = await get_config("web_fetch.timeout", 30)

    # Clamp max_chars to reasonable range
    max_chars = min(max(max_chars, 500), 50000)

    current = url
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SyneBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        ) as client:
            raw = b""
            truncated = False
            content_type = ""
            encoding = "utf-8"
            got_response = False

            for _ in range(MAX_REDIRECTS + 1):
                # Re-validate every hop (scheme + SSRF + DNS-rebinding).
                ok, why = await is_url_safe_async(current)
                if not ok:
                    return f"Error: Redirect blocked ({why})"

                # Stream so we can enforce the byte cap DURING download.
                async with client.stream("GET", current) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        loc = response.headers.get("location")
                        if not loc:
                            return "Error: Redirect without Location header"
                        current = urljoin(current, loc)
                        continue  # don't read the redirect body

                    got_response = True

                    if response.status_code >= 400:
                        return f"Error: HTTP {response.status_code} - {response.reason_phrase}"

                    content_type = (response.headers.get("content-type") or "").lower()
                    if not ("text/html" in content_type
                            or "application/xhtml" in content_type
                            or "text/" in content_type
                            or "application/json" in content_type):
                        return f"Error: Unsupported content type: {content_type or 'unknown'}"

                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        raw += chunk
                        if len(raw) >= DEFAULT_MAX_BYTES:
                            raw = raw[:DEFAULT_MAX_BYTES]
                            truncated = True
                            break

                    encoding = response.encoding or "utf-8"
                    break
            else:
                return "Error: Too many redirects"

            if not got_response:
                return "Error: No response"

            try:
                body = raw.decode(encoding, errors="replace")
            except (LookupError, TypeError):
                body = raw.decode("utf-8", errors="replace")

            if "text/html" in content_type or "application/xhtml" in content_type:
                text = strip_html_tags(body)
            else:
                text = body

            # Truncate if needed
            if len(text) > max_chars:
                text = text[:max_chars]
                truncated = True
            if truncated:
                text += "\n\n[... truncated ...]"

            return f"Content from {current}:\n\n{text}"

    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout_seconds}s"
    except httpx.ConnectError:
        return f"Error: Could not connect to {current}"
    except httpx.TooManyRedirects:
        return "Error: Too many redirects"
    except Exception as e:
        logger.error(f"Web fetch error for {current}: {e}")
        return f"Error: {str(e)}"


# Tool metadata for registration
WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": "Fetch a URL and extract text content.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch (http:// or https://)",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 4000)",
                "default": 4000,
            },
        },
        "required": ["url"],
    },
    "handler": web_fetch_handler,
    "permission": 0o555,
}
