"""Web Fetch Tool — fetch a URL and extract readable content."""

import logging
import re
import httpx

from ..db.models import get_config

logger = logging.getLogger("syne.tools.web_fetch")


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
    
    # Decode common HTML entities
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&#39;', "'")
    
    # Clean up whitespace
    html = re.sub(r'\n\s*\n+', '\n\n', html)  # Multiple newlines to double
    html = re.sub(r' +', ' ', html)  # Multiple spaces to single
    
    return html.strip()


async def web_fetch_handler(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and extract readable text content.
    
    Args:
        url: URL to fetch (must be http:// or https://)
        max_chars: Maximum characters to return (default 4000)
        
    Returns:
        Extracted text content or error message
    """
    if not url:
        return "Error: URL is required."
    
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        return "Error: URL must start with http:// or https://"
    
    # SSRF protection: block internal/private URLs
    from ..security import is_url_safe
    safe, reason = is_url_safe(url)
    if not safe:
        return f"Error: URL blocked ({reason})"
    
    # Get timeout from config
    timeout_seconds = await get_config("web_fetch.timeout", 30)
    
    # Clamp max_chars to reasonable range
    max_chars = min(max(max_chars, 500), 50000)
    
    try:
        async with httpx.AsyncClient(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SyneBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        ) as client:
            response = await client.get(url)
            
            # Check for HTTP errors
            if response.status_code >= 400:
                return f"Error: HTTP {response.status_code} - {response.reason_phrase}"
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            
            if "text/html" in content_type or "application/xhtml" in content_type:
                # HTML content - extract text
                text = strip_html_tags(response.text)
            elif "text/" in content_type:
                # Plain text content
                text = response.text
            elif "application/json" in content_type:
                # JSON content - return as-is
                text = response.text
            else:
                return f"Error: Unsupported content type: {content_type}"
            
            # Truncate if needed
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... truncated ...]"
            
            return f"Content from {url}:\n\n{text}"
            
    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout_seconds}s"
    except httpx.ConnectError:
        return f"Error: Could not connect to {url}"
    except httpx.TooManyRedirects:
        return "Error: Too many redirects"
    except Exception as e:
        logger.error(f"Web fetch error for {url}: {e}")
        return f"Error: {str(e)}"


# Tool metadata for registration
WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": "Fetch a URL and extract readable text content. Use for reading web pages, articles, documentation.",
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
    "requires_access_level": "public",
}
