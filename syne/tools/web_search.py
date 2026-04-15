"""Web Search Tool — search the web via Tavily or Brave Search API."""

import asyncio
import logging
import time
import httpx

from ..db.models import get_config

logger = logging.getLogger("syne.tools.web_search")

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds
_MIN_INTERVAL = 1.1  # seconds between requests (Brave free: 1 req/sec)
_last_request_time: float = 0.0


async def _throttle():
    """Ensure minimum interval between search API requests."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


async def _request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """Make HTTP request with retry on 429/5xx. Throttled to 1 req/sec."""
    for attempt in range(_MAX_RETRIES):
        await _throttle()
        response = await client.request(method, url, **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < _MAX_RETRIES - 1:
                # Parse retry-after header if present
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else _BASE_DELAY * (2 ** attempt)
                delay = min(delay, 30.0)
                logger.warning(f"Web search {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
        return response
    return response  # return last response even if failed


async def _search_tavily(query: str, api_key: str, count: int) -> str:
    """Search using Tavily API (POST, Bearer auth)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await _request_with_retry(
            client, "POST",
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "max_results": count,
            },
        )

        if response.status_code != 200:
            error_text = response.text[:200]
            return f"Error: Tavily API returned {response.status_code}: {error_text}"

        data = response.json()
        results = data.get("results", [])

        if not results:
            return f"No results found for: {query}"

        formatted = []
        for i, result in enumerate(results[:count], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description")
            formatted.append(f"{i}. **{title}**\n   {url}\n   {content}")

        return f"Search results for: {query}\n\n" + "\n\n".join(formatted)


async def _search_brave(query: str, api_key: str, count: int) -> str:
    """Search using Brave Search API (GET, X-Subscription-Token)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await _request_with_retry(
            client, "GET",
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": query,
                "count": count,
            },
        )

        if response.status_code != 200:
            error_text = response.text[:200]
            return f"Error: Brave API returned {response.status_code}: {error_text}"

        data = response.json()
        web_results = data.get("web", {}).get("results", [])

        if not web_results:
            return f"No results found for: {query}"

        formatted = []
        for i, result in enumerate(web_results[:count], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "No description")
            formatted.append(f"{i}. **{title}**\n   {url}\n   {description}")

        return f"Search results for: {query}\n\n" + "\n\n".join(formatted)


async def web_search_handler(query: str, count: int = 5) -> str:
    """Search the web using Tavily or Brave Search API.

    Detects the provider from the API key prefix (tvly- = Tavily)
    or web_search.driver config.
    """
    if not query:
        return "Error: Query is required."

    api_key = await get_config("web_search.api_key", "")

    if not api_key:
        return (
            "Web search not configured. Ask the owner to set the API key via: "
            "update_config(key='web_search.api_key', value='YOUR_KEY')\n"
            "Supported: Tavily (tvly-...) or Brave Search API key."
        )

    count = min(max(count, 1), 10)

    # Auto-detect provider from key prefix or explicit driver config
    driver = await get_config("web_search.driver", "")
    if not driver:
        driver = "tavily" if api_key.startswith("tvly-") else "brave"

    try:
        if driver == "tavily":
            return await _search_tavily(query, api_key, count)
        else:
            return await _search_brave(query, api_key, count)
    except httpx.TimeoutException:
        return "Error: Search request timed out (30s)."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Error: {str(e)}"


# Tool metadata for registration
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "count": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    "handler": web_search_handler,
    "permission": 0o555,
}
