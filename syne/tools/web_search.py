"""Web Search Tool — search the web via Brave Search API."""

import logging
import httpx

from ..db.models import get_config

logger = logging.getLogger("syne.tools.web_search")


async def web_search_handler(query: str, count: int = 5) -> str:
    """Search the web using Brave Search API.
    
    Args:
        query: Search query string
        count: Number of results to return (1-10)
        
    Returns:
        Formatted search results or error message
    """
    if not query:
        return "Error: Query is required."
    
    # Get API key from config
    api_key = await get_config("web_search.api_key", "")
    
    if not api_key:
        return (
            "Web search not configured. Ask the owner to set the Brave API key via: "
            "update_config(key='web_search.api_key', value='YOUR_KEY')"
        )
    
    # Clamp count to valid range
    count = min(max(count, 1), 10)
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
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
                return f"Error: API returned {response.status_code}: {error_text}"
            
            data = response.json()
            web_results = data.get("web", {}).get("results", [])
            
            if not web_results:
                return f"No results found for: {query}"
            
            # Format results
            formatted = []
            for i, result in enumerate(web_results[:count], 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                description = result.get("description", "No description")
                formatted.append(f"{i}. **{title}**\n   {url}\n   {description}")
            
            return f"Search results for: {query}\n\n" + "\n\n".join(formatted)
            
    except httpx.TimeoutException:
        return "Error: Search request timed out (30s)."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Error: {str(e)}"


# Tool metadata for registration
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information using Brave Search",
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
    "requires_access_level": "public",
}
