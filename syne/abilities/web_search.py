"""Web Search Ability — search the web via Brave Search API."""

import os
import httpx

from .base import Ability


class WebSearchAbility(Ability):
    """Search the web using Brave Search API."""
    
    name = "web_search"
    description = "Search the web for information using Brave Search"
    version = "1.0"
    
    async def execute(self, params: dict, context: dict) -> dict:
        """Search the web with the given query.
        
        Args:
            params: Must contain 'query' key with search terms
            context: Must contain 'config' with 'BRAVE_API_KEY'
            
        Returns:
            dict with success, result, error keys
        """
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required"}
        
        # Get API key from config or environment (accept multiple key names)
        config = context.get("config", {})
        api_key = (
            config.get("BRAVE_API_KEY")
            or config.get("brave_api_key")
            or config.get("api_key")
            or os.environ.get("BRAVE_API_KEY")
        )
        
        if not api_key:
            return {"success": False, "error": "Brave Search API key not configured"}
        
        # Optional parameters
        count = min(params.get("count", 5), 10)  # Max 10 results
        freshness = params.get("freshness", "")  # pw, pm, py
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                request_params = {
                    "q": query,
                    "count": count,
                }
                if freshness:
                    request_params["freshness"] = freshness
                
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                    params=request_params,
                )
                
                if response.status_code != 200:
                    error_text = response.text[:200]
                    return {"success": False, "error": f"API error ({response.status_code}): {error_text}"}
                
                data = response.json()
                web_results = data.get("web", {}).get("results", [])
                
                if not web_results:
                    return {
                        "success": True,
                        "result": f"No results found for: {query}",
                    }
                
                # Format results
                formatted = []
                for i, result in enumerate(web_results[:count], 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "")
                    description = result.get("description", "No description")
                    
                    formatted.append(f"{i}. **{title}**\n   {url}\n   {description}")
                
                result_text = f"Search results for: {query}\n\n" + "\n\n".join(formatted)
                
                return {
                    "success": True,
                    "result": result_text,
                }
                
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out (30s)"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
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
                        "freshness": {
                            "type": "string",
                            "enum": ["pd", "pw", "pm", "py"],
                            "description": "Filter by freshness: pd=past day, pw=past week, pm=past month, py=past year",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    
    def get_required_config(self) -> list[str]:
        """API key is required but can come from environment."""
        return []
