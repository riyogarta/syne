"""Image Analysis Ability — analyze images via multiple vision backends.

Backends (tried in order):
1. Google Gemini via OAuth (free, rate-limited)
2. Together AI via API key (paid, from ability config)
"""

import base64
import logging
import httpx
from typing import Optional

from .base import Ability

logger = logging.getLogger("syne.abilities.image_analysis")


class ImageAnalysisAbility(Ability):
    """Analyze and describe images using AI vision models.
    
    Supports multiple backends with automatic fallback:
    - Google Gemini (OAuth) — preferred, free
    - Together AI (API key) — fallback, uses config api_key
    
    Supports ability-first pre-processing: when an image is received,
    this ability runs BEFORE the LLM sees the raw image data.
    """
    
    name = "image_analysis"
    description = "Analyze and describe images using AI vision"
    version = "1.1"
    
    def handles_input_type(self, input_type: str) -> bool:
        """This ability handles image inputs."""
        return input_type == "image"
    
    async def pre_process(self, input_type: str, input_data: dict, user_prompt: str) -> Optional[str]:
        """Pre-process image before LLM sees it."""
        prompt = user_prompt if user_prompt and user_prompt.lower() not in (
            "what's in this image?", "describe this image"
        ) else "Describe this image in detail. If there is text, transcribe it."
        
        result = await self.execute(
            params={
                "image_base64": input_data.get("base64", ""),
                "mime_type": input_data.get("mime_type", "image/jpeg"),
                "prompt": prompt,
            },
            context={"config": {}},
        )
        
        if result.get("success"):
            return result["result"]
        return None
    
    async def execute(self, params: dict, context: dict) -> dict:
        """Analyze an image and return a description.
        
        Tries backends in order: Google Gemini OAuth → Together AI API key.
        
        Args:
            params: Must contain either:
                - 'image_url': URL of image to analyze
                - 'image_base64': Base64-encoded image data
            context: Contains 'config' with optional Together AI api_key
            
        Returns:
            dict with success, result, error keys
        """
        image_url = params.get("image_url", "")
        image_base64 = params.get("image_base64", "")
        prompt = params.get("prompt", "Describe this image in detail.")
        
        if not image_url and not image_base64:
            return {"success": False, "error": "Either image_url or image_base64 is required"}
        
        try:
            # Resolve image data from URL if needed
            image_base64, mime_type = await self._resolve_image(
                image_url, image_base64, params.get("mime_type", "image/jpeg")
            )
            if not image_base64:
                return {"success": False, "error": "Failed to get image data"}
        except Exception as e:
            return {"success": False, "error": f"Image resolution error: {str(e)}"}
        
        # Try backends — prioritize what's actually configured
        errors = []
        
        # Check Together AI key first (from ability config or DB)
        config = context.get("config", {})
        together_key = config.get("api_key", "")
        if not together_key:
            together_key = await self._get_together_key_from_db()
        
        # Check Google OAuth availability
        google_available = False
        try:
            from ..auth.google_oauth import get_credentials
            creds = await get_credentials()
            google_available = creds is not None
        except Exception:
            pass
        
        # 1. Together AI — if key exists, use it first (owner configured it)
        if together_key:
            result = await self._try_together(image_base64, mime_type, prompt, together_key)
            if result.get("success"):
                return result
            errors.append(f"Together: {result.get('error', 'unknown')}")
        
        # 2. Google Gemini via OAuth — fallback
        if google_available:
            result = await self._try_gemini(image_base64, mime_type, prompt)
            if result.get("success"):
                return result
            errors.append(f"Gemini: {result.get('error', 'unknown')}")
        elif not together_key:
            errors.append("No vision backend configured (no API key, no Google OAuth)")
        
        return {
            "success": False,
            "error": f"All vision backends failed: {'; '.join(errors)}"
        }
    
    async def _resolve_image(
        self, image_url: str, image_base64: str, default_mime: str
    ) -> tuple[str, str]:
        """Resolve image data — download from URL if needed."""
        if image_url:
            from ..security import is_url_safe
            safe, reason = is_url_safe(image_url)
            if not safe:
                raise ValueError(f"URL blocked: {reason}")
            
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_url)
                if resp.status_code != 200:
                    raise ValueError(f"HTTP {resp.status_code} fetching image")
                
                content_type = resp.headers.get("content-type", "image/jpeg")
                if "png" in content_type or image_url.endswith(".png"):
                    mime_type = "image/png"
                elif "gif" in content_type or image_url.endswith(".gif"):
                    mime_type = "image/gif"
                elif "webp" in content_type or image_url.endswith(".webp"):
                    mime_type = "image/webp"
                else:
                    mime_type = "image/jpeg"
                
                return base64.b64encode(resp.content).decode(), mime_type
        
        return image_base64, default_mime
    
    async def _try_gemini(self, image_b64: str, mime_type: str, prompt: str) -> dict:
        """Try Google Gemini via OAuth."""
        try:
            from ..auth.google_oauth import get_credentials
            creds = await get_credentials()
            if not creds:
                return {"success": False, "error": "No Google OAuth credentials"}
            access_token = await creds.get_token()
        except Exception as e:
            return {"success": False, "error": f"Google auth failed: {str(e)}"}
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inlineData": {"mimeType": mime_type, "data": image_b64}},
                            ]
                        }],
                        "generationConfig": {
                            "temperature": 0.4,
                            "maxOutputTokens": 1024,
                        },
                    },
                )
                
                if response.status_code != 200:
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
                
                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"success": True, "result": text}
                
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except (KeyError, IndexError):
            return {"success": False, "error": "Unexpected response format"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _try_together(
        self, image_b64: str, mime_type: str, prompt: str, api_key: str
    ) -> dict:
        """Try Together AI vision model."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "meta-llama/Llama-Vision-Free",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_b64}"
                                    },
                                },
                            ],
                        }],
                        "max_tokens": 1024,
                        "temperature": 0.4,
                    },
                )
                
                if response.status_code != 200:
                    return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
                
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                return {"success": True, "result": text}
                
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except (KeyError, IndexError):
            return {"success": False, "error": "Unexpected response format"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_together_key_from_db(self) -> str:
        """Try to load Together AI API key from ability config in DB."""
        try:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT config FROM abilities WHERE name = $1", "image_analysis"
                )
                if row and row["config"]:
                    import json
                    config = json.loads(row["config"]) if isinstance(row["config"], str) else row["config"]
                    return config.get("api_key", "")
        except Exception as e:
            logger.debug(f"Could not load Together key from DB: {e}")
        return ""
    
    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "image_analysis",
                "description": "Analyze an image and describe its contents using AI vision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to analyze",
                        },
                        "image_base64": {
                            "type": "string",
                            "description": "Base64-encoded image data (alternative to URL)",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Specific question or instruction about the image",
                            "default": "Describe this image in detail.",
                        },
                    },
                    "required": [],
                },
            },
        }
    
    def get_required_config(self) -> list[str]:
        """No hard requirements — uses Google OAuth or Together AI key."""
        return []
