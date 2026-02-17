"""Image Analysis Ability — analyze images via Google Gemini vision."""

import base64
import httpx

from .base import Ability


class ImageAnalysisAbility(Ability):
    """Analyze and describe images using Google Gemini vision model."""
    
    name = "image_analysis"
    description = "Analyze and describe images using AI vision"
    version = "1.0"
    
    async def execute(self, params: dict, context: dict) -> dict:
        """Analyze an image and return a description.
        
        Args:
            params: Must contain either:
                - 'image_url': URL of image to analyze
                - 'image_base64': Base64-encoded image data
            context: Must contain 'config' with Google OAuth credentials
            
        Returns:
            dict with success, result, error keys
        """
        image_url = params.get("image_url", "")
        image_base64 = params.get("image_base64", "")
        prompt = params.get("prompt", "Describe this image in detail.")
        
        if not image_url and not image_base64:
            return {"success": False, "error": "Either image_url or image_base64 is required"}
        
        try:
            # Get image data
            if image_url:
                async with httpx.AsyncClient(timeout=30) as client:
                    img_response = await client.get(image_url)
                    if img_response.status_code != 200:
                        return {"success": False, "error": f"Failed to fetch image from URL"}
                    
                    image_bytes = img_response.content
                    # Detect mime type from content-type header or URL
                    content_type = img_response.headers.get("content-type", "image/jpeg")
                    if "png" in content_type or image_url.endswith(".png"):
                        mime_type = "image/png"
                    elif "gif" in content_type or image_url.endswith(".gif"):
                        mime_type = "image/gif"
                    elif "webp" in content_type or image_url.endswith(".webp"):
                        mime_type = "image/webp"
                    else:
                        mime_type = "image/jpeg"
                    
                    image_base64 = base64.b64encode(image_bytes).decode()
            else:
                # Assume JPEG if not specified
                mime_type = params.get("mime_type", "image/jpeg")
            
            # Get Google OAuth credentials
            # Import here to avoid circular imports
            from ..auth.google_oauth import get_credentials
            
            try:
                creds = await get_credentials()
                access_token = await creds.get_token()
            except Exception as e:
                return {"success": False, "error": f"Failed to get Google credentials: {str(e)}"}
            
            # Call Gemini API with vision
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt},
                                    {
                                        "inlineData": {
                                            "mimeType": mime_type,
                                            "data": image_base64,
                                        }
                                    },
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.4,
                            "maxOutputTokens": 1024,
                        },
                    },
                )
                
                if response.status_code != 200:
                    error_text = response.text[:200]
                    return {"success": False, "error": f"Gemini API error ({response.status_code}): {error_text}"}
                
                data = response.json()
                
                # Extract text from response
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return {"success": False, "error": "Unexpected response format from Gemini"}
                
                return {
                    "success": True,
                    "result": text,
                }
                
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "image_analysis",
                "description": "Analyze an image and describe its contents",
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
                    "required": [],  # One of image_url or image_base64 required
                },
            },
        }
    
    def get_required_config(self) -> list[str]:
        """Google OAuth credentials are loaded automatically."""
        return []
