"""Image Generation Ability — generates images via Together AI FLUX.1-schnell."""

import os
import time
import httpx

from .base import Ability


class ImageGenAbility(Ability):
    """Generate images from text prompts using Together AI's FLUX.1-schnell model."""
    
    name = "image_gen"
    description = "Generate images from text descriptions using AI"
    version = "1.0"
    
    async def execute(self, params: dict, context: dict) -> dict:
        """Generate an image from the given prompt.
        
        Args:
            params: Must contain 'prompt' key with image description
            context: Must contain 'config' with 'TOGETHER_API_KEY'
            
        Returns:
            dict with success, result, error, media keys
        """
        prompt = params.get("prompt", "")
        if not prompt:
            return {"success": False, "error": "Prompt is required"}
        
        # Get API key from config (accept multiple key names)
        config = context.get("config", {})
        api_key = (
            config.get("TOGETHER_API_KEY")
            or config.get("together_api_key")
            or config.get("api_key")
            or os.environ.get("TOGETHER_API_KEY")
        )
        
        if not api_key:
            return {"success": False, "error": "Together AI API key not configured"}
        
        # Optional style parameter
        style = params.get("style", "")
        if style:
            prompt = f"{prompt}, {style} style"
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.together.xyz/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "black-forest-labs/FLUX.1-schnell",
                        "prompt": prompt,
                        "n": 1,
                        "width": 1024,
                        "height": 1024,
                    },
                )
                
                if response.status_code != 200:
                    error_text = response.text[:200]  # Truncate long errors
                    return {"success": False, "error": f"API error ({response.status_code}): {error_text}"}
                
                data = response.json()
                
                # Together AI returns b64_json or url depending on response_format
                image_data = data.get("data", [{}])[0]
                
                if "b64_json" in image_data:
                    # Decode base64 image
                    import base64
                    image_bytes = base64.b64decode(image_data["b64_json"])
                elif "url" in image_data:
                    # Download from URL (SSRF check even for API responses)
                    from ..security import is_url_safe
                    safe, reason = is_url_safe(image_data["url"])
                    if not safe:
                        return {"success": False, "error": f"Image URL blocked: {reason}"}
                    img_response = await client.get(image_data["url"])
                    if img_response.status_code != 200:
                        return {"success": False, "error": "Failed to download generated image"}
                    image_bytes = img_response.content
                else:
                    return {"success": False, "error": "No image data in response"}
                
                # Save to workspace/outputs/
                session_id = context.get("session_id", int(time.time()))
                outdir = self.get_output_dir()
                filepath = os.path.join(outdir, f"syne_image_{session_id}_{int(time.time())}.png")
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                return {
                    "success": True,
                    "result": f"Generated image for: {prompt}",
                    "media": filepath,
                }
                
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out (60s)"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_schema(self) -> dict:
        """Return OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "image_gen",
                "description": "Generate an image from a text description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image to generate",
                        },
                        "style": {
                            "type": "string",
                            "enum": ["realistic", "artistic", "anime", "sketch", "photographic"],
                            "description": "Style of the generated image (optional)",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        }
    
    def get_required_config(self) -> list[str]:
        """API key is required but can come from environment."""
        # Not strictly required from config since we fallback to env
        return []
