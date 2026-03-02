"""Image Analysis Ability — analyze images using configured vision provider.

All configuration comes from the ability's config in DB.
No hardcoded providers or models. Owner decides everything.

Example configs:
  Together AI: {"provider": "together", "api_key": "...", "model": "Qwen/Qwen2.5-VL-72B-Instruct"}
  Google:      {"provider": "google"}  (uses OAuth, model defaults to gemini-2.0-flash)
  OpenAI:      {"provider": "openai", "api_key": "...", "model": "gpt-4o"}
"""

import base64
import logging
import httpx
from typing import Optional

from .base import Ability

logger = logging.getLogger("syne.abilities.image_analysis")


class ImageAnalysisAbility(Ability):
    """Analyze images using whatever vision provider the owner configured."""

    name = "image_analysis"
    description = "Analyze and describe images using AI vision"
    version = "2.0"

    def handles_input_type(self, input_type: str) -> bool:
        return input_type == "image"

    async def pre_process(
        self, input_type: str, input_data: dict, user_prompt: str,
        config: Optional[dict] = None
    ) -> Optional[str]:
        prompt = user_prompt if user_prompt and user_prompt.lower() not in (
            "what's in this image?", "describe this image"
        ) else "Describe this image in detail. If there is text, transcribe it."

        result = await self.execute(
            params={
                "image_base64": input_data.get("base64", ""),
                "mime_type": input_data.get("mime_type", "image/jpeg"),
                "prompt": prompt,
            },
            context={"config": config or {}},
        )

        if result.get("success"):
            return result["result"]
        logger.warning(f"Image analysis failed: {result.get('error', 'unknown')}")
        return None

    async def execute(self, params: dict, context: dict) -> dict:
        image_url = params.get("image_url", "")
        image_base64 = params.get("image_base64", "")
        prompt = params.get("prompt", "Describe this image in detail.")

        if not image_url and not image_base64:
            return {"success": False, "error": "Either image_url or image_base64 is required"}

        # Resolve image from URL if needed
        try:
            image_base64, mime_type = await self._resolve_image(
                image_url, image_base64, params.get("mime_type", "image/jpeg")
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

        # Read config — owner decides provider, model, key
        config = context.get("config", {})
        if not config:
            config = await self._load_config_from_db()

        provider = config.get("provider", "").lower()
        api_key = config.get("api_key", "")
        model = config.get("model", "")

        # Dispatch to provider
        if provider == "together" and api_key:
            return await self._call_openai_compatible(
                image_base64, mime_type, prompt,
                api_key=api_key,
                model=model or "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                base_url="https://api.together.xyz/v1",
            )

        if provider == "openai" and api_key:
            return await self._call_openai_compatible(
                image_base64, mime_type, prompt,
                api_key=api_key,
                model=model or "gpt-4o",
                base_url="https://api.openai.com/v1",
            )

        if provider == "google" or (not provider and not api_key):
            # Google Gemini via OAuth — no API key needed
            return await self._call_gemini(
                image_base64, mime_type, prompt,
                model=model or "gemini-2.0-flash",
            )

        return {"success": False, "error": f"Unknown provider '{provider}' or missing api_key"}

    async def _resolve_image(self, url: str, b64: str, default_mime: str) -> tuple[str, str]:
        if not url:
            return b64, default_mime

        from ..security import is_url_safe
        safe, reason = is_url_safe(url)
        if not safe:
            raise ValueError(f"URL blocked: {reason}")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            ct = resp.headers.get("content-type", "image/jpeg")
            if "png" in ct:
                mime = "image/png"
            elif "gif" in ct:
                mime = "image/gif"
            elif "webp" in ct:
                mime = "image/webp"
            else:
                mime = "image/jpeg"

            return base64.b64encode(resp.content).decode(), mime

    async def _call_openai_compatible(
        self, b64: str, mime: str, prompt: str,
        api_key: str, model: str, base_url: str,
    ) -> dict:
        """Call any OpenAI-compatible vision API (Together, OpenAI, etc.)."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:{mime};base64,{b64}"
                                }},
                            ],
                        }],
                        "max_tokens": 1024,
                        "temperature": 0.4,
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

                text = resp.json()["choices"][0]["message"]["content"]
                return {"success": True, "result": text}

        except httpx.TimeoutException:
            return {"success": False, "error": "Timeout"}
        except (KeyError, IndexError):
            return {"success": False, "error": "Unexpected response format"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _call_gemini(self, b64: str, mime: str, prompt: str, model: str) -> dict:
        """Call Google Gemini via OAuth."""
        try:
            from ..auth.google_oauth import get_credentials
            creds = await get_credentials()
            if not creds:
                return {"success": False, "error": "No Google OAuth credentials"}
            token = await creds.get_token()
        except Exception as e:
            return {"success": False, "error": f"Google auth: {str(e)}"}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "contents": [{"parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": mime, "data": b64}},
                        ]}],
                        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1024},
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return {"success": True, "result": text}

        except httpx.TimeoutException:
            return {"success": False, "error": "Timeout"}
        except (KeyError, IndexError):
            return {"success": False, "error": "Unexpected response format"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _load_config_from_db(self) -> dict:
        try:
            from ..db.connection import get_connection
            import json
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT config FROM abilities WHERE name = $1", "image_analysis"
                )
                if row and row["config"]:
                    return json.loads(row["config"]) if isinstance(row["config"], str) else row["config"]
        except Exception as e:
            logger.debug(f"Could not load config from DB: {e}")
        return {}

    def get_schema(self) -> dict:
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
                            "description": "Base64-encoded image data",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Question or instruction about the image",
                            "default": "Describe this image in detail.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def get_guide(self, enabled: bool, config: dict) -> str:
        provider = config.get("provider", "")
        has_key = bool(config.get("api_key"))
        ready = enabled and (provider == "google" or has_key)
        if ready:
            label = provider or "google"
            return (
                f"- Status: **ready** (provider: {label})\n"
                "- Automatically analyzes images sent by users (ability-first)\n"
                "- Also callable: `image_analysis(image_url='...', prompt='...')`"
            )
        return (
            "- Status: **not ready**\n"
            "- Providers: google (OAuth, no key needed), together, openai (need api_key)\n"
            "- Setup: `update_ability(action='config', name='image_analysis', "
            "config='{\"provider\": \"google\"}')`"
        )

    def get_required_config(self) -> list[str]:
        return []
