"""Image Generation Ability — generate images via multiple providers.

Supported providers:
  Vertex AI: {"provider": "vertex", "model": "imagen-3.0-generate-002"}
  Together AI: {"provider": "together", "api_key": "...", "model": "black-forest-labs/FLUX.1-schnell"}
  OpenAI: {"provider": "openai", "api_key": "...", "model": "dall-e-3"}
"""

import base64
import logging
import os
import time
import httpx

from .base import Ability

logger = logging.getLogger("syne.abilities.image_gen")


class ImageGenAbility(Ability):
    """Generate images from text prompts using configurable providers."""

    name = "image_gen"
    description = "Generate images from text descriptions using AI"
    version = "2.0"
    permission = 0o777

    async def execute(self, params: dict, context: dict) -> dict:
        prompt = params.get("prompt", "")
        if not prompt:
            return {"success": False, "error": "Prompt is required"}

        style = params.get("style", "")
        if style:
            prompt = f"{prompt}, {style} style"

        config = context.get("config", {})
        if not config:
            config = await self._load_config_from_db()

        provider = config.get("provider", "").lower()
        api_key = config.get("api_key", "")
        model = config.get("model", "")

        if provider == "vertex":
            image_bytes = await self._gen_vertex(
                prompt, model=model or "imagen-3.0-generate-002",
                api_key=api_key, region=config.get("region", ""),
            )
        elif provider == "together":
            api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
            if not api_key:
                return {"success": False, "error": "Together AI API key not configured"}
            image_bytes = await self._gen_together(
                prompt, api_key=api_key,
                model=model or "black-forest-labs/FLUX.1-schnell",
            )
        elif provider == "openai":
            if not api_key:
                return {"success": False, "error": "OpenAI API key not configured"}
            image_bytes = await self._gen_openai(
                prompt, api_key=api_key, model=model or "dall-e-3",
            )
        else:
            return {"success": False, "error": f"Unknown provider '{provider}'. Use /imagegen to configure."}

        if isinstance(image_bytes, dict):
            return image_bytes  # error dict

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

    async def _gen_vertex(self, prompt: str, model: str, api_key: str = "", region: str = "") -> bytes | dict:
        """Generate image via Vertex AI Imagen."""
        if not api_key or not region:
            try:
                _key, _region = await self._load_vertex_credentials()
                api_key = api_key or _key
                region = region or _region
            except Exception as e:
                return {"success": False, "error": f"Vertex credentials: {e}"}

        if not api_key:
            return {"success": False, "error": "No Vertex API key. Set via /imagegen or add Vertex model in /models."}

        url = f"https://{region}-aiplatform.googleapis.com/v1/publishers/google/models/{model}:predict"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "instances": [{"prompt": prompt}],
                        "parameters": {"sampleCount": 1, "aspectRatio": "1:1"},
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"Vertex HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                predictions = data.get("predictions", [])
                if not predictions:
                    return {"success": False, "error": "Vertex returned no predictions"}

                b64 = predictions[0].get("bytesBase64Encoded", "")
                if not b64:
                    return {"success": False, "error": "No image data in Vertex response"}

                return base64.b64decode(b64)

        except httpx.TimeoutException:
            return {"success": False, "error": "Vertex timeout (60s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _gen_together(self, prompt: str, api_key: str, model: str) -> bytes | dict:
        """Generate image via Together AI."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.together.xyz/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "n": 1,
                        "width": 1024,
                        "height": 1024,
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"Together HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                image_data = data.get("data", [{}])[0]

                if "b64_json" in image_data:
                    return base64.b64decode(image_data["b64_json"])
                elif "url" in image_data:
                    from ..security import is_url_safe
                    safe, reason = is_url_safe(image_data["url"])
                    if not safe:
                        return {"success": False, "error": f"Image URL blocked: {reason}"}
                    img_resp = await client.get(image_data["url"])
                    if img_resp.status_code != 200:
                        return {"success": False, "error": "Failed to download generated image"}
                    return img_resp.content
                else:
                    return {"success": False, "error": "No image data in response"}

        except httpx.TimeoutException:
            return {"success": False, "error": "Together timeout (60s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _gen_openai(self, prompt: str, api_key: str, model: str) -> bytes | dict:
        """Generate image via OpenAI DALL-E."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "prompt": prompt,
                        "n": 1,
                        "size": "1024x1024",
                        "response_format": "b64_json",
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                b64 = data.get("data", [{}])[0].get("b64_json", "")
                if not b64:
                    return {"success": False, "error": "No image data in OpenAI response"}

                return base64.b64decode(b64)

        except httpx.TimeoutException:
            return {"success": False, "error": "OpenAI timeout (60s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _load_vertex_credentials(self) -> tuple[str, str]:
        """Auto-load Vertex API key and region from model registry."""
        from ..db.connection import get_connection
        import json as _json

        async with get_connection() as conn:
            row = await conn.fetchrow("SELECT value FROM config WHERE key = 'provider.models'")
            if not row:
                raise ValueError("No model registry found")

            models = _json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
            for m in models:
                if m.get("driver") == "vertex":
                    credential_key = m.get("credential_key", "")
                    region = m.get("region", "")
                    api_key = ""
                    if credential_key:
                        key_row = await conn.fetchrow(
                            "SELECT value FROM config WHERE key = $1", credential_key
                        )
                        if key_row:
                            val = key_row["value"]
                            api_key = _json.loads(val) if isinstance(val, str) and val.startswith('"') else val
                    return api_key, region

        raise ValueError("No Vertex model found in registry")

    async def _load_config_from_db(self) -> dict:
        try:
            from ..db.connection import get_connection
            import json
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT config FROM abilities WHERE name = $1", "image_gen"
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

    def get_guide(self, enabled: bool, config: dict) -> str:
        provider = config.get("provider", "")
        has_key = bool(config.get("api_key"))
        ready = enabled and (provider == "vertex" or has_key)
        if ready:
            return (
                f"- Status: **ready** (provider: {provider})\n"
                "- Use: `image_gen(prompt='...')` to generate images"
            )
        return (
            "- Status: **not ready**\n"
            "- Providers: vertex (auto-reads key from /models), together, openai (need api_key)\n"
            "- Setup via /imagegen command or: `update_ability(action='config', name='image_gen', "
            "config='{\"provider\": \"vertex\"}')`"
        )

    def get_required_config(self) -> list[str]:
        return []
