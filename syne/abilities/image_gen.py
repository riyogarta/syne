"""Image Generation Ability — generate images with a model from /models.

Config: {"model_key": "<key of a /models entry>"}  (set via /imagegen)
The entry decides the provider:
  driver vertex                         → Vertex AI (Gemini image / Imagen)
  driver openai_compat, api.openai.com  → OpenAI Image API
  driver openai_compat, together        → Together AI images
API key comes from the entry's credential_key.
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
    version = "2.1"
    # 0o770 — owner + family rwx. Digits contain x → consent gate fires
    # (calls external API + spends money + writes new image file).
    permission = 0o770

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

        model_key = config.get("model_key", "")
        if not model_key:
            return {"success": False, "error": "Image generation model not set. Choose one in /imagegen."}

        try:
            entry, api_key = await self._resolve_model_entry(model_key)
        except Exception as e:
            return {"success": False, "error": f"Could not read /models registry: {e}"}
        if not entry:
            return {"success": False, "error": f"Image generation model '{model_key}' not found in /models. Choose one in /imagegen."}

        driver = entry.get("driver", "")
        model = entry.get("model_id", "")
        base_url = (entry.get("base_url") or "").lower()

        if driver == "vertex":
            image_bytes = await self._gen_vertex(
                prompt, model=model, api_key=api_key,
                region=entry.get("region", ""),
                project_id=entry.get("project_id", ""),
            )
        elif driver == "openai_compat" and "api.openai.com" in base_url:
            if not api_key:
                return {"success": False, "error": "OpenAI API key not configured for this model in /models"}
            image_bytes = await self._gen_openai(prompt, api_key=api_key, model=model)
        elif driver == "openai_compat" and "together" in base_url:
            if not api_key:
                return {"success": False, "error": "Together AI API key not configured for this model in /models"}
            image_bytes = await self._gen_together(prompt, api_key=api_key, model=model)
        else:
            return {"success": False, "error": f"Model '{entry.get('label', model_key)}' (driver {driver}) cannot generate images. Choose another in /imagegen."}

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

    async def _gen_vertex(self, prompt: str, model: str, api_key: str = "", region: str = "", project_id: str = "") -> bytes | dict:
        """Generate image via Vertex AI.

        Two different APIs live behind the same provider:
          - Imagen models  → `:predict`         (instances / bytesBase64Encoded)
          - Gemini models  → `:generateContent` (contents / inlineData)
        They are not interchangeable, so dispatch on the model family.
        """
        if not api_key:
            return {"success": False, "error": "No Vertex API key for this model. Check its credential in /models."}
        if not region:
            return {"success": False, "error": "No Vertex region for this model in /models."}

        # Chat models in the registry may use the global endpoint (needs the
        # project ID), so build URLs the same way the Vertex driver does.
        from ..llm.vertex import _vertex_endpoint
        try:
            base = _vertex_endpoint(region, model, project_id)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        if model.startswith("gemini-"):
            return await self._gen_vertex_gemini(prompt, base, api_key)

        url = f"{base}:predict"

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

    async def _gen_vertex_gemini(self, prompt: str, base: str, api_key: str) -> bytes | dict:
        """Generate image via Vertex AI Gemini image models (`:generateContent`).

        Response shape:
          candidates[0].content.parts[*].inlineData.data  → base64 image
        Text parts may be interleaved, so scan parts for the first inlineData.
        """
        url = f"{base}:generateContent"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"responseModalities": ["IMAGE"]},
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"Vertex HTTP {resp.status_code}: {resp.text[:200]}"}

                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    # Prompt blocked by safety filters returns no candidates.
                    feedback = data.get("promptFeedback", {})
                    reason = feedback.get("blockReason", "no candidates returned")
                    return {"success": False, "error": f"Vertex returned no image ({reason})"}

                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    b64 = part.get("inlineData", {}).get("data", "")
                    if b64:
                        return base64.b64decode(b64)

                finish = candidates[0].get("finishReason", "")
                return {
                    "success": False,
                    "error": f"No image data in Vertex response (finishReason: {finish or 'unknown'})",
                }

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
                    from ..security import is_url_safe_async
                    safe, reason = await is_url_safe_async(image_data["url"])
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
        """Generate image via the OpenAI Image API (gpt-image models)."""
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
                        # gpt-image models always return b64_json; only the
                        # retired DALL-E models took response_format.
                        **({"response_format": "b64_json"} if model.startswith("dall-e") else {}),
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

    async def _resolve_model_entry(self, model_key: str) -> tuple[dict | None, str]:
        """Look up model_key in the /models registry → (entry, api_key)."""
        from ..db.connection import get_connection
        import json as _json

        async with get_connection() as conn:
            row = await conn.fetchrow("SELECT value FROM config WHERE key = 'provider.models'")
            if not row:
                return None, ""
            models = _json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
            entry = next((m for m in (models or []) if m.get("key") == model_key), None)
            if not entry:
                return None, ""

            api_key = ""
            credential_key = entry.get("credential_key", "")
            if credential_key:
                key_row = await conn.fetchrow(
                    "SELECT value FROM config WHERE key = $1", credential_key
                )
                if key_row:
                    val = key_row["value"]
                    api_key = _json.loads(val) if isinstance(val, str) and val.startswith('"') else val
            return entry, api_key or ""

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
        model_key = config.get("model_key", "")
        if enabled and model_key:
            return (
                f"- Status: **ready** (model: {model_key})\n"
                "- Use: `image_gen(prompt='...')` to generate images"
            )
        return (
            "- Status: **not ready**\n"
            "- Setup: owner picks an image model (from /models) via /imagegen"
        )

    def get_required_config(self) -> list[str]:
        return []
