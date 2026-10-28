"""Image Analysis Ability — analyze images using configured vision provider.

All configuration comes from the ability's config in DB.
No hardcoded providers or models. Owner decides everything.

Example configs:
  Ollama:      {"provider": "ollama", "model": "gemma3:4b"}
  Together AI: {"provider": "together", "api_key": "...", "model": "Qwen/Qwen3.5-9B"}
  OpenAI:      {"provider": "openai", "api_key": "...", "model": "gpt-6-luna"}
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
    # 0o444 — everyone may invoke; r-only (image → text), no side effect.
    # Was 0o555 which had x bit set — spurious under the new gate rule.
    permission = 0o444

    def handles_input_type(self, input_type: str) -> bool:
        return input_type in ("image", "images")

    async def pre_process(
        self, input_type: str, input_data, user_prompt: str,
        config: Optional[dict] = None
    ) -> Optional[str]:
        base_instruction = (
            "Describe this image in detail. If there is text, transcribe it."
        )
        generic = ("what's in this image?", "describe this image")
        if user_prompt and user_prompt.lower() not in generic:
            # The caption ADDS context, it must never REPLACE the describe
            # instruction. A caption that is a statement rather than a
            # question (e.g. "left is me, middle is Agha") used to become
            # the entire prompt, so the model just echoed it back and
            # returned no description of the image at all.
            prompt = (
                f"{base_instruction}\n\n"
                f"User's note about this image: {user_prompt}"
            )
        else:
            prompt = base_instruction

        # Album path: input_type 'images' carries a LIST of image dicts.
        # Analyze each one and concatenate, so a multi-photo message doesn't
        # silently drop every photo but the first.
        if input_type == "images" or isinstance(input_data, list):
            images = input_data if isinstance(input_data, list) else [input_data]
            parts: list[str] = []
            total = len(images)
            for idx, img in enumerate(images, start=1):
                b64 = img.get("base64", "") if isinstance(img, dict) else ""
                if not b64:
                    parts.append(f"[Foto {idx}/{total}: gagal dibaca (data kosong)]")
                    continue
                result = await self.execute(
                    params={
                        "image_base64": b64,
                        "mime_type": (img.get("mime_type", "image/jpeg")
                                      if isinstance(img, dict) else "image/jpeg"),
                        "prompt": prompt,
                    },
                    context={"config": config or {}},
                )
                if result.get("success"):
                    parts.append(f"=== Foto {idx}/{total} ===\n{result['result']}")
                else:
                    err = result.get("error", "unknown")
                    logger.warning(f"Image analysis failed (photo {idx}/{total}): {err}")
                    parts.append(f"=== Foto {idx}/{total} ===\n[Analisis gagal: {err}]")
            # Only return text if at least one photo produced a real analysis.
            if any("[Analisis gagal" not in p and "gagal dibaca" not in p for p in parts):
                return "\n\n".join(parts)
            return None

        # Single-image path (unchanged).
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
        if provider == "vertex":
            return await self._call_vertex(
                image_base64, mime_type, prompt,
                model=model or "gemini-3.8-flash",
                api_key=api_key,
                region=config.get("region", ""),
                project_id=config.get("project_id", ""),
            )

        if provider == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")
            return await self._call_ollama(
                image_base64, mime_type, prompt,
                model=model or "gemma3:4b",
                base_url=base_url,
            )

        if provider == "together" and api_key:
            return await self._call_openai_compatible(
                image_base64, mime_type, prompt,
                api_key=api_key,
                model=model or "Qwen/Qwen3.5-9B",
                base_url="https://api.together.xyz/v1",
            )

        if provider == "openai" and api_key:
            return await self._call_openai_compatible(
                image_base64, mime_type, prompt,
                api_key=api_key,
                model=model or "gpt-6-luna",
                base_url="https://api.openai.com/v1",
            )

        if provider == "google":
            # Google has closed all OAuth access paths for Gemini.
            return {"success": False, "error": "Gemini OAuth is no longer available from Google. Choose another vision provider via /vision."}

        if not provider:
            return {"success": False, "error": "No vision provider configured. Choose one via /vision."}

        return {"success": False, "error": f"Unknown provider '{provider}' or missing api_key"}

    async def _resolve_image(self, url: str, b64: str, default_mime: str) -> tuple[str, str]:
        if not url:
            return b64, default_mime

        from ..security import is_url_safe_async
        safe, reason = await is_url_safe_async(url)
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

    async def _call_vertex(
        self, b64: str, mime: str, prompt: str,
        model: str, api_key: str = "", region: str = "", project_id: str = "",
    ) -> dict:
        """Call Vertex AI Gemini vision API.

        Auto-reads API key, region and project ID from the model registry if
        not provided (the global endpoint needs the project ID).
        """
        # Auto-load API key + region from existing Vertex model config
        if not api_key or not region:
            try:
                _key, _region, _project = await self._load_vertex_credentials()
                api_key = api_key or _key
                if not region:
                    region, project_id = _region, project_id or _project
            except Exception as e:
                return {"success": False, "error": f"Vertex credentials: {e}"}

        if not api_key:
            return {"success": False, "error": "No Vertex API key. Set in ability config or add a Vertex model in /models."}
        if not region:
            return {"success": False, "error": "No Vertex region. Set in ability config or add a Vertex model in /models."}

        from ..llm.vertex import _vertex_endpoint
        try:
            url = f"{_vertex_endpoint(region, model, project_id)}:generateContent"
        except ValueError as e:
            return {"success": False, "error": str(e)}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"role": "user", "parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": mime, "data": b64}},
                        ]}],
                        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 4096},
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"Vertex HTTP {resp.status_code}: {resp.text[:200]}"}

                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return {"success": True, "result": text}

        except httpx.TimeoutException:
            return {"success": False, "error": "Vertex timeout"}
        except (KeyError, IndexError):
            return {"success": False, "error": "Unexpected Vertex response format"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _load_vertex_credentials(self) -> tuple[str, str, str]:
        """Auto-load Vertex API key, region and project ID from model registry."""
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

                    # Load API key
                    api_key = ""
                    if credential_key:
                        key_row = await conn.fetchrow(
                            "SELECT value FROM config WHERE key = $1", credential_key
                        )
                        if key_row:
                            val = key_row["value"]
                            api_key = _json.loads(val) if isinstance(val, str) and val.startswith('"') else val

                    return api_key, region, m.get("project_id", "")

        raise ValueError("No Vertex model found in registry")

    async def _call_ollama(
        self, b64: str, mime: str, prompt: str,
        model: str, base_url: str,
    ) -> dict:
        """Call Ollama vision API (images field with base64 array)."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{base_url.rstrip('/')}/api/chat",
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": prompt,
                            "images": [b64],
                        }],
                        "stream": False,
                        "options": {"num_predict": 4096},
                    },
                )

                if resp.status_code != 200:
                    return {"success": False, "error": f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"}

                text = resp.json().get("message", {}).get("content", "")
                if not text:
                    return {"success": False, "error": "Ollama returned empty response"}
                return {"success": True, "result": text}

        except httpx.TimeoutException:
            return {"success": False, "error": "Ollama timeout (120s) — model may be loading"}
        except httpx.ConnectError:
            return {"success": False, "error": f"Cannot connect to Ollama at {base_url}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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
                        "max_tokens": 4096,
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
        # vertex reads its key from /models, ollama needs none; the others
        # need an api_key in the ability config.
        ready = enabled and (
            provider in ("vertex", "ollama")
            or (provider in ("together", "openai") and has_key)
        )
        if ready:
            return (
                f"- Status: **ready** (provider: {provider})\n"
                "- Automatically analyzes images sent by users (ability-first)\n"
                "- Also callable: `image_analysis(image_url='...', prompt='...')`"
            )
        return (
            "- Status: **not ready**\n"
            "- Providers: vertex (auto-reads key from /models), ollama (local), together, openai\n"
            "- Setup: `update_ability(action='config', name='image_analysis', "
            "config='{\"provider\": \"vertex\"}')`"
        )

    def get_required_config(self) -> list[str]:
        return []
