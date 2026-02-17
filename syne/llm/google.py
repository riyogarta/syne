"""Google Gemini provider via Cloud Code Assist (OAuth) or Gemini API (key)."""

import asyncio
import json
import logging
import time
import math
import random
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse
from ..auth.google_oauth import GoogleCredentials

logger = logging.getLogger("syne.llm.google")


def _convert_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """Convert OpenAI function calling format to Gemini functionDeclarations format.

    Input (OpenAI format):
        [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Output (Gemini format):
        [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]
    """
    if not tools:
        return []

    declarations = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            decl = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                decl["parameters"] = func["parameters"]
            declarations.append(decl)

    if not declarations:
        return []

    return [{"functionDeclarations": declarations}]

# Cloud Code Assist endpoint (used with OAuth tokens)
_CCA_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_CCA_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}

# Standard Gemini API endpoint (used with API keys)
_GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(LLMProvider):
    """Google Gemini via Cloud Code Assist OAuth (free) or Gemini API key (paid).

    OAuth mode uses the Cloud Code Assist wrapper API.
    API key mode uses the standard Gemini API directly.
    """

    def __init__(
        self,
        credentials: Optional[GoogleCredentials] = None,
        api_key: Optional[str] = None,
        chat_model: str = "gemini-2.5-pro",
        embedding_model: str = "text-embedding-004",
    ):
        self.credentials = credentials
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model

        if not credentials and not api_key:
            raise ValueError("Either credentials (OAuth) or api_key is required")

        self._use_cca = credentials is not None

    @property
    def name(self) -> str:
        return "google"

    @property
    def supports_vision(self) -> bool:
        return True

    def _format_messages(self, messages: list[ChatMessage]) -> tuple[Optional[str], list[dict]]:
        """Convert ChatMessages to Gemini format. Returns (system_text, contents)."""
        system_text = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                # Accumulate system messages
                if system_text is None:
                    system_text = msg.content
                else:
                    system_text += "\n\n" + msg.content

            elif msg.role == "user":
                parts = [{"text": msg.content}]
                # Include image data for vision if present in metadata
                if msg.metadata and "image" in msg.metadata:
                    img = msg.metadata["image"]
                    parts.append({
                        "inlineData": {
                            "mimeType": img.get("mime_type", "image/jpeg"),
                            "data": img.get("base64", ""),
                        }
                    })
                contents.append({"role": "user", "parts": parts})

            elif msg.role == "assistant":
                # Check if assistant message has tool_calls in metadata
                tool_calls = msg.metadata.get("tool_calls") if msg.metadata else None
                if tool_calls:
                    # Assistant message with function calls
                    parts = []
                    if msg.content:
                        parts.append({"text": msg.content})
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {}
                        parts.append({
                            "functionCall": {
                                "name": func.get("name", ""),
                                "args": args,
                            }
                        })
                    contents.append({"role": "model", "parts": parts})
                else:
                    contents.append({"role": "model", "parts": [{"text": msg.content}]})

            elif msg.role == "tool":
                # Tool result — format as functionResponse
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                # Try to parse content as JSON; if not, wrap as result string
                try:
                    result_data = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    result_data = {"result": msg.content}

                fr_part = {
                    "functionResponse": {
                        "name": tool_name,
                        "response": result_data,
                    }
                }

                # Merge consecutive tool results into one "user" message
                # Gemini requires all functionResponse parts from one round
                # to be in a single user message
                if contents and contents[-1].get("role") == "user" and \
                   contents[-1]["parts"] and "functionResponse" in contents[-1]["parts"][0]:
                    contents[-1]["parts"].append(fr_part)
                else:
                    contents.append({
                        "role": "user",
                        "parts": [fr_part],
                    })

        return system_text, contents

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        model = model or self.chat_model
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self._use_cca:
                    return await self._chat_cca(messages, model, temperature, max_tokens, tools)
                else:
                    return await self._chat_api(messages, model, temperature, max_tokens, tools)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 2  # 2s, 4s, 8s
                    logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
                raise

    async def _chat_cca(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Chat via Cloud Code Assist (OAuth, free)."""
        token = await self.credentials.get_token()
        system_text, contents = self._format_messages(messages)

        # Build inner request
        inner = {"contents": contents}
        if system_text:
            inner["systemInstruction"] = {"parts": [{"text": system_text}]}

        gen_config = {"temperature": temperature}
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        inner["generationConfig"] = gen_config

        # Add tools if provided
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools)
            if gemini_tools:
                inner["tools"] = gemini_tools

        # Wrap in CCA envelope
        request_id = f"syne-{int(time.time())}-{random.randint(1000, 9999)}"
        body = {
            "project": self.credentials.project_id,
            "model": model,
            "request": inner,
            "userAgent": "syne-agent",
            "requestId": request_id,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **_CCA_HEADERS,
        }

        # Use non-streaming endpoint for simplicity
        url = f"{_CCA_ENDPOINT}/v1internal:generateContent"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)

            if resp.status_code == 404 or resp.status_code == 405:
                # Fallback to streaming and collect
                logger.debug("Non-streaming endpoint unavailable, using streaming...")
                return await self._chat_cca_streaming(body, headers)

            resp.raise_for_status()
            data = resp.json()

        return self._parse_cca_response(data, model)

    async def _chat_cca_streaming(self, body: dict, headers: dict) -> ChatResponse:
        """Chat via CCA streaming endpoint, collecting full response."""
        url = f"{_CCA_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"
        headers["Accept"] = "text/event-stream"
        model = body["model"]

        text_parts = []
        tool_calls = []
        input_tokens = 0
        output_tokens = 0

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                        # Extract text and function calls from candidates
                        for candidate in chunk.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    text_parts.append(part["text"])
                                elif "functionCall" in part:
                                    fc = part["functionCall"]
                                    tool_calls.append({
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": json.dumps(fc.get("args", {})),
                                        }
                                    })
                        # Extract usage
                        usage = chunk.get("usageMetadata", {})
                        if usage.get("promptTokenCount"):
                            input_tokens = usage["promptTokenCount"]
                        if usage.get("candidatesTokenCount"):
                            output_tokens = usage["candidatesTokenCount"]
                    except json.JSONDecodeError:
                        continue

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _parse_cca_response(self, data: dict, model: str) -> ChatResponse:
        """Parse CCA response (may be wrapped or direct)."""
        # CCA may wrap in a response envelope
        if "response" in data:
            data = data["response"]

        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])

        # Extract text parts
        text_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        usage = data.get("usageMetadata", {})

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            tool_calls=tool_calls if tool_calls else None,
        )

    async def _chat_api(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Chat via standard Gemini API (API key, paid)."""
        system_text, contents = self._format_messages(messages)

        body: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if max_tokens:
            body["generationConfig"]["maxOutputTokens"] = max_tokens

        # Add tools if provided
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools)
            if gemini_tools:
                body["tools"] = gemini_tools

        url = f"{_GEMINI_API}/models/{model}:generateContent"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, params={"key": self.api_key})
            resp.raise_for_status()
            data = resp.json()

        candidate = data["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])

        # Extract text parts and function calls
        text_parts = []
        tool_calls = []

        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    }
                })

        usage = data.get("usageMetadata", {})

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            tool_calls=tool_calls if tool_calls else None,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Embedding — always uses Gemini API (CCA doesn't have embed endpoint)."""
        model = model or self.embedding_model
        url = f"{_GEMINI_API}/models/{model}:embedContent"

        body = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
        }

        # For OAuth, we use the access token as Bearer
        # For API key, we use the key param
        if self._use_cca:
            token = await self.credentials.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            params = {}
        else:
            headers = {}
            params = {"key": self.api_key}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=body, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        vector = data["embedding"]["values"]
        return EmbeddingResponse(vector=vector, model=model, dimensions=len(vector))

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        model = model or self.embedding_model
        url = f"{_GEMINI_API}/models/{model}:batchEmbedContents"

        requests = [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]

        if self._use_cca:
            token = await self.credentials.get_token()
            headers = {"Authorization": f"Bearer {token}"}
            params = {}
        else:
            headers = {}
            params = {"key": self.api_key}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url, json={"requests": requests}, headers=headers, params=params
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            EmbeddingResponse(vector=emb["values"], model=model, dimensions=len(emb["values"]))
            for emb in data["embeddings"]
        ]
