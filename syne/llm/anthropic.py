"""Anthropic Claude provider — OAuth via claude.ai subscription."""

import json
import logging
import os
import time
from typing import Optional

import httpx

from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.anthropic")

# Anthropic Messages API
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

# Claude CLI credentials path
CLAUDE_CREDENTIALS_PATH = os.path.expanduser("~/.claude/.credentials.json")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using OAuth tokens from claude.ai subscription.
    
    Reads OAuth tokens from ~/.claude/.credentials.json (shared with Claude CLI).
    Auto-refreshes when token is expired or near expiry.
    """

    def __init__(
        self,
        chat_model: str = "claude-sonnet-4-20250514",
        credentials_path: str = CLAUDE_CREDENTIALS_PATH,
    ):
        self.chat_model = chat_model
        self.credentials_path = credentials_path
        self._access_token: Optional[str] = None
        self._expires_at: float = 0
        self._last_read: float = 0

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_vision(self) -> bool:
        return True

    def _load_token(self) -> str:
        """Load OAuth token from credentials file. Re-reads if stale."""
        now = time.time()
        
        # Re-read file at most every 30 seconds
        if self._access_token and (now - self._last_read) < 30 and now < self._expires_at:
            return self._access_token
        
        try:
            with open(self.credentials_path) as f:
                creds = json.load(f)
            
            oauth = creds.get("claudeAiOauth", {})
            self._access_token = oauth.get("accessToken", "")
            expires_at_ms = oauth.get("expiresAt", 0)
            self._expires_at = expires_at_ms / 1000 if expires_at_ms > 1e12 else expires_at_ms
            self._last_read = now
            
            if not self._access_token:
                raise RuntimeError("No accessToken in Claude credentials")
            
            # Check if expired
            if now >= self._expires_at:
                logger.warning(
                    f"Claude OAuth token expired at {time.strftime('%Y-%m-%d %H:%M', time.localtime(self._expires_at))}. "
                    "Run 'claude' CLI to refresh, or OpenClaw will refresh automatically."
                )
                # Still return the token — Anthropic may accept it briefly,
                # or the caller needs to handle the 401
            
            return self._access_token
            
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude credentials not found: {self.credentials_path}. "
                "Run 'claude' CLI to authenticate first."
            )
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in {self.credentials_path}")

    def _headers(self) -> dict:
        token = self._load_token()
        return {
            "Authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
    ) -> ChatResponse:
        model = model or self.chat_model
        
        # Separate system message from conversation messages
        system_text = None
        conversation = []
        # Collect tool results to merge into user messages
        pending_tool_results = []
        
        for m in messages:
            if m.role == "system":
                system_text = m.content
            elif m.role == "tool":
                # Anthropic expects tool results inside a "user" role message
                tool_call_id = (m.metadata or {}).get("tool_call_id", "unknown")
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": m.content or "",
                })
            elif m.role == "assistant":
                # Flush any pending tool results before assistant message
                if pending_tool_results:
                    conversation.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                
                # Check if assistant had tool_calls — need to format as content blocks
                tool_calls_meta = (m.metadata or {}).get("tool_calls", [])
                if tool_calls_meta:
                    content_blocks = []
                    if m.content:
                        content_blocks.append({"type": "text", "text": m.content})
                    for tc in tool_calls_meta:
                        if "function" in tc:
                            func = tc["function"]
                            input_data = func.get("arguments", "{}")
                            if isinstance(input_data, str):
                                input_data = json.loads(input_data)
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", "unknown"),
                                "name": func.get("name", ""),
                                "input": input_data,
                            })
                    conversation.append({"role": "assistant", "content": content_blocks})
                else:
                    conversation.append({"role": "assistant", "content": m.content or ""})
            else:
                # Flush pending tool results before user message
                if pending_tool_results:
                    conversation.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                
                # Build content — handle vision (base64 images via metadata)
                img_meta = (m.metadata or {}).get("image")
                if img_meta:
                    content_parts = []
                    media_type = img_meta.get("mime_type", "image/jpeg")
                    img_data = img_meta.get("base64", "")
                    if img_data:
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_data,
                            }
                        })
                    if m.content:
                        content_parts.append({"type": "text", "text": m.content})
                    conversation.append({"role": m.role, "content": content_parts})
                else:
                    conversation.append({"role": m.role, "content": m.content})
        
        # Flush remaining tool results
        if pending_tool_results:
            conversation.append({"role": "user", "content": pending_tool_results})
        
        # Build request body
        body: dict = {
            "model": model,
            "messages": conversation,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_text:
            body["system"] = system_text
        
        # Thinking support
        if thinking_budget and thinking_budget > 0:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # When thinking is enabled, temperature must be 1
            body["temperature"] = 1
        else:
            body["temperature"] = temperature
        
        # Tool use
        if tools:
            body["tools"] = self._convert_tools(tools)
        
        # Make request
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(3):
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    json=body,
                    headers=self._headers(),
                )
                
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/3)")
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                
                if resp.status_code == 401:
                    logger.error("Claude OAuth token rejected (401). Token may be expired.")
                    raise RuntimeError(
                        "Claude OAuth token expired or invalid. "
                        "Run 'claude' CLI to refresh authentication."
                    )
                
                resp.raise_for_status()
                break
            else:
                raise RuntimeError("Rate limited after 3 retries")
        
        data = resp.json()
        
        # Parse response
        content_text = ""
        thinking_text = ""
        tool_calls = []
        
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "thinking":
                thinking_text += block.get("thinking", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    }
                })
        
        usage = data.get("usage", {})
        
        return ChatResponse(
            content=content_text,
            model=data.get("model", model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            tool_calls=tool_calls if tool_calls else None,
            thinking=thinking_text if thinking_text else None,
        )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Anthropic does not support embeddings."""
        raise NotImplementedError("Anthropic does not provide an embedding API")

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
        """Anthropic does not support embeddings."""
        raise NotImplementedError("Anthropic does not provide an embedding API")

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)
        return anthropic_tools
