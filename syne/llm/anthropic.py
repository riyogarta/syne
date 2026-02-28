"""Anthropic Claude provider — OAuth via claude.ai subscription or API key."""

import asyncio
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
_BETA_HEADER = "oauth-2025-04-20"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using OAuth tokens from DB or API key.
    
    Token sources (priority order):
    1. DB (Claude OAuth credentials from `syne init`)
    2. Environment (ANTHROPIC_API_KEY)
    """

    def __init__(
        self,
        chat_model: str = "claude-sonnet-4-20250514",
    ):
        self.chat_model = chat_model
        self._access_token: Optional[str] = None
        self._is_oauth: bool = False
        self._expires_at: float = 0
        self._last_load: float = 0
        self._claude_creds = None  # ClaudeCredentials instance

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def context_window(self) -> int:
        return 200_000  # Claude Sonnet 4 default; Opus 4.6 = 1M (set in model registry)

    @property
    def reserved_output_tokens(self) -> int:
        return self.DEFAULT_MAX_TOKENS + self.DEFAULT_THINKING_BUDGET  # 26624

    async def _load_token(self) -> str:
        """Load Anthropic token from DB or environment.
        
        Priority:
        1. DB — Claude OAuth credentials (from `syne init`)
        2. Environment — ANTHROPIC_API_KEY
        """
        now = time.time()
        
        # Cache: re-check at most every 30 seconds
        if self._access_token and (now - self._last_load) < 30:
            # If OAuth, check expiry and auto-refresh
            if self._is_oauth and self._claude_creds:
                if self._claude_creds.is_expired:
                    token = await self._claude_creds.get_token()
                    self._access_token = token
                    return token
            return self._access_token
        
        # Try DB first
        try:
            from ..auth.claude_oauth import ClaudeCredentials
            creds = await ClaudeCredentials.load_from_db()
            if creds:
                if creds.is_expired:
                    await creds.get_token()
                self._claude_creds = creds
                self._access_token = creds.access_token
                self._is_oauth = True
                self._last_load = now
                return creds.access_token
        except Exception as e:
            logger.debug(f"DB credential load failed: {e}")
        
        # Try DB API key
        try:
            from ..db.models import get_config
            db_api_key = await get_config("credential.anthropic_api_key", None)
            if db_api_key:
                self._access_token = db_api_key
                self._is_oauth = False
                self._last_load = now
                return db_api_key
        except Exception as e:
            logger.debug(f"DB API key load failed: {e}")

        # Try environment
        env_token = os.environ.get("ANTHROPIC_API_KEY")
        if env_token:
            self._access_token = env_token
            self._is_oauth = env_token.startswith("sk-ant-oat")
            self._last_load = now
            return env_token
        
        raise RuntimeError(
            "No Claude/Anthropic credentials found. Options:\n"
            "  1. Run 'syne init' and select Claude (OAuth)\n"
            "  2. Set ANTHROPIC_API_KEY environment variable"
        )

    def _build_headers(self, token: str) -> dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        # OAuth tokens require the beta header
        if token.startswith("sk-ant-oat") or self._is_oauth:
            headers["anthropic-beta"] = _BETA_HEADER
        return headers

    @staticmethod
    def _sanitize_conversation(conversation: list[dict]) -> list[dict]:
        """Sanitize conversation to ensure Anthropic tool_use/tool_result pairing.
        
        Anthropic requires:
        1. Every tool_result must reference a tool_use_id from the immediately preceding assistant message
        2. Every tool_use in an assistant message must have a matching tool_result in the next user message
        
        This removes orphaned tool_results and converts orphaned tool_use assistant
        messages to plain text to prevent 400 errors from context trimming/compaction.
        """
        if not conversation:
            return conversation
        
        sanitized = []
        i = 0
        while i < len(conversation):
            msg = conversation[i]
            
            # Check if this is an assistant message with tool_use blocks
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                tool_use_ids = {
                    block["id"]
                    for block in msg["content"]
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                }
                
                if tool_use_ids:
                    # Look ahead: next message must be user with matching tool_results
                    next_msg = conversation[i + 1] if i + 1 < len(conversation) else None
                    
                    if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
                        result_ids = {
                            block.get("tool_use_id")
                            for block in next_msg["content"]
                            if isinstance(block, dict) and block.get("type") == "tool_result"
                        }
                        
                        if tool_use_ids & result_ids:
                            matched_ids = tool_use_ids & result_ids
                            
                            filtered_assistant = [
                                block for block in msg["content"]
                                if block.get("type") != "tool_use" or block.get("id") in matched_ids
                            ]
                            filtered_results = [
                                block for block in next_msg["content"]
                                if block.get("type") != "tool_result" or block.get("tool_use_id") in matched_ids
                            ]
                            
                            sanitized.append({"role": "assistant", "content": filtered_assistant})
                            if filtered_results:
                                sanitized.append({"role": "user", "content": filtered_results})
                            i += 2
                            continue
                    
                    # No matching tool_results — convert to plain text
                    text_parts = [
                        block.get("text", "")
                        for block in msg["content"]
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    text = " ".join(t for t in text_parts if t) or "[tool calls without results — trimmed]"
                    sanitized.append({"role": "assistant", "content": text})
                    i += 1
                    continue
            
            # Check if this is a user message with tool_results (orphaned)
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_tool_results = any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in msg["content"]
                )
                
                if has_tool_results:
                    prev = sanitized[-1] if sanitized else None
                    if prev and prev.get("role") == "assistant" and isinstance(prev.get("content"), list):
                        prev_tool_ids = {
                            block["id"]
                            for block in prev["content"]
                            if isinstance(block, dict) and block.get("type") == "tool_use"
                        }
                        filtered = [
                            block for block in msg["content"]
                            if block.get("type") != "tool_result" or block.get("tool_use_id") in prev_tool_ids
                        ]
                        if filtered:
                            sanitized.append({"role": "user", "content": filtered})
                    else:
                        logger.debug("Dropping orphaned tool_result message (no preceding tool_use)")
                    i += 1
                    continue
            
            sanitized.append(msg)
            i += 1
        
        # Final pass: merge consecutive same-role messages
        merged = []
        for msg in sanitized:
            if merged and merged[-1].get("role") == msg.get("role"):
                prev_content = merged[-1].get("content", "")
                new_content = msg.get("content", "")
                if isinstance(prev_content, str) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + "\n" + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, list):
                    merged[-1]["content"] = prev_content + new_content
                elif isinstance(prev_content, str) and isinstance(new_content, list):
                    merged[-1]["content"] = [{"type": "text", "text": prev_content}] + new_content
                elif isinstance(prev_content, list) and isinstance(new_content, str):
                    merged[-1]["content"] = prev_content + [{"type": "text", "text": new_content}]
            else:
                merged.append(msg)
        
        return merged

    # Claude-specific defaults — tuned for quality over creativity
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 64000
    DEFAULT_THINKING_BUDGET = 32000  # generous default; None=use this, 0=off, >0=use that

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        thinking_budget: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> ChatResponse:
        model = model or self.chat_model
        
        # Separate system message from conversation messages
        system_text = None
        conversation = []
        pending_tool_results = []
        
        for m in messages:
            if m.role == "system":
                system_text = m.content
            elif m.role == "tool":
                tool_call_id = (m.metadata or {}).get("tool_call_id", "unknown")
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": m.content or "",
                })
            elif m.role == "assistant":
                if pending_tool_results:
                    conversation.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                
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
                        else:
                            # Normalized format: {"name", "args"(dict), "id"}
                            input_data = tc.get("args", {})
                            if isinstance(input_data, str):
                                input_data = json.loads(input_data)
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.get("id", "unknown"),
                                "name": tc.get("name", ""),
                                "input": input_data,
                            })
                    conversation.append({"role": "assistant", "content": content_blocks})
                else:
                    conversation.append({"role": "assistant", "content": m.content or ""})
            else:
                if pending_tool_results:
                    conversation.append({"role": "user", "content": pending_tool_results})
                    pending_tool_results = []
                
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
        
        if pending_tool_results:
            conversation.append({"role": "user", "content": pending_tool_results})
        
        conversation = self._sanitize_conversation(conversation)
        
        body: dict = {
            "model": model,
            "messages": conversation,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }

        if system_text:
            body["system"] = system_text

        # Thinking: None=default ON, 0=explicitly OFF, >0=use that value
        effective_budget = thinking_budget if thinking_budget is not None else self.DEFAULT_THINKING_BUDGET
        if effective_budget > 0:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": effective_budget,
            }
            body["temperature"] = 1  # required by Anthropic when thinking is on
        else:
            body["temperature"] = temperature
        
        if top_p is not None:
            body["top_p"] = top_p
        # Anthropic: top_k is not allowed when thinking is enabled
        if top_k is not None and effective_budget <= 0:
            body["top_k"] = top_k

        if tools:
            body["tools"] = self._convert_tools(tools)

        token = await self._load_token()
        headers = self._build_headers(token)
        
        async with httpx.AsyncClient(timeout=120) as client:
            for attempt in range(2):
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    json=body,
                    headers=headers,
                    timeout=30 if attempt > 0 else 120,
                )

                if resp.status_code == 429:
                    if attempt < 1:
                        logger.warning(f"Rate limited (429), retrying in 1s (attempt {attempt + 1}/2)")
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError("Rate limited (429) after 2 attempts")

                if resp.status_code == 401:
                    # Try refreshing token once
                    if self._claude_creds and attempt == 0:
                        logger.info("Got 401 — attempting token refresh...")
                        try:
                            token = await self._claude_creds.get_token()
                            self._access_token = token
                            headers = self._build_headers(token)
                            continue
                        except Exception as e:
                            logger.error(f"Token refresh failed: {e}")
                    raise RuntimeError(
                        "Claude OAuth token expired or invalid. "
                        "Run 'syne init' to re-authenticate."
                    )

                if resp.status_code == 400:
                    error_text = resp.text[:500]
                    logger.error(f"Anthropic 400 Bad Request: {error_text}")
                    raise RuntimeError(f"Anthropic API error 400: {error_text}")

                if 500 <= resp.status_code < 600:
                    error_text = resp.text[:500]
                    logger.error(f"Anthropic {resp.status_code} Server Error: {error_text}")
                    logger.debug(f"Request body model={body.get('model')}, messages_count={len(body.get('messages', []))}, has_tools={bool(body.get('tools'))}")
                    if attempt < 1:
                        logger.warning(f"Server error ({resp.status_code}), retrying in 1s (attempt {attempt + 1}/2)")
                        await asyncio.sleep(1)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                break
            else:
                raise RuntimeError("Rate limited (429) after 2 attempts")
        
        data = resp.json()
        
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
        raise NotImplementedError("Anthropic does not provide an embedding API")

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[EmbeddingResponse]:
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
                anthropic_tools.append(tool)
        return anthropic_tools
