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

# Credential paths (in priority order)
OPENCLAW_AUTH_PROFILES = os.path.expanduser("~/.openclaw/agents/main/agent/auth-profiles.json")
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
        """Load token from OpenClaw auth-profiles or Claude CLI credentials.
        
        Priority:
        1. OpenClaw auth-profiles.json (anthropic:default or anthropic:*)
        2. Claude CLI ~/.claude/.credentials.json (claudeAiOauth)
        """
        now = time.time()
        
        # Re-read file at most every 30 seconds
        if self._access_token and (now - self._last_read) < 30:
            if self._expires_at == 0 or now < self._expires_at:
                return self._access_token
        
        # Try OpenClaw auth-profiles first
        token = self._load_from_openclaw_profiles()
        if token:
            self._access_token = token
            self._expires_at = 0  # OpenClaw manages refresh, no expiry tracking needed
            self._last_read = now
            return token
        
        # Fallback: Claude CLI credentials
        token = self._load_from_claude_cli()
        if token:
            return token
        
        raise RuntimeError(
            "No Claude/Anthropic token found. Check:\n"
            f"  1. OpenClaw auth-profiles: {OPENCLAW_AUTH_PROFILES}\n"
            f"  2. Claude CLI credentials: {self.credentials_path}"
        )
    
    def _load_from_openclaw_profiles(self) -> Optional[str]:
        """Load token from OpenClaw auth-profiles.json."""
        try:
            with open(OPENCLAW_AUTH_PROFILES) as f:
                data = json.load(f)
            
            profiles = data.get("profiles", {})
            # Try anthropic:default first, then any anthropic:* profile
            for key in ["anthropic:default", *[k for k in profiles if k.startswith("anthropic:")]]:
                profile = profiles.get(key, {})
                token = profile.get("token", "")
                if token:
                    logger.debug(f"Loaded Anthropic token from OpenClaw profile: {key}")
                    return token
            
            return None
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _load_from_claude_cli(self) -> Optional[str]:
        """Load token from Claude CLI credentials."""
        try:
            with open(self.credentials_path) as f:
                creds = json.load(f)
            
            oauth = creds.get("claudeAiOauth", {})
            token = oauth.get("accessToken", "")
            if not token:
                return None
            
            expires_at_ms = oauth.get("expiresAt", 0)
            self._expires_at = expires_at_ms / 1000 if expires_at_ms > 1e12 else expires_at_ms
            self._access_token = token
            self._last_read = time.time()
            
            if time.time() >= self._expires_at:
                logger.warning("Claude CLI token expired. Run 'claude' to refresh.")
            
            return token
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _headers(self) -> dict:
        token = self._load_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        # OAuth tokens (sk-ant-oat*) require the beta header
        if token.startswith("sk-ant-oat"):
            headers["anthropic-beta"] = "oauth-2025-04-20"
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
                            # Has matching pairs — keep both, but filter to only matched IDs
                            matched_ids = tool_use_ids & result_ids
                            
                            # Filter assistant content: keep text + matched tool_use
                            filtered_assistant = [
                                block for block in msg["content"]
                                if block.get("type") != "tool_use" or block.get("id") in matched_ids
                            ]
                            # Filter tool results: keep only matched
                            filtered_results = [
                                block for block in next_msg["content"]
                                if block.get("type") != "tool_result" or block.get("tool_use_id") in matched_ids
                            ]
                            
                            sanitized.append({"role": "assistant", "content": filtered_assistant})
                            if filtered_results:
                                sanitized.append({"role": "user", "content": filtered_results})
                            i += 2
                            continue
                    
                    # No matching tool_results found — convert to plain text
                    text_parts = [
                        block.get("text", "")
                        for block in msg["content"]
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    text = " ".join(t for t in text_parts if t) or "[tool calls without results — trimmed]"
                    sanitized.append({"role": "assistant", "content": text})
                    i += 1
                    continue
            
            # Check if this is a user message with tool_results (orphaned — no preceding tool_use)
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_tool_results = any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in msg["content"]
                )
                
                if has_tool_results:
                    # Check if previous message was assistant with matching tool_use
                    prev = sanitized[-1] if sanitized else None
                    if prev and prev.get("role") == "assistant" and isinstance(prev.get("content"), list):
                        prev_tool_ids = {
                            block["id"]
                            for block in prev["content"]
                            if isinstance(block, dict) and block.get("type") == "tool_use"
                        }
                        # Filter to only matching results
                        filtered = [
                            block for block in msg["content"]
                            if block.get("type") != "tool_result" or block.get("tool_use_id") in prev_tool_ids
                        ]
                        if filtered:
                            sanitized.append({"role": "user", "content": filtered})
                    else:
                        # Orphaned tool_results with no preceding tool_use — skip entirely
                        logger.debug("Dropping orphaned tool_result message (no preceding tool_use)")
                    i += 1
                    continue
            
            sanitized.append(msg)
            i += 1
        
        # Final pass: merge consecutive same-role messages (Anthropic requires alternating)
        merged = []
        for msg in sanitized:
            if merged and merged[-1].get("role") == msg.get("role"):
                # Merge content
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
        
        # Sanitize: ensure all tool_use/tool_result pairs are valid
        conversation = self._sanitize_conversation(conversation)
        
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
                
                if resp.status_code == 400:
                    error_text = resp.text[:500]
                    logger.error(f"Anthropic 400 Bad Request: {error_text}")
                    raise RuntimeError(f"Anthropic API error 400: {error_text}")
                
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
