"""OpenAI Codex Responses API provider (ChatGPT OAuth, free)."""

import json
import logging
import os
import time
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.codex")

_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
_AUTH_FILE_PATH = os.path.expanduser("~/.openclaw/agents/main/agent/auth.json")

# Token refresh buffer — refresh if expires within this many seconds
_REFRESH_BUFFER_SECONDS = 300  # 5 minutes


class CodexProvider(LLMProvider):
    """OpenAI Codex via ChatGPT backend (Responses API, OAuth).
    
    Uses the same OAuth flow as OpenClaw/Codex CLI.
    Free via ChatGPT subscription (Plus/Pro/Team).
    
    Token refresh: Before each request, checks if the token is expired or about
    to expire (<5 min). If so, re-reads from the auth file (OpenClaw refreshes it).
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        chat_model: str = "gpt-5.2",
        base_url: str = _CODEX_BASE_URL,
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.chat_model = chat_model
        self.base_url = base_url.rstrip("/")
        
        # Cache for parsed auth file
        self._auth_cache: Optional[dict] = None
        self._auth_cache_time: float = 0
        self._token_expires_at: float = 0  # Unix timestamp

    @property
    def name(self) -> str:
        return "codex"

    @property
    def supports_vision(self) -> bool:
        return True

    def _refresh_token(self) -> bool:
        """Re-read token from auth file if expired or about to expire.
        
        Returns:
            True if token was refreshed, False otherwise
        """
        now = time.time()
        
        # Check if we need to refresh
        if self._token_expires_at > 0 and (now + _REFRESH_BUFFER_SECONDS) < self._token_expires_at:
            # Token still valid, no refresh needed
            return False
        
        # Check cache validity (re-read file at most every 30 seconds)
        if self._auth_cache and (now - self._auth_cache_time) < 30:
            # Cache is fresh enough, check if token changed
            codex_auth = self._auth_cache.get("openai-codex", {})
            new_token = codex_auth.get("access", "")
            if new_token and new_token != self.access_token:
                self.access_token = new_token
                self.refresh_token = codex_auth.get("refresh", "")
                # Parse expires timestamp (ms → s)
                expires = codex_auth.get("expires", 0)
                self._token_expires_at = expires / 1000 if expires > 1e12 else expires
                logger.debug("Codex token refreshed from cached auth file")
                return True
            return False
        
        # Read fresh from file
        try:
            with open(_AUTH_FILE_PATH) as f:
                auth_data = json.load(f)
            self._auth_cache = auth_data
            self._auth_cache_time = now
            
            codex_auth = auth_data.get("openai-codex", {})
            new_token = codex_auth.get("access", "")
            
            if new_token and new_token != self.access_token:
                self.access_token = new_token
                self.refresh_token = codex_auth.get("refresh", "")
                # Parse expires timestamp (ms → s)
                expires = codex_auth.get("expires", 0)
                self._token_expires_at = expires / 1000 if expires > 1e12 else expires
                logger.info("Codex token refreshed from auth file")
                return True
            elif new_token:
                # Same token, but update expiry
                expires = codex_auth.get("expires", 0)
                self._token_expires_at = expires / 1000 if expires > 1e12 else expires
                
        except FileNotFoundError:
            logger.warning(f"Auth file not found: {_AUTH_FILE_PATH}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse auth file: {e}")
        
        return False

    def _get_headers(self) -> dict:
        # Refresh token if needed before building headers
        self._refresh_token()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
        }

    def _format_input(self, messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        """Convert ChatMessages to Codex Responses format.
        
        Returns:
            (system_instructions, input_messages)
        """
        instructions = ""
        input_msgs = []

        for msg in messages:
            if msg.role == "system":
                instructions = msg.content
            elif msg.role == "user":
                input_msgs.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                if msg.metadata and msg.metadata.get("tool_calls"):
                    # Assistant with tool calls — add as function_call items
                    if msg.content:
                        input_msgs.append({"role": "assistant", "content": msg.content})
                    for tc in msg.metadata["tool_calls"]:
                        # Codex requires IDs starting with "fc_"
                        raw_id = tc.get("id", tc.get("call_id", ""))
                        fc_id = raw_id if raw_id.startswith("fc") else f"fc_{raw_id or tc.get('name', 'unknown')}"
                        args = tc.get("args", tc.get("arguments", {}))
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        input_msgs.append({
                            "type": "function_call",
                            "id": fc_id,
                            "call_id": fc_id,
                            "name": tc.get("name"),
                            "arguments": args,
                        })
                else:
                    input_msgs.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "unknown") if msg.metadata else "unknown"
                raw_call_id = msg.metadata.get("tool_call_id", "") if msg.metadata else ""
                call_id = raw_call_id if raw_call_id.startswith("fc") else f"fc_{raw_call_id or tool_name}"
                input_msgs.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": msg.content,
                })

        return instructions, input_msgs

    def _format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-style tool schemas to Responses API format."""
        formatted = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                formatted.append({
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                formatted.append(tool)
        return formatted

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
        instructions, input_msgs = self._format_input(messages)

        body: dict = {
            "model": model,
            "instructions": instructions or "You are a helpful assistant.",
            "input": input_msgs,
            "stream": True,
            "store": False,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        # Note: Codex backend does NOT support temperature parameter
        if max_tokens:
            body["max_output_tokens"] = max_tokens
        if tools:
            body["tools"] = self._format_tools(tools)

        url = f"{self.base_url}/codex/responses"
        headers = self._get_headers()
        
        logger.debug(f"Codex request: model={model}, input={len(input_msgs)} msgs, tools={len(tools) if tools else 0}")

        # Stream and collect response
        content_text = ""
        tool_calls = []
        current_tool_call = None
        usage = {"input_tokens": 0, "output_tokens": 0}
        thinking_text = ""
        actual_model = model

        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST", url, json=body, headers=headers,
            ) as resp:
                if resp.status_code == 429:
                    error_text = ""
                    async for chunk in resp.aiter_text():
                        error_text += chunk
                    logger.error(f"Rate limited (429): {error_text[:200]}")
                    raise httpx.HTTPStatusError(
                        f"Rate limited: {error_text[:200]}",
                        request=resp.request,
                        response=resp,
                    )
                
                if resp.status_code != 200:
                    error_text = ""
                    async for chunk in resp.aiter_text():
                        error_text += chunk
                    logger.error(f"Codex error {resp.status_code}: {error_text[:300]}")
                    raise httpx.HTTPStatusError(
                        f"Codex API error {resp.status_code}: {error_text[:200]}",
                        request=resp.request,
                        response=resp,
                    )

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        
                        if not line or line.startswith("event:"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        
                        data_str = line[5:].strip()
                        if not data_str:
                            continue

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        # Text content
                        if event_type == "response.output_text.delta":
                            content_text += event.get("delta", "")

                        # Function call start
                        elif event_type == "response.output_item.added":
                            item = event.get("item", {})
                            if item.get("type") == "function_call":
                                current_tool_call = {
                                    "id": item.get("call_id", item.get("id", "")),
                                    "name": item.get("name", ""),
                                    "arguments": "",
                                }

                        # Function call arguments
                        elif event_type == "response.function_call_arguments.delta":
                            if current_tool_call is not None:
                                current_tool_call["arguments"] += event.get("delta", "")

                        elif event_type == "response.function_call_arguments.done":
                            if current_tool_call is not None:
                                current_tool_call["arguments"] = event.get("arguments", current_tool_call["arguments"])
                                tool_calls.append(current_tool_call)
                                current_tool_call = None

                        # Reasoning/thinking
                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            if item.get("type") == "reasoning":
                                summaries = item.get("summary", [])
                                if summaries:
                                    thinking_text = "\n\n".join(s.get("text", "") for s in summaries)

                        # Completion with usage
                        elif event_type == "response.completed":
                            resp_data = event.get("response", {})
                            actual_model = resp_data.get("model", model)
                            u = resp_data.get("usage", {})
                            usage["input_tokens"] = u.get("input_tokens", 0)
                            usage["output_tokens"] = u.get("output_tokens", 0)

        # Convert tool calls to Syne format
        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_calls.append({
                    "name": tc["name"],
                    "args": args,
                    "id": tc["id"],
                })

        return ChatResponse(
            content=content_text,
            model=actual_model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            tool_calls=parsed_tool_calls,
            thinking=thinking_text or None,
        )

    async def embed(self, text: str, model: Optional[str] = None) -> EmbeddingResponse:
        raise NotImplementedError("Codex provider does not support embeddings. Use HybridProvider with Together AI.")

    async def embed_batch(self, texts: list[str], model: Optional[str] = None) -> list[EmbeddingResponse]:
        raise NotImplementedError("Codex provider does not support embeddings. Use HybridProvider with Together AI.")
