"""OpenAI Codex Responses API provider (ChatGPT OAuth, free)."""

import asyncio
import json
import logging
import os
import time
import httpx
from typing import Optional
from .provider import LLMProvider, ChatMessage, ChatResponse, EmbeddingResponse

logger = logging.getLogger("syne.llm.codex")

_CODEX_BASE_URL = "https://chatgpt.com/backend-api"

# Codex OAuth (same client ID as OpenClaw/Codex CLI — public app)
_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Token refresh buffer — refresh if expires within this many seconds
_REFRESH_BUFFER_SECONDS = 300  # 5 minutes


class CodexProvider(LLMProvider):
    """OpenAI Codex via ChatGPT backend (Responses API, OAuth).
    
    Free via ChatGPT subscription (Plus/Pro/Team).
    Tokens stored in DB (credential.codex_access_token, credential.codex_refresh_token).
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
        self._auth_failure: Optional[str] = None  # Set when token refresh fails

    @property
    def name(self) -> str:
        return "codex"

    @property
    def context_window(self) -> int:
        return 1_000_000  # GPT-5.2 default

    @property
    def supports_vision(self) -> bool:
        return True

    def _refresh_token(self) -> bool:
        """Refresh Codex OAuth token if expired or about to expire.
        
        Uses the same refresh endpoint as OpenClaw/Codex CLI:
        POST https://auth.openai.com/oauth/token
        
        Returns:
            True if token was refreshed, False otherwise
        """
        now = time.time()
        
        # Check if refresh needed
        if self._token_expires_at > 0 and (now + _REFRESH_BUFFER_SECONDS) < self._token_expires_at:
            return False
        
        if not self.refresh_token:
            logger.warning("No refresh token available for Codex")
            return False
        
        try:
            # Synchronous refresh using httpx (called from sync context)
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    _CODEX_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.refresh_token,
                        "client_id": _CODEX_CLIENT_ID,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                
                if not resp.is_success:
                    logger.error(f"Codex token refresh failed: {resp.status_code} {resp.text[:200]}")
                    self._auth_failure = f"Codex OAuth token refresh failed ({resp.status_code}). Run `syne reauth` to re-authenticate."
                    return False
                
                data = resp.json()
            
            new_access = data.get("access_token")
            new_refresh = data.get("refresh_token")
            expires_in = data.get("expires_in", 0)
            
            if not new_access or not new_refresh:
                logger.error("Codex token refresh response missing fields")
                return False
            
            self.access_token = new_access
            self.refresh_token = new_refresh
            self._token_expires_at = now + expires_in
            
            # Save refreshed tokens to DB (fire and forget via thread)
            self._save_tokens_to_db(new_access, new_refresh, self._token_expires_at)
            
            logger.info("Codex OAuth token refreshed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Codex token refresh error: {e}")
            return False
    
    def _save_tokens_to_db(self, access_token: str, refresh_token: str, expires_at: float):
        """Save refreshed tokens to DB in background."""
        import asyncio
        import threading
        
        async def _save():
            try:
                from ..db.models import set_config
                await set_config("credential.codex_access_token", access_token)
                await set_config("credential.codex_refresh_token", refresh_token)
                await set_config("credential.codex_expires_at", expires_at)
                logger.debug("Codex tokens saved to DB")
            except Exception as e:
                logger.warning(f"Failed to save Codex tokens to DB: {e}")
        
        def _run():
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_save())
                loop.close()
            except Exception:
                pass
        
        threading.Thread(target=_run, daemon=True).start()

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
                # Check for image metadata (vision)
                if msg.metadata and msg.metadata.get("image"):
                    img = msg.metadata["image"]
                    content_parts = [
                        {"type": "input_text", "text": msg.content},
                        {
                            "type": "input_image",
                            "image_url": f"data:{img.get('mime_type', 'image/jpeg')};base64,{img['base64']}",
                        },
                    ]
                    input_msgs.append({"role": "user", "content": content_parts})
                else:
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

        # Validate: remove function_call_output entries without matching function_call
        # This can happen after compaction removes part of the conversation
        fc_ids = {m["call_id"] for m in input_msgs if m.get("type") == "function_call"}
        input_msgs = [
            m for m in input_msgs
            if m.get("type") != "function_call_output" or m.get("call_id") in fc_ids
        ]

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
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> ChatResponse:
        model = model or self.chat_model
        instructions, input_msgs = self._format_input(messages)

        body: dict = {
            "model": model,
            "instructions": instructions or "You are a helpful assistant.",
            "input": input_msgs,
            "stream": True,
            "store": False,
        }

        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if thinking_budget is not None and thinking_budget > 0:
            effort = "high" if thinking_budget >= 8192 else "medium" if thinking_budget >= 2048 else "low"
            body["reasoning"] = {"effort": effort}

        # max_output_tokens — must always be set for Codex reasoning models,
        # otherwise API uses a small default and tool calls get truncated.
        # Value comes from model params JSON (default 32768 for codex driver).
        if max_tokens is not None and max_tokens > 0:
            body["max_output_tokens"] = max_tokens

        if tools:
            body["tools"] = self._format_tools(tools)
            body["tool_choice"] = "auto"
            body["parallel_tool_calls"] = True

        url = f"{self.base_url}/codex/responses"
        headers = self._get_headers()
        
        # Log request body (exclude input/instructions for brevity)
        body_log = {k: v for k, v in body.items() if k not in ("input", "instructions")}
        body_log["input_count"] = len(input_msgs)
        body_log["instructions_len"] = len(instructions) if instructions else 0
        logger.info(f"Codex request body: {body_log}")
        if tools:
            tool_names = [t.get("function", t).get("name", "?") if isinstance(t.get("function"), dict) else t.get("name", "?") for t in tools]
            logger.info(f"Codex tools available ({len(tools)}): {tool_names}")

        # Stream and collect response
        content_text = ""
        tool_calls = []
        current_tool_call = None
        usage = {"input_tokens": 0, "output_tokens": 0}
        thinking_text = ""
        actual_model = model
        event_counts: dict[str, int] = {}  # track all event types

        async with httpx.AsyncClient(timeout=180) as client:
          for attempt in range(2):
            # Shorter timeout for retry attempt to avoid long waits
            req_timeout = 30 if attempt > 0 else 180
            async with client.stream(
                "POST", url, json=body, headers=headers,
                timeout=req_timeout,
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

                if 500 <= resp.status_code < 600 and attempt < 1:
                    error_text = ""
                    async for chunk in resp.aiter_text():
                        error_text += chunk
                    logger.warning(f"Codex {resp.status_code}, retrying in 1s (attempt {attempt + 1}/2): {error_text[:200]}")
                    await asyncio.sleep(1)
                    continue

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
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1

                        # Text content
                        if event_type == "response.output_text.delta":
                            content_text += event.get("delta", "")

                        # Function call start
                        elif event_type == "response.output_item.added":
                            item = event.get("item", {})
                            item_type = item.get("type", "")
                            logger.info(f"Codex output_item.added: type={item_type}, id={item.get('id', '')}, call_id={item.get('call_id', '')}, name={item.get('name', '')}")
                            if item_type == "function_call":
                                current_tool_call = {
                                    "id": item.get("call_id", item.get("id", "")),
                                    "name": item.get("name", ""),
                                    "arguments": "",
                                }
                                logger.info(f"Codex tool call START: {current_tool_call['name']} (id={current_tool_call['id']})")

                        # Function call arguments
                        elif event_type == "response.function_call_arguments.delta":
                            if current_tool_call is not None:
                                current_tool_call["arguments"] += event.get("delta", "")

                        elif event_type == "response.function_call_arguments.done":
                            if current_tool_call is not None:
                                current_tool_call["arguments"] = event.get("arguments", current_tool_call["arguments"])
                                tool_calls.append(current_tool_call)
                                logger.info(f"Codex tool call DONE: {current_tool_call['name']} args={current_tool_call['arguments'][:200]}")
                                current_tool_call = None
                            else:
                                logger.warning(f"Codex function_call_arguments.done but no current_tool_call! args={event.get('arguments', '')[:200]}")

                        # Reasoning/thinking
                        elif event_type == "response.output_item.done":
                            item = event.get("item", {})
                            item_type = item.get("type", "")
                            if item_type == "reasoning":
                                summaries = item.get("summary", [])
                                if summaries:
                                    thinking_text = "\n\n".join(s.get("text", "") for s in summaries)
                            elif item_type == "function_call":
                                # Log completed function call item for verification
                                logger.info(f"Codex output_item.done: function_call name={item.get('name', '')}, call_id={item.get('call_id', '')}, status={item.get('status', '')}")

                        # Completion with usage
                        elif event_type == "response.completed":
                            resp_data = event.get("response", {})
                            actual_model = resp_data.get("model", model)
                            resp_status = resp_data.get("status", "")
                            # Log ALL output items for debugging
                            output_items = resp_data.get("output", [])
                            output_summary = [f"{o.get('type', '?')}({o.get('name', '')})" for o in output_items] if output_items else []
                            logger.info(f"Codex response.completed: status={resp_status}, output_items={output_summary}")
                            if resp_status == "incomplete":
                                reason = resp_data.get("incomplete_details", {})
                                logger.warning(f"Codex response INCOMPLETE: {reason}")
                            elif resp_status == "failed":
                                err = resp_data.get("error", {})
                                logger.error(f"Codex response FAILED: {err}")
                            u = resp_data.get("usage", {})
                            usage["input_tokens"] = u.get("input_tokens", 0)
                            usage["output_tokens"] = u.get("output_tokens", 0)

            # Log stream summary
            logger.info(
                f"Codex stream summary: text={len(content_text)} chars, "
                f"tool_calls={len(tool_calls)}, thinking={len(thinking_text)} chars, "
                f"events={dict(event_counts)}, "
                f"usage=in:{usage['input_tokens']}/out:{usage['output_tokens']}"
            )
            if current_tool_call is not None:
                logger.warning(f"Codex ORPHANED tool call (never completed): {current_tool_call}")
            if content_text and not tool_calls:
                # Log first 300 chars of text when no tool calls — helps diagnose "narrate without acting"
                logger.info(f"Codex text-only response (no tools): {content_text[:300]}")

            break  # success — exit retry loop
          else:
            raise RuntimeError("Codex API failed after 2 attempts")

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
