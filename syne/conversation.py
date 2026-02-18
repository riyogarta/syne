"""Conversation manager — session handling, message history, context building."""

import json
import logging
from typing import Optional
from .db.connection import get_connection
from .llm.provider import LLMProvider, ChatMessage, ChatResponse
from .memory.engine import MemoryEngine
from .memory.evaluator import evaluate_and_store
from .context import ContextManager, estimate_messages_tokens
from .compaction import auto_compact_check
from .boot import get_full_prompt
from .tools.registry import ToolRegistry
from .abilities import AbilityRegistry
from .security import (
    get_group_context_restrictions,
    filter_tools_for_group,
    should_filter_tools_for_group,
)

logger = logging.getLogger("syne.conversation")


class Conversation:
    """Manages a single conversation session."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryEngine,
        tools: ToolRegistry,
        context_mgr: ContextManager,
        session_id: int,
        user: dict,
        system_prompt: str,
        abilities: Optional[AbilityRegistry] = None,
        is_group: bool = False,
    ):
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.abilities = abilities
        self.context_mgr = context_mgr
        self.session_id = session_id
        self.user = user
        self.system_prompt = system_prompt
        self.is_group = is_group
        self.thinking_budget: Optional[int] = None  # None = model default, 0 = off
        self._message_cache: list[ChatMessage] = []
        self._processing: bool = False

    async def load_history(self, limit: int = 50) -> list[ChatMessage]:
        """Load recent message history from database."""
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, self.session_id, limit)

        messages = []
        for row in reversed(rows):
            messages.append(ChatMessage(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            ))

        self._message_cache = messages
        return messages

    async def save_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Save a message to the database."""
        meta_json = json.dumps(metadata) if metadata else "{}"

        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
            """, self.session_id, role, content, meta_json)

            await conn.execute("""
                UPDATE sessions
                SET message_count = message_count + 1, updated_at = NOW()
                WHERE id = $1
            """, self.session_id)

        self._message_cache.append(ChatMessage(role=role, content=content, metadata=metadata))

    async def build_context(self, user_message: str) -> list[ChatMessage]:
        """Build full context: system prompt + memories + history + current message."""
        messages = []
        access_level = self.user.get("access_level", "public")

        # 1. System prompt
        prompt = self.system_prompt
        
        # ═══════════════════════════════════════════════════════════════
        # SECURITY: Add group context restrictions to system prompt
        # This reinforces that owner tools are DM-only when in groups
        # ═══════════════════════════════════════════════════════════════
        if self.is_group:
            group_restrictions = get_group_context_restrictions(access_level, is_group=True)
            prompt = prompt + group_restrictions
        
        messages.append(ChatMessage(role="system", content=prompt))

        # 2. Recall relevant memories (with Rule 760 filtering via access_level)
        memories = await self.memory.recall(
            query=user_message,
            limit=10,
            user_id=self.user.get("id"),
            requester_access_level=access_level,  # Pass access level for Rule 760
        )

        if memories:
            memory_lines = ["# Relevant Memories"]
            for mem in memories:
                score = f"({mem['similarity']:.0%})"
                memory_lines.append(f"- [{mem['category']}] {mem['content']} {score}")
            messages.append(ChatMessage(role="system", content="\n".join(memory_lines)))

        # 3. Conversation history
        if not self._message_cache:
            await self.load_history()
        messages.extend(self._message_cache)

        # 4. Current user message (with optional image metadata)
        msg_metadata = getattr(self, '_message_metadata', None)
        messages.append(ChatMessage(role="user", content=user_message, metadata=msg_metadata))

        # 5. Trim to fit context window
        messages = self.context_mgr.trim_context(messages)

        return messages

    async def chat(self, user_message: str, message_metadata: Optional[dict] = None) -> str:
        """Process a user message and return agent response.
        
        Args:
            user_message: The user's text message
            message_metadata: Optional metadata (e.g. {"image": {"mime_type": "...", "base64": "..."}})
        """
        # Reset media collector for this turn
        self._pending_media: list[str] = []
        self._message_metadata = message_metadata
        self._processing = True

        # Save user message
        await self.save_message("user", user_message)

        # Build context (with auto-trim)
        context = await self.build_context(user_message)

        # Log context usage
        usage = self.context_mgr.get_usage(context)
        logger.debug(f"Context: {usage['used_tokens']} tokens ({usage['usage_percent']}%)")

        # Get available tools + abilities
        access_level = self.user.get("access_level", "public")
        
        # ═══════════════════════════════════════════════════════════════
        # SECURITY: In group chats, cap effective access to "public" for tools
        # Owner can still use all tools via DM, but group context = restricted
        # ═══════════════════════════════════════════════════════════════
        effective_access_level = access_level
        if self.is_group:
            # In groups, even owners get restricted tool access
            # This prevents prompt injection attacks in group context
            effective_access_level = "public"
            logger.debug(f"Group context: capping tool access to 'public' (user was: {access_level})")
        
        tool_schemas = self.tools.to_openai_schema(effective_access_level)
        if self.abilities:
            tool_schemas = tool_schemas + self.abilities.to_openai_schema(effective_access_level)
        
        # Additional filter: remove owner-only tools entirely in group context
        if self.is_group and should_filter_tools_for_group(self.is_group):
            tool_schemas = filter_tools_for_group(tool_schemas)

        # Call LLM
        response = await self.provider.chat(
            messages=context,
            tools=tool_schemas if tool_schemas else None,
            thinking_budget=self.thinking_budget,
        )

        # Handle tool calls
        if response.tool_calls:
            response = await self._handle_tool_calls(response, context, access_level, tool_schemas)

        # Save assistant response
        await self.save_message("assistant", response.content)

        # Store thinking for the channel to optionally display
        self._last_thinking = response.thinking

        # Attach any media collected during tool calls to the final response
        final_response = response.content
        if self._pending_media:
            # Append the last media path (most relevant — usually the final image/file)
            # If LLM response already contains MEDIA:, don't duplicate
            if "MEDIA: " not in final_response:
                last_media = self._pending_media[-1]
                final_response = f"{final_response}\n\nMEDIA: {last_media}"
                logger.info(f"Attached pending media to response: {last_media}")

        # Evaluate memory (only if auto_capture enabled)
        from .db.models import get_config
        auto_capture = await get_config("memory.auto_capture", False)
        if auto_capture:
            await evaluate_and_store(
                provider=self.provider,
                memory_engine=self.memory,
                user_message=user_message,
                user_id=self.user.get("id"),
            )

        # Check if compaction needed
        if self.context_mgr.should_compact(
            self._message_cache,
            threshold=0.75,
        ):
            logger.info(f"Context at 75%+, triggering compaction for session {self.session_id}")
            result = await auto_compact_check(
                session_id=self.session_id,
                provider=self.provider,
            )
            if result:
                logger.info(
                    f"Auto-compacted: {result['messages_before']} → {result['messages_after']} messages"
                )
                # Reload history after compaction
                await self.load_history()

        self._processing = False
        return final_response

    async def _handle_tool_calls(
        self,
        response: ChatResponse,
        context: list[ChatMessage],
        access_level: str,
        tool_schemas: Optional[list[dict]] = None,
        max_rounds: int = 10,
    ) -> ChatResponse:
        """Execute tool calls and get final response. Loops for multi-step tool use."""
        current = response

        for round_num in range(max_rounds):
            if not current.tool_calls:
                break

            context.append(ChatMessage(
                role="assistant",
                content=current.content or "",
                metadata={"tool_calls": current.tool_calls},
            ))

            for tool_call in current.tool_calls:
                # Handle both normalized format (name/args) and OpenAI raw (function.name/arguments)
                if "function" in tool_call:
                    func = tool_call["function"]
                    name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                else:
                    name = tool_call.get("name", "")
                    args = tool_call.get("args", {})
                    if isinstance(args, str):
                        args = json.loads(args)

                tool_call_id = tool_call.get("id")

                logger.info(f"Tool call (round {round_num + 1}): {name}({args})")

                # Check if this is a built-in tool or an ability
                if self.tools.get(name):
                    result = await self.tools.execute(name, args, access_level)
                elif self.abilities and self.abilities.get(name):
                    # Execute ability with context
                    ability_context = {
                        "user_id": self.user.get("id"),
                        "session_id": self.session_id,
                        "access_level": access_level,
                        "config": self.abilities.get(name).config or {},
                    }
                    ability_result = await self.abilities.execute(name, args, ability_context)
                    if ability_result.get("success"):
                        result = ability_result.get("result", "")
                        # Handle media output (images, audio)
                        if ability_result.get("media"):
                            media_path = ability_result["media"]
                            result = f"{result}\n\nMEDIA: {media_path}"
                            # Collect media for the channel to deliver
                            if hasattr(self, '_pending_media'):
                                self._pending_media.append(media_path)
                    else:
                        result = f"Error: {ability_result.get('error', 'Unknown error')}"
                else:
                    result = f"Error: Unknown tool or ability '{name}'"

                tool_meta = {"tool_name": name}
                if tool_call_id:
                    tool_meta["tool_call_id"] = tool_call_id
                await self.save_message("tool", result, metadata=tool_meta)
                context.append(ChatMessage(role="tool", content=str(result), metadata=tool_meta))

            # Get next response — may contain more tool calls
            current = await self.provider.chat(
                messages=context,
                thinking_budget=self.thinking_budget,
                tools=tool_schemas if tool_schemas else None,
            )

        return current


class ConversationManager:
    """Manages multiple conversations across users and platforms."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryEngine,
        tools: ToolRegistry,
        abilities: Optional[AbilityRegistry] = None,
        context_mgr: Optional[ContextManager] = None,
        subagents=None,
    ):
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.abilities = abilities
        self.context_mgr = context_mgr or ContextManager()
        self.subagents = subagents
        self._active: dict[str, Conversation] = {}
        self._delivery_callback = None  # Set by channel for delivering sub-agent results

        # Wire up sub-agent completion callback
        if self.subagents:
            self.subagents.set_completion_callback(self._on_subagent_complete)

    async def _on_subagent_complete(self, run_id: str, status: str, result: str):
        """Called when a sub-agent finishes. Delivers result to the parent session."""
        if status == "completed":
            msg = f"✅ Sub-agent completed (run: {run_id[:8]})\n\n{result}"
        else:
            msg = f"❌ Sub-agent failed (run: {run_id[:8]})\n\n{result}"

        # Deliver via callback if set (e.g., send Telegram message)
        if self._delivery_callback:
            try:
                await self._delivery_callback(msg)
            except Exception as e:
                logger.error(f"Failed to deliver sub-agent result: {e}")
        else:
            logger.info(f"Sub-agent result (no delivery callback): {msg[:200]}")

    def set_delivery_callback(self, callback):
        """Set callback for delivering sub-agent results to the user.
        
        callback(message: str) -> None
        """
        self._delivery_callback = callback

    def _session_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    async def get_or_create_session(
        self,
        platform: str,
        chat_id: str,
        user: dict,
        is_group: bool = False,
    ) -> Conversation:
        """Get existing conversation or create new one.
        
        Args:
            platform: Platform identifier (telegram, etc.)
            chat_id: Chat/conversation ID
            user: User dict with access_level, id, etc.
            is_group: Whether this is a group chat (affects security restrictions)
            
        Returns:
            Conversation instance
        """
        key = self._session_key(platform, chat_id)

        if key in self._active:
            # Update is_group flag in case it changed
            self._active[key].is_group = is_group
            return self._active[key]

        async with get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM sessions
                WHERE platform = $1 AND platform_chat_id = $2 AND status = 'active'
                ORDER BY updated_at DESC LIMIT 1
            """, platform, chat_id)

            if row:
                session_id = row["id"]
            else:
                row = await conn.fetchrow("""
                    INSERT INTO sessions (user_id, platform, platform_chat_id)
                    VALUES ($1, $2, $3)
                    RETURNING id
                """, user.get("id"), platform, chat_id)
                session_id = row["id"]

        # Build system prompt with tools and abilities for this user's access level
        access_level = user.get("access_level", "public")
        tool_schemas = self.tools.to_openai_schema(access_level)
        ability_schemas = self.abilities.to_openai_schema(access_level) if self.abilities else []
        system_prompt = await get_full_prompt(
            user=user,
            tools=tool_schemas,
            abilities=ability_schemas,
        )

        conv = Conversation(
            provider=self.provider,
            memory=self.memory,
            tools=self.tools,
            context_mgr=self.context_mgr,
            session_id=session_id,
            user=user,
            system_prompt=system_prompt,
            abilities=self.abilities,
            is_group=is_group,
        )

        # Load thinking budget from DB config
        from .db.models import get_config as _get_config
        saved_budget = await _get_config("session.thinking_budget", None)
        if saved_budget is not None:
            conv.thinking_budget = int(saved_budget)

        self._active[key] = conv
        return conv

    async def handle_message(
        self,
        platform: str,
        chat_id: str,
        user: dict,
        message: str,
        is_group: bool = False,
        message_metadata: Optional[dict] = None,
    ) -> str:
        """Handle an incoming message. Returns agent response.
        
        Args:
            platform: Platform identifier
            chat_id: Chat/conversation ID
            user: User dict
            message: Message content
            is_group: Whether this is a group chat (affects security)
            message_metadata: Optional metadata (e.g. image data for vision)
            
        Returns:
            Agent response string
        """
        conv = await self.get_or_create_session(platform, chat_id, user, is_group=is_group)
        return await conv.chat(message, message_metadata=message_metadata)
