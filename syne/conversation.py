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
        self._mgr: Optional["ConversationManager"] = None  # Back-reference, set by manager

    async def load_history(self) -> list[ChatMessage]:
        """Load ALL message history from database for this session.
        
        No artificial limits — compaction controls session size.
        If the session has been compacted, there are fewer messages.
        If not, load everything and let compaction handle it before
        the next LLM call.
        """
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = $1
                ORDER BY created_at ASC
            """, self.session_id)

        messages = []
        for row in rows:
            messages.append(ChatMessage(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            ))

        logger.info(f"load_history: loaded {len(messages)} messages for session {self.session_id}")
        self._message_cache = messages
        return messages

    async def save_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Save a message to the database."""
        # Strip null bytes — PostgreSQL text columns reject 0x00
        if content and "\x00" in content:
            content = content.replace("\x00", "")
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
            memory_lines = ["# Relevant Memories (auto-retrieved, scores = similarity confidence)"]
            for mem in memories:
                score = f"(confidence: {mem['similarity']:.0%})"
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
        # Reset per-turn state
        self._pending_media: list[str] = []
        self._cached_input_data: dict = {}  # For ability-first retry via tool call
        self._message_metadata = message_metadata
        self._processing = True

        # ═══════════════════════════════════════════════════════════════
        # ABILITY-FIRST PRE-PROCESSING
        # For any input type (image, audio, document, etc.):
        #   1. Try matching ability's pre_process() first
        #   2. If ability succeeds → inject result as text, strip raw input
        #   3. If ability fails → fallback to native LLM capability
        # This applies to ALL abilities (bundled + self-created).
        # ═══════════════════════════════════════════════════════════════
        if message_metadata:
            user_message, message_metadata = await self._ability_first_preprocess(
                user_message, message_metadata
            )

        # Save user message
        await self.save_message("user", user_message)

        # ═══════════════════════════════════════════════════════════════
        # PRE-FLIGHT COMPACTION CHECK
        # Compact BEFORE building context so the LLM never sees a
        # silently-trimmed conversation.  Old flow had compaction after
        # the response, which meant trim_context could silently drop
        # messages and cause amnesia.
        # ═══════════════════════════════════════════════════════════════
        if self._message_cache and self.context_mgr.should_compact(
            self._message_cache,
            threshold=0.90,
        ):
            logger.info(f"Context at 75%+, compacting BEFORE LLM call for session {self.session_id}")
            if self._mgr and self._mgr._status_callback:
                try:
                    await self._mgr._status_callback(
                        self.session_id,
                        "🧹 Compacting memory... please wait a moment."
                    )
                except Exception as e:
                    logger.debug(f"Status callback failed: {e}")
            result = await auto_compact_check(
                session_id=self.session_id,
                provider=self.provider,
            )
            if result:
                logger.info(
                    f"Auto-compacted: {result['messages_before']} → {result['messages_after']} messages"
                )
                await self.load_history()
                if self._mgr and self._mgr._status_callback:
                    try:
                        await self._mgr._status_callback(
                            self.session_id,
                            f"✅ Compaction done ({result['messages_before']} → {result['messages_after']} messages)"
                        )
                    except Exception as e:
                        logger.debug(f"Status callback failed: {e}")

        # Build context
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

        # Check for auth failures (expired OAuth tokens etc.)
        auth_failure = getattr(self.provider, '_auth_failure', None)
        if auth_failure:
            logger.warning(f"Provider auth failure: {auth_failure}")
            # Notify once, then clear
            self.provider._auth_failure = None
            await self.save_message("assistant", f"⚠️ {auth_failure}")
            return f"⚠️ {auth_failure}"

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

        self._processing = False
        return final_response

    async def _handle_tool_calls(
        self,
        response: ChatResponse,
        context: list[ChatMessage],
        access_level: str,
        tool_schemas: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Execute tool calls and get final response. Loops for multi-step tool use.
        
        max_rounds loaded from DB config `session.max_tool_rounds` (default: 25).
        If limit is reached, appends a notice to the response.
        """
        from .db.models import get_config

        max_rounds = await get_config("session.max_tool_rounds", 100)
        if isinstance(max_rounds, str):
            max_rounds = int(max_rounds)

        current = response
        limit_reached = False

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

                # Notify channel about tool activity
                if self._mgr and self._mgr._tool_callback:
                    try:
                        await self._mgr._tool_callback(name)
                    except Exception:
                        pass

                # Check if this is a built-in tool or an ability
                if self.tools.get(name):
                    result = await self.tools.execute(name, args, access_level)
                elif self.abilities and self.abilities.get(name):
                    # Inject cached input data if ability needs it
                    # (e.g. image_analysis retry after pre-process failed)
                    cached = getattr(self, '_cached_input_data', {})
                    if cached:
                        registered = self.abilities.get(name)
                        if registered and registered.instance:
                            for itype, idata in cached.items():
                                if registered.instance.handles_input_type(itype):
                                    # Auto-fill missing params from cache
                                    if itype == "image":
                                        if not args.get("image_base64") and not args.get("image_url"):
                                            args["image_base64"] = idata.get("base64", "")
                                            args.setdefault("mime_type", idata.get("mime_type", "image/jpeg"))
                                    elif itype == "audio":
                                        if not args.get("audio_base64"):
                                            args["audio_base64"] = idata.get("base64", "")
                                            args.setdefault("mime_type", idata.get("mime_type", "audio/ogg"))
                                    elif itype == "document":
                                        if not args.get("document_base64"):
                                            args["document_base64"] = idata.get("base64", "")
                                            args.setdefault("mime_type", idata.get("mime_type", "application/pdf"))
                                    logger.debug(f"Injected cached {itype} data into ability '{name}' tool call")
                                    break
                    
                    # Execute ability with context
                    ability_context = {
                        "user_id": self.user.get("id"),
                        "session_id": self.session_id,
                        "access_level": access_level,
                        "config": self.abilities.get(name).config or {},
                        "workspace": getattr(self._mgr, 'workspace_outputs', None) if self._mgr else None,
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

                # ═══════════════════════════════════════════════════════
                # GLOBAL TOOL RESULT SCRUBBER
                # Each tool declares its own scrub_level:
                #   "aggressive" — full regex (Cookie, PEM, querystring...)
                #   "safe" — high-confidence only (JWT, sk-*, bot tokens)
                #   "none" — tool has its own dedicated scrubber
                # Default = "aggressive" (safest for new tools)
                # ═══════════════════════════════════════════════════════
                tool_obj = self.tools.get(name)
                scrub = tool_obj.scrub_level if tool_obj else "aggressive"
                if scrub == "none":
                    pass  # tool handles its own scrubbing
                elif scrub == "safe":
                    from .security import redact_content_output
                    result = redact_content_output(str(result))
                else:  # "aggressive" (default)
                    from .security import redact_secrets_in_text
                    result = redact_secrets_in_text(str(result))

                # Collect MEDIA: from tool results (same as ability results)
                result_str = str(result)
                if "\n\nMEDIA: " in result_str or result_str.startswith("MEDIA: "):
                    if "\n\nMEDIA: " in result_str:
                        media_path = result_str.rsplit("\n\nMEDIA: ", 1)[1].strip()
                    else:
                        media_path = result_str[7:].strip()
                    if media_path and hasattr(self, '_pending_media'):
                        self._pending_media.append(media_path)

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
        else:
            # Loop exhausted without breaking — limit reached
            if current.tool_calls:
                limit_reached = True

        if limit_reached:
            notice = (
                f"\n\n⚠️ Tool call limit reached ({max_rounds} rounds). "
                f"Some steps may be incomplete. Use /compact or continue the conversation to proceed."
            )
            if current.content:
                current = ChatResponse(
                    content=current.content + notice,
                    tool_calls=None,
                    thinking=current.thinking,
                )
            else:
                # LLM didn't produce text — generate a final response without tools
                context.append(ChatMessage(
                    role="system",
                    content=f"STOP. You have used {max_rounds} tool rounds. Summarize what you've done so far and what remains.",
                ))
                current = await self.provider.chat(
                    messages=context,
                    thinking_budget=self.thinking_budget,
                    tools=None,  # No tools — force text response
                )
                current = ChatResponse(
                    content=(current.content or "") + notice,
                    tool_calls=None,
                    thinking=current.thinking,
                )

        return current


    async def _ability_first_preprocess(
        self, user_message: str, metadata: dict
    ) -> tuple[str, Optional[dict]]:
        """Ability-first pre-processing for all input types.
        
        Scans metadata for known input types (image, audio, document, etc.)
        and checks if any registered ability can handle them via pre_process().
        
        If an ability handles the input:
        - Its text result is appended to user_message
        - The raw input is stripped from metadata (prevents double-processing)
        
        If no ability handles it:
        - Metadata stays intact for native LLM processing (if provider supports it)
        
        This is the ENGINE-LEVEL implementation of the ability-first principle.
        Works for ALL abilities — bundled and self-created.
        
        Returns:
            Tuple of (updated_user_message, updated_metadata)
        """
        if not self.abilities:
            return user_message, metadata
        
        # Known input types to check in metadata
        # Each maps to: metadata key, native LLM capability check, label for logging
        INPUT_TYPES = {
            "image": {
                "meta_key": "image",
                "native_check": lambda: self.provider.supports_vision,
                "label": "Image analysis",
            },
            "audio": {
                "meta_key": "audio",
                "native_check": lambda: False,  # No LLM natively handles audio input (yet)
                "label": "Audio transcription",
            },
            "document": {
                "meta_key": "document",
                "native_check": lambda: False,  # Documents need parsing
                "label": "Document processing",
            },
        }
        
        for input_type, spec in INPUT_TYPES.items():
            input_data = metadata.get(spec["meta_key"])
            if not input_data:
                continue
            
            # Find a priority ability that handles this input type
            result_text = None
            for registered in self.abilities.list_enabled("owner"):
                # Skip abilities that opted out of priority pre-processing
                if not getattr(registered.instance, 'priority', True):
                    continue
                if not registered.instance.handles_input_type(input_type):
                    continue
                
                try:
                    result_text = await registered.instance.pre_process(
                        input_type, input_data, user_message,
                        config=registered.config,
                    )
                    if result_text:
                        logger.info(
                            f"{spec['label']} via ability '{registered.name}': "
                            f"{len(result_text)} chars"
                        )
                        break  # First successful ability wins
                except Exception as e:
                    logger.error(f"Ability '{registered.name}' pre_process error: {e}")
                    continue
            
            if result_text:
                # Ability succeeded — inject result, strip raw input
                user_message = f"{user_message}\n\n[{spec['label']} result: {result_text}]"
                metadata = {k: v for k, v in metadata.items() if k != spec["meta_key"]}
                self._message_metadata = metadata if metadata else None
            elif spec["native_check"]():
                # Ability failed but provider has native capability — let it through
                logger.info(
                    f"{input_type} ability failed, falling back to native LLM"
                )
            else:
                # Ability failed AND no native support — nothing we can do
                logger.warning(
                    f"{input_type} received but no handler available"
                )
        
        return user_message, metadata if metadata else None


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
        self.workspace_outputs: Optional[str] = None  # Set by agent after init
        self._active: dict[str, Conversation] = {}
        self._delivery_callback = None  # Set by channel for delivering sub-agent results
        self._status_callback = None  # Set by channel for status notifications (e.g., compaction)
        self._tool_callback = None  # Set by channel for tool activity notifications

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

    def set_status_callback(self, callback):
        """Set callback for status notifications (compaction, etc.).
        
        callback(session_id: str, message: str) -> None
        """
        self._status_callback = callback

    def set_tool_callback(self, callback):
        """Set callback for tool activity notifications.
        
        callback(tool_name: str) -> None
        """
        self._tool_callback = callback

    def _session_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    async def get_or_create_session(
        self,
        platform: str,
        chat_id: str,
        user: dict,
        is_group: bool = False,
        extra_context: str = None,
    ) -> Conversation:
        """Get existing conversation or create new one.
        
        Args:
            platform: Platform identifier (telegram, etc.)
            chat_id: Chat/conversation ID
            user: User dict with access_level, id, etc.
            is_group: Whether this is a group chat (affects security restrictions)
            extra_context: Optional extra context for system prompt (e.g., CLI cwd)
            
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
            extra_context=extra_context,
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

        conv._mgr = self  # Back-reference for status callbacks

        # Load history EAGERLY — before any chat() call can save a message
        # and make _message_cache non-empty (which would skip load_history
        # in build_context). This is the fix for restart amnesia.
        await conv.load_history()

        # Load thinking budget from DB config
        from .db.models import get_config as _get_config
        saved_budget = await _get_config("session.thinking_budget", None)
        if saved_budget is not None:
            conv.thinking_budget = int(saved_budget)

        self._active[key] = conv
        return conv

    async def refresh_system_prompts(self):
        """Rebuild system prompts for all active conversations.

        Called after ability changes (create/enable/disable) so that the
        LLM sees updated tool descriptions without requiring a restart.
        """
        from .boot import get_full_prompt

        for key, conv in self._active.items():
            try:
                access_level = conv.user.get("access_level", "public")
                tool_schemas = self.tools.to_openai_schema(access_level)
                ability_schemas = (
                    self.abilities.to_openai_schema(access_level) if self.abilities else []
                )
                new_prompt = await get_full_prompt(
                    user=conv.user,
                    tools=tool_schemas,
                    abilities=ability_schemas,
                )
                conv.system_prompt = new_prompt
                logger.debug(f"Refreshed system prompt for session {key}")
            except Exception as e:
                logger.error(f"Failed to refresh prompt for {key}: {e}")

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
        # Extract extra context from metadata (e.g., CLI working directory)
        extra_context = None
        if message_metadata and message_metadata.get("cwd"):
            cwd = message_metadata["cwd"]
            extra_context = (
                f"\n## Working Directory\n"
                f"You are running in CLI mode. The user's current working directory is: {cwd}\n"
                f"When using exec tool, commands run in this directory by default.\n"
                f"Focus on the project/files in this directory unless asked otherwise."
            )
        conv = await self.get_or_create_session(
            platform, chat_id, user, is_group=is_group, extra_context=extra_context
        )
        return await conv.chat(message, message_metadata=message_metadata)
