"""Conversation manager — session handling, message history, context building."""

import asyncio
import json
import logging
from typing import Optional
from .db.connection import get_connection
from .llm.provider import LLMProvider, ChatMessage, ChatResponse, UsageAccumulator
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
        inbound: Optional["InboundContext"] = None,
    ):
        self.provider = provider
        self.memory = memory
        self.tools = tools
        self.abilities = abilities
        self.context_mgr = context_mgr
        self.session_id = session_id
        self.user = user
        self.system_prompt = system_prompt
        # InboundContext is the single source of truth for chat context
        self.inbound: Optional["InboundContext"] = inbound
        self.is_group: bool = inbound.is_group if inbound else False
        self.chat_name: Optional[str] = inbound.group_subject if inbound else None
        self.chat_id: Optional[str] = inbound.chat_id if inbound else None
        self.model_params: dict = {}  # Per-model LLM params (all 7: temperature, max_tokens, thinking_budget, top_p, top_k, frequency_penalty, presence_penalty)
        self.reasoning_visible: bool = False  # Per-model reasoning visibility
        self._message_cache: list[ChatMessage] = []
        self._processing: bool = False
        self._lock = asyncio.Lock()  # Prevent concurrent chat() on same session
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

    async def build_context(self, user_message: str, recall_query: Optional[str] = None) -> list[ChatMessage]:
        """Build full context: system prompt + memories + history + current message.

        Args:
            user_message: Full message (may include context prefix) for history.
            recall_query: Clean text (without prefix) for memory recall. Falls back to user_message.
        """
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
        #    Memory is GLOBAL — all sessions (DM + group) recall from the same pool.
        #    Rule 760 filtering ensures family-private info only goes to family.
        #    History is STRICTLY per-session (enforced by session_id in load_history).
        from .db.models import get_config as _get_recall_config
        recall_limit = await _get_recall_config("memory.recall_limit", 5)
        if isinstance(recall_limit, str):
            recall_limit = int(recall_limit)
        memories = await self.memory.recall(
            query=recall_query or user_message,
            limit=recall_limit,
            user_id=self.user.get("id"),
            requester_access_level=access_level,  # Pass access level for Rule 760
        )

        # 3. Conversation history
        if not self._message_cache:
            await self.load_history()
        messages.extend(self._message_cache)

        # 4. Inject recalled memories AFTER history, close to the user message.
        #    This positioning ensures the LLM "sees" memories near the question,
        #    preventing long conversation history from drowning out memory context.
        if memories:
            memory_lines = [
                "# Relevant Memories",
                "Your stored facts about the owner and their world. Use these to answer questions.",
                "",
            ]
            for mem in memories:
                score = f"(confidence: {mem['similarity']:.0%})"
                # Conflict flags injected by memory engine (code-enforced)
                conflict_status = mem.get("_conflict_status", "")
                if conflict_status == "conflicted":
                    ref_id = mem.get("_conflicts_with", "?")
                    flag = f" ⚠️ CONFLICTED — superseded by memory #{ref_id}"
                elif conflict_status == "authoritative":
                    flag = " ✅ AUTHORITATIVE"
                else:
                    flag = ""
                memory_lines.append(f"- [{mem['category']}] {mem['content']} {score}{flag}")
            messages.append(ChatMessage(role="system", content="\n".join(memory_lines)))
            # Log recalled memories for debugging
            _actual_query = recall_query or user_message
            logger.info(f"Recalled {len(memories)} memories for query: {_actual_query[:80]}")
            for mem in memories[:3]:
                logger.info(f"  Memory #{mem['id']} (sim={mem['similarity']:.3f}): {mem['content'][:60]}")
        else:
            _actual_query = recall_query or user_message
            logger.info(f"No memories recalled for query: {_actual_query[:80]}")

        # 5. Current user message (with optional image metadata)
        msg_metadata = getattr(self, '_message_metadata', None)
        messages.append(ChatMessage(role="user", content=user_message, metadata=msg_metadata))

        # 5. Trim to fit context window
        messages = self.context_mgr.trim_context(messages)

        return messages

    def _build_chat_kwargs(self) -> dict:
        """Extract all LLM parameters from self.model_params for provider.chat()."""
        kwargs = {}
        p = self.model_params
        # Int params
        for key in ("thinking_budget", "top_k"):
            v = p.get(key)
            if v is not None:
                kwargs[key] = int(v)
        # Float params
        for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            v = p.get(key)
            if v is not None:
                kwargs[key] = float(v)
        v = p.get("max_tokens")
        if v is not None:
            kwargs["max_tokens"] = int(v)
        logger.debug(f"chat_kwargs from model_params: {kwargs}")
        return kwargs

    async def chat(self, user_message: str, message_metadata: Optional[dict] = None) -> str:
        """Process a user message and return agent response.

        Args:
            user_message: The user's text message
            message_metadata: Optional metadata (e.g. {"image": {"mime_type": "...", "base64": "..."}})
        """
        # Wait for lock with timeout — prevents permanent queue if previous request hangs
        locked = self._lock.locked()
        if locked:
            logger.warning(f"Session {self.session_id}: lock is held, waiting (up to 30s)...")
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning(f"Session {self.session_id}: lock timeout — previous request still processing, dropping this message silently")
            return None  # None = silently dropped; typing indicator from first request stays visible
        logger.debug(f"Session {self.session_id}: lock acquired")
        try:
            # Reset per-turn state
            self._pending_media: list[str] = []
            self._cached_input_data: dict = {}  # For ability-first retry via tool call
            self._message_metadata = message_metadata
            self._processing = True
            try:
                return await self._chat_inner(user_message, message_metadata)
            finally:
                self._processing = False
        finally:
            self._lock.release()

    async def _chat_inner(self, user_message: str, message_metadata: Optional[dict] = None) -> str:
        """Inner chat logic. Wrapped by chat() with try/finally for _processing flag."""

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
        # Compaction gate: trigger on EITHER context fullness OR message count/char threshold
        # Two paths: (1) context window >90% full, (2) message count or chars exceed config thresholds
        context_full = self._message_cache and self.context_mgr.should_compact(
            self._message_cache,
            threshold=0.90,
        )
        from .db.models import get_config as _gc
        _msg_thresh = int(await _gc("session.max_messages", 100))
        _chr_thresh = int(await _gc("session.compaction_threshold", 150000))
        msg_count = len(self._message_cache) if self._message_cache else 0
        count_exceeded = msg_count >= _msg_thresh
        # Quick char estimate from message cache
        char_total = sum(len(m.content or "") for m in self._message_cache) if self._message_cache else 0
        chars_exceeded = char_total >= _chr_thresh

        if context_full or count_exceeded or chars_exceeded:
            logger.info(f"Compaction triggered for session {self.session_id}: context_full={context_full}, msgs={msg_count}/{_msg_thresh}, chars={char_total}/{_chr_thresh}")
            if self._mgr and self._mgr._status_callbacks:
                for cb in self._mgr._status_callbacks:
                    try:
                        await cb(
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
                if self._mgr and self._mgr._status_callbacks:
                    for cb in self._mgr._status_callbacks:
                        try:
                            await cb(
                                self.session_id,
                                f"✅ Compaction done ({result['messages_before']} → {result['messages_after']} messages)"
                            )
                        except Exception as e:
                            logger.debug(f"Status callback failed: {e}")

        # Build context — use original text (without context prefix) for memory recall
        recall_query = (message_metadata or {}).get("original_text", user_message)
        context = await self.build_context(user_message, recall_query=recall_query)

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

        # Call LLM with all params from model registry
        response = await self.provider.chat(
            messages=context,
            tools=tool_schemas if tool_schemas else None,
            **self._build_chat_kwargs(),
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
        logger.info(f"LLM response: content={len(response.content or '')} chars, tool_calls={len(response.tool_calls) if response.tool_calls else 0}, model={response.model}")
        if response.tool_calls:
            logger.info(f"Tool calls: {[tc.get('name') for tc in response.tool_calls]}")
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
        # Rule 760: Only owner and family can write to global memory.
        # Non-family messages stay in session history only.
        # Fire-and-forget. Delay only when using main provider (rate limit).
        from .db.models import get_config
        access_level = self.user.get("access_level", "public")
        can_write_memory = access_level in ("owner", "family")
        auto_capture = await get_config("memory.auto_capture", False)
        if auto_capture and can_write_memory:
            eval_driver = await get_config("memory.evaluator_driver", "ollama")
            eval_model = await get_config("memory.evaluator_model", "qwen3:0.6b")
            # Use original text (without context prefix) for memory evaluation
            eval_text = (message_metadata or {}).get("original_text", user_message)
            async def _deferred_evaluate():
                try:
                    if eval_driver != "ollama":
                        await asyncio.sleep(40)  # CCA rate limit window (~36s)
                    result = await evaluate_and_store(
                        provider=self.provider,
                        memory_engine=self.memory,
                        user_message=eval_text,
                        user_id=self.user.get("id"),
                        evaluator_driver=eval_driver,
                        evaluator_model=eval_model,
                    )
                    logger.debug(f"Evaluator done: {'stored #' + str(result) if result else 'skipped'}")
                except Exception as e:
                    logger.warning(f"Deferred memory evaluation failed: {type(e).__name__}: {e}")
            asyncio.create_task(_deferred_evaluate())

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
        If limit is reached or a tool loop is detected, stops early with a notice.
        Token usage is accumulated across all rounds.
        """
        from .db.models import get_config
        from .tools.loop_detection import ToolLoopDetector

        # Set owner-DM context for file_ops security bypass.
        # Owner identity is platform-verified (Telegram ID), not message content.
        from .tools.file_ops import set_owner_dm
        is_owner_dm = (access_level == "owner" and not self.is_group)
        set_owner_dm(is_owner_dm)

        # Set current user context for scheduler (auto-fill created_by).
        from .tools.scheduler import set_current_user
        user_platform_id = self.user.get("platform_id")
        try:
            set_current_user(int(user_platform_id))
        except (TypeError, ValueError):
            set_current_user(None)

        max_rounds = int(await get_config("session.max_tool_rounds", 100))

        current = response
        limit_reached = False
        loop_stuck = False

        # Loop detection + usage accumulation
        detector = ToolLoopDetector()
        usage = UsageAccumulator()
        usage.add(response)  # Initial response that triggered tool calls

        for round_num in range(max_rounds):
            if not current.tool_calls:
                break

            context.append(ChatMessage(
                role="assistant",
                content=current.content or "",
                metadata={"tool_calls": current.tool_calls},
            ))

            tool_calls_list = current.tool_calls
            for tc_idx, tool_call in enumerate(tool_calls_list):
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

                # ── Loop detection: record + check BEFORE execution ──
                loop_record = detector.record_call(name, args, round_num)
                loop_check = detector.detect()

                if loop_check.stuck:
                    logger.warning(f"Tool loop detected (stuck): {loop_check.message}")
                    # Append synthetic results for remaining tool calls in this batch
                    for skip_tc in tool_calls_list[tc_idx:]:
                        skip_id = skip_tc.get("id")
                        skip_name = skip_tc.get("name", skip_tc.get("function", {}).get("name", ""))
                        skip_meta = {"tool_name": skip_name}
                        if skip_id:
                            skip_meta["tool_call_id"] = skip_id
                        skip_msg = "Skipped: tool loop detected"
                        await self.save_message("tool", skip_msg, metadata=skip_meta)
                        context.append(ChatMessage(role="tool", content=skip_msg, metadata=skip_meta))
                    loop_stuck = True
                    break

                if loop_check.level == "warning":
                    logger.warning(f"Tool loop warning: {loop_check.message}")
                    context.append(ChatMessage(
                        role="system",
                        content=f"WARNING: {loop_check.message}. Try a different approach or stop using this tool.",
                    ))

                # Notify channel about tool activity
                if self._mgr and self._mgr._tool_callback:
                    try:
                        await self._mgr._tool_callback(name)
                    except Exception:
                        pass

                # Check if this is a built-in tool or an ability
                if self.tools.get(name):
                    is_scheduled = bool((self._message_metadata or {}).get("scheduled"))
                    result = await self.tools.execute(
                        name, args, access_level,
                        scheduled=is_scheduled,
                        provider=self.provider,
                    )
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
                        "_registry": self.abilities,  # For call_ability() support
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

                # Record result for loop detection
                detector.record_result(loop_record, str(result))

                # ═══════════════════════════════════════════════════════
                # GLOBAL TOOL RESULT SCRUBBER
                # Each tool declares its own scrub_level:
                #   "aggressive" — full regex (Cookie, PEM, querystring...)
                #   "safe" — high-confidence only (JWT, sk-*, bot tokens)
                #   "none" — tool has its own dedicated scrubber
                # Default = "aggressive" (safest for new tools)
                #
                # BYPASS: Owner DM — owner identity is platform-verified,
                # no scrubbing needed. Owner can see all output.
                # ═══════════════════════════════════════════════════════
                if not is_owner_dm:
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

                # Strip server paths from tool results before LLM sees them
                # (prevents LLM from echoing /home/syne/... paths to users)
                from .communication.outbound import strip_server_paths
                result = strip_server_paths(str(result))

                tool_meta = {"tool_name": name}
                if tool_call_id:
                    tool_meta["tool_call_id"] = tool_call_id
                await self.save_message("tool", result, metadata=tool_meta)
                context.append(ChatMessage(role="tool", content=str(result), metadata=tool_meta))

            # If loop was detected mid-batch, break out of the round loop
            if loop_stuck:
                break

            # Get next response — may contain more tool calls
            current = await self.provider.chat(
                messages=context,
                tools=tool_schemas if tool_schemas else None,
                **self._build_chat_kwargs(),
            )
            usage.add(current)
        else:
            # Loop exhausted without breaking — limit reached
            if current.tool_calls:
                limit_reached = True

        if loop_stuck:
            # Force a final text response with no tools
            context.append(ChatMessage(
                role="system",
                content="STOP. A tool call loop was detected — you keep calling the same tool(s) with identical arguments. Summarize what you've done so far and try a completely different approach if the task is incomplete.",
            ))
            current = await self.provider.chat(
                messages=context,
                tools=None,
                **self._build_chat_kwargs(),
            )
            usage.add(current)

        if limit_reached:
            notice = (
                f"\n\n⚠️ Tool call limit reached ({max_rounds} rounds). "
                f"Some steps may be incomplete. Use /compact or continue the conversation to proceed."
            )
            if current.content:
                current = ChatResponse(
                    content=current.content + notice,
                    model=current.model,
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
                    tools=None,  # No tools — force text response
                    **self._build_chat_kwargs(),
                )
                usage.add(current)
                current = ChatResponse(
                    content=(current.content or "") + notice,
                    model=current.model,
                    tool_calls=None,
                    thinking=current.thinking,
                )

        # Reset context after tool execution
        set_owner_dm(False)
        set_current_user(None)

        logger.info(f"Tool loop done: {usage.rounds} rounds, in={usage.last_input}, out={usage.total_output}")
        return usage.apply_to(current)


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
        self._delivery_callbacks: list = []  # Multi-slot: channels register via add/remove
        self._status_callbacks: list = []  # Multi-slot: channels register via add/remove
        self._tool_callback = None  # Single-slot (CLI only, per-cycle)

        # Wire up sub-agent completion callback
        if self.subagents:
            self.subagents.set_completion_callback(self._on_subagent_complete)

    async def _on_subagent_complete(self, run_id: str, status: str, result: str, parent_session_id: int):
        """Called when a sub-agent finishes. Delivers result to the parent session."""
        if status == "completed":
            msg = f"✅ Sub-agent completed (run: {run_id[:8]})\n\n{result}"
        else:
            msg = f"❌ Sub-agent failed (run: {run_id[:8]})\n\n{result}"

        # Deliver via callbacks (e.g., Telegram, WhatsApp)
        if self._delivery_callbacks:
            for cb in self._delivery_callbacks:
                try:
                    await cb(msg, parent_session_id)
                except Exception as e:
                    logger.error(f"Failed to deliver sub-agent result via {cb}: {e}")
        else:
            logger.info(f"Sub-agent result (no delivery callback): {msg[:200]}")

    def add_delivery_callback(self, callback):
        """Register a callback for delivering sub-agent results.

        callback(message: str, parent_session_id: int) -> None
        Multiple channels can register (Telegram, WhatsApp, etc.).
        """
        if callback not in self._delivery_callbacks:
            self._delivery_callbacks.append(callback)

    def remove_delivery_callback(self, callback):
        """Unregister a delivery callback."""
        try:
            self._delivery_callbacks.remove(callback)
        except ValueError:
            pass

    def add_status_callback(self, callback):
        """Register a callback for status notifications (compaction, etc.).

        callback(session_id: int, message: str) -> None
        Multiple channels can register (Telegram, WhatsApp, CLI, etc.).
        """
        if callback not in self._status_callbacks:
            self._status_callbacks.append(callback)

    def remove_status_callback(self, callback):
        """Unregister a status callback."""
        try:
            self._status_callbacks.remove(callback)
        except ValueError:
            pass

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
        extra_context: str = None,
        inbound: Optional["InboundContext"] = None,
    ) -> Conversation:
        """Get existing conversation or create new one.

        InboundContext is the SINGLE SOURCE OF TRUTH for chat context.
        is_group, chat_name, etc. are all derived from it.

        Args:
            platform: Platform identifier (telegram, etc.)
            chat_id: Chat/conversation ID
            user: User dict with access_level, id, etc.
            extra_context: Optional extra context for system prompt (e.g., CLI cwd)
            inbound: InboundContext with full message metadata

        Returns:
            Conversation instance
        """
        key = self._session_key(platform, chat_id)

        if key in self._active:
            # Update inbound context (may change per message, e.g. different sender in group)
            existing = self._active[key]
            if inbound:
                existing.inbound = inbound
                existing.is_group = inbound.is_group
            return existing

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
            inbound=inbound,
        )

        # ═══════════════════════════════════════════════════════════════
        # Per-group/per-user model override
        # ═══════════════════════════════════════════════════════════════
        conv_provider = self.provider  # default
        if inbound and inbound.is_group and inbound.chat_id:
            from .db.models import get_group
            group = await get_group(platform, str(inbound.chat_id))
            if group:
                group_model = (group.get("settings") or {}).get("model")
                if group_model and hasattr(self, '_agent') and self._agent:
                    override = await self._agent.create_provider_for_model(group_model)
                    if override:
                        conv_provider = override
                        logger.info(f"Group {inbound.chat_id} using model override: {group_model}")

        # Per-user model override (DM only — group model takes precedence)
        if conv_provider is self.provider and user:
            user_model = (user.get("preferences") or {}).get("model")
            if user_model and hasattr(self, '_agent') and self._agent:
                override = await self._agent.create_provider_for_model(user_model)
                if override:
                    conv_provider = override
                    logger.info(f"User {user.get('platform_id')} using model override: {user_model}")

        conv = Conversation(
            provider=conv_provider,
            memory=self.memory,
            tools=self.tools,
            context_mgr=self.context_mgr,
            session_id=session_id,
            user=user,
            system_prompt=system_prompt,
            abilities=self.abilities,
            inbound=inbound,
        )

        conv._mgr = self  # Back-reference for status callbacks

        # Load history EAGERLY — before any chat() call can save a message
        # and make _message_cache non-empty (which would skip load_history
        # in build_context). This is the fix for restart amnesia.
        await conv.load_history()

        # Load per-model LLM params from model registry
        from .db.models import get_config as _get_config
        active_models = await _get_config("provider.models", [])
        # Resolve effective model key: group override > user override > global default
        effective_key = await _get_config("provider.active_model", "gemini-pro")
        if inbound and inbound.is_group and inbound.chat_id:
            from .db.models import get_group as _get_group
            _group = await _get_group(platform, str(inbound.chat_id))
            if _group:
                _gm = (_group.get("settings") or {}).get("model")
                if _gm:
                    effective_key = _gm
        elif user:
            _um = (user.get("preferences") or {}).get("model")
            if _um:
                effective_key = _um
        model_entry = next((m for m in active_models if m.get("key") == effective_key), {})
        conv.model_params = model_entry.get("params") or {}
        # If no params stored, fall back to driver defaults
        if not conv.model_params:
            _driver = model_entry.get("driver", "")
            _defaults = {
                "google_cca":   {"temperature": 0.7, "max_tokens": None, "thinking_budget": -1,    "top_p": 0.95, "top_k": 40,   "frequency_penalty": None, "presence_penalty": None},
                "codex":        {"temperature": 0.7, "max_tokens": None, "thinking_budget": 10000, "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
                "anthropic":    {"temperature": 0.3, "max_tokens": None, "thinking_budget": 32000, "top_p": 0.99, "top_k": 50,   "frequency_penalty": None, "presence_penalty": None},
                "openai_compat":{"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
                "together":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
                "ollama":       {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 0.9,  "top_k": 40,   "frequency_penalty": None, "presence_penalty": None},
                "_default":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
            }
            conv.model_params = _defaults.get(_driver, _defaults["_default"]).copy()
        # Load per-model reasoning_visible
        conv.reasoning_visible = bool(model_entry.get("reasoning_visible", False))

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
                    inbound=conv.inbound,
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
        message_metadata: Optional[dict] = None,
    ) -> str:
        """Handle an incoming message. Returns agent response.

        InboundContext must be present in message_metadata["inbound"].
        All chat context (is_group, chat_name, sender, etc.) comes from it.

        Args:
            platform: Platform identifier
            chat_id: Chat/conversation ID
            user: User dict
            message: Message content
            message_metadata: Metadata dict. Must contain "inbound" (InboundContext).

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

        # Extract InboundContext from metadata (set by channel)
        inbound = message_metadata.get("inbound") if message_metadata else None

        conv = await self.get_or_create_session(
            platform, chat_id, user,
            extra_context=extra_context,
            inbound=inbound,
        )
        response = await conv.chat(message, message_metadata=message_metadata)

        if response is None:
            return None  # Silent drop (lock timeout)

        # Run memory decay (conversation-based, fire-and-forget)
        if self.memory:
            try:
                await self.memory.run_decay()
            except Exception as e:
                import logging
                logging.getLogger("syne.memory").warning(f"Memory decay error: {e}")
        
        return response
