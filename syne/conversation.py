"""Conversation manager — session handling, message history, context building."""

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from typing import Optional


try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

from .db.connection import get_connection
from .llm.provider import LLMProvider, ChatMessage, ChatResponse, UsageAccumulator, StreamCallbacks, LLMContextWindowError, LLMBadRequestError
from .memory.engine import MemoryEngine
from .memory.evaluator import evaluate_and_store
from .context import ContextManager, estimate_messages_tokens, DEFAULT_CHARS_PER_TOKEN
from .compaction import compact_session, _build_preservation_context
from .boot import get_full_prompt
from .tools.registry import ToolRegistry, ToolResult
from .abilities import AbilityRegistry
from .security import (
    get_group_context_restrictions,
    filter_tools_for_group,
    should_filter_tools_for_group,
)

logger = logging.getLogger("syne.conversation")

# Node-side tools — these execute on the remote node, not the server
# Canonical definition is in gateway/protocol.py; import to avoid divergence.
try:
    from .gateway.protocol import NODE_TOOLS as _NODE_TOOLS
except ImportError:
    _NODE_TOOLS = frozenset({"exec", "file_read", "file_write", "read_source"})

# ── Time context helpers (module-level, not recreated per call) ──

_TIME_CFG_KEYS = ['system.timezone', 'time.locale', 'time.format.full']
_TIME_CFG_DEFAULTS = {
    'system.timezone': 'UTC',
    'time.locale': 'id',
    'time.format.full': '{day_name}, {date} {time}',
}

_ID_DAYS = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
_ID_MONTHS = [
    'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember',
]
_EN_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
_EN_MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]


def _resolve_tz(tz_name: str):
    tz_name = (tz_name or 'UTC').strip() if isinstance(tz_name, str) else 'UTC'
    if ZoneInfo:
        try:
            return ZoneInfo(tz_name), tz_name
        except Exception:
            return ZoneInfo('UTC'), 'UTC'
    return timezone.utc, 'UTC'


def _fmt_offset(td: timedelta) -> str:
    if td is None:
        return "+00:00"
    total = int(td.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hh, rem = divmod(total, 3600)
    mm = rem // 60
    return f"{sign}{hh:02d}:{mm:02d}"


def _fmt_components(dt: datetime, locale: str) -> dict:
    loc = (locale or 'id').lower()
    if loc.startswith('id'):
        days, months = _ID_DAYS, _ID_MONTHS
    else:
        days, months = _EN_DAYS, _EN_MONTHS
    day_name = days[dt.weekday()]
    month_name = months[dt.month - 1]
    date_str = f"{dt.day} {month_name} {dt.year}"
    time_str = dt.strftime('%H:%M:%S')
    return {
        'day_name': day_name, 'month_name': month_name,
        'year': dt.year, 'month': dt.month, 'day': dt.day,
        'hour': dt.hour, 'minute': dt.minute, 'second': dt.second,
        'date': date_str, 'time': time_str,
        'full': f"{day_name}, {date_str} {time_str}",
        'iso': dt.isoformat(timespec='seconds'),
    }


def _apply_template(tpl: str, components: dict) -> str:
    try:
        return (tpl or '').format(**components)
    except Exception:
        return components.get('full') or components.get('iso')


async def _load_time_config() -> dict:
    """Fetch all time config in a single DB query (was 6 separate queries)."""
    result = dict(_TIME_CFG_DEFAULTS)
    try:
        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config WHERE key = ANY($1::text[])",
                _TIME_CFG_KEYS,
            )
        for row in rows:
            result[row['key']] = json.loads(row['value'])
    except Exception:
        pass
    return result


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
        self.model_params: dict = {}  # Per-model LLM params (all 8: temperature, max_tokens, thinking_budget, top_p, top_k, frequency_penalty, presence_penalty, chars_per_token)
        self.reasoning_visible: bool = False  # Per-model reasoning visibility
        self.stream_callbacks: Optional[StreamCallbacks] = None  # Set by ConversationManager for CLI streaming
        self._message_cache: list[ChatMessage] = []
        self._processing: bool = False
        self._lock = asyncio.Lock()  # Prevent concurrent chat() on same session
        self._last_saved_hash: str = ""  # Dedup consecutive save_message calls
        self._mgr: Optional["ConversationManager"] = None  # Back-reference, set by manager

    async def run_compact(self) -> Optional[dict]:
        """Single compact implementation — used by auto-compact, manual /compact, and emergency compact.

        Always uses the agent's base provider (via ConversationManager),
        never the conversation's overridden provider.
        """
        _ctx_tokens = self.context_mgr.available
        _keep = max(20, min(200, _ctx_tokens // 5000))
        _recent = self._message_cache[-_keep:] if self._message_cache else []
        _preservation = _build_preservation_context(_recent)

        # Agent's base provider — never conversation override
        _provider = self._mgr.provider if self._mgr else self.provider

        result = await compact_session(
            session_id=self.session_id,
            provider=_provider,
            keep_recent=_keep,
            recent_context=_preservation,
            chars_per_token=self.context_mgr.chars_per_token,
        )
        if result:
            await self.load_history()
        return result

    async def load_history(self) -> list[ChatMessage]:
        """Load recent message history from database for this session.

        Loads at most N messages (from session.history_limit config, default 100).
        Compaction controls total session size; this limits what enters context.
        """
        from .db.models import get_config as _get_config
        limit = await _get_config("session.history_limit", 100)
        if isinstance(limit, str):
            limit = int(limit)

        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, self.session_id, limit)

        # Reverse to chronological order
        rows = list(reversed(rows))

        messages = []
        for row in rows:
            messages.append(ChatMessage(
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            ))

        logger.info(f"load_history: loaded {len(messages)} messages (limit={limit}) for session {self.session_id}")
        self._message_cache = messages
        return messages

    async def save_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Save a message to the database.

        If the session row was deleted externally (e.g. manual reset),
        the INSERT will fail with an FK violation.  We recreate the
        session row and retry once, then mark the conversation for
        eviction from the manager cache so the next call creates a
        fresh Conversation.
        """
        # Strip null bytes — PostgreSQL text columns reject 0x00
        if content and "\x00" in content:
            content = content.replace("\x00", "")

        # Dedup — skip immediate consecutive duplicates
        msg_hash = hashlib.md5(f"{role}:{content}:{json.dumps(metadata, sort_keys=True) if metadata else ''}".encode()).hexdigest()
        if msg_hash == self._last_saved_hash:
            logger.debug(f"Skipping duplicate save: {role}")
            return
        self._last_saved_hash = msg_hash

        meta_json = json.dumps(metadata) if metadata else "{}"

        for attempt in range(2):
            try:
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
                break  # success
            except Exception as e:
                err = str(e).lower()
                if attempt == 0 and ("foreign key" in err or "fk" in err or "messages_session_id_fkey" in err):
                    logger.warning(f"Session {self.session_id} missing from DB, recreating")
                    try:
                        async with get_connection() as conn:
                            await conn.execute("""
                                INSERT INTO sessions (id, user_id, platform, platform_chat_id)
                                VALUES ($1, $2, $3, $4)
                                ON CONFLICT (id) DO NOTHING
                            """, self.session_id, self.user.get("id"),
                                self.inbound.platform if self.inbound else "unknown",
                                self.inbound.chat_id if self.inbound else "unknown",
                            )
                    except Exception as re_err:
                        logger.error(f"Failed to recreate session: {re_err}")
                        raise e
                    continue  # retry the INSERT
                raise

        self._message_cache.append(ChatMessage(role=role, content=content, metadata=metadata))

    async def build_context(self, user_message: str, recall_query: Optional[str] = None) -> list[ChatMessage]:
        """Build full context: system prompt + memories + history + current message.

        Args:
            user_message: Full message (may include context prefix) for history.
            recall_query: Clean text (without prefix) for memory recall. Falls back to user_message.
        """
        messages = []
        access_level = self.user.get("access_level", "public")
        # In group chats, the effective access level for memory is the SENDER (who asked),
        # resolved from the group member registry in InboundContext.sender_access.
        # If not available, fall back to the user record.
        if self.is_group and self.inbound and self.inbound.sender_access:
            access_level = self.inbound.sender_access

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


        # 1b. Runtime time context (ground truth, compact format)
        # Policy: default=SYNE (no tz label), 'server'→SERVER, 'UTC'→UTC
        try:
            cfg = await _load_time_config()
            syne_tz, syne_tz_name = _resolve_tz(cfg['system.timezone'])
            locale = cfg['time.locale']
            fmt_full = cfg['time.format.full']

            # Single base time (UTC) to avoid drift between the three clocks
            base_utc = datetime.now(timezone.utc)
            server_tz = datetime.now().astimezone().tzinfo or timezone.utc
            syne_now = base_utc.astimezone(syne_tz)
            server_now = base_utc.astimezone(server_tz)

            syne_off = _fmt_offset(syne_now.utcoffset())
            server_off = _fmt_offset(server_now.utcoffset())
            same_tz = syne_off == server_off

            syne_c = _fmt_components(syne_now, locale)
            syne_full = _apply_template(fmt_full, syne_c)

            # UTC always English locale
            utc_c = _fmt_components(base_utc, 'en')
            utc_full = _apply_template(fmt_full, utc_c)

            time_lines = [
                '# Current Time (authoritative — DO NOT GUESS)',
                f"SYNE ({syne_tz_name}, UTC{syne_off}): {syne_full} | {syne_c['iso']}",
            ]
            if same_tz:
                time_lines.append("SERVER: same as SYNE")
            else:
                server_c = _fmt_components(server_now, locale)
                server_full = _apply_template(fmt_full, server_c)
                time_lines.append(f"SERVER (UTC{server_off}): {server_full} | {server_c['iso']}")
            time_lines.append(f"UTC: {utc_full} | {utc_c['iso']}")
            time_lines.append("Time policy: default=SYNE (no tz label), 'server'=SERVER+' (Server)', 'UTC'=UTC+' UTC', multiple=answer all")

            messages.append(ChatMessage(role='system', content='\n'.join(time_lines)))
        except Exception as e:
            logger.debug(f"Time context injection failed: {e}")

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

        # 4b. Knowledge graph context (entity-relation traversal)
        try:
            from .memory.graph import recall_graph
            graph_lines = await recall_graph(recall_query or user_message)
            if graph_lines:
                graph_block = "\n".join([
                    "# Knowledge Graph",
                    "Related entities and relationships from stored knowledge.",
                    "",
                ] + graph_lines)
                messages.append(ChatMessage(role="system", content=graph_block))
                logger.info(f"Graph: injected {len(graph_lines)} relations")
        except Exception as e:
            logger.debug(f"Graph recall skipped: {e}")

        # 5. Prune oversized tool results
        # NOTE: user message is already in _message_cache (added by save_message in _chat_inner)
        # Do NOT append it again here — that caused the LLM to see duplicate user messages.
        messages = self.context_mgr.prune_tool_results(messages)

        # 6. Adaptive history reduction — drop oldest 4 non-system messages per
        # iteration until context fits. Gentler than trim_context's emergency drop.
        from .context import estimate_messages_tokens
        _est = estimate_messages_tokens(messages, self.context_mgr.chars_per_token)
        while _est > self.context_mgr.available:
            _new = []
            _skip = 4
            for m in messages:
                if m.role == "system" or _skip <= 0:
                    _new.append(m)
                else:
                    _skip -= 1
            if len(_new) == len(messages):
                break  # nothing left to drop
            messages = _new
            _est = estimate_messages_tokens(messages, self.context_mgr.chars_per_token)
            logger.info(f"Adaptive reduction: {len(messages)} msgs, ~{_est} tokens")

        # Safety net — should rarely trigger after adaptive reduction
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
            # Replace stuck lock with a fresh one — old lock is abandoned
            logger.error(f"Session {self.session_id}: lock stuck for 30s, replacing with new lock")
            self._lock = asyncio.Lock()
            self._processing = False
            await self._lock.acquire()
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

        # Save user message (redact credentials from history if flagged)
        if message_metadata and message_metadata.get("has_credential"):
            from .security import redact_content_output
            await self.save_message("user", redact_content_output(user_message))
        else:
            await self.save_message("user", user_message)

        # Attach media metadata (image/audio/doc) to the cached message for LLM context.
        # This is NOT persisted to DB — only needed for the current turn.
        if message_metadata and self._message_cache:
            media_meta = {}
            for key in ("image", "audio", "document"):
                if key in message_metadata:
                    media_meta[key] = message_metadata[key]
            if media_meta:
                self._message_cache[-1] = ChatMessage(
                    role="user",
                    content=self._message_cache[-1].content,
                    metadata=media_meta,
                )

        # ═══════════════════════════════════════════════════════════════
        # PRE-FLIGHT COMPACTION CHECK
        # Compact BEFORE building context so the LLM never sees a
        # silently-trimmed conversation.  Old flow had compaction after
        # the response, which meant trim_context could silently drop
        # messages and cause amnesia.
        # ═══════════════════════════════════════════════════════════════
        # Compaction gate: use REAL DB totals (not _message_cache which only has last N)
        from .compaction import get_session_stats
        _real_stats = await get_session_stats(self.session_id)
        _real_msg_count = _real_stats["message_count"]
        _real_char_total = _real_stats["total_chars"]

        # Derive thresholds from effective context (minus reserved output tokens)
        _ctx_tokens = self.context_mgr.available
        _msg_thresh = max(100, min(300, _ctx_tokens // 1000))
        _chr_thresh = int(_ctx_tokens * 0.75 * 3.5)
        count_exceeded = _real_msg_count >= _msg_thresh
        chars_exceeded = _real_char_total >= _chr_thresh
        # Also check if cached messages (what LLM will see) fill 90% of context
        context_full = self._message_cache and self.context_mgr.should_compact(
            self._message_cache,
            threshold=0.90,
        )

        if context_full or count_exceeded or chars_exceeded:
            logger.info(f"Compaction triggered for session {self.session_id}: context_full={context_full}, db_msgs={_real_msg_count}/{_msg_thresh}, db_chars={_real_char_total}/{_chr_thresh}")
            if self._mgr and self._mgr._status_callbacks:
                for cb in self._mgr._status_callbacks:
                    try:
                        await cb(
                            self.session_id,
                            "🧹 Compacting memory... please wait a moment."
                        )
                    except Exception as e:
                        logger.debug(f"Status callback failed: {e}")
            try:
                result = await self.run_compact()
            except Exception as e:
                _err_msg = f"❌ Compaction failed: {e}"
                logger.error(f"Auto-compact failed for session {self.session_id}: {e}")
                if self._mgr and self._mgr._status_callbacks:
                    for cb in self._mgr._status_callbacks:
                        try:
                            await cb(self.session_id, _err_msg)
                        except Exception:
                            pass
                result = None
            if result:
                logger.info(
                    f"Auto-compacted: {result['messages_before']} → {result['messages_after']} messages"
                )
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
        
        # In group chats, tool access is based on the SENDER (member registry),
        # not the cached session user.
        if self.is_group and self.inbound and self.inbound.sender_access:
            access_level = self.inbound.sender_access

        # SECURITY (Groups): tools are sender-based, but owner-only tools stay DM-only.
        # We enforce DM-only for owner tools by filtering them out from tool schemas below.
        effective_access_level = access_level
        
        tool_schemas = self.tools.to_openai_schema(effective_access_level)
        if self.abilities:
            tool_schemas = tool_schemas + self.abilities.to_openai_schema(effective_access_level)
        
        # Additional filter: remove owner-only tools entirely in group context
        if self.is_group and should_filter_tools_for_group(self.is_group):
            tool_schemas = filter_tools_for_group(tool_schemas)

        # Call LLM with all params from model registry.
        # Retry once on *empty* non-tool responses — some providers occasionally
        # return "" with no tool_calls, which would otherwise surface as a
        # channel-level fallback message.
        # Google driver already retries empty streams internally — skip conversation-level retry
        chat_kwargs = self._build_chat_kwargs()
        response = None
        max_attempts = 1 if self.provider.name in ("google", "vertex") else 3
        for attempt in range(max_attempts):
            try:
                # Auto-retry on vague 400 errors (e.g. concurrent KG extraction)
                _vague_retries = 3
                for _vague_attempt in range(_vague_retries):
                    try:
                        response = await self.provider.chat(
                            messages=context,
                            tools=tool_schemas if tool_schemas else None,
                            stream_callbacks=self.stream_callbacks,
                            **chat_kwargs,
                        )
                        break  # success
                    except (RuntimeError, LLMBadRequestError) as _re:
                        _msg = str(_re)
                        _is_vague = '"message":"Error"' in _msg or '"message": "Error"' in _msg
                        if _is_vague and _vague_attempt < _vague_retries - 1:
                            logger.warning(f"Vague 400 error, retrying in 2s ({_vague_attempt + 1}/{_vague_retries})")
                            await asyncio.sleep(2.0)
                            continue
                        raise  # not vague or last attempt — let outer handler deal with it
            except LLMContextWindowError:
                # Input tokens exceeded model limit — emergency compact and retry once
                logger.warning(f"Context window exceeded for session {self.session_id}, triggering emergency compaction")
                if self._mgr and self._mgr._status_callbacks:
                    for cb in self._mgr._status_callbacks:
                        try:
                            await cb(self.session_id, "Context limit hit — compacting memory...")
                        except Exception:
                            pass
                result = await self.run_compact()
                if result:
                    logger.info(f"Emergency compaction: {result['messages_before']} → {result['messages_after']} messages")
                    # Rebuild context after compaction
                    context = await self.build_context(user_message, recall_query=recall_query)
                    tool_schemas = self.tools.to_openai_schema(effective_access_level)
                    if self.abilities:
                        tool_schemas = tool_schemas + self.abilities.to_openai_schema(effective_access_level)
                    if self.is_group and should_filter_tools_for_group(self.is_group):
                        tool_schemas = filter_tools_for_group(tool_schemas)
                    # Retry the chat call (no further catch — let it fail if still too big)
                    response = await self.provider.chat(
                        messages=context,
                        tools=tool_schemas if tool_schemas else None,
                        stream_callbacks=self.stream_callbacks,
                        **chat_kwargs,
                    )
                else:
                    raise

            # Check for auth failures (expired OAuth tokens etc.)
            auth_failure = getattr(self.provider, '_auth_failure', None)
            if auth_failure:
                logger.warning(f"Provider auth failure: {auth_failure}")
                # Notify once, then clear
                self.provider._auth_failure = None
                await self.save_message("assistant", f"⚠️ {auth_failure}")
                return f"⚠️ {auth_failure}"

            # Stop retrying if we got tool calls or any non-whitespace content.
            if (response.tool_calls) or ((response.content or "").strip()):
                break

            if attempt < max_attempts - 1:
                logger.warning(
                    f"LLM returned empty content (no tool calls). Retrying after 1s (attempt {attempt + 1}/{max_attempts})..."
                )
                await asyncio.sleep(1.0)

        # Handle tool calls
        logger.info(f"LLM response: content={len(response.content or '')} chars, tool_calls={len(response.tool_calls) if response.tool_calls else 0}, model={response.model}")
        if response.tool_calls:
            logger.info(f"Tool calls: {[tc.get('name') for tc in response.tool_calls]}")
            response = await self._handle_tool_calls(response, context, access_level, tool_schemas)

        # Save assistant response
        await self.save_message("assistant", response.content)

        # Store thinking for the channel to optionally display
        self._last_thinking = response.thinking

        # Store last response for CLI token display
        self._last_chat_response = response

        # Attach any media collected during tool calls to the final response
        final_response = response.content
        if self._pending_media:
            # Strip any "MEDIA:" the LLM may have echoed (it has no valid path)
            import re
            final_response = re.sub(r'\n*MEDIA:\s*\S*', '', final_response).rstrip()
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
                    result = await evaluate_and_store(
                        provider=self.provider,
                        memory_engine=self.memory,
                        user_message=eval_text,
                        user_id=self.user.get("id"),
                        evaluator_driver=eval_driver,
                        evaluator_model=eval_model,
                        speaker_name=self.user.get("display_name") or self.user.get("name", ""),
                    )
                    logger.debug(f"Evaluator done: {'stored #' + str(result) if result else 'skipped'}")
                except Exception as e:
                    logger.warning(f"Deferred memory evaluation failed: {type(e).__name__}: {e}")
            asyncio.create_task(_deferred_evaluate())

        return final_response

    @staticmethod
    def _get_on_demand_guide(tool_name: str, args: dict, injected: set) -> Optional[str]:
        """Return a reference guide to inject when a relevant tool is called.

        Returns None if the guide was already injected this turn or the tool
        doesn't need one.
        """
        guide_key: Optional[str] = None
        guide_text: Optional[str] = None

        if tool_name == "update_config":
            guide_key = "config"
        elif tool_name == "update_ability" and args.get("action") == "create":
            guide_key = "ability_create"
        elif tool_name in ("read_source", "exec"):
            guide_key = "system"

        if not guide_key or guide_key in injected:
            return None

        if guide_key == "config":
            from .config_guide import CONFIG_GUIDE
            guide_text = CONFIG_GUIDE
        elif guide_key == "ability_create":
            from .abilities.ability_guide import get_creation_guide
            guide_text = get_creation_guide()
        elif guide_key == "system":
            from .system_guide import SYSTEM_GUIDE
            guide_text = SYSTEM_GUIDE

        if guide_text:
            injected.add(guide_key)
        return guide_text

    def _get_node_connection(self, node_id: str = ""):
        """Get a remote node connection.

        Args:
            node_id: Specific node to target (for multi-node from Telegram etc.)
                     If empty, falls back to current session's node connection.

        Returns:
            NodeConnection or None.
        """
        if not self._mgr:
            return None

        # Explicit node_id — look up from gateway's connected nodes
        if node_id:
            agent = getattr(self._mgr, '_agent', None)
            if agent and agent.gateway:
                return agent.gateway.get_node(node_id)
            return None

        # Implicit — this session is a node session
        return self._mgr._node_connections.get(self.chat_id)

    async def _execute_tool_on_node(self, node_conn, tool_name: str, args: dict):
        """Execute a tool on the remote node via WebSocket.

        Sends a tool_request to the node and waits for tool_result.
        Returns a ToolResult.
        """
        try:
            result = await node_conn.request_tool(tool_name, args, timeout=120)
            if result.get("success"):
                return ToolResult(str(result.get("result", "")), ok=True)
            else:
                return ToolResult(
                    f"Error (node): {result.get('result', 'Unknown error')}",
                    ok=False, error_type="node_error",
                )
        except asyncio.TimeoutError:
            return ToolResult(
                f"Error: Tool '{tool_name}' timed out on remote node",
                ok=False, error_type="timeout",
            )
        except ConnectionError:
            return ToolResult(
                f"Error: Remote node disconnected while executing '{tool_name}'",
                ok=False, error_type="disconnected",
            )

    async def _handle_tool_calls(
        self,
        response: ChatResponse,
        context: list[ChatMessage],
        access_level: str,
        tool_schemas: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """Execute tool calls and get final response. Loops for multi-step tool use.

        Hard limit of max_rounds (default 30, configurable via session.max_tool_rounds).
        Token usage is accumulated across all rounds.
        """
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

        current = response
        loop_stuck = False
        round_num = 0

        # Hard round limit — prevents infinite tool loops from holding session lock
        from .db.models import get_config as _gc_rounds
        _max_rounds = await _gc_rounds("session.max_tool_rounds", 30)
        if isinstance(_max_rounds, str):
            _max_rounds = int(_max_rounds)

        # Track which on-demand guides have been injected this turn
        _injected_guides: set = set()

        # Loop detection + usage accumulation
        detector = ToolLoopDetector()
        usage = UsageAccumulator()
        usage.add(response)  # Initial response that triggered tool calls

        while current.tool_calls and round_num < _max_rounds:

            context.append(ChatMessage(
                role="assistant",
                content=current.content or "",
                metadata={"tool_calls": current.tool_calls},
            ))

            tool_calls_list = current.tool_calls

            # ── Phase 1: Parse + Loop Detection (sequential, cheap) ──
            parsed_calls = []
            for tc_idx, tool_call in enumerate(tool_calls_list):
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

                loop_record = detector.record_call(name, args, round_num)
                loop_check = detector.detect()

                if loop_check.stuck:
                    logger.warning(f"Tool loop detected (stuck): {loop_check.message}")
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

                # Notify channel about tool activity (before execution for fast typing indicator)
                if self._mgr and self._mgr._tool_callback:
                    try:
                        await self._mgr._tool_callback(name)
                    except Exception:
                        pass

                parsed_calls.append((name, args, tool_call_id, loop_record))

            if loop_stuck:
                break  # Skip Phase 2 & 3

            # ── Phase 2: Execute tools in parallel ──
            is_scheduled = bool((self._message_metadata or {}).get("scheduled"))

            async def _execute_single_tool(t_name, t_args, t_call_id):
                """Execute one tool/ability and return ToolResult."""
                # Auto-inject chat_id for send_reaction if not provided by LLM
                if t_name == "send_reaction" and not t_args.get("chat_id") and self.chat_id:
                    t_args["chat_id"] = str(self.chat_id)

                # Remote node tool routing:
                # 1. Explicit: LLM passes `node` param → route to that specific node
                # 2. Implicit: this is a node CLI session → route to connected node
                target_node_id = t_args.pop("node", "") if t_name in _NODE_TOOLS else ""
                if target_node_id:
                    node_conn = self._get_node_connection(target_node_id)
                    if not node_conn:
                        return ToolResult(
                            f"Error: Node '{target_node_id}' is not connected. "
                            f"Use node_status to check online nodes.",
                            ok=False, error_type="node_offline",
                        )
                    return await self._execute_tool_on_node(node_conn, t_name, t_args)
                node_conn = self._get_node_connection()
                if node_conn and t_name in _NODE_TOOLS:
                    return await self._execute_tool_on_node(node_conn, t_name, t_args)

                if self.tools.get(t_name):
                    return await self.tools.execute(
                        t_name, t_args, access_level,
                        scheduled=is_scheduled,
                        provider=self.provider,
                    )
                elif self.abilities and self.abilities.get(t_name):
                    cached = getattr(self, '_cached_input_data', {})
                    if cached:
                        registered = self.abilities.get(t_name)
                        if registered and registered.instance:
                            for itype, idata in cached.items():
                                if registered.instance.handles_input_type(itype):
                                    if itype == "image":
                                        if not t_args.get("image_base64") and not t_args.get("image_url"):
                                            t_args["image_base64"] = idata.get("base64", "")
                                            t_args.setdefault("mime_type", idata.get("mime_type", "image/jpeg"))
                                    elif itype == "audio":
                                        if not t_args.get("audio_base64"):
                                            t_args["audio_base64"] = idata.get("base64", "")
                                            t_args.setdefault("mime_type", idata.get("mime_type", "audio/ogg"))
                                    elif itype == "document":
                                        if not t_args.get("document_base64"):
                                            t_args["document_base64"] = idata.get("base64", "")
                                            t_args.setdefault("mime_type", idata.get("mime_type", "application/pdf"))
                                    logger.debug(f"Injected cached {itype} data into ability '{t_name}' tool call")
                                    break

                    ability_context = {
                        "user_id": self.user.get("id"),
                        "session_id": self.session_id,
                        "access_level": access_level,
                        "config": self.abilities.get(t_name).config or {},
                        "workspace": getattr(self._mgr, 'workspace_outputs', None) if self._mgr else None,
                        "_registry": self.abilities,
                    }
                    ability_result = await self.abilities.execute(t_name, t_args, ability_context)
                    if ability_result.get("success"):
                        content = ability_result.get("result", "")
                        if ability_result.get("media"):
                            media_path = ability_result["media"]
                            if hasattr(self, '_pending_media'):
                                self._pending_media.append(media_path)
                        return ToolResult(str(content), ok=True)
                    else:
                        return ToolResult(f"Error: {ability_result.get('error', 'Unknown error')}", ok=False, error_type="unknown")
                else:
                    return ToolResult(f"Error: Unknown tool or ability '{t_name}'", ok=False, error_type="not_found")

            tasks = [
                _execute_single_tool(name, args, tc_id)
                for name, args, tc_id, _ in parsed_calls
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Wrap exceptions as ToolResult
            results = []
            for r in raw_results:
                if isinstance(r, Exception):
                    results.append(ToolResult(f"Error: {r}", ok=False, error_type="unknown"))
                elif isinstance(r, ToolResult):
                    results.append(r)
                else:
                    # Legacy str return from abilities
                    results.append(ToolResult(str(r), ok=True))

            # ── Automatic retry for retryable failures (max 1 per tool per round) ──
            for i, result in enumerate(results):
                if result.retryable:
                    name, args, tc_id, _ = parsed_calls[i]
                    logger.info(f"Retrying retryable tool '{name}' (error_type={result.error_type})")
                    try:
                        retry_result = await _execute_single_tool(name, args, tc_id)
                        if isinstance(retry_result, ToolResult):
                            results[i] = retry_result
                        else:
                            results[i] = ToolResult(str(retry_result), ok=True)
                    except Exception as e:
                        results[i] = ToolResult(f"Error (retry failed): {e}", ok=False, error_type="unknown")

            # ── Phase 3: Post-process results (sequential) ──
            from .security import redact_content_output, redact_secrets_in_text
            from .communication.outbound import strip_server_paths

            for i, result in enumerate(results):
                name, args, tool_call_id, loop_record = parsed_calls[i]

                # Record result for loop detection
                detector.record_result(loop_record, result.content)

                # Inject system hint for permanent failures
                if not result.ok and not result.retryable:
                    context.append(ChatMessage(
                        role="system",
                        content=f"Tool '{name}' failed permanently ({result.error_type}). Try an alternative approach.",
                    ))

                # Anti-hallucination: flag exec results with non-zero exit code
                # The LLM sometimes claims success despite error output.
                if name == "exec" and result.ok:
                    import re as _re
                    _ec_match = _re.search(r'exit_code:\s*(\d+)', result.content)
                    if _ec_match and _ec_match.group(1) != "0":
                        context.append(ChatMessage(
                            role="system",
                            content=(
                                f"IMPORTANT: The exec command exited with code {_ec_match.group(1)} "
                                f"(non-zero = error). Report the ACTUAL output to the user. "
                                f"Do NOT claim the command succeeded."
                            ),
                        ))

                result_str = result.content

                # ═══════════════════════════════════════════════════════
                # GLOBAL TOOL RESULT SCRUBBER
                # BYPASS: Owner DM — owner identity is platform-verified
                # ═══════════════════════════════════════════════════════
                if not is_owner_dm:
                    tool_obj = self.tools.get(name)
                    scrub = tool_obj.scrub_level if tool_obj else "aggressive"
                    if scrub == "none":
                        pass
                    elif scrub == "safe":
                        result_str = redact_content_output(result_str)
                    else:
                        result_str = redact_secrets_in_text(result_str)

                # Collect MEDIA: from tool results and strip from result
                if "\n\nMEDIA: " in result_str or result_str.startswith("MEDIA: "):
                    if "\n\nMEDIA: " in result_str:
                        media_path = result_str.rsplit("\n\nMEDIA: ", 1)[1].strip()
                        result_str = result_str.rsplit("\n\nMEDIA: ", 1)[0]
                    else:
                        media_path = result_str[7:].strip()
                        result_str = ""
                    if media_path and hasattr(self, '_pending_media'):
                        self._pending_media.append(media_path)

                # Strip server paths
                result_str = strip_server_paths(result_str)

                # Notify CLI about tool execution details
                if self._mgr and self._mgr._tool_detail_callback:
                    try:
                        preview = result_str[:200]
                        await self._mgr._tool_detail_callback(name, args, preview)
                    except Exception:
                        pass

                tool_meta = {"tool_name": name}
                if tool_call_id:
                    tool_meta["tool_call_id"] = tool_call_id
                await self.save_message("tool", result_str, metadata=tool_meta)
                context.append(ChatMessage(role="tool", content=result_str, metadata=tool_meta))

            # ── On-demand guide injection ──
            # Inject reference guides as system messages when relevant tools are called.
            # Each guide is injected at most once per turn to avoid bloating context.
            for name, args, _, _ in parsed_calls:
                guide = self._get_on_demand_guide(name, args, _injected_guides)
                if guide:
                    context.append(ChatMessage(role="system", content=guide))

                # Anti-hallucination: after spawning a sub-agent, inject hard constraint
                if name == "spawn_subagent":
                    context.append(ChatMessage(
                        role="system",
                        content=(
                            "CRITICAL: A sub-agent has been spawned and is running in the background. "
                            "You do NOT know its progress or results yet. "
                            "NEVER claim the task is done, report numbers (e.g. '110 processed'), "
                            "or fabricate progress. Only say: the task has been delegated and the user "
                            "will be notified when it completes. If the user asks for progress, "
                            "use the subagent_status tool to check — do NOT guess."
                        ),
                    ))

            # Small delay between tool call rounds to avoid rate limiting
            await asyncio.sleep(1.0)

            # Get next response — may contain more tool calls
            # Auto-retry on vague 400 errors
            for _vague_attempt in range(3):
                try:
                    current = await self.provider.chat(
                        messages=context,
                        tools=tool_schemas if tool_schemas else None,
                        stream_callbacks=self.stream_callbacks,
                        **self._build_chat_kwargs(),
                    )
                    break
                except (RuntimeError, LLMBadRequestError) as _re:
                    _msg = str(_re)
                    _is_vague = '"message":"Error"' in _msg or '"message": "Error"' in _msg
                    if _is_vague and _vague_attempt < 2:
                        logger.warning(f"Vague 400 in tool loop, retrying in 2s ({_vague_attempt + 1}/3)")
                        await asyncio.sleep(2.0)
                        continue
                    raise
            usage.add(current)
            round_num += 1

        if loop_stuck or (round_num >= _max_rounds and current.tool_calls):
            # Force a final text response with no tools
            reason = "tool call loop detected" if loop_stuck else f"max tool rounds ({_max_rounds}) reached"
            logger.warning(f"Forcing final response: {reason}")
            context.append(ChatMessage(
                role="system",
                content=f"STOP. {reason.capitalize()}. Summarize what you've done so far and respond to the user with what you have.",
            ))
            current = await self.provider.chat(
                messages=context,
                tools=None,
                stream_callbacks=self.stream_callbacks,
                **self._build_chat_kwargs(),
            )
            usage.add(current)

        # Reset context after tool execution
        set_owner_dm(False)
        set_current_user(None)

        # If final response is empty (e.g. thinking-only), use fallback
        if not (current.content or "").strip():
            fallback = f"[Completed {usage.rounds} tool rounds but could not generate a text response. Please try asking again.]"
            current = ChatResponse(
                content=fallback,
                model=current.model,
                input_tokens=current.input_tokens,
                output_tokens=current.output_tokens,
                tool_calls=None,
                thinking=current.thinking,
            )

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
        
        # Use original_text (without channel instructions) as ability prompt
        ability_prompt = metadata.get("original_text") or user_message

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
                        input_type, input_data, ability_prompt,
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
            else:
                # Ability failed — strip raw input, inject error so user knows
                logger.warning(f"{spec['label']} failed — no silent fallback")
                user_message = f"{user_message}\n\n[{spec['label']} failed. Tell the user the analysis could not be completed and to try again later.]"
                metadata = {k: v for k, v in metadata.items() if k != spec["meta_key"]}
                self._message_metadata = metadata if metadata else None
        
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
        self._stream_callbacks: Optional[StreamCallbacks] = None  # CLI streaming
        self._tool_detail_callback = None  # async (name, args, result_preview) — CLI tool display
        self._node_connections: dict[str, object] = {}  # chat_id → NodeConnection (remote nodes)

        # Wire up sub-agent completion callback
        if self.subagents:
            self.subagents.set_completion_callback(self._on_subagent_complete)

    def is_any_chat_active(self) -> bool:
        """Check if any conversation is currently processing a chat request."""
        return any(conv._processing for conv in self._active.values())

    async def _on_subagent_complete(self, run_id: str, status: str, result: str, parent_session_id: int):
        """Called when a sub-agent finishes. Delivers result to the parent session."""
        if status == "completed":
            msg = f"✅ Sub-agent completed (run: {run_id[:8]})\n\n{result}"
        else:
            msg = f"❌ Sub-agent failed (run: {run_id[:8]})\n\n{result}"

        # Save to conversation history so the LLM knows the sub-agent finished.
        # Without this, the bot has no memory of sub-agent results and can't
        # answer questions about what happened.
        conv = self._find_conversation_by_session(parent_session_id)
        if conv:
            # Truncate for history — execution reports can be long
            history_msg = msg[:2000] if len(msg) > 2000 else msg
            try:
                await conv.save_message("system", history_msg, metadata={
                    "type": "subagent_result",
                    "run_id": run_id,
                    "status": status,
                })
            except Exception as e:
                logger.error(f"Failed to save sub-agent result to history: {e}")

        # Deliver via callbacks (e.g., Telegram, WhatsApp)
        if self._delivery_callbacks:
            for cb in self._delivery_callbacks:
                try:
                    await cb(msg, parent_session_id)
                except Exception as e:
                    logger.error(f"Failed to deliver sub-agent result via {cb}: {e}")
        else:
            logger.info(f"Sub-agent result (no delivery callback): {msg[:200]}")

    def _find_conversation_by_session(self, session_id: int) -> Optional[Conversation]:
        """Find an active conversation by session ID."""
        for conv in self._active.values():
            if conv.session_id == session_id:
                return conv
        return None

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

    def set_stream_callbacks(self, callbacks: Optional[StreamCallbacks]):
        """Set streaming callbacks (CLI only). Applied to conversations in handle_message."""
        self._stream_callbacks = callbacks

    def set_tool_detail_callback(self, callback):
        """Set callback for detailed tool execution display.

        callback(tool_name: str, args: dict, result_preview: str) -> Awaitable
        """
        self._tool_detail_callback = callback

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

        # With Limit N history loading, only last N messages are loaded.
        # Auto-compact on session load is no longer needed — the pre-flight
        # compaction check in _chat_inner uses real DB totals.

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
                "google_cca":   {"temperature": 0.7, "max_tokens": None, "thinking_budget": -1,    "top_p": 0.95, "top_k": 40,   "frequency_penalty": None, "presence_penalty": None, "chars_per_token": 4.0},
                "codex":        {"temperature": 0.7, "max_tokens": None, "thinking_budget": 10000, "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0,    "chars_per_token": 4.0},
                "anthropic":    {"temperature": 0.3, "max_tokens": None, "thinking_budget": 32000, "top_p": 0.99, "top_k": 50,   "frequency_penalty": None, "presence_penalty": None, "chars_per_token": 4.0},
                "openai_compat":{"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0,    "chars_per_token": 4.0},
                "together":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0,    "chars_per_token": 4.0},
                "ollama":       {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 0.9,  "top_k": 40,   "frequency_penalty": None, "presence_penalty": None, "chars_per_token": 4.0},
                "_default":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0,    "chars_per_token": 4.0},
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

        # Per-message WA model override (applies even on cached sessions)
        wa_override = (message_metadata or {}).get("wa_model_override")
        if wa_override and hasattr(self, '_agent') and self._agent:
            override_provider = await self._agent.create_provider_for_model(wa_override)
            if override_provider:
                conv.provider = override_provider
                # Load matching model_params + reasoning_visible
                from .db.models import get_config as _gc_wa
                _wa_models = await _gc_wa("provider.models", [])
                _wa_entry = next((m for m in _wa_models if m.get("key") == wa_override), {})
                conv.model_params = _wa_entry.get("params") or conv.model_params
                conv.reasoning_visible = bool(_wa_entry.get("reasoning_visible", False))
                # Update context_mgr to match override model's context_window
                _ctx = int(_wa_entry.get("context_window", 0)) or override_provider.context_window
                _reserved = override_provider.reserved_output_tokens
                _cpt_wa = float(conv.model_params.get("chars_per_token", 0)) or DEFAULT_CHARS_PER_TOKEN
                conv.context_mgr = ContextManager(max_context_tokens=_ctx, reserved_output_tokens=_reserved, chars_per_token=_cpt_wa)
                logger.info(f"WA model override for {chat_id}: {wa_override} (ctx={_ctx}, cpt={_cpt_wa})")

        # Per-node model override (remote node with custom model)
        node_override = (message_metadata or {}).get("node_model_override")
        if node_override and hasattr(self, '_agent') and self._agent:
            override_provider = await self._agent.create_provider_for_model(node_override)
            if override_provider:
                conv.provider = override_provider
                from .db.models import get_config as _gc_node
                _node_models = await _gc_node("provider.models", [])
                _node_entry = next((m for m in _node_models if m.get("key") == node_override), {})
                conv.model_params = _node_entry.get("params") or conv.model_params
                conv.reasoning_visible = bool(_node_entry.get("reasoning_visible", False))
                # Update context_mgr to match override model's context_window
                _ctx = int(_node_entry.get("context_window", 0)) or override_provider.context_window
                _reserved = override_provider.reserved_output_tokens
                _cpt_node = float(conv.model_params.get("chars_per_token", 0)) or DEFAULT_CHARS_PER_TOKEN
                conv.context_mgr = ContextManager(max_context_tokens=_ctx, reserved_output_tokens=_reserved, chars_per_token=_cpt_node)
                logger.info(f"Node model override for {chat_id}: {node_override} (ctx={_ctx}, cpt={_cpt_node})")

        # Apply streaming callbacks (CLI only — None for Telegram/WA)
        conv.stream_callbacks = self._stream_callbacks

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
