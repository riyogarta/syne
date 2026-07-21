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
import re as _re

# Tools/abilities that pull UNTRUSTED external content into the turn. When any
# of these runs mid-turn, the turn is tainted → the consent gate stops skipping
# for owner/family. Provenance defense: content fetched from web/file/image
# must never silently drive a destructive act. See _detect_untrusted (input
# side) and the taint-set after tool/ability dispatch (mid-turn side).
_UNTRUSTED_TOOLS = frozenset({
    "web_search", "fetch_url", "website_screenshot", "image_analysis",
    "file_read", "pdf", "office",
})


_NON_ANTHROPIC_GUARDRAILS = """

# Accuracy & Tool Discipline (CRITICAL — VIOLATIONS = WRONG ANSWERS)

## NEVER CLAIM ACTIONS YOU DID NOT PERFORM
- NEVER say "I saved", "I wrote", "I ran", "I executed", "I searched", "I stored" unless you ACTUALLY called the corresponding tool AND received a success response.
- If you did NOT call a tool, you did NOT perform the action. Period.
- Saying "done" without a tool call is LYING. The user will check and you will be caught.

## TOOL RESULTS ARE SACRED
- When a tool returns a result, report ONLY what the tool returned. Do not add, embellish, or supplement with information from your training data.
- If web_search returns 3 results, report those 3 results. Do not add a 4th from your imagination.
- If memory_search returns nothing, say "no memories found". Do not fabricate memories.
- If a tool fails, say it failed. Do not invent an alternative result.

## FACTS MUST COME FROM TOOLS
- NEVER answer factual questions (prices, dates, events, statistics, addresses, phone numbers) from training data alone. Use web_search first.
- If you cannot use a tool, say "I don't have that information" instead of guessing.
- Your training data may be outdated or wrong. Tools give real-time truth.

## DO NOT FABRICATE
- Never invent URLs, email addresses, phone numbers, prices, or statistics.
- Never claim a file exists unless file_read confirmed it.
- Never claim data is in the database unless db_query confirmed it.
- When uncertain, say "I'm not sure" — never present uncertainty as fact.
"""

# Regex for detecting phantom action claims (used post-response). Three
# independent branches, any hit ⇒ phantom claim:
#
#   (a) Claim marker + action verb, ≤40 chars apart. ID (sudah/telah/
#       berhasil) + EN (I have/I've/I already/successfully/done). Verb
#       list is a stem/short form so it catches inflected variants
#       (save/saved/saving; delet-ed/-ing). Expanded with dev/ops verbs
#       (push/pull/commit/install/deploy/…) — the actions Syne actually
#       gets asked to perform and is most likely to phantom-claim.
#
#   (b) Indonesian passive: `ter-<verb>` intrinsically claims completion,
#       no marker needed ("Tersimpan di memori."). Vocabulary is narrow
#       on purpose — only ter- forms where the prefix unambiguously means
#       'action performed' (never a benign adjective/preposition like
#       terima, terlihat, termasuk, terjadi, terlalu).
#
#   (c) English standalone participle at end of clause. Requires trailing
#       .!, or end-of-string via lookahead, so 'committed to helping you'
#       (no punctuation after 'committed') does NOT match. Catches terse
#       status reports like "Message sent." / "File saved." / "Deployed!"
#
# Note: current runtime gates the entire guard with `tools_ran_this_turn`
# in Conversation, so a broader regex here CAN'T fire on a legitimate
# tool-result summary — it still needs 'no tool ran this turn' to trigger.
_PHANTOM_ACTION_RE = _re.compile(
    r"(?:sudah|telah|berhasil|sudah saya|sudah ku|I have|I've|I already|successfully|done)"
    r"[^.!?\n]{0,40}"
    r"(?:simpan|tulis|jalankan|eksekusi|kirim|hapus|buat|catat|proses|unggah|unduh"
    r"|save|writ|ran|execut|sent|delet|creat|search|stor"
    r"|upload|download|commit|push|pull|install|deploy|publish|restart|start|stop)"
    r"|"
    r"\bter(?:simpan|kirim|hapus|instal(?:l)?|update|catat|jalankan|proses|unggah|unduh|upload|download|selesaikan)\b"
    r"|"
    r"\b(?:sent|saved|deleted|created|stored|executed|uploaded|downloaded|installed|deployed|published|restarted|committed|pushed|completed)(?=\s*[.!,]|\s*$)",
    _re.IGNORECASE,
)


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
    _NODE_TOOLS = frozenset({"exec", "shell", "file_read", "file_write", "read_source"})

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
        # Consent-system pending state — populated by consent.check_and_hold
        # when a tool/ability call is held pending confirmation. The
        # deterministic bypass at the top of chat() checks these on every turn
        # and, if the user replied ya/yes, grants + dispatches the pending
        # call WITHOUT invoking the LLM. Immune to hallucination/injection:
        # the LLM never sees or decides on the confirmation itself.
        self._pending_consent_kind: Optional[str] = None       # "tool" | "ability"
        self._pending_consent_op: Optional[str] = None         # "x" | "w"
        self._pending_consent_tool: Optional[str] = None       # tool/ability name
        self._pending_consent_args: dict = {}                  # full args dict
        self._pending_consent_hash: str = ""                   # sha256[:12] of payload
        self._pending_consent_at: float = 0.0                  # set-time
        # Per-turn provenance taint. True when the current turn's INPUT carries
        # untrusted external content (image / uploaded file / URL in text) OR an
        # untrusted tool (web_search, fetch_url, file_read, pdf/office read,
        # image_analysis, website_screenshot) runs mid-turn. Read by the consent
        # gate: a CLEAN turn from owner/family skips the gate (conscious direct
        # command); a tainted turn still requires the Yes button so injection
        # from web/file/image content cannot silently trigger a destructive act.
        self._turn_untrusted: bool = False

    @staticmethod
    def _detect_untrusted(user_message, message_metadata) -> bool:
        """Provenance check on a turn's INPUT: does it carry untrusted external
        content the owner did not consciously type?

        True when EITHER:
          - metadata carries an attachment of ANY kind: image(s), an uploaded
            document/file (any extension — pdf, docx, xls, csv, txt, md, ...),
            detected via the `image`/`images`/`document` keys or the
            `_temp_upload_paths` staging list; OR
          - the text contains a URL (http(s):// or www.).

        Voice/audio is the owner's own transcribed speech — a conscious command,
        NOT untrusted — so it is deliberately excluded.
        """
        meta = message_metadata or {}
        if (meta.get("image") or meta.get("images") or meta.get("document")
                or meta.get("_temp_upload_paths")):
            return True
        import re as _re
        if _re.search(r'https?://|www\.', user_message or '', _re.IGNORECASE):
            return True
        return False

    async def run_compact(self) -> Optional[dict]:
        """Single compact implementation — used by auto-compact, manual /compact, and emergency compact.

        Uses the conversation's own provider (same model as chat).
        """
        _ctx_tokens = self.context_mgr.available
        from .db.models import get_config as _gc_hl
        _history_limit = await _gc_hl("session.history_limit", 100)
        if isinstance(_history_limit, str):
            _history_limit = int(_history_limit)
        # keep_recent must be less than history_limit so resume + recent all fit in loaded history
        _keep = max(20, min(_history_limit - 10, _ctx_tokens // 5000))
        _recent = self._message_cache[-_keep:] if self._message_cache else []
        _preservation = _build_preservation_context(_recent)

        _provider = self.provider

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
                WHERE session_id = $1 AND status = 'active'
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

        inserted_id: Optional[int] = None
        for attempt in range(2):
            try:
                async with get_connection() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO messages (session_id, role, content, metadata)
                        VALUES ($1, $2, $3, $4::jsonb)
                        RETURNING id
                    """, self.session_id, role, content, meta_json)
                    inserted_id = row["id"] if row else None

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

        # Fire-and-forget: embed user rows as anchors for history_search.
        # Only user role — see design note in migration v20 for rationale
        # (single canonical anchor per topic, 3-5x storage saving, reads
        # around anchor recover the assistant/tool context anyway).
        # Compaction_summary rows are role='system' so they're excluded
        # by the role check — no metadata inspection needed.
        if role == "user" and inserted_id is not None and content:
            asyncio.create_task(self._embed_message_row(inserted_id, content))

    async def _embed_message_row(self, message_id: int, content: str) -> None:
        """Background task: embed a user-role message and UPDATE messages.embedding.

        Fail-quiet: any error (kill switch off, provider missing, embed call
        fails, DB write fails) is logged at info/warning and swallowed. The
        content row is already durably stored; a NULL embedding just means
        this row is invisible to history_search until the backfill CLI
        (`syne memory reembed-history`) picks it up. Never let embed
        failures surface as user-visible chat errors.
        """
        try:
            from .db.models import get_config
            if not bool(await get_config("messages.embedding_enabled", True)):
                return
            if self.memory is None or getattr(self.memory, "provider", None) is None:
                logger.debug(
                    f"_embed_message_row skipped id={message_id}: no memory/provider"
                )
                return

            max_chars = int(await get_config("history_search.max_content_chars", 4000))
            body = content if len(content) <= max_chars else content[:max_chars]
            resp = await self.memory.provider.embed(body)
            vector = getattr(resp, "vector", None)
            if not vector:
                logger.warning(
                    f"_embed_message_row id={message_id}: provider returned empty vector"
                )
                return

            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE messages SET embedding = $1::vector WHERE id = $2",
                    str(vector), message_id,
                )
            logger.debug(
                f"_embed_message_row id={message_id}: embedded ({len(vector)} dims)"
            )
        except Exception as e:
            # Do NOT re-raise from a fire-and-forget task — an uncaught
            # exception would show up as 'Task exception was never retrieved'
            # in the log without any actionable context.
            logger.warning(f"_embed_message_row id={message_id} failed: {e}")

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
        # PROVIDER-SPECIFIC: Anti-hallucination for non-Anthropic models
        # Gemini/Vertex and GPT tend to hallucinate facts and ignore
        # tool-calling instructions more than Claude. These extra rules
        # reduce (not eliminate) that tendency.
        # ═══════════════════════════════════════════════════════════════
        _pname = getattr(self.provider, 'name', '')
        if _pname and 'anthropic' not in _pname:
            prompt += _NON_ANTHROPIC_GUARDRAILS

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

        # 2b. Run memory recall + graph recall IN PARALLEL (saves ~400ms)
        from .memory.graph import recall_graph as _recall_graph

        async def _do_memory_recall():
            return await self.memory.recall(
                query=recall_query or user_message,
                limit=recall_limit,
                user_id=self.user.get("id"),
                requester_access_level=access_level,
            )

        async def _do_graph_recall():
            try:
                return await _recall_graph(recall_query or user_message)
            except Exception as e:
                logger.debug(f"Graph recall skipped: {e}")
                return []

        memories, graph_lines = await asyncio.gather(
            _do_memory_recall(), _do_graph_recall()
        )

        # 3. Conversation history
        if not self._message_cache:
            await self.load_history()
        messages.extend(self._message_cache)

        # 4. Inject recalled memories AFTER history, close to the user message.
        #    This positioning ensures the LLM "sees" memories near the question,
        #    preventing long conversation history from drowning out memory context.
        if memories:
            from .memory.engine import format_relative_time
            # Pick locale from time.locale config (same source as the time context
            # block above). Falls back to 'id' to keep prior behavior intact when
            # config is unreachable.
            try:
                _mem_locale = (await _load_time_config())['time.locale']
            except Exception:
                _mem_locale = 'id'
            memory_lines = [
                "# Relevant Memories",
                "Your stored facts about the owner and their world. Use these to answer questions.",
                "Each line includes when the memory was stored so you can tell recent events from old ones.",
                "",
            ]
            for mem in memories:
                score = f"(confidence: {mem['similarity']:.0%})"
                rel = format_relative_time(mem.get("created_at"), locale=_mem_locale)
                when = f" [{rel}]" if rel else ""
                # Conflict flags injected by memory engine (code-enforced)
                conflict_status = mem.get("_conflict_status", "")
                if conflict_status == "conflicted":
                    ref_id = mem.get("_conflicts_with", "?")
                    flag = f" ⚠️ CONFLICTED — superseded by memory #{ref_id}"
                elif conflict_status == "authoritative":
                    flag = " ✅ AUTHORITATIVE"
                else:
                    flag = ""
                memory_lines.append(f"- [{mem['category']}]{when} {mem['content']} {score}{flag}")
            messages.append(ChatMessage(role="system", content="\n".join(memory_lines)))
            # Log recalled memories for debugging
            _actual_query = recall_query or user_message
            logger.info(f"Recalled {len(memories)} memories for query: {_actual_query[:80]}")
            for mem in memories[:3]:
                logger.info(f"  Memory #{mem['id']} (sim={mem['similarity']:.3f}): {mem['content'][:60]}")
        else:
            _actual_query = recall_query or user_message
            logger.info(f"No memories recalled for query: {_actual_query[:80]}")

        # 4b. Knowledge graph context (already fetched in parallel above)
        if graph_lines:
            graph_block = "\n".join([
                "# Knowledge Graph",
                "Related entities and relationships from stored knowledge.",
                "",
            ] + graph_lines)
            messages.append(ChatMessage(role="system", content=graph_block))
            logger.info(f"Graph: injected {len(graph_lines)} relations")

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

    def _clear_pending_consent(self) -> None:
        """Drop any pending consent state on this conversation."""
        self._pending_consent_kind = None
        self._pending_consent_op = None
        self._pending_consent_tool = None
        self._pending_consent_args = {}
        self._pending_consent_hash = ""
        self._pending_consent_at = 0.0

    async def _maybe_execute_pending_consent(
        self, user_message: str,
    ) -> Optional[str]:
        """Deterministic consent-confirmation bypass — LLM-free.

        If a pending consent (any tool or ability) exists on this conversation
        AND the current user message is a strict "ya"/"yes" (after stripping
        channel metadata), we:
          1. Grant the ConsentStore key for the pending (tool, args).
          2. Dispatch through the correct registry (tools or abilities) with
             conv=self so the gate short-circuits on the just-granted key.
          3. Save the turn (user reply + tool result) to the transcript.
          4. Return the result to the caller.

        The LLM is NEVER invoked for this turn — that's the whole point. Every
        one of the deny paths below returns None so control falls back to the
        normal LLM flow:
          - no pending consent
          - pending older than 120 s (matches other confirm TTLs)
          - reply not exactly ya/yes
          - security.consent_enabled False
          - agent/manager unreachable
          - unknown pending kind
        """
        import time as _time

        pending_kind = self._pending_consent_kind
        pending_op = self._pending_consent_op
        pending_tool = self._pending_consent_tool
        pending_args = self._pending_consent_args or {}
        pending_hash = self._pending_consent_hash
        pending_at = self._pending_consent_at
        if not pending_kind or not pending_tool or not pending_at:
            logger.debug(
                f"Consent bypass skip (no pending): kind={pending_kind}, "
                f"tool={pending_tool}, at={pending_at}"
            )
            return None
        # TTL 120s — same as sudo-guard, keeps confirm windows uniform.
        _age = _time.time() - pending_at
        if _age > 120:
            logger.info(
                f"Consent bypass: pending expired (age={_age:.1f}s > 120s), "
                f"tool={pending_tool}, hash={pending_hash}"
            )
            self._clear_pending_consent()
            return None

        from .consent import (
            last_reply_token, make_key, canonical_payload, DEFAULT_CONSENT_ENABLED,
        )
        token = last_reply_token(user_message)
        if token not in ("ya", "yes"):
            logger.debug(
                f"Consent bypass skip: token={token!r} (raw={user_message[:80]!r})"
            )
            return None
        logger.info(
            f"Consent bypass firing: tool={pending_tool}, hash={pending_hash}, "
            f"token={token!r}, session={self.session_id}"
        )

        # Feature flag last so a lingering pending on a consent-off instance
        # doesn't accidentally intercept legit user text.
        try:
            from .db.models import get_config as _get_config
            enabled = await _get_config("security.consent_enabled", DEFAULT_CONSENT_ENABLED)
        except Exception:
            enabled = DEFAULT_CONSENT_ENABLED
        if not enabled:
            return None

        agent = getattr(self._mgr, "_agent", None) if self._mgr else None
        if agent is None:
            logger.error(
                f"Consent bypass: no agent ref via manager for session {self.session_id}; "
                "clearing pending and falling back to LLM path"
            )
            self._clear_pending_consent()
            return None

        access_level = self.user.get("access_level", "public")
        sid = str(self.session_id)
        uid = str(self.user.get("id", ""))

        # Grant the exact key BEFORE dispatch so the gate in the registry sees
        # an existing grant and short-circuits to the run path — no recursive
        # "held" prompt.
        payload = canonical_payload(pending_tool, pending_args)
        ckey = make_key(uid, sid, op=(pending_op or "x"), target=pending_tool, payload=payload)
        try:
            agent._consent.grant(ckey)
        except Exception as e:
            logger.error(f"Consent bypass: grant failed: {e}")

        # Persist the user's "ya" so the transcript reflects reality even
        # though the LLM never processed this turn. Do it BEFORE running so
        # a dispatch crash doesn't lose the confirmation record.
        try:
            await self.save_message("user", user_message)
        except Exception as e:
            logger.warning(f"Consent bypass: save_message('user') failed: {e}")

        # Clear pending BEFORE dispatch so a slow tool + a second incoming
        # "ya" cannot double-fire.
        self._clear_pending_consent()

        logger.info(
            f"Consent bypass: granting + running {pending_kind}={pending_tool} "
            f"hash={pending_hash} (session={sid}, user={uid})"
        )

        result_str = ""
        try:
            if pending_kind == "tool":
                tool_result = await self.tools.execute(
                    pending_tool,
                    dict(pending_args),
                    access_level,
                    provider=self.provider,
                    conv=self,
                )
                result_str = str(tool_result)
            elif pending_kind == "ability" and self.abilities is not None:
                ability_context = {
                    "user_id": self.user.get("id"),
                    "session_id": self.session_id,
                    "access_level": access_level,
                    "config": (self.abilities.get(pending_tool).config or {}) if self.abilities.get(pending_tool) else {},
                    "workspace": getattr(self._mgr, "workspace_outputs", None) if self._mgr else None,
                    "_registry": self.abilities,
                    "conv": self,
                    "scheduled": False,
                }
                ability_result = await self.abilities.execute(
                    pending_tool, dict(pending_args), ability_context,
                )
                if ability_result.get("success"):
                    result_str = str(ability_result.get("result", ""))
                else:
                    result_str = f"Error: {ability_result.get('error', 'Unknown error')}"
            else:
                result_str = f"Error: cannot dispatch pending {pending_kind}/{pending_tool}"
        except Exception as e:
            logger.error(f"Consent bypass dispatch failed: {e}", exc_info=True)
            result_str = f"Error running action after consent: {e}"
        finally:
            # Single-use grant (root fix for consent double-exec): revoke the
            # grant immediately after dispatch. If the transcript surgery below
            # fails to resolve the LLM's original tool_use, the LLM may re-emit
            # the identical payload next turn — without this revoke that re-emit
            # hits the still-cached grant and runs a SECOND time with no prompt.
            # Revoking here forces the re-emit back through the consent gate.
            try:
                agent._consent.revoke(ckey)
            except Exception as _e:
                logger.warning(f"Consent bypass: revoke failed: {_e}")

        # ─── Transcript surgery ────────────────────────────────────────────
        # When the gate held earlier, the tool loop saved the "balas ya"
        # PROMPT as the tool_result for the LLM's original tool_use (with
        # the LLM-issued tool_call_id, adjacent to the tool_use). We now
        # OVERWRITE that same row with the ACTUAL result. Two wins:
        #   1. Anthropic's sanitizer (which drops orphan tool_results whose
        #      tool_use_id doesn't match a preceding tool_use) keeps this
        #      pair — the LLM will see the real output on the next turn
        #      instead of the stale "balas ya" prompt.
        #   2. No infinite re-invocation loop: the LLM never re-issues the
        #      same tool because from its point of view the tool already
        #      ran with a real result.
        # We locate the row by walking _message_cache in reverse for the
        # first tool-role message whose metadata.tool_name matches the
        # pending tool. The "balas ya" prompt is almost always the most
        # recent tool message for this tool name at this point in the
        # session (bypass ran less than TTL seconds after the hold), so
        # this heuristic is robust in practice. If no match is found we
        # fall back to appending a fresh tool message.
        _patched = False
        try:
            _patched = await self._patch_held_tool_result(
                pending_tool, pending_hash, result_str,
            )
        except Exception as e:
            logger.warning(f"Consent bypass: patch_held_tool_result failed: {e}")

        if not _patched:
            # Surgery couldn't find the held row. With the assistant(tool_use)
            # now correctly persisted during the tool loop, this branch should
            # be extremely rare (missed only when compaction archived the
            # source row between hold and bypass). Log for observability.
            logger.warning(
                f"Consent bypass: surgery failed to locate held row for "
                f"tool={pending_tool} hash={pending_hash!r}. Continuation "
                f"will run without the actual output paired to the LLM's "
                f"original tool_use — the LLM will likely re-issue the "
                f"tool_use next turn."
            )

        # ─── LLM continuation ───────────────────────────────────────────────
        # This is what makes multi-step workflows finish. After surgery,
        # the transcript now shows:
        #     Assistant: [tool_use cat file.py, id=abc]
        #     Tool:      [ACTUAL output] (id=abc, properly paired)
        #     Assistant: [previous "waiting for consent" text]
        #     User:      "ya"
        # A fresh _chat_inner turn feeds this to the LLM. It sees the real
        # cmd1 output (not the stale "balas ya" prompt — that's what
        # _patch_held_tool_result overwrote) and can plan cmd2 naturally.
        # Each subsequent op=x call still hits the consent gate on its own
        # — but the LLM won't re-issue cmd1 because from its point of view
        # cmd1 already succeeded.
        #
        # We prime _last_saved_hash so _chat_inner's internal
        # save_message("user", user_message) is deduplicated with the
        # "ya" bypass just saved. Fallback to canned response if the
        # continuation crashes.
        import hashlib as _hashlib, json as _json
        _ya_hash = _hashlib.md5(
            f"user:{user_message}:".encode()
        ).hexdigest()
        self._last_saved_hash = _ya_hash

        # The tool already ran via the consent-bypass path above; this
        # continuation only summarizes it. Flag it one-shot so the phantom-
        # action guard in _chat_inner does not flag the summary as fake.
        self._tool_ran_via_bypass = True
        try:
            llm_response = await self._chat_inner(user_message, message_metadata=None)
            if llm_response:
                _has_marker = "[[CONSENT_BUTTONS:hash=" in llm_response
                _preview = llm_response[:200].replace("\n", " ⏎ ")
                logger.info(
                    f"Consent bypass continuation returned {len(llm_response)} chars, "
                    f"contains_marker={_has_marker}, "
                    f"post_pending_hash={self._pending_consent_hash!r}, "
                    f"preview={_preview!r}"
                )
                return llm_response
            logger.warning(
                "Consent bypass continuation returned empty response — "
                "falling back to canned reply with raw output"
            )
        except Exception as e:
            logger.error(
                f"Consent bypass continuation failed: {e}", exc_info=True
            )

        # Fallback: LLM turn crashed or returned nothing — at least surface
        # the raw tool output so the owner still sees what actually ran.
        head = f"✅ Consent granted (`{pending_hash}`). Ran: `{pending_tool}`"
        return f"{head}\n\n{result_str}"

    async def _patch_held_tool_result(
        self, tool_name: str, expected_hash: str, actual_result: str,
    ) -> bool:
        """Rewrite the most recent held-prompt tool_result for `tool_name`
        to the actual replay output — both in DB and in _message_cache.

        Returns True if a row was patched. Returns False if no candidate
        was found (caller falls back to appending a new tool message).
        Only matches messages whose content contains the pending hash so
        we can't clobber a legitimate real tool_result by accident.
        """
        from .consent import CONSENT_BUTTON_MARKER
        # Walk cache in reverse for the first tool-role hit whose metadata
        # names our tool AND whose content still carries the hash marker
        # (i.e. it was the held prompt, not a real prior tool_result).
        # STRICT: require the SPECIFIC pending hash to be in body. A generic
        # marker match would clobber a stale held row from a different pending.
        target_idx = None
        _tool_rows_seen = 0
        for i in range(len(self._message_cache) - 1, -1, -1):
            m = self._message_cache[i]
            if m.role != "tool":
                continue
            _tool_rows_seen += 1
            # NOTE: we deliberately do NOT gate on metadata.tool_name here.
            # Held rows are saved with the LLM-issued tool name (e.g. 'shell')
            # while the consent gate tracks the canonical name (e.g. 'exec'),
            # so a name comparison produces false negatives and re-loops the
            # tool. The marker+hash below is already a globally-unique key for
            # exactly one held-prompt row, so it is sufficient and safe.
            body = m.content or ""
            if (CONSENT_BUTTON_MARKER + expected_hash) in body:
                target_idx = i
                break
        if target_idx is None:
            logger.warning(
                f"patch_held_tool_result: NO MATCH for tool={tool_name} "
                f"expected_hash={expected_hash!r} in {len(self._message_cache)} cache msgs "
                f"({_tool_rows_seen} tool rows scanned). LLM will see synthetic-ID "
                f"orphan tool_result that the sanitizer WILL drop — this is why "
                f"'output tidak sampai ke LLM'."
            )
            return False

        target_msg = self._message_cache[target_idx]
        old_content = target_msg.content or ""
        target_msg.content = actual_result
        _tool_call_id_preview = (target_msg.metadata or {}).get("tool_call_id", "<missing>")
        logger.info(
            f"patch_held_tool_result: PATCHED cache[{target_idx}] tool={tool_name} "
            f"hash={expected_hash!r} tool_call_id={_tool_call_id_preview!r} "
            f"old_len={len(old_content)} new_len={len(actual_result)}"
        )

        # Mirror to DB. Use the metadata's tool_call_id (if present) plus
        # role + session to pinpoint the row. Fall back to matching by
        # exact old content within this session.
        tool_call_id = (target_msg.metadata or {}).get("tool_call_id") or ""
        try:
            from .db.connection import get_connection
            async with get_connection() as conn:
                if tool_call_id:
                    await conn.execute(
                        """
                        UPDATE messages
                        SET content = $1
                        WHERE id = (
                            SELECT id FROM messages
                            WHERE session_id = $2
                              AND role = 'tool'
                              AND (metadata->>'tool_call_id') = $3
                            ORDER BY id DESC
                            LIMIT 1
                        )
                        """,
                        actual_result, self.session_id, tool_call_id,
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE messages
                        SET content = $1
                        WHERE id = (
                            SELECT id FROM messages
                            WHERE session_id = $2
                              AND role = 'tool'
                              AND content = $3
                            ORDER BY id DESC
                            LIMIT 1
                        )
                        """,
                        actual_result, self.session_id, old_content,
                    )
        except Exception as e:
            # DB write failed — cache is already patched, so the CURRENT
            # session sees the fix. Only future reloads would see stale
            # content. Log loudly so operators notice.
            logger.error(
                f"patch_held_tool_result: DB update failed for session "
                f"{self.session_id}, tool={tool_name}: {e}"
            )
        return True

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
        # Save reference to the lock we acquired — release THIS one, not self._lock
        # (which may be replaced by another task during our execution)
        _my_lock = self._lock
        logger.debug(f"Session {self.session_id}: lock acquired")
        try:
            # Reset per-turn state
            self._pending_media: list[str] = []
            # Per-turn cache of raw media input (image / audio / document).
            # Populated by _ability_first_preprocess as it walks INPUT_TYPES,
            # BEFORE that function strips the raw bytes from metadata on a
            # successful ability run. Read by _handle_tool_calls to auto-fill
            # image_base64 / audio_base64 / document_base64 when the LLM
            # emits image_analysis(...) / audio_transcription(...) / etc.
            # without providing pixels itself. Reset each turn (this line).
            self._cached_input_data: dict = {}
            self._message_metadata = message_metadata
            # Provenance taint: seed from this turn's INPUT (image / uploaded
            # file / URL). May also flip True later if an untrusted tool runs
            # mid-turn (see taint-set after tool/ability dispatch). The consent
            # gate reads this to decide whether owner/family skip applies.
            self._turn_untrusted = self._detect_untrusted(user_message, message_metadata)
            self._processing = True
            try:
                # ─── Deterministic consent-confirmation bypass ───
                # Runs INSIDE the lock so bypass's tool dispatch + LLM
                # continuation are serialized with any concurrent chat()
                # calls on this session. The bypass will either:
                #   - replay the pending call, feed the actual output back
                #     to the LLM via _chat_inner("ya"), and return the LLM's
                #     continuation response, OR
                #   - short-circuit (no pending, TTL expired, wrong token,
                #     etc.) → we fall through to the normal LLM path with
                #     `user_message` as the input.
                _bypass_response = await self._maybe_execute_pending_consent(user_message)
                if _bypass_response is not None:
                    return _bypass_response
                return await self._chat_inner(user_message, message_metadata)
            finally:
                self._processing = False
                # Cleanup transient upload files (Option B from design discussion).
                # Disk staging in workspace/uploads/ is needed during the turn
                # so tools like memory_store_file / send_file / file_read can
                # reference it via path. After the turn, the canonical store
                # is either memory_blobs (if user asked to save) or nowhere
                # (transient). Either way, disk file is orphan — delete.
                temp_paths = (message_metadata or {}).get("_temp_upload_paths") or []
                import os as _os
                for p in temp_paths:
                    try:
                        if p and _os.path.isfile(p):
                            _os.remove(p)
                            logger.debug(f"Cleaned up transient upload: {p}")
                    except OSError as e:
                        logger.warning(f"Failed to delete transient upload {p}: {e}")
        finally:
            try:
                _my_lock.release()
            except RuntimeError:
                pass  # lock already released or replaced

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
        # Compaction gate: SINGLE token-based trigger, configurable via
        # config `compaction.trigger_percent` (1-100, default 40). Compact when
        # the context the LLM will see reaches that % of the model's context window.
        from .db.models import get_config as _gc_cp
        _trig_pct = await _gc_cp("compaction.trigger_percent", 40)
        try:
            _trig_pct = float(_trig_pct)
        except (TypeError, ValueError):
            _trig_pct = 40.0
        _trig_pct = max(1.0, min(100.0, _trig_pct))
        _threshold = _trig_pct / 100.0

        context_full = bool(self._message_cache) and self.context_mgr.should_compact(
            self._message_cache,
            threshold=_threshold,
        )

        if context_full:
            logger.info(f"Compaction triggered for session {self.session_id}: context usage >= {_trig_pct:.0f}% of model context window")
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
            # Build ability permission map so abilities aren't wrongly filtered as owner-only
            ability_perms = {}
            if self.abilities:
                for ab in self.abilities._abilities.values():
                    ability_perms[ab.name] = ab.permission
            tool_schemas = filter_tools_for_group(tool_schemas, extra_permissions=ability_perms)

        # Tool routing removed: send the full access-level-filtered toolset every
        # turn. Prior regex routing (_TOOL_SIGNALS) was lossy — English/paraphrase
        # gaps meant the LLM sometimes did not see meta-tools like update_soul or
        # memory_delete and would tell the user "I don't have that tool" while
        # the tool actually existed. Worse, the routing varied the toolset
        # turn-to-turn, destabilising the model's mental map of its own
        # capabilities and feeding tool-name hallucinations, AND it killed the
        # Anthropic prompt-cache breakpoint on system+tools (that cache pays for
        # itself many times over per session). Now: stable, correct, cheaper.
        logger.debug(f"Toolset: {len(tool_schemas)} tools (routing dropped)")

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
                # Last retry: disable thinking so all output goes to text
                if attempt == max_attempts - 2:
                    chat_kwargs["thinking_budget"] = 0
                    chat_kwargs.pop("top_p", None)
                    chat_kwargs.pop("top_k", None)
                    logger.info("Final empty retry: thinking disabled")
                await asyncio.sleep(1.0)

        # Handle tool calls
        logger.info(f"LLM response: content={len(response.content or '')} chars, tool_calls={len(response.tool_calls) if response.tool_calls else 0}, model={response.model}")
        # Capture BEFORE _handle_tool_calls: the final response it returns always
        # has empty .tool_calls (the closing answer doesn't call another tool), so
        # checking response.tool_calls afterwards can't tell "no tool ran" apart
        # from "tool ran, now summarizing". This flag preserves that distinction.
        # A tool that ran via the consent-bypass path executed in a PRIOR
        # turn; this continuation only summarizes it. Consume the one-shot
        # flag so the phantom guard below does not flag the summary (fixes
        # "push sukses" false-positive on bypass continuations, 13 Jul 2026).
        _bypass_ran = getattr(self, "_tool_ran_via_bypass", False)
        self._tool_ran_via_bypass = False
        tools_ran_this_turn = bool(response.tool_calls) or _bypass_ran
        if response.tool_calls:
            logger.info(f"Tool calls: {[tc.get('name') for tc in response.tool_calls]}")
            response = await self._handle_tool_calls(response, context, access_level, tool_schemas)

        # Phantom action detection — ANY model can claim an action it never
        # performed (no tool calls). Provider-agnostic on purpose: Claude is
        # less prone than Gemini/GPT but NOT immune, and it's the owner's
        # preferred model. Deterministic regex guard, runs for all providers.
        # BUT: skip if a tool actually ran this turn — summarizing a real
        # tool_result with words like "berhasil/executed" is NOT a phantom claim
        # (false positive fixed 11 Jul 2026: speedtest summary got flagged).
        if not response.tool_calls and not tools_ran_this_turn:
            content = response.content or ""
            if _PHANTOM_ACTION_RE.search(content):
                logger.warning(f"Phantom action detected in response (no tool calls): {content[:200]}")
                response = ChatResponse(
                    content=content + "\n\n⚠️ _Warning: the above claims may not have been executed. No tool was actually called. Please verify._",
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    tool_calls=response.tool_calls,
                    thinking=response.thinking,
                )

        # Save assistant response
        await self.save_message("assistant", response.content)

        # Store thinking for the channel to optionally display
        self._last_thinking = response.thinking

        # Store last response for CLI token display
        self._last_chat_response = response

        # Attach any media collected during tool calls to the final response
        final_response = response.content

        # ─── Ensure the consent prompt + buttons reach the channel ─────────
        # If any tool call in this turn hit the consent gate, we now have a
        # pending consent on the conversation. The registry handed the LLM
        # the "balas ya" prompt (with the [[CONSENT_BUTTONS:hash=…]] marker
        # the Telegram adapter scans for) as the tool_result, but the LLM's
        # final text is free to paraphrase, translate, or drop the marker
        # entirely — and if it does, the adapter has no way to know to
        # attach the Yes/No inline buttons. Worse, the LLM sometimes
        # returns an empty final text after a held tool call, which would
        # send NOTHING at all to the user.
        #
        # Guarantee both the prompt content AND the marker are present
        # whenever there is a pending consent: regenerate the canonical
        # prompt from the pending state when the LLM's own text is empty
        # or clearly missing the confirmation ask; otherwise just append
        # the marker so the adapter can find it.
        _pending_hash_for_buttons = self._pending_consent_hash
        logger.info(
            f"Consent marker check: pending_hash={_pending_hash_for_buttons!r}, "
            f"pending_tool={self._pending_consent_tool!r}, "
            f"final_response_len={len(final_response or '')}"
        )
        if _pending_hash_for_buttons:
            from .consent import format_consent_prompt, CONSENT_BUTTON_MARKER
            _fresh_prompt = format_consent_prompt(
                self._pending_consent_tool or "",
                self._pending_consent_args or {},
                _pending_hash_for_buttons,
            )
            _resp = (final_response or "").strip()
            # Rule: if there is ANY pending consent at end-of-turn, the outgoing
            # message MUST carry the canonical Yes/No prompt so the owner has a
            # way to approve. Two cases:
            #   1. LLM's own final text is empty OR doesn't already contain the
            #      marker → append the canonical prompt (with marker) at the
            #      end. No heuristics — the guarantee has to be bulletproof.
            #   2. LLM already emitted the marker verbatim → leave as-is.
            if CONSENT_BUTTON_MARKER in _resp:
                logger.info("Consent marker: LLM's response already contains marker, keeping as-is")
            else:
                # Pending consent at end-of-turn: the outgoing message MUST be
                # ONLY the canonical Yes/No prompt. Any LLM narration around a
                # held gate ("held - hash - type yes") is noise and gets
                # DISCARDED entirely - not appended. The system speaks for the
                # gate; the LLM does not.
                final_response = _fresh_prompt
                logger.info("Consent marker: discarded LLM narration, using canonical prompt only")

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
        elif tool_name in ("read_source", "exec", "shell"):
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

        Shell tools ("shell" / "exec") pass through the server-side shell_guard
        BEFORE the WebSocket forward: HARD_DENY blocks here, NEEDS_CONSENT
        goes through the same check_and_hold flow every other gated tool
        uses, and only ALLOWED commands actually reach the node. This gives
        node-routed exec the same security posture as local exec — the node's
        own blocklist stays as defense-in-depth for a spoofed/replayed
        gateway message, but the primary gate lives on the server where the
        runtime allowlist and consent state live.
        """
        if tool_name in ("shell", "exec"):
            guard_result = await self._gate_shell_for_node(
                args.get("command", ""), tool_name,
            )
            if guard_result is not None:
                return guard_result
        else:
            # Non-shell node tools (file_write, file_read, read_source): fire
            # the same check_and_hold gate the server-side tools.registry.execute()
            # would run. Without this, routing a file_write to a node would
            # skip the owner's Yes/No prompt that the identical file_write
            # tool triggers when run locally — same tool, same permission
            # bits, wildly different security posture based on where it
            # happens to execute.
            gate_result = await self._gate_generic_tool_for_node(tool_name, args)
            if gate_result is not None:
                return gate_result

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

    async def _gate_shell_for_node(self, command: str, tool_name: str):
        """Run shell_guard against a command about to be forwarded to a node.

        Returns:
            None if the guard cleared — caller proceeds with the WebSocket
            forward.
            ToolResult if execution must short-circuit — either HARD_DENY
            (ok=False) or a held consent prompt (ok=True, LLM surfaces it
            to the owner).
        """
        from .shell_exec import run_shell, Outcome
        from .db.connection import get_pool
        from .db.models import get_config as _gc

        if not command:
            return None  # let the node-side handler produce its own error

        try:
            _pool = get_pool()
        except Exception:
            _pool = None

        _consent_enabled = await _gc("security.consent_enabled", True)
        if isinstance(_consent_enabled, str):
            _consent_enabled = _consent_enabled.strip().lower() in ("1", "true", "yes", "on")

        async def _guard(approved: bool):
            return await run_shell(
                command,
                source="llm",
                db_pool=_pool,
                consent_enabled=bool(_consent_enabled),
                approved=approved,
                check_only=True,
            )

        res = await _guard(approved=False)

        if res.outcome == Outcome.DENIED:
            cand = f" (unrecognized: {', '.join(res.candidates)})" if res.candidates else ""
            logger.warning(
                f"shell_guard DENIED (node route): {command[:100]} — {res.reason}"
            )
            return ToolResult(
                f"⛔ Blocked by shell guard: {res.reason}{cand}\n\n"
                "This command is hard-denied and cannot be approved. If a listed "
                "binary is legitimate, the owner can add it via /add-allowlist.",
                ok=False, error_type="permission",
            )

        if res.outcome == Outcome.NEEDS_CONSENT:
            from .consent import check_and_hold as _consent_gate
            logger.info(
                f"shell_guard NEEDS_CONSENT (node route): {command[:100]} — {res.reason}"
            )
            _access = self.user.get("access_level", "public")
            _agent = getattr(self._mgr, "_agent", None) if self._mgr else None
            action, prompt = await _consent_gate(
                conv=self, agent=_agent,
                tool_name=tool_name, args={"command": command},
                access_level=_access, permission=0o700,
                pre_decided=True,
            )
            if action == "held":
                return ToolResult(prompt or "", ok=True)
            # Owner approved — re-run the guard with approved=True so a stale
            # allowlist / denylist change between the two calls is honoured.
            res2 = await _guard(approved=True)
            if res2.outcome != Outcome.ALLOWED:
                return ToolResult(
                    f"Error: shell guard rejected after consent: {res2.reason}",
                    ok=False, error_type="permission",
                )

        # ALLOWED — safe to forward to the node.
        return None

    async def _gate_generic_tool_for_node(self, tool_name: str, args: dict):
        """Run the standard check_and_hold gate for a node-routed non-shell tool.

        Mirrors the consent flow that tools.registry.execute() runs for local
        calls, so a tool like file_write can't reach the node without the
        owner's explicit Yes just because the WebSocket path skipped
        registry.execute(). Returns None if the gate cleared, ToolResult if
        held (LLM surfaces the prompt to the owner).
        """
        tool = self.tools.get(tool_name) if self.tools else None
        if tool is None:
            # Not a registered server-side tool — nothing to gate against.
            # Should not happen for NODE_TOOLS in normal flow.
            return None

        from .consent import check_and_hold as _consent_gate
        _access = self.user.get("access_level", "public")
        _agent = getattr(self._mgr, "_agent", None) if self._mgr else None
        is_scheduled = bool((self._message_metadata or {}).get("scheduled"))

        action, prompt = await _consent_gate(
            conv=self, agent=_agent,
            tool_name=tool_name, args=args,
            access_level=_access, permission=tool.permission,
            scheduled=is_scheduled,
        )
        if action == "held":
            return ToolResult(prompt or "", ok=True)
        return None

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
        from .tools.scheduler import set_current_user, set_current_chat
        user_platform_id = self.user.get("platform_id")
        try:
            set_current_user(int(user_platform_id))
        except (TypeError, ValueError):
            set_current_user(None)
        # Set current chat context — so manage_schedule can auto-target the
        # current group when a reminder is created from a group conversation.
        _chat_type = "group" if self.is_group else "direct"
        _chat_id_for_ctx = None
        if self.inbound and self.inbound.chat_id:
            _chat_id_for_ctx = str(self.inbound.chat_id)
        elif self.chat_id:
            _chat_id_for_ctx = str(self.chat_id)
        set_current_chat(_chat_id_for_ctx, _chat_type)

        current = response
        loop_stuck = False
        round_num = 0

        # Timeout-based limit (like OpenClaw) — default 30 min, configurable
        import time as _time
        from .db.models import get_config as _gc_timeout
        _timeout_sec = await _gc_timeout("session.tool_loop_timeout", 1800)
        if isinstance(_timeout_sec, str):
            _timeout_sec = int(_timeout_sec)
        _loop_deadline = _time.monotonic() + _timeout_sec

        # Track which on-demand guides have been injected this turn
        _injected_guides: set = set()

        # Loop detection + usage accumulation
        detector = ToolLoopDetector()
        usage = UsageAccumulator()
        usage.add(response)  # Initial response that triggered tool calls

        while current.tool_calls and _time.monotonic() < _loop_deadline:

            # Persist the assistant tool_use turn — NOT just add to context.
            # Skipping save_message here was the source of every "output tidak
            # sampai ke LLM" bug: without a persisted assistant(tool_use), the
            # tool_result row below has no preceding tool_use in DB/cache. On
            # the NEXT turn, Anthropic's sanitizer sees an orphaned
            # tool_result and drops it. The next-turn LLM literally never
            # sees the tool's output regardless of surgery or markers.
            _assistant_meta = {"tool_calls": current.tool_calls}
            await self.save_message(
                "assistant",
                current.content or "",
                metadata=_assistant_meta,
            )
            context.append(ChatMessage(
                role="assistant",
                content=current.content or "",
                metadata=_assistant_meta,
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

            # ── Phase 1.5: Serialize gate-triggering calls ──
            # If the LLM emitted multiple tool_uses in ONE assistant message and
            # more than one would trigger the consent gate (op=x), executing
            # them all in parallel breaks the single-pending-consent model:
            # `_pending_consent_hash` gets overwritten on each check_and_hold
            # so only the LAST held call is reachable via the Yes button, and
            # the LLM's continuation ends up seeing a mix of "actual output"
            # and stale "balas ya" prompts. The observable failure is the LLM
            # confusing itself: user asks for cmd1+cmd2+cmd3, only cmd3 (last)
            # gets a button, cmd1/cmd2 never do.
            #
            # Enforce serial: keep the FIRST gate-triggering call (it holds
            # the gate + sets pending as usual). Every subsequent gate call
            # in the same batch is intercepted before registry dispatch and
            # gets a "queued" ToolResult that the LLM will see and re-emit
            # on the next turn after the user confirms the first one. Non-
            # gate-triggering calls (read-only) still run in parallel — no
            # reason to serialize them.
            from .security import needs_consent as _needs_consent
            _first_gate_seen = False
            _queued_indices: set = set()
            # Provenance skip mirrors consent.check_and_hold: on a CLEAN turn
            # (owner/family, not tainted) the gate does NOT fire, so there is
            # no single-pending-consent constraint to protect — serializing
            # would spuriously queue legit parallel exec/file_write calls and
            # make the LLM think they "hadn't run yet". Only serialize when the
            # gate will actually hold something this turn.
            _gate_will_skip = (
                access_level in ("owner", "family")
                and not getattr(self, "_turn_untrusted", False)
            )
            if not _gate_will_skip:
                for _idx, (_pname, _pargs, _pcid, _pl) in enumerate(parsed_calls):
                    _ptool = self.tools.get(_pname)
                    if _ptool is None:
                        continue  # abilities routed elsewhere; skip for this check
                    if not _needs_consent(access_level, _ptool.permission):
                        continue  # read-only — no serialization needed
                    if not _first_gate_seen:
                        _first_gate_seen = True  # this one goes through as usual
                        continue
                    _queued_indices.add(_idx)
                    logger.info(
                        f"Serialize gate: queueing tool_use[{_idx}] {_pname} "
                        f"(prior gate-triggering call in same batch will run first)"
                    )

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
                    # Auto-inject file content for memory_store_file when user
                    # uploaded a file in this turn. LLM can call with just
                    # content+category; we fill file_base64/filename/mime_type
                    # from the message metadata.
                    if t_name == "memory_store_file":
                        meta = self._message_metadata or {}
                        if not t_args.get("file_base64") and not t_args.get("file_path"):
                            # document = PDF/Office, image = photo, audio = voice
                            for key in ("document", "image", "audio"):
                                src = meta.get(key)
                                if isinstance(src, dict):
                                    # Prefer file_path (disk) over base64 (memory) when available
                                    if src.get("path"):
                                        t_args["file_path"] = src["path"]
                                    elif src.get("base64"):
                                        t_args["file_base64"] = src["base64"]
                                    else:
                                        continue
                                    if not t_args.get("filename"):
                                        t_args["filename"] = src.get("filename") or ""
                                    if not t_args.get("mime_type"):
                                        t_args["mime_type"] = src.get("mime_type") or ""
                                    break
                    # Provenance taint (mid-turn): an untrusted tool pulling
                    # external content into context flips the turn tainted, so
                    # any exec/file_write that follows in THIS turn re-arms the
                    # consent gate even if the user's own input was clean.
                    if t_name in _UNTRUSTED_TOOLS:
                        self._turn_untrusted = True
                    _tool_result = await self.tools.execute(
                        t_name, t_args, access_level,
                        scheduled=is_scheduled,
                        provider=self.provider,
                        conv=self,
                    )
                    return _tool_result
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
                        # Conversation ref + scheduled flag so the consent gate
                        # in abilities.execute can find the pending state and
                        # apply the send_* hybrid rule uniformly with tools.
                        "conv": self,
                        "scheduled": is_scheduled,
                    }
                    # Provenance taint (mid-turn): untrusted abilities (fetch_url,
                    # website_screenshot, image_analysis, pdf/office read) flip
                    # the turn tainted before dispatch, re-arming the consent
                    # gate for any destructive action later in this turn.
                    if t_name in _UNTRUSTED_TOOLS:
                        self._turn_untrusted = True
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

            async def _queued_gate_result():
                """Placeholder ToolResult for a gate-triggering tool that was
                queued behind another gate call in the same batch. Explains to
                the LLM what to do next so it re-emits the tool_use after the
                first gate is confirmed."""
                return ToolResult(
                    "Queued — sequential consent mode.\n\n"
                    "This tool call was NOT executed. Another gate-triggering "
                    "tool call in the same turn was placed at the head of the "
                    "queue and is awaiting the user's Yes/No. After the user "
                    "confirms that first call and its actual output appears, "
                    "re-emit THIS tool_use in your next assistant message and "
                    "it will be gated normally.\n\n"
                    "Do NOT claim this call succeeded or failed — it simply "
                    "hasn't run yet.",
                    ok=True,
                )

            tasks = [
                _queued_gate_result() if _i in _queued_indices
                else _execute_single_tool(name, args, tc_id)
                for _i, (name, args, tc_id, _) in enumerate(parsed_calls)
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
                if name in ("exec", "shell") and result.ok:
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

                # Strip server paths (bypass for owner DM — platform-verified identity)
                if not is_owner_dm:
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

            # Early exit — a consent gate held mid-loop. Do NOT let the
            # model iterate again in the same turn: its next tool_result
            # (the "balas ya" prompt) would be its own signal, and misreads
            # of that signal have driven duplicate re-emits before. The
            # marker-injection at the tail of _chat_inner will replace the
            # outgoing response with the canonical consent prompt for the
            # channel, so the user still sees Yes/No properly.
            if getattr(self, "_pending_consent_hash", ""):
                logger.info(
                    f"Tool loop: gate held mid-turn "
                    f"(hash={self._pending_consent_hash}) — terminating loop early"
                )
                break

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

        _timed_out = _time.monotonic() >= _loop_deadline and current.tool_calls
        if loop_stuck or _timed_out:
            # Force a final text response with no tools
            reason = "tool call loop detected" if loop_stuck else f"tool loop timeout ({_timeout_sec}s, {round_num} rounds)"
            logger.warning(f"Forcing final response: {reason}")

            context.append(ChatMessage(
                role="system",
                content=f"STOP. {reason.capitalize()}. Summarize what you've done so far and respond to the user with what you have.",
            ))

            # Disable thinking for forced final — all output budget goes to text
            _forced_kwargs = self._build_chat_kwargs()
            _forced_kwargs["thinking_budget"] = 0
            _forced_kwargs.pop("top_p", None)
            _forced_kwargs.pop("top_k", None)
            current = await self.provider.chat(
                messages=context,
                tools=None,
                stream_callbacks=self.stream_callbacks,
                **_forced_kwargs,
            )
            usage.add(current)
            logger.info(f"Forced response: content={len(current.content or '')} chars, thinking={len(current.thinking or '')} chars, tool_calls={len(current.tool_calls or [])}, in={current.input_tokens}, out={current.output_tokens}")

        # Reset context after tool execution
        set_owner_dm(False)
        set_current_user(None)
        set_current_chat(None, None)

        # If final response is empty (e.g. thinking-only), retry once without thinking
        if not (current.content or "").strip():
            logger.warning(f"Tool loop final response empty after {usage.rounds} rounds — retrying without thinking")
            try:
                _retry_kwargs = self._build_chat_kwargs()
                _retry_kwargs["thinking_budget"] = 0
                _retry_kwargs.pop("top_p", None)  # can't send both temperature + top_p
                _retry_kwargs.pop("top_k", None)
                current = await self.provider.chat(
                    messages=context,
                    tools=None,
                    stream_callbacks=self.stream_callbacks,
                    **_retry_kwargs,
                )
                usage.add(current)
                if (current.content or "").strip():
                    logger.info(f"Retry without thinking succeeded: {len(current.content)} chars")
                else:
                    logger.warning(f"Retry without thinking also empty")
            except Exception as e:
                logger.error(f"Retry without thinking failed: {e}")

        # Still empty after retry — use fallback message
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

            # Snapshot the raw input BEFORE any stripping so a later
            # tool-call retry can access it. The tool loop (in
            # _handle_tool_calls) auto-injects from self._cached_input_data
            # when the LLM emits e.g. image_analysis(prompt="…") without
            # pixels — common flow: pre_process succeeds → text description
            # injected → LLM decides to re-analyze from a different angle →
            # calls the tool → tool loop pulls the raw bytes from HERE.
            # Without this snapshot the tool call would fail with
            # "Either image_url or image_base64 is required" and the model
            # would surface "aku nggak bisa baca gambarnya" to the user
            # even though the pixels are literally one dict away.
            self._cached_input_data[input_type] = input_data

            # Find a priority ability that handles this input type
            result_text = None
            for registered in self.abilities.list_enabled("owner"):
                # Skip abilities that opted out of priority pre-processing
                if not getattr(registered.instance, 'priority', True):
                    continue
                if not registered.instance.handles_input_type(input_type):
                    continue

                # Lazy dependency install on first pre_process call.
                # Same logic as registry.execute() — bundled abilities
                # enabled-by-default never trigger enable() so deps would
                # otherwise stay uninstalled until first real failure.
                if not getattr(registered, 'deps_ensured', False):
                    try:
                        dep_ok, dep_msg = await registered.instance.ensure_dependencies()
                        if dep_ok:
                            registered.deps_ensured = True
                            if dep_msg:
                                logger.info(f"Ability '{registered.name}' deps: {dep_msg}")
                        else:
                            logger.warning(f"Ability '{registered.name}' deps not ready, skipping pre_process: {dep_msg}")
                            continue
                    except Exception as e:
                        logger.warning(f"Ability '{registered.name}' ensure_dependencies crashed: {e}")
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
                # Ability failed — check if LLM can handle natively
                if spec["native_check"]():
                    # LLM supports this natively — keep metadata, let LLM handle it
                    logger.info(f"{spec['label']} ability failed, falling back to native LLM")
                else:
                    # Plain-text documents (.md/.txt/.csv/.json/source code)
                    # need no ability at all: their bytes ARE the content.
                    # Read and inline directly before giving up.
                    inlined = None
                    if input_type == "document" and isinstance(input_data, dict):
                        inlined = self._inline_text_document(input_data)
                    if inlined is not None:
                        fname = input_data.get("filename") or "file"
                        user_message = (
                            f"{user_message}\n\n"
                            f"[Contents of {fname}:]\n{inlined}"
                        )
                        logger.info(
                            f"{spec['label']}: inlined text document "
                            f"({len(inlined)} chars, no ability needed)"
                        )
                        metadata = {k: v for k, v in metadata.items() if k != spec["meta_key"]}
                        self._message_metadata = metadata if metadata else None
                    else:
                        # No native support — strip and show error
                        logger.warning(f"{spec['label']} failed — no native fallback available")
                        user_message = f"{user_message}\n\n[{spec['label']} failed. Tell the user the analysis could not be completed and to try again later.]"
                        metadata = {k: v for k, v in metadata.items() if k != spec["meta_key"]}
                        self._message_metadata = metadata if metadata else None
        
        return user_message, metadata if metadata else None

    # Text-like MIME types and extensions whose raw bytes are directly
    # readable as UTF-8 - no parsing ability required.
    _TEXT_DOC_EXTS = (
        ".md", ".markdown", ".txt", ".text", ".csv", ".tsv", ".log",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".xml", ".html", ".htm", ".rst", ".py", ".js", ".ts", ".sh",
        ".sql", ".c", ".h", ".cpp", ".go", ".rs", ".java", ".rb",
    )
    _TEXT_DOC_MAX = 200_000  # ~200 KB safety cap for inlining

    def _inline_text_document(self, doc: dict) -> Optional[str]:
        """If a document is plain text, return its decoded content.

        Returns None when the document is not text-like or cannot be read,
        so the caller can fall through to the normal error path.
        """
        mime = (doc.get("mime_type") or "").lower()
        fname = (doc.get("filename") or "").lower()
        path = doc.get("path")
        is_text = (
            mime.startswith("text/")
            or mime in ("application/json", "application/xml",
                        "application/x-yaml", "application/yaml")
            or fname.endswith(self._TEXT_DOC_EXTS)
        )
        if not is_text:
            return None
        # Prefer base64 (survives disk cleanup); fall back to on-disk path.
        raw = None
        b64 = doc.get("base64")
        if b64:
            try:
                import base64 as _b64
                raw = _b64.b64decode(b64)
            except Exception:
                raw = None
        if raw is None and path:
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
            except Exception as e:
                logger.warning(f"_inline_text_document: read failed: {e}")
                return None
        if not raw:
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                return None
        if len(text) > self._TEXT_DOC_MAX:
            text = text[: self._TEXT_DOC_MAX] + "\n\n[... truncated ...]"
        return text


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
            self.subagents.set_start_callback(self._on_subagent_start)

    def is_any_chat_active(self) -> bool:
        """Check if any conversation is currently processing a chat request."""
        return any(conv._processing for conv in self._active.values())

    async def _on_subagent_start(self, run_id: str, task: str, parent_session_id: int, resumed_from: str = None):
        """Called when a sub-agent starts. Notifies the parent session."""
        task_preview = task[:200] + ("…" if len(task) > 200 else "")
        if resumed_from:
            msg = (
                f"🔄 Sub-agent resumed (run: {run_id[:8]}, from: {str(resumed_from)[:8]})\n\n"
                f"Continuing: {task_preview}"
            )
        else:
            msg = f"🚀 Sub-agent started (run: {run_id[:8]})\n\nTask: {task_preview}"

        # Deliver via callbacks (e.g., Telegram, WhatsApp)
        if self._delivery_callbacks:
            for cb in self._delivery_callbacks:
                try:
                    await cb(msg, parent_session_id)
                except Exception as e:
                    logger.error(f"Failed to deliver sub-agent start via {cb}: {e}")
        else:
            logger.info(f"Sub-agent started (no delivery callback): {msg[:200]}")

    async def _on_subagent_complete(self, run_id: str, status: str, result: str, parent_session_id: int):
        """Called when a sub-agent finishes. Delivers result to the parent session."""
        if status == "completed":
            msg = f"✅ Sub-agent completed (run: {run_id[:8]})\n\n{result}"
        elif status == "incomplete":
            msg = (
                f"⚠️ Sub-agent incomplete (run: {run_id[:8]}) — hit max rounds "
                f"before finishing. Partial result below; continue manually or "
                f"spawn another sub-agent.\n\n{result}"
            )
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
