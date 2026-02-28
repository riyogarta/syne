"""Telegram channel adapter."""

import asyncio
import httpx
import json
import logging
import re
from collections import deque
from typing import Optional, Tuple
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ReactionTypeEmoji
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    MessageHandler,
    MessageReactionHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

from ..agent import SyneAgent
from .tags import parse_reply_tag, parse_react_tags
from .outbound import strip_server_paths, extract_media, split_message, process_outbound
from ..llm.provider import LLMRateLimitError, LLMAuthError, LLMBadRequestError, LLMEmptyResponseError
from ..db.models import (
    get_group,
    get_user,
    get_or_create_user,
    get_config,
    get_identity,
    get_user_alias,
    set_config,
    update_user,
)


logger = logging.getLogger("syne.telegram")


def _classify_error(e: Exception) -> str:
    """Classify any exception into a user-friendly message for Telegram."""
    # 1-4: Typed LLM exceptions (from CCA streaming)
    if isinstance(e, LLMRateLimitError):
        return "⚠️ Rate limited. Please wait a moment and try again."
    if isinstance(e, LLMAuthError):
        return "⚠️ Authentication error. Owner may need to refresh credentials."
    if isinstance(e, LLMBadRequestError):
        return "⚠️ LLM rejected the request. This may be a conversation format issue — try /clear to start fresh."
    if isinstance(e, LLMEmptyResponseError):
        return "⚠️ LLM returned an empty response. Please try again."

    # 5: httpx HTTP status errors (from non-CCA drivers)
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return "⚠️ Rate limited. Please wait a moment and try again."
        if code in (401, 403):
            return "⚠️ Authentication error. Owner may need to refresh credentials."
        if code == 400:
            return "⚠️ LLM rejected the request. This may be a conversation format issue — try /clear to start fresh."
        if 500 <= code < 600:
            return "⚠️ LLM provider is having server issues. Please try again later."
        return f"⚠️ LLM provider returned HTTP {code}. Please try again later."

    # 6: RuntimeError with known message patterns (from Anthropic/other drivers)
    if isinstance(e, RuntimeError):
        msg = str(e)
        if "429" in msg or "rate" in msg.lower():
            return "⚠️ Rate limited. Please wait a moment and try again."
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            return "⚠️ Authentication error. Owner may need to refresh credentials."
        if "400" in msg or "bad request" in msg.lower():
            return "⚠️ LLM rejected the request. This may be a conversation format issue — try /clear to start fresh."
        if "overloaded" in msg.lower() or "529" in msg:
            return "⚠️ LLM provider is overloaded. Please try again later."

    # 7-8: Database errors
    try:
        import asyncpg
        if isinstance(e, asyncpg.InterfaceError):
            return "⚠️ Database connection pool exhausted. Please try again in a moment."
        if isinstance(e, asyncpg.PostgresError):
            return "⚠️ Database error. Please try again later."
    except ImportError:
        pass

    # 9-10: Network / timeout errors
    if isinstance(e, httpx.ConnectError):
        return "⚠️ Cannot connect to LLM provider. Please check connectivity and try again."
    if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout)):
        return "⚠️ Request timed out. Please try again."

    # 11: asyncio timeout
    if isinstance(e, asyncio.TimeoutError):
        return "⚠️ Request timed out. Please try again."

    # 12: Unexpected response shape
    if isinstance(e, (KeyError, IndexError)):
        return "⚠️ Unexpected response format from LLM provider. Please try again."

    # 13: Not implemented
    if isinstance(e, NotImplementedError):
        return "⚠️ This feature is not supported by the current LLM provider."

    # 14: Fallback — include type name for debugging
    type_name = type(e).__name__
    return f"⚠️ Something went wrong ({type_name}). Check logs for details."


class _TypingIndicator:
    """Keeps sending 'typing' action every 4s until cancelled.

    Usage:
        async with _TypingIndicator(bot, chat_id):
            await long_running_work()

    Safety: auto-stops after max_duration seconds even if the wrapped
    coroutine hangs (e.g. LLM API timeout).  Default = 5 minutes.
    """

    def __init__(self, bot: Bot, chat_id: int, interval: float = 4.0, max_duration: float = 300.0):
        self._bot = bot
        self._chat_id = chat_id
        self._interval = interval
        self._max_duration = max_duration
        self._task: Optional[asyncio.Task] = None

    async def _loop(self):
        import time
        start = time.monotonic()
        try:
            while True:
                if time.monotonic() - start > self._max_duration:
                    logging.getLogger("syne.telegram").warning(
                        f"Typing indicator timeout ({self._max_duration}s) for chat {self._chat_id}"
                    )
                    break
                await self._bot.send_chat_action(self._chat_id, "typing")
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass  # silently ignore — typing is best-effort

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class TelegramChannel:
    """Telegram bot adapter for Syne."""

    def __init__(self, agent: SyneAgent, bot_token: str):
        self.agent = agent
        self.bot_token = bot_token
        self.app: Optional[Application] = None
        # Track recent message IDs per chat for reaction context
        # Format: {chat_id: deque([(message_id, preview_text), ...], maxlen=10)}
        self._recent_messages: dict[int, deque] = {}
        # Browse mode: {telegram_user_id: path_string} — when set, messages use CLI session
        self._browse_cwd: dict[int, str] = {}
        # Path lookup for browse callbacks (short hash → full path)
        self._browse_paths: dict[str, str] = {}
        # Auth flow state: {telegram_user_id: {"type": "oauth"|"apikey", "provider": str, ...}}
        self._auth_state: dict[int, dict] = {}
        # Active processing tasks per chat_id — for /cancel support
        self._active_tasks: dict[int, asyncio.Task] = {}

    async def _build_inbound(self, update: Update, is_group: bool) -> "InboundContext":
        """Build InboundContext from a Telegram Update. Used by ALL handlers.

        This is the SINGLE place where Telegram-specific data is mapped to
        the channel-agnostic InboundContext. Group settings are loaded here
        so downstream code never needs to query the DB for chat context.
        """
        from .inbound import InboundContext, load_group_settings
        msg = update.message
        chat = msg.chat
        user = msg.from_user
        reply_raw = self._extract_reply_context_raw(update)
        ctx = InboundContext(
            channel="telegram",
            platform="telegram",
            chat_type="group" if is_group else "direct",
            conversation_label=self._get_display_name(user) if user else None,
            group_subject=chat.title if is_group else None,
            chat_id=str(chat.id),
            sender_name=self._get_display_name(user) if is_group and user else None,
            sender_id=str(user.id) if is_group and user else None,
            sender_username=user.username if is_group and user else None,
            was_mentioned=is_group,
            has_reply_context=reply_raw is not None,
            reply_to_sender=reply_raw["sender"] if reply_raw else None,
            reply_to_body=reply_raw["body"] if reply_raw else None,
        )
        # Load group settings from DB (owner_alias, context_notes, etc.)
        await load_group_settings(ctx)
        return ctx

    def _register_handlers(self):
        """Register all Telegram handlers on self.app."""
        # Command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("version", self._cmd_version))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("memory", self._cmd_memory))
        self.app.add_handler(CommandHandler("compact", self._cmd_compact))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("autocapture", self._cmd_autocapture))
        self.app.add_handler(CommandHandler("identity", self._cmd_identity))
        self.app.add_handler(CommandHandler("restart", self._cmd_restart))
        self.app.add_handler(CommandHandler("models", self._cmd_models))
        self.app.add_handler(CommandHandler("embedding", self._cmd_embedding))
        self.app.add_handler(CommandHandler("evaluator", self._cmd_evaluator))
        self.app.add_handler(CommandHandler("browse", self._cmd_browse))
        self.app.add_handler(CommandHandler("groups", self._cmd_groups))
        self.app.add_handler(CommandHandler("members", self._cmd_members))
        self.app.add_handler(CommandHandler("cancel", self._cmd_cancel))
        # Message handlers
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.app.add_handler(MessageHandler(filters.PHOTO & ~filters.LOCATION, self._handle_photo))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._handle_voice))
        self.app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self.app.add_handler(MessageHandler(filters.LOCATION | filters.Regex(r'^/location'), self._handle_location))
        # Reaction handler
        self.app.add_handler(MessageReactionHandler(self._handle_reaction_update))
        # Chat member handler
        self.app.add_handler(ChatMemberHandler(self._handle_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
        # Callback query handler
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))
        # Error handler
        self.app.add_error_handler(self._handle_error)

    async def start(self):
        """Start the Telegram bot."""
        self.app = (
            Application.builder()
            .token(self.bot_token)
            .concurrent_updates(256)
            .build()
        )

        self._register_handlers()

        logger.info("Starting Telegram bot...")
        # Retry initialization (getMe) — transient network timeouts shouldn't kill the bot
        for attempt in range(5):
            try:
                await self.app.initialize()
                break
            except Exception as e:
                if attempt < 4:
                    delay = [2, 5, 10, 15][attempt]
                    logger.warning(f"Telegram init failed (attempt {attempt + 1}/5): {type(e).__name__}: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise
        await self.app.start()
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=[
                "message", "edited_message", "callback_query",
                "my_chat_member", "message_reaction",
            ],
        )

        # Register bot commands menu (the "/" button in Telegram)
        from telegram import BotCommand
        await self.app.bot.set_my_commands([
            BotCommand("start", "Welcome message"),
            BotCommand("help", "Available commands"),
            BotCommand("version", "Version info"),
            BotCommand("status", "Agent status"),
            BotCommand("memory", "Memory statistics"),
            BotCommand("compact", "Compact conversation history"),
            BotCommand("reasoning", "Toggle reasoning visibility (on/off)"),
            BotCommand("autocapture", "Toggle auto memory capture (on/off)"),
            BotCommand("clear", "Clear current conversation"),
            BotCommand("identity", "Show agent identity"),
            BotCommand("models", "Manage LLM models (owner only)"),
            BotCommand("embedding", "Manage embedding models (owner only)"),
            BotCommand("evaluator", "Manage evaluator model (owner only)"),
            BotCommand("groups", "Manage groups, members & aliases"),
            BotCommand("members", "Manage global user access levels"),
            BotCommand("restart", "Restart Syne (owner only)"),
            BotCommand("browse", "Browse directories (share session with CLI)"),
            BotCommand("cancel", "Cancel active operation"),
        ])

        # Wire sub-agent delivery: when a sub-agent completes, send result to the last active chat
        if self.agent.conversations:
            self._bot = self.app.bot
            self.agent.conversations.set_delivery_callback(self._deliver_subagent_result)
            self.agent.conversations.set_status_callback(self._send_status_message)

        # Wire up reactions tool with this channel reference
        from ..tools.reactions import set_telegram_channel
        set_telegram_channel(self)

        # Wire up send_message tool with this channel reference
        from ..tools.send_message import set_telegram_channel as set_msg_channel
        set_msg_channel(self)

        logger.info("Telegram bot started.")

        # Check if this startup follows a /restart command → notify user
        await self._notify_restart_complete()

    async def _notify_restart_complete(self):
        """Send 'restart done' notification if this startup follows a /restart command."""
        import json, os, tempfile
        restart_flag = os.path.join(tempfile.gettempdir(), "syne_restart_flag.json")
        if not os.path.exists(restart_flag):
            return
        try:
            with open(restart_flag) as f:
                data = json.load(f)
            chat_id = data.get("chat_id")
            if chat_id:
                await self.app.bot.send_message(
                    chat_id=chat_id,
                    text="✅ Syne restarted successfully.",
                )
                logger.info(f"Sent restart-complete notification to chat {chat_id}")
        except Exception as e:
            logger.warning(f"Failed to send restart notification: {e}")
        finally:
            try:
                os.remove(restart_flag)
            except OSError:
                pass

    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped.")

    async def process_scheduled_message(self, chat_id: int, payload: str):
        """Process a scheduled task payload as if the user sent it.
        
        This is called by the scheduler when a task executes.
        
        Args:
            chat_id: Telegram chat ID (user ID for DMs)
            payload: Message payload to process
        """
        if not self.app or not self.app.bot:
            logger.warning("Cannot process scheduled message: bot not initialized")
            return
        
        # Get user info
        user = await get_user("telegram", str(chat_id))
        if not user:
            logger.warning(f"Scheduled message: user {chat_id} not found")
            return
        
        user_name = user.get("display_name") or user.get("name") or str(chat_id)
        access_level = user.get("access_level", "public")
        
        logger.info(f"[scheduled] {user_name} ({chat_id}): {payload[:100]}")
        
        # Send typing indicator
        try:
            await self.app.bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass  # Best-effort
        
        try:
            # Build InboundContext for scheduled messages (DM context)
            from .inbound import InboundContext
            sched_inbound = InboundContext(
                channel="telegram",
                platform="telegram",
                chat_type="direct",
                conversation_label=user_name,
                chat_id=str(chat_id),
            )
            # Process the message via agent (DM context, not group)
            response = await self.agent.handle_message(
                platform="telegram",
                chat_id=str(chat_id),
                user_name=user_name,
                user_platform_id=str(chat_id),
                message=payload,
                display_name=user_name,
                message_metadata={"scheduled": True, "inbound": sched_inbound},
            )
            
            if response:
                # Parse reply tags (no incoming message for cron)
                response, reply_to = parse_reply_tag(response)
                await self._send_response_with_media(chat_id, response, None, reply_to_message_id=reply_to)
        
        except Exception as e:
            logger.error(f"Error processing scheduled message: {e}", exc_info=True)
            try:
                await self.app.bot.send_message(chat_id, _classify_error(e))
            except Exception:
                pass

    # ── Message handlers ─────────────────────────────────────

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        if not update.message or not update.message.text:
            return

        user = update.effective_user
        chat = update.effective_chat
        text = update.message.text
        is_group = chat.type in ("group", "supergroup")

        # ═══════════════════════════════════════════════════════════════
        # AUTH FLOW INTERCEPT: If user is in auth flow, handle credential
        # input here and SKIP history/memory entirely.
        # ═══════════════════════════════════════════════════════════════
        if user.id in self._auth_state and not is_group:
            handled = await self._handle_auth_input(update, context, text)
            if handled:
                return  # Credential processed — do NOT save to history

        # Handle group messages with registration and mention checks
        if is_group:
            result = await self._process_group_message(update, context, text)
            if result is None:
                return  # Message filtered out — not for us, no reaction
            # Message IS for us (mentioned/replied) — send 👀 read receipt
            try:
                await self.send_reaction(chat.id, update.message.message_id, "👀")
            except Exception:
                pass  # Best-effort, don't fail on reaction errors
            text = result

        # Handle DMs - auto-create user
        else:
            db_user = await self._ensure_user(user, is_dm=True)
            # Block rejected/blocked users silently
            if db_user.get("access_level") == "blocked":
                logger.debug(f"Ignoring message from blocked user {user.id}")
                return
            # Check approval policy for DMs
            if db_user.get("access_level") == "pending":
                await self._handle_pending_user(update, db_user)
                return
            # DM received — send 👀 read receipt
            try:
                await self.send_reaction(chat.id, update.message.message_id, "👀")
            except Exception:
                pass

        if not text:
            return

        # ═══════════════════════════════════════════════════════════════
        # CREDENTIAL LEAK PREVENTION: Detect credential patterns in
        # normal chat. Warn user and skip history/memory.
        # ═══════════════════════════════════════════════════════════════
        if not is_group and self._contains_credential(text):
            await update.message.reply_text(
                "⚠️ Credential detected in message. Use `/models` to manage credentials.\n"
                "This message was NOT saved to history or memory.",
                parse_mode="Markdown",
            )
            logger.warning(f"Credential pattern detected in chat from user {user.id} — skipped history")
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): {text[:100]}")

        # Track incoming message ID for reaction context
        message_id = update.message.message_id
        self._track_message(chat.id, message_id, text[:100])

        # Build InboundContext (OpenClaw-style — core module)
        from .inbound import build_user_context_prefix
        inbound = await self._build_inbound(update, is_group)

        # Prepend user context prefix to message (untrusted, per-message)
        original_text = text  # Keep original for memory evaluator
        user_prefix = build_user_context_prefix(inbound)
        if user_prefix:
            text = f"{user_prefix}\n\n{text}"

        # Track this as an active task for /cancel support
        self._active_tasks[chat.id] = asyncio.current_task()
        # Keep typing indicator alive throughout the entire processing
        async with _TypingIndicator(context.bot, chat.id):
            try:
                # Include message_id and inbound context in metadata
                metadata = {
                    "message_id": message_id,
                    "chat_id": str(chat.id),
                    "inbound": inbound,  # Single source of truth for all context
                    "original_text": original_text,  # Without context prefix, for evaluator
                }

                # Browse mode: route to CLI-compatible session with cwd
                browse_cwd = self._browse_cwd.get(user.id) if not is_group else None
                if browse_cwd:
                    import getpass
                    from .inbound import InboundContext
                    username = getpass.getuser()
                    cli_chat_id = f"cli:{username}:{browse_cwd}"
                    metadata["cwd"] = browse_cwd
                    # Override inbound for CLI context
                    metadata["inbound"] = InboundContext(
                        channel="cli",
                        platform="cli",
                        chat_type="direct",
                        conversation_label=self._get_display_name(user),
                        chat_id=cli_chat_id,
                    )

                    # Get the existing Telegram user (don't create a CLI user)
                    tg_user = await get_user("telegram", str(user.id))
                    if tg_user:
                        # Set exec working directory to browse path
                        prev_cwd = self.agent._cli_cwd
                        self.agent._cli_cwd = browse_cwd
                        try:
                            response = await self.agent.conversations.handle_message(
                                platform="cli",
                                chat_id=cli_chat_id,
                                user=tg_user,
                                message=text,
                                message_metadata=metadata,
                            )
                        finally:
                            self.agent._cli_cwd = prev_cwd
                    else:
                        response = "⚠️ User not found. Send any message first to register."
                else:
                    response = await self.agent.handle_message(
                        platform="telegram",
                        chat_id=str(chat.id),
                        user_name=user.first_name or user.username or str(user.id),
                        user_platform_id=str(user.id),
                        message=text,
                        display_name=self._get_display_name(user),
                        message_metadata=metadata,
                        is_dm=not is_group,
                    )

                if not response:
                    response = "⚠️ LLM returned an empty response. Please try again."
                    logger.warning(f"Empty response for chat {chat.id} — sending fallback")

                if response:
                    # Check if per-model reasoning visibility is ON — prepend thinking if available
                    if browse_cwd:
                        import getpass as _gp
                        key = f"cli:cli:{_gp.getuser()}:{browse_cwd}"
                    else:
                        key = f"telegram:{chat.id}"
                    conv = self.agent.conversations._active.get(key)
                    if conv and conv.reasoning_visible:
                        thinking = getattr(conv, '_last_thinking', None)
                        if thinking:
                            thinking_block = f"💭 **Thinking:**\n_{thinking[:3000]}_\n\n"
                            response = thinking_block + response

                    # Parse reply and react tags from LLM response
                    response, reply_to = parse_reply_tag(response, message_id)
                    response, react_emojis = parse_react_tags(response)

                    # Send reaction to incoming message if requested
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, message_id, emoji)

                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    # Track bot's response for reaction context
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:100])

            except asyncio.CancelledError:
                logger.info(f"Processing cancelled by user for chat {chat.id}")
                return
            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                await update.message.reply_text(_classify_error(e))
            finally:
                self._active_tasks.pop(chat.id, None)

    async def _process_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str
    ) -> Optional[str]:
        """Process a group message with registration and mention checks.
        
        Returns:
            Processed message text if should be handled, None if filtered out.
        """
        chat = update.effective_chat
        user = update.effective_user
        group_id = str(chat.id)
        
        # Check group registration
        group = await get_group("telegram", group_id)
        
        # Get group policy
        group_policy = await get_config("telegram.group_policy", "allowlist")
        
        if group_policy == "allowlist" and not group:
            logger.debug(f"Ignoring message from unregistered group {group_id}")
            return None
        
        if group and not group.get("enabled", True):
            logger.debug(f"Ignoring message from disabled group {group_id}")
            return None
        
        # Check allow_from policy
        if group and group.get("allow_from") == "registered":
            existing_user = await get_user("telegram", str(user.id))
            if not existing_user:
                logger.debug(f"Ignoring message from unregistered user {user.id} in group {group_id}")
                return None
        
        # Note: group users are NOT registered in global users table.
        # Only DM users appear in /members. Group members are tracked
        # separately via update_group_member below.
        
        # Auto-collect group member — store from.id + name + username
        # Never overwrites alias or access (those are manual-only)
        try:
            from ..db.models import update_group_member
            await update_group_member(
                platform="telegram",
                platform_group_id=group_id,
                member_id=str(user.id),
                name=user.first_name or user.username or str(user.id),
                username=user.username,
            )
        except Exception as e:
            logger.debug(f"Auto-collect member failed: {e}")
        
        # Check if mention is required
        require_mention = True
        if group:
            require_mention = group.get("require_mention", True)
        else:
            require_mention = await get_config("telegram.require_mention", True)
        
        if require_mention:
            # Check for bot @username mention
            bot_username = (await context.bot.get_me()).username
            
            # Get trigger name from identity
            trigger_name = await self._get_trigger_name()
            
            # Check if mentioned by @username or trigger name
            is_mentioned, processed_text = self._check_and_strip_mention(
                text, bot_username, trigger_name
            )
            
            # Also allow replies to bot
            if not is_mentioned and not self._is_reply_to_bot(update):
                return None
            
            return processed_text
        
        return text

    async def _ensure_user(self, tg_user, is_dm: bool = False) -> dict:
        """Ensure user exists in database, creating if needed.
        
        Args:
            tg_user: Telegram user object
            is_dm: True if this is from a direct/private message.
                   Only DM interactions can auto-promote to owner.
        """
        return await get_or_create_user(
            name=tg_user.first_name or tg_user.username or str(tg_user.id),
            platform="telegram",
            platform_id=str(tg_user.id),
            display_name=self._get_display_name(tg_user),
            is_dm=is_dm,
        )

    async def _handle_pending_user(self, update: Update, db_user: dict):
        """Handle messages from pending (unapproved) users.
        
        Sends a polite waiting message to the user and notifies the owner
        with approve/reject inline buttons.
        """
        user = update.effective_user
        chat = update.effective_chat
        user_name = self._get_display_name(user)
        
        # Reply to user
        await update.message.reply_text(
            "⏳ Your access is pending approval from the owner. Please wait."
        )
        
        # Find owner to notify
        from ..db.connection import get_connection
        async with get_connection() as conn:
            owner_row = await conn.fetchrow(
                "SELECT platform_id FROM users WHERE platform = 'telegram' AND access_level = 'owner' LIMIT 1"
            )
        
        if not owner_row:
            logger.warning("No owner found to notify about pending user")
            return
        
        owner_chat_id = int(owner_row["platform_id"])
        
        # Don't spam owner — check if we already notified recently
        # Use a simple in-memory tracker
        pending_key = f"pending_notify:{user.id}"
        if not hasattr(self, '_pending_notified'):
            self._pending_notified = set()
        
        if pending_key in self._pending_notified:
            return  # Already notified owner about this user
        
        self._pending_notified.add(pending_key)
        
        # Build notification with approve/reject buttons
        username_str = f" (@{user.username})" if user.username else ""
        buttons = [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve:{user.id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject:{user.id}"),
            ]
        ]
        
        for attempt in range(3):
            try:
                await self.app.bot.send_message(
                    chat_id=owner_chat_id,
                    text=(
                        f"🔔 **New user wants access:**\n\n"
                        f"• Name: {user_name}{username_str}\n"
                        f"• ID: `{user.id}`\n\n"
                        f"Approve or reject?"
                    ),
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(buttons),
                )
                break
            except Exception as e:
                logger.error(f"Failed to notify owner about pending user (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

    async def _handle_my_chat_member(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle when bot is added to or removed from a group.
        
        When added to a new group, notify owner with approve/reject buttons.
        Bot stays silent in the group until approved.
        """
        member_update = update.my_chat_member
        if not member_update:
            return
        
        chat = member_update.chat
        new_status = member_update.new_chat_member.status
        old_status = member_update.old_chat_member.status
        
        # Only handle group additions
        if chat.type not in ("group", "supergroup"):
            return
        
        # Bot was added to group (status changed to "member" or "administrator")
        # Check old_status broadly — covers "left", "kicked", AND "restricted"
        if new_status in ("member", "administrator") and old_status not in ("member", "administrator"):
            logger.info(f"Bot added to group: {chat.title} ({chat.id})")
            
            # Check if group is already registered
            group = await get_group("telegram", str(chat.id))
            if group:
                logger.info(f"Group {chat.id} already registered, enabled={group.get('enabled')}")
                return
            
            # New group — notify owner for approval
            added_by = member_update.from_user
            added_by_name = self._get_display_name(added_by) if added_by else "Unknown"
            added_by_username = f" (@{added_by.username})" if added_by and added_by.username else ""
            
            from ..db.connection import get_connection
            async with get_connection() as conn:
                owner_row = await conn.fetchrow(
                    "SELECT platform_id FROM users WHERE platform = 'telegram' AND access_level = 'owner' LIMIT 1"
                )
            
            if not owner_row:
                logger.warning("No owner found to notify about group addition")
                return
            
            owner_chat_id = int(owner_row["platform_id"])
            
            buttons = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"group_approve:{chat.id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"group_reject:{chat.id}"),
                ]
            ]
            
            for attempt in range(3):
                try:
                    await self.app.bot.send_message(
                        chat_id=owner_chat_id,
                        text=(
                            f"🔔 **Bot added to a new group:**\n\n"
                            f"• Group: {chat.title}\n"
                            f"• ID: `{chat.id}`\n"
                            f"• Added by: {added_by_name}{added_by_username}\n\n"
                            f"Approve this group?"
                        ),
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(buttons),
                    )
                    break
                except Exception as e:
                    logger.error(f"Failed to notify owner about group addition (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
        
        # Bot was removed from group (covers "member", "administrator", AND "restricted")
        elif new_status in ("left", "kicked") and old_status not in ("left", "kicked"):
            logger.info(f"Bot removed from group: {chat.title} ({chat.id})")
            
            # Delete group from DB — clean slate if re-added later
            from ..db.connection import get_connection
            try:
                async with get_connection() as conn:
                    await conn.execute(
                        "DELETE FROM groups WHERE platform = 'telegram' AND platform_group_id = $1",
                        str(chat.id),
                    )
                logger.info(f"Deleted group {chat.id} from database")
            except Exception as e:
                logger.error(f"Failed to delete group {chat.id}: {e}")


    async def _get_trigger_name(self) -> str:
        """Get the bot trigger name from config or identity."""
        # First check explicit config
        trigger = await get_config("telegram.bot_trigger_name", None)
        if trigger:
            return trigger
        
        # Fall back to identity name
        identity = await get_identity()
        return identity.get("name", "Syne")

    def _check_and_strip_mention(
        self, text: str, bot_username: str, trigger_name: str
    ) -> Tuple[bool, str]:
        """Check if bot is mentioned and strip the mention from text.
        
        Returns:
            Tuple of (is_mentioned, processed_text)
        """
        original_text = text
        is_mentioned = False
        
        # Check @username (case-insensitive)
        username_pattern = re.compile(rf"@{re.escape(bot_username)}\b", re.IGNORECASE)
        if username_pattern.search(text):
            is_mentioned = True
            text = username_pattern.sub("", text).strip()
        
        # Check trigger name (case-insensitive, word boundary)
        if trigger_name:
            # Match trigger name at word boundaries, with optional comma/colon after
            trigger_pattern = re.compile(
                rf"\b{re.escape(trigger_name)}[,:]?\s*", re.IGNORECASE
            )
            if trigger_pattern.search(text):
                is_mentioned = True
                text = trigger_pattern.sub("", text).strip()
        
        return is_mentioned, text if text else original_text

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages — download, encode, and send to LLM with vision."""
        if not update.message:
            return

        import base64
        import os

        caption = update.message.caption or ""
        user = update.effective_user
        chat = update.effective_chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting

        # Group checks (same as text messages)
        if is_group:
            # For photos in groups, only process if replying to bot or caption mentions bot
            if caption:
                result = await self._process_group_message(update, context, caption)
                if result is None:
                    return
                caption = result
            elif not self._is_reply_to_bot(update):
                return  # Ignore photos in groups without mention/reply

        # Only register DM users; group users stay in group_members only
        if not is_group:
            db_user = await self._ensure_user(user, is_dm=True)
        else:
            db_user = {"access_level": "public"}

        # Block pending users
        if not is_group and db_user.get("access_level") == "pending":
            await self._handle_pending_user(update, db_user)
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): [photo] {caption[:100]}")

        # Reply context is now handled by InboundContext (build_user_context_prefix)

        # Keep typing indicator alive throughout photo processing
        async with _TypingIndicator(context.bot, chat.id):
            try:
                # Download the largest photo version
                photo = update.message.photo[-1]  # Largest size
                tg_file = await photo.get_file()

                # Download to memory and encode as base64
                photo_bytes = await tg_file.download_as_bytearray()

                if not photo_bytes or len(photo_bytes) == 0:
                    logger.error(f"Downloaded photo is empty (0 bytes)")
                    await update.message.reply_text("Sorry, couldn't download the photo. Try sending again.")
                    return

                photo_b64 = base64.b64encode(bytes(photo_bytes)).decode("utf-8")
                logger.info(f"Photo downloaded: {len(photo_bytes)} bytes, base64: {len(photo_b64)} chars")

                # Build message with image metadata for vision
                user_text = caption if caption else "What's in this image?"
                inbound = await self._build_inbound(update, is_group)
                from .inbound import build_user_context_prefix
                prefix = build_user_context_prefix(inbound)
                if prefix:
                    user_text = f"{prefix}\n\n{user_text}"
                metadata = {
                    "image": {
                        "mime_type": "image/jpeg",
                        "base64": photo_b64,
                    },
                    "inbound": inbound,
                }

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_text,
                    display_name=self._get_display_name(user),
                    message_metadata=metadata,
                    is_dm=not is_group,
                )

                if response:
                    response, reply_to = parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)

            except Exception as e:
                logger.error(f"Error handling photo: {e}", exc_info=True)
                await update.message.reply_text(_classify_error(e))

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages and audio files — transcribe via STT and process as text."""
        if not update.message:
            return

        user = update.effective_user
        chat = update.effective_chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting

        # Group checks — voice in groups requires mention/reply context
        if is_group:
            if not self._is_reply_to_bot(update):
                return  # Ignore voice in groups unless replying to bot

        # Only register DM users; group users stay in group_members only
        if not is_group:
            db_user = await self._ensure_user(user, is_dm=True)
        else:
            db_user = {"access_level": "public"}

        # Block pending users
        if not is_group and db_user.get("access_level") == "pending":
            await self._handle_pending_user(update, db_user)
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): [voice message]")

        # Keep typing indicator while transcribing
        async with _TypingIndicator(context.bot, chat.id):
            try:
                # Get the voice/audio file
                voice = update.message.voice or update.message.audio
                if not voice:
                    return

                tg_file = await voice.get_file()
                audio_bytes = await tg_file.download_as_bytearray()

                if not audio_bytes or len(audio_bytes) == 0:
                    logger.error("Downloaded voice is empty (0 bytes)")
                    await update.message.reply_text("Sorry, couldn't download the voice message.")
                    return

                logger.info(f"Voice downloaded: {len(audio_bytes)} bytes")

                # Transcribe using STT
                from ..tools.voice import transcribe_audio
                
                # Use file_unique_id as filename hint
                filename = f"voice_{voice.file_unique_id}.ogg"
                success, result = await transcribe_audio(bytes(audio_bytes), filename)

                if not success:
                    logger.error(f"Transcription failed: {result}")
                    await update.message.reply_text(f"⚠️ Couldn't transcribe: {result[:200]}")
                    return

                transcribed_text = result
                logger.info(f"Transcribed: {transcribed_text[:100]}")

                # Track this message ID
                self._track_message(chat.id, update.message.message_id, f"[voice] {transcribed_text[:50]}")

                # Reply context handled by InboundContext

                # Process the transcribed text as a normal message
                # Include transcription indicator so agent knows it's from voice
                user_message = f"[Voice message transcription]: {transcribed_text}"
                inbound = await self._build_inbound(update, is_group)
                from .inbound import build_user_context_prefix
                prefix = build_user_context_prefix(inbound)
                if prefix:
                    user_message = f"{prefix}\n\n{user_message}"
                
                metadata = {
                    "message_id": update.message.message_id,
                    "voice_transcription": True,
                    "original_text": transcribed_text,
                    "inbound": inbound,
                }

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_message,
                    display_name=self._get_display_name(user),
                    message_metadata=metadata,
                    is_dm=not is_group,
                )

                if response:
                    response, reply_to = parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    # Track bot's response message ID
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:50])

            except Exception as e:
                logger.error(f"Error handling voice: {e}", exc_info=True)
                await update.message.reply_text(_classify_error(e))

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document/file uploads — download, save to disk, pass path to LLM."""
        if not update.message or not update.message.document:
            return

        import base64
        import os
        import tempfile

        caption = update.message.caption or ""
        user = update.effective_user
        chat = update.effective_chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting

        # Group checks — documents in groups require mention/reply
        if is_group:
            if caption:
                result = await self._process_group_message(update, context, caption)
                if result is None:
                    return
                caption = result
            elif not self._is_reply_to_bot(update):
                return

        db_user = await self._ensure_user(user, is_dm=not is_group)

        # Block pending users
        if not is_group and db_user.get("access_level") == "pending":
            await self._handle_pending_user(update, db_user)
            return

        doc = update.message.document
        filename = doc.file_name or "unknown_file"
        mime_type = doc.mime_type or "application/octet-stream"
        file_size = doc.file_size or 0

        # Telegram Bot API file download limit: 20MB
        if file_size > 20 * 1024 * 1024:
            await update.message.reply_text("⚠️ File too large (max 20 MB for download).")
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): [document] {filename} ({mime_type}, {file_size} bytes)")

        # Extract reply/quote context
        # Reply context handled by InboundContext

        async with _TypingIndicator(context.bot, chat.id):
            try:
                tg_file = await doc.get_file()
                file_bytes = await tg_file.download_as_bytearray()

                if not file_bytes or len(file_bytes) == 0:
                    await update.message.reply_text("Sorry, couldn't download the file. Try sending again.")
                    return

                # Save to workspace/uploads/
                uploads_dir = self.agent.workspace_uploads
                os.makedirs(uploads_dir, exist_ok=True)

                # Use unique name to avoid collisions
                safe_name = filename.replace("/", "_").replace("..", "_")
                save_path = os.path.join(uploads_dir, f"{doc.file_unique_id}_{safe_name}")
                with open(save_path, "wb") as f:
                    f.write(file_bytes)

                logger.info(f"Document saved: {save_path} ({len(file_bytes)} bytes)")

                # Build metadata for the agent
                metadata = {
                    "document": {
                        "path": save_path,
                        "filename": filename,
                        "mime_type": mime_type,
                        "size": len(file_bytes),
                    },
                    "message_id": update.message.message_id,
                }

                # For images sent as documents (uncompressed), treat as image
                if mime_type and mime_type.startswith("image/"):
                    photo_b64 = base64.b64encode(bytes(file_bytes)).decode("utf-8")
                    metadata["image"] = {
                        "mime_type": mime_type,
                        "base64": photo_b64,
                    }

                # For PDF files, also include base64 for ability pre-processing
                if mime_type == "application/pdf":
                    doc_b64 = base64.b64encode(bytes(file_bytes)).decode("utf-8")
                    metadata["document"]["base64"] = doc_b64

                # Construct user message
                if caption:
                    user_text = f"[User sent a file: {filename} ({mime_type}, {len(file_bytes)} bytes), saved at: {save_path}]\n\n{caption}"
                else:
                    user_text = f"[User sent a file: {filename} ({mime_type}, {len(file_bytes)} bytes), saved at: {save_path}]"

                self._track_message(chat.id, update.message.message_id, f"[doc] {filename}")

                inbound = await self._build_inbound(update, is_group)
                from .inbound import build_user_context_prefix
                prefix = build_user_context_prefix(inbound)
                if prefix:
                    user_text = f"{prefix}\n\n{user_text}"
                metadata["inbound"] = inbound

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_text,
                    display_name=self._get_display_name(user),
                    message_metadata=metadata,
                    is_dm=not is_group,
                )

                if response:
                    response, reply_to = parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:50])

            except Exception as e:
                logger.error(f"Error handling document: {e}", exc_info=True)
                await update.message.reply_text(_classify_error(e))

    async def _handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle location messages — reverse geocode and pass address to LLM."""
        if not update.message or not update.message.location:
            return

        user = update.message.from_user
        chat = update.message.chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting

        # Group checks
        if is_group and not self._is_reply_to_bot(update):
            return  # Ignore location in groups without reply to bot

        db_user = await self._ensure_user(user, is_dm=not is_group)

        # Block pending users
        if not is_group and db_user.get("access_level") == "pending":
            await self._handle_pending_user(update, db_user)
            return

        location = update.message.location
        lat = location.latitude
        lng = location.longitude
        logger.info(f"Location received from Telegram API: lat={lat}, lng={lng}")

        # Check for venue (location with name/address)
        venue = update.message.venue
        caption = ""
        if update.message.caption:
            caption = update.message.caption

        # Reverse geocode to get actual address — don't trust LLM with raw coords
        address = await self._reverse_geocode(lat, lng)
        logger.info(f"Reverse geocode result: {address}")

        # Build message with location data
        if venue:
            location_text = (
                f"[Location shared: {venue.title}"
                f"{f' — {venue.address}' if venue.address else ''}"
                f" (lat: {lat}, lng: {lng})]"
            )
        elif address:
            location_text = f"[Location shared: {address} (lat: {lat}, lng: {lng})]"
        else:
            location_text = f"[Location shared: lat: {lat}, lng: {lng}]"

        # Build inbound context
        inbound = await self._build_inbound(update, is_group)
        from .inbound import build_user_context_prefix
        prefix = build_user_context_prefix(inbound)

        user_text = ""
        if prefix:
            user_text += prefix + "\n\n"
        if caption:
            user_text += caption + "\n\n"
        user_text += location_text

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): {location_text}")

        metadata = {
            "message_id": update.message.message_id,
            "location": {"latitude": lat, "longitude": lng},
            "inbound": inbound,
        }
        if venue:
            metadata["venue"] = {
                "title": venue.title,
                "address": venue.address or "",
            }

        async with _TypingIndicator(context.bot, chat.id):
            try:
                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_text,
                    display_name=self._get_display_name(user),
                    message_metadata=metadata,
                    is_dm=not is_group,
                )

                if response:
                    response, reply_to = parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)

            except Exception as e:
                logger.error(f"Error handling location: {e}", exc_info=True)
                await update.message.reply_text(_classify_error(e))

    async def _reverse_geocode(self, lat: float, lng: float) -> str | None:
        """Reverse geocode coordinates to address using Google Maps API."""
        try:
            api_key = None
            # Try to get API key from maps ability config
            if self.agent and hasattr(self.agent, 'db'):
                row = await self.agent.db.fetchrow(
                    "SELECT config FROM abilities WHERE name = 'maps' AND enabled = true"
                )
                if row and row['config']:
                    import json
                    config = json.loads(row['config']) if isinstance(row['config'], str) else row['config']
                    api_key = (
                        config.get('api_key')
                        or config.get('GOOGLE_MAPS_API_KEY')
                        or config.get('GOOGLE_PLACES_API_KEY')
                    )
            # Fallback to environment
            if not api_key:
                import os
                api_key = os.environ.get('GOOGLE_PLACES_API_KEY') or os.environ.get('GOOGLE_MAPS_API_KEY')
            if not api_key:
                return None

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={"latlng": f"{lat},{lng}", "key": api_key, "language": "id"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK" and data.get("results"):
                        return data["results"][0].get("formatted_address", None)
        except Exception as e:
            logger.warning(f"Reverse geocode failed: {e}")
        return None

    async def _handle_reaction_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle reaction updates on messages."""
        reaction = update.message_reaction
        if not reaction:
            return

        chat = reaction.chat
        user = reaction.user
        message_id = reaction.message_id

        # Get the new reactions (what was added)
        new_reactions = reaction.new_reaction or []
        old_reactions = reaction.old_reaction or []

        if not new_reactions and not old_reactions:
            return

        # Determine what changed
        added_emojis = []
        removed_emojis = []

        new_emoji_set = {r.emoji if hasattr(r, 'emoji') else str(r) for r in new_reactions}
        old_emoji_set = {r.emoji if hasattr(r, 'emoji') else str(r) for r in old_reactions}

        added_emojis = list(new_emoji_set - old_emoji_set)
        removed_emojis = list(old_emoji_set - new_emoji_set)

        if not added_emojis and not removed_emojis:
            return

        # Build reaction event message
        user_name = self._get_display_name(user) if user else "Someone"
        user_id = str(user.id) if user else "unknown"

        # Get message preview from our tracking
        msg_preview = self._get_message_preview(chat.id, message_id)
        
        if added_emojis:
            emoji_str = " ".join(added_emojis)
            event_text = f"[Reaction: {emoji_str} from {user_name} on your message: '{msg_preview}']"
            logger.info(f"Reaction received: {emoji_str} from {user_id} on msg {message_id}")

            # Route reaction to agent so it can decide to respond
            try:
                from .inbound import InboundContext
                is_group = chat.type != "private"
                react_inbound = InboundContext(
                    channel="telegram",
                    platform="telegram",
                    chat_type="group" if is_group else "direct",
                    conversation_label=user_name,
                    group_subject=chat.title if is_group else None,
                    chat_id=str(chat.id),
                    sender_name=user_name if is_group else None,
                    sender_id=user_id if is_group else None,
                )
                if is_group:
                    from .inbound import load_group_settings
                    await load_group_settings(react_inbound)
                metadata = {
                    "message_id": message_id,
                    "chat_id": str(chat.id),
                    "is_reaction": True,
                    "reaction_emojis": added_emojis,
                    "inbound": react_inbound,
                }
                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user_name,
                    user_platform_id=user_id,
                    message=event_text,
                    display_name=user_name,
                    message_metadata=metadata,
                    is_dm=not is_group,
                )
                if response and response.strip().upper() != "NO_REPLY":
                    response, reply_to = parse_reply_tag(response, message_id)
                    sent = await self._send_response_with_media(
                        chat.id, response, context, reply_to_message_id=reply_to
                    )
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:100])
            except Exception as e:
                logger.error(f"Error handling reaction event: {e}", exc_info=True)

    def _track_message(self, chat_id: int, message_id: int, preview: str):
        """Track a message ID for reaction context."""
        if chat_id not in self._recent_messages:
            self._recent_messages[chat_id] = deque(maxlen=20)
        self._recent_messages[chat_id].append((message_id, preview[:100]))

    def _get_message_preview(self, chat_id: int, message_id: int) -> str:
        """Get a preview of a tracked message."""
        if chat_id not in self._recent_messages:
            return f"(msg #{message_id})"
        for mid, preview in self._recent_messages[chat_id]:
            if mid == message_id:
                return preview
        return f"(msg #{message_id})"

    async def send_reaction(self, chat_id: int, message_id: int, emoji: str) -> bool:
        """Send a reaction to a message.
        
        Args:
            chat_id: The chat ID
            message_id: The message ID to react to
            emoji: The reaction emoji
            
        Returns:
            True if successful, False otherwise
        """
        if not self.app or not self.app.bot:
            logger.error("Bot not initialized")
            return False

        try:
            await self.app.bot.set_message_reaction(
                chat_id=chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
            logger.info(f"Sent reaction {emoji} to message {message_id} in chat {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send reaction: {e}")
            return False

    def get_recent_messages(self, chat_id: int) -> list[tuple[int, str]]:
        """Get recent message IDs and previews for a chat.
        
        Returns:
            List of (message_id, preview) tuples, most recent first
        """
        if chat_id not in self._recent_messages:
            return []
        return list(reversed(self._recent_messages[chat_id]))

    # ── Command handlers ─────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        from ..db.models import get_identity
        identity = await get_identity()
        name = identity.get("name", "Syne")
        motto = identity.get("motto", "")

        welcome = f"Hi! I'm **{name}**."
        if motto:
            welcome += f"\n_{motto}_"
        welcome += "\n\nSend me a message to get started!"

        await update.message.reply_text(welcome, parse_mode="Markdown")

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """**Available commands:**

/start — Welcome message
/help — This help message
/version — Version info
/status — Show agent status
/memory — Show memory stats
/compact — Compact conversation history
/autocapture — Toggle auto memory capture (on/off)
/models — Manage LLM models
/embedding — Manage embedding models
/evaluator — Manage evaluator model
/groups — Manage groups & members
/members — Manage global user access
/clear — Clear conversation history
/identity — Show agent identity
/browse — Browse directories (share session with CLI)
/cancel — Cancel active auth flow

Or just send me a message!"""

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /v command — show version."""
        from .. import __version__
        from ..db.models import get_identity

        identity = await get_identity()
        name = identity.get("name", "Syne")

        await update.message.reply_text(
            f"🧬 **{name}** v{__version__}",
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command — comprehensive agent status."""
        from ..db.connection import get_connection
        from ..db.models import get_identity, get_config
        from ..compaction import get_session_stats

        chat_id = str(update.effective_chat.id)

        async with get_connection() as conn:
            mem_count = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
            user_count = await conn.fetchrow("SELECT COUNT(*) as c FROM users")
            session_count = await conn.fetchrow(
                "SELECT COUNT(*) as c FROM sessions WHERE status = 'active'"
            )
            group_count = await conn.fetchrow(
                "SELECT COUNT(*) as c FROM groups WHERE enabled = true"
            )
            ability_count = await conn.fetchrow(
                "SELECT COUNT(*) as c FROM abilities WHERE enabled = true"
            )

            # Current session info
            session_row = await conn.fetchrow("""
                SELECT id, message_count, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE session_id = s.id 
                        AND metadata @> '{"type": "compaction_summary"}'::jsonb) as compactions
                FROM sessions s
                WHERE platform = 'telegram' AND platform_chat_id = $1 AND status = 'active'
                ORDER BY updated_at DESC LIMIT 1
            """, chat_id)

        # Identity
        identity = await get_identity()
        name = identity.get("name", "Syne")

        # Get model registry and active model
        models = await get_config("provider.models", [])
        active_model_key = await get_config("provider.active_model", None)
        active_model_entry = next((m for m in models if m.get("key") == active_model_key), None) if models and active_model_key else None
        
        # Model name — prefer registry, fallback to legacy key
        if active_model_entry:
            chat_model = active_model_entry.get("model_id", active_model_key)
        else:
            chat_model = await get_config("provider.chat_model", "unknown")
        
        auto_capture = await get_config("memory.auto_capture", False)
        # Read thinking/reasoning from active model entry (per-model, not global)
        _model_params = (active_model_entry.get("params") or {}) if active_model_entry else {}
        thinking_budget = _model_params.get("thinking_budget")
        reasoning_visible = bool(active_model_entry.get("reasoning_visible", False)) if active_model_entry else False

        # Context window and driver name from registry
        context_window = int(active_model_entry.get("context_window", 128000)) if active_model_entry else 128000
        provider_name = active_model_entry.get("driver", self.agent.provider.name) if active_model_entry else self.agent.provider.name
        
        # Get active embedding info
        embed_models = await get_config("provider.embedding_models", [])
        active_embed_key = await get_config("provider.active_embedding", None)
        active_embed_entry = next((m for m in embed_models if m.get("key") == active_embed_key), None) if embed_models and active_embed_key else None
        embed_label = active_embed_entry.get("label", "Together AI") if active_embed_entry else "Together AI"
        
        # Tools & abilities
        tool_count = len(self.agent.tools.list_tools("owner"))
        abilities = ability_count["c"] if ability_count else 0

        # Session details
        session_info = ""
        if session_row:
            msg_count = session_row["message_count"]
            compactions = session_row["compactions"]

            # Estimate context usage
            stats = await get_session_stats(session_row["id"])
            chars = stats["total_chars"]
            est_tokens = int(chars / 3.5)
            reserved = self.agent.provider.reserved_output_tokens if self.agent else 4096
            available = context_window - reserved
            pct = round(est_tokens / available * 100) if available > 0 else 0

            # Format context window for display
            if context_window >= 1000000:
                ctx_display = f"{context_window / 1000000:.1f}M"
            else:
                ctx_display = f"{context_window // 1000}K"

            # Compaction thresholds
            msg_thresh = int(await get_config("session.max_messages", 100))
            chr_thresh = int(await get_config("session.compaction_threshold", 150000))

            # Build progress bar (10 blocks)
            filled = min(10, pct // 10)
            bar = "█" * filled + "░" * (10 - filled)

            # Compact trigger indicators
            token_trigger = pct >= 90
            msg_trigger = msg_count >= msg_thresh
            char_trigger = chars >= chr_thresh

            session_info = (
                f"📐 Context: {ctx_display} ({available:,} usable)\n"
                f"`[{bar}]` {pct}% — ~{est_tokens:,}/{available:,} tokens\n"
                f"📋 Messages: {msg_count}/{msg_thresh}"
                f"{' ⚠️' if msg_trigger else ''}\n"
                f"📝 Chars: {chars:,}/{chr_thresh:,}"
                f"{' ⚠️' if char_trigger else ''}\n"
                f"🧹 Compactions: {compactions}"
                f"{' | ⚡ compact soon' if (token_trigger or msg_trigger or char_trigger) else ''}"
            )

        from .. import __version__ as syne_version

        status_lines = [
            f"🧠 **{name} Status** · Syne v{syne_version}",
            "",
            f"🤖 Model: `{chat_model}` ({provider_name})",
            f"🧬 Embedding: {embed_label}",
            f"📚 Memories: {mem_count['c']}",
            f"👥 Users: {user_count['c']} | Groups: {group_count['c']}",
            f"💬 Active sessions: {session_count['c']}",
            f"🔧 Tools: {tool_count} | Abilities: {abilities}",
            f"💭 Thinking: {self._format_thinking_level(thinking_budget)} | Reasoning: {'ON' if reasoning_visible else 'OFF'}",
            f"📝 Auto-capture: {'ON' if auto_capture else 'OFF'}",
        ]

        # Evaluator model
        eval_models = await get_config("memory.evaluator_models", [])
        active_eval_key = await get_config("memory.active_evaluator", None)
        active_eval_entry = next((m for m in eval_models if m.get("key") == active_eval_key), None) if eval_models and active_eval_key else None
        eval_label = active_eval_entry.get("label", active_eval_key or "?") if active_eval_entry else (active_eval_key or "none")
        status_lines.append(f"🔬 Evaluator: {eval_label}")

        # Credential summary
        try:
            cred_parts = []
            from ..db.credentials import get_google_oauth_credentials, get_credential
            import time as _time
            g_creds = await get_google_oauth_credentials()
            if g_creds and g_creds.get("refresh_token"):
                exp = g_creds.get("expires_at", 0)
                cred_parts.append("Google " + ("✅" if _time.time() < exp else "⚠️"))
            codex_token = await get_credential("credential.codex_access_token")
            if codex_token:
                exp = float(await get_credential("credential.codex_expires_at", 0) or 0)
                cred_parts.append("Codex " + ("✅" if _time.time() < exp else "⚠️"))
            together_key = await get_credential("credential.together_api_key")
            if together_key:
                cred_parts.append("Together ✅")
            groq_key = await get_credential("credential.groq_api_key")
            if groq_key:
                cred_parts.append("Groq ✅")
            if cred_parts:
                status_lines.append(f"🔐 Credentials: {' | '.join(cred_parts)}")
        except Exception:
            pass

        # Browse mode indicator
        browse_cwd = self._browse_cwd.get(update.effective_user.id)
        if browse_cwd:
            status_lines.append(f"📂 Browse: `{browse_cwd}`")

        if session_info:
            status_lines.append("")
            status_lines.append("**Current session:**")
            status_lines.append(session_info)

        # Escape underscores for Telegram Markdown (e.g., google_cca → google\_cca)
        status_text = "\n".join(status_lines)
        # Only escape underscores NOT inside backtick blocks
        import re
        def _escape_md_underscores(text: str) -> str:
            """Escape underscores outside of backtick-delimited code spans."""
            parts = text.split("`")
            for i in range(len(parts)):
                if i % 2 == 0:  # Outside backticks
                    parts[i] = parts[i].replace("_", "\\_")
            return "`".join(parts)
        
        status_text = _escape_md_underscores(status_text)
        
        await update.message.reply_text(status_text, parse_mode="Markdown")

    async def _cmd_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /memory command — show memory and session stats."""
        from ..db.connection import get_connection

        async with get_connection() as conn:
            total = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
            by_cat = await conn.fetch("""
                SELECT category, COUNT(*) as c
                FROM memory GROUP BY category ORDER BY c DESC
            """)
            session_count = await conn.fetchrow("SELECT COUNT(*) as c FROM sessions")
            message_count = await conn.fetchrow("SELECT COUNT(*) as c FROM messages")

        cat_parts = " • ".join(f"{row['category']}: {row['c']}" for row in by_cat)
        lines = [
            f"🧠 Memory: {total['c']} items",
            cat_parts,
            "",
            f"💬 Sessions: {session_count['c']}",
            f"📨 Messages: {message_count['c']}",
        ]

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_compact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compact command — summarize old messages to free context."""
        from ..compaction import compact_session, get_session_stats
        from ..db.connection import get_connection

        chat_id = str(update.effective_chat.id)
        user = update.effective_user

        # Only owner can compact
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can compact sessions.")
            return

        # Find active session
        async with get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT id FROM sessions
                WHERE platform = 'telegram' AND platform_chat_id = $1 AND status = 'active'
                ORDER BY updated_at DESC LIMIT 1
            """, chat_id)

        if not row:
            await update.message.reply_text("No active session to compact.")
            return

        session_id = row["id"]

        # Get pre-compact stats
        stats = await get_session_stats(session_id)
        if stats["message_count"] <= 25:
            await update.message.reply_text(
                f"📊 Session only has {stats['message_count']} messages ({stats['total_chars']:,} chars). "
                f"Not enough to compact."
            )
            return

        await update.message.reply_text(
            f"🔄 Compacting session...\n"
            f"Messages: {stats['message_count']} | Chars: {stats['total_chars']:,}"
        )

        try:
            result = await compact_session(
                session_id=session_id,
                provider=self.agent.provider,
            )

            if result:
                # Clear cached conversation so it reloads
                key = f"telegram:{chat_id}"
                if key in self.agent.conversations._active:
                    await self.agent.conversations._active[key].load_history()

                await update.message.reply_text(
                    f"✅ **Compaction complete**\n\n"
                    f"Messages: {result['messages_before']} → {result['messages_after']}\n"
                    f"Chars: {result['chars_before']:,} → {result['chars_after']:,}\n"
                    f"Summarized: {result['messages_summarized']} messages → {result['summary_length']:,} char summary",
                    parse_mode="Markdown",
                )
            else:
                await update.message.reply_text("Nothing to compact.")

        except Exception as e:
            logger.error(f"Compaction failed: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Compaction failed: {str(e)[:200]}")

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command — archive current session and start fresh."""
        from ..db.connection import get_connection

        chat_id = str(update.effective_chat.id)

        async with get_connection() as conn:
            await conn.execute("""
                UPDATE sessions SET status = 'archived', updated_at = NOW()
                WHERE platform = 'telegram' AND platform_chat_id = $1 AND status = 'active'
            """, chat_id)

        # Clear from active conversations
        key = f"telegram:{chat_id}"
        self.agent.conversations._active.pop(key, None)

        await update.message.reply_text("Session cleared. Starting fresh! 🔄")

    async def _cmd_evaluator(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /evaluator command — manage evaluator model for auto-capture."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can manage the evaluator.")
            return
        await self._eval_menu_main(update)

    async def _eval_menu_main(self, update_or_query):
        """Render evaluator model list + auto-capture toggle."""
        eval_models = await get_config("memory.evaluator_models", [])
        active_key = await get_config("memory.active_evaluator", "qwen3-0-6b")
        auto_capture = await get_config("memory.auto_capture", False)
        ac_label = "ON" if auto_capture else "OFF"

        buttons = []
        for m in eval_models:
            key = m.get("key", "")
            label = m.get("label", key)
            prefix = "✅ " if key == active_key else ""
            buttons.append([InlineKeyboardButton(
                f"{prefix}{label}", callback_data=f"eval:detail:{key}"
            )])
        buttons.append([InlineKeyboardButton("➕ Add Evaluator", callback_data="eval:add_menu")])
        buttons.append([InlineKeyboardButton(
            f"{'✅ ' if auto_capture else '⬜ '}Auto-capture: {ac_label}",
            callback_data="eval:toggle_ac",
        )])

        count = len(eval_models)
        text = f"🔬 <b>Evaluator</b> — {count} registered | Auto-capture: <b>{ac_label}</b>"
        markup = InlineKeyboardMarkup(buttons)

        if hasattr(update_or_query, 'message') and update_or_query.message:
            await update_or_query.message.reply_text(text, parse_mode="HTML", reply_markup=markup)
        else:
            await update_or_query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)

    async def _eval_menu_detail(self, query, eval_key: str):
        """Show evaluator model detail submenu."""
        eval_models = await get_config("memory.evaluator_models", [])
        active_key = await get_config("memory.active_evaluator", "qwen3-0-6b")
        entry = next((m for m in eval_models if m.get("key") == eval_key), None)
        if not entry:
            await query.edit_message_text("❌ Evaluator model not found.")
            return

        label = entry.get("label", eval_key)
        driver = entry.get("driver", "?")
        model_id = entry.get("model_id", "?")
        is_active = eval_key == active_key

        text = (
            f"🔬 <b>{label}</b>\n\n"
            f"Driver: <code>{driver}</code> | Model: <code>{model_id}</code>"
        )
        if is_active:
            text += "\n✅ <b>Active</b>"

        buttons = []
        if not is_active:
            buttons.append([InlineKeyboardButton("✅ Set Active", callback_data=f"eval:set_active:{eval_key}")])
        buttons.append([InlineKeyboardButton("🗑 Delete", callback_data=f"eval:delete_confirm:{eval_key}")])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="eval:main")])

        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _handle_eval_callback(self, query, data: str):
        """Handle all eval:* callback routing."""
        user = query.from_user

        if data == "eval:main":
            await self._eval_menu_main(query)

        elif data.startswith("eval:detail:"):
            eval_key = data.split(":", 2)[2]
            await self._eval_menu_detail(query, eval_key)

        elif data.startswith("eval:set_active:"):
            eval_key = data.split(":", 2)[2]
            eval_models = await get_config("memory.evaluator_models", [])
            entry = next((m for m in eval_models if m.get("key") == eval_key), None)
            if not entry:
                await query.answer("Evaluator not found", show_alert=True)
                return
            await set_config("memory.active_evaluator", eval_key)
            await set_config("memory.evaluator_driver", entry.get("driver", "ollama"))
            await set_config("memory.evaluator_model", entry.get("model_id", ""))
            await query.answer(f"Active → {entry.get('label', eval_key)}")
            await self._eval_menu_main(query)

        elif data == "eval:toggle_ac":
            auto_capture = await get_config("memory.auto_capture", False)
            new_val = not auto_capture
            if new_val:
                # Check if evaluator model is available before enabling
                eval_driver = await get_config("memory.evaluator_driver", "ollama")
                if eval_driver == "ollama":
                    eval_model = await get_config("memory.evaluator_model", "qwen3:0.6b")
                    from ..memory.evaluator import check_model_available
                    available = await check_model_available(model=eval_model)
                    if not available:
                        await query.answer(
                            f"Evaluator model '{eval_model}' not available. Run: ollama pull {eval_model}",
                            show_alert=True,
                        )
                        return
            await set_config("memory.auto_capture", new_val)
            await query.answer(f"Auto-capture {'ON' if new_val else 'OFF'}")
            await self._eval_menu_main(query)

        elif data.startswith("eval:delete_confirm:"):
            eval_key = data.split(":", 2)[2]
            eval_models = await get_config("memory.evaluator_models", [])
            entry = next((m for m in eval_models if m.get("key") == eval_key), None)
            if not entry:
                await query.answer("Evaluator not found", show_alert=True)
                return
            await query.edit_message_text(
                f"🗑 <b>Delete {entry.get('label', eval_key)}?</b>",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("🗑 Yes, Delete", callback_data=f"eval:delete:{eval_key}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"eval:detail:{eval_key}"),
                    ],
                ]),
            )

        elif data.startswith("eval:delete:"):
            eval_key = data.split(":", 2)[2]
            eval_models = await get_config("memory.evaluator_models", [])
            active_key = await get_config("memory.active_evaluator", "qwen3-0-6b")
            eval_models = [m for m in eval_models if m.get("key") != eval_key]
            await set_config("memory.evaluator_models", eval_models)
            if eval_key == active_key and eval_models:
                new_active = eval_models[0]
                await set_config("memory.active_evaluator", new_active["key"])
                await set_config("memory.evaluator_driver", new_active.get("driver", "ollama"))
                await set_config("memory.evaluator_model", new_active.get("model_id", ""))
            await query.answer("Evaluator deleted")
            await self._eval_menu_main(query)

        elif data == "eval:add_menu":
            drivers = [
                ("ollama", "Ollama (local, FREE)"),
                ("provider", "Use main chat LLM"),
            ]
            buttons = [
                InlineKeyboardButton(label, callback_data=f"eval:add:{key}")
                for key, label in drivers
            ]
            buttons.append(InlineKeyboardButton("⬅️ Back", callback_data="eval:main"))
            keyboard = [[btn] for btn in buttons]
            await query.edit_message_text(
                "🔬 <b>Add Evaluator</b>\n\n"
                "Choose a driver:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        elif data.startswith("eval:add:"):
            driver = data.split(":", 2)[2]
            if driver == "provider":
                # Use the active chat model directly
                active_model_key = await get_config("provider.active_model", "gemini-pro")
                models = await get_config("provider.models", [])
                chat_entry = next((m for m in models if m.get("key") == active_model_key), None)
                chat_label = chat_entry.get("label", active_model_key) if chat_entry else active_model_key
                chat_model_id = chat_entry.get("model_id", active_model_key) if chat_entry else active_model_key

                entry = {
                    "key": f"provider-{active_model_key}",
                    "label": f"{chat_label} (main LLM)",
                    "driver": "provider",
                    "model_id": chat_model_id,
                }
                eval_models = await get_config("memory.evaluator_models", [])
                existing = next((m for m in eval_models if m["key"] == entry["key"]), None)
                if existing:
                    await query.answer("Already registered", show_alert=True)
                    return
                eval_models.append(entry)
                await set_config("memory.evaluator_models", eval_models)
                await query.answer(f"Added: {entry['label']}")
                await self._eval_menu_main(query)
            else:
                # Ollama — multi-step: ask for model_id
                self._auth_state[user.id] = {
                    "type": "eval_add",
                    "driver": driver,
                    "step": "model_id",
                }
                try:
                    await query.answer()
                except Exception:
                    pass
                await (query._bot or self.app.bot).send_message(
                    chat_id=query.message.chat_id,
                    text=(
                        "🔬 <b>Add Evaluator — Ollama</b>\n\n"
                        "Send the model ID (e.g. <code>qwen3:0.6b</code>, <code>gemma3:1b</code>).\n\n"
                        "Send /cancel to abort."
                    ),
                    parse_mode="HTML",
                )

    async def _process_addeval_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Multi-step evaluator model addition flow."""
        bot = context.bot if context else self.app.bot
        step = state.get("step", "")
        text = text.strip()

        if step == "model_id":
            state["model_id"] = text
            state["step"] = "label"
            await bot.send_message(
                chat_id=chat_id,
                text=f"Model ID: <code>{text}</code>\n\n"
                     f"Send a <b>display label</b> (e.g. <code>qwen3:0.6b (Ollama)</code>).\n"
                     f"Or send <code>.</code> to auto-generate.",
                parse_mode="HTML",
            )
            return True

        elif step == "label":
            model_id = state.get("model_id", "")
            driver = state.get("driver", "ollama")
            if text == ".":
                label = f"{model_id} (Ollama)"
            else:
                label = text

            entry = {
                "key": model_id.replace("/", "-").replace(".", "-").replace(":", "-").lower(),
                "label": label,
                "driver": driver,
                "model_id": model_id,
            }
            if driver == "ollama":
                entry["base_url"] = "http://localhost:11434"

            eval_models = await get_config("memory.evaluator_models", [])
            existing = next((m for m in eval_models if m["key"] == entry["key"]), None)
            if existing:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Evaluator <code>{entry['key']}</code> already exists.",
                    parse_mode="HTML",
                )
                self._auth_state.pop(user_id, None)
                return True

            eval_models.append(entry)
            await set_config("memory.evaluator_models", eval_models)

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ <b>Evaluator added:</b>\n\n"
                    f"• Key: <code>{entry['key']}</code>\n"
                    f"• Label: {entry['label']}\n"
                    f"• Driver: {driver}\n\n"
                    f"Use /evaluator to manage."
                ),
                parse_mode="HTML",
            )
            logger.info(f"Evaluator added: {entry}")
            self._auth_state.pop(user_id, None)
            return True

        return False

    async def _cmd_autocapture(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /autocapture command — toggle with inline buttons."""
        user = update.effective_user

        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can change auto-capture settings.")
            return

        args = update.message.text.split(maxsplit=1)
        toggle = args[1].strip().lower() if len(args) > 1 else None

        if toggle is None:
            current = await get_config("memory.auto_capture", False)
            state = "ON" if current else "OFF"
            buttons = [
                InlineKeyboardButton(f"{'✅ ' if current else ''}ON", callback_data="autocapture:on"),
                InlineKeyboardButton(f"{'✅ ' if not current else ''}OFF", callback_data="autocapture:off"),
            ]
            await update.message.reply_text(
                f"📝 **Auto-capture:** {state}\n"
                f"⚠️ ON = extra LLM + embedding per message",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([buttons]),
            )
            return

        if toggle not in ("on", "off"):
            await update.message.reply_text(f"❌ Use: `on` or `off`", parse_mode="Markdown")
            return

        enabled = toggle == "on"
        if enabled:
            # Check if evaluator model is available before enabling
            eval_driver = await get_config("memory.evaluator_driver", "ollama")
            if eval_driver == "ollama":
                eval_model = await get_config("memory.evaluator_model", "qwen3:0.6b")
                from ..memory.evaluator import check_model_available
                available = await check_model_available(model=eval_model)
                if not available:
                    await update.message.reply_text(
                        f"❌ Evaluator model `{eval_model}` not available.\n"
                        f"Run `ollama pull {eval_model}` first, or use /evaluator to configure.",
                        parse_mode="Markdown",
                    )
                    return
        await set_config("memory.auto_capture", enabled)
        if enabled:
            await update.message.reply_text(
                "📝 Auto-capture **ON** ⚠️\nExtra LLM + embedding calls per message.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "📝 Auto-capture **OFF**\nMemory only stored on explicit request.",
                parse_mode="Markdown",
            )

    async def _cmd_identity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /identity command."""
        from ..db.models import get_identity
        identity = await get_identity()

        lines = ["🪪 **Identity**\n"]
        for k, v in identity.items():
            lines.append(f"• **{k}**: {v}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /models command — CRUD model management menu."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can manage models.")
            return
        await self._models_menu_main(update)

    async def _models_menu_main(self, update_or_query):
        """Render CRUD model list — reusable for command + back button."""
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")

        buttons = []
        for m in models:
            key = m.get("key", "")
            label = m.get("label", key)
            prefix = "⭐ " if key == active_key else ""
            buttons.append([InlineKeyboardButton(
                f"{prefix}{label}", callback_data=f"models:detail:{key}"
            )])
        buttons.append([InlineKeyboardButton("➕ Add Model", callback_data="models:add_menu")])

        count = len(models)
        text = f"🤖 <b>Models</b> — {count} registered"
        markup = InlineKeyboardMarkup(buttons)

        if hasattr(update_or_query, 'message') and update_or_query.message:
            await update_or_query.message.reply_text(text, parse_mode="HTML", reply_markup=markup)
        else:
            await update_or_query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)

    async def _models_menu_detail(self, query, model_key: str):
        """Show model detail submenu."""
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        entry = next((m for m in models if m.get("key") == model_key), None)
        if not entry:
            await query.edit_message_text("❌ Model not found.")
            return

        label = entry.get("label", model_key)
        driver = entry.get("driver", "?")
        model_id = entry.get("model_id", "?")
        ctx = int(entry.get("context_window", 0))
        ctx_display = f"{ctx/1000000:.1f}M" if ctx >= 1000000 else f"{ctx//1000}K" if ctx else "?"
        is_default = model_key == active_key

        # Thinking / reasoning info from model entry
        params = entry.get("params") or {}
        thinking_budget = params.get("thinking_budget")
        thinking_label = self._format_thinking_level(thinking_budget)
        reasoning_on = bool(entry.get("reasoning_visible", False))

        text = (
            f"🤖 <b>{label}</b>\n\n"
            f"Driver: <code>{driver}</code> | Model: <code>{model_id}</code>\n"
            f"Context: {ctx_display} tokens\n"
            f"💭 Thinking: {thinking_label}\n"
            f"💬 Reasoning: {'ON' if reasoning_on else 'OFF'}"
        )
        if is_default:
            text += "\n⭐ <b>Default model</b>"

        auth_type = entry.get("auth", "")
        buttons = []
        if not is_default:
            buttons.append([InlineKeyboardButton("⭐ Set as Default", callback_data=f"models:set_default:{model_key}")])
        buttons.append([InlineKeyboardButton(f"📐 Context Window: {ctx_display}", callback_data="models:ctx")])
        buttons.append([InlineKeyboardButton(f"💭 Thinking: {thinking_label}", callback_data=f"models:thinking:{model_key}")])
        buttons.append([InlineKeyboardButton(f"💬 Reasoning: {'ON' if reasoning_on else 'OFF'}", callback_data=f"models:reasoning:{model_key}")])
        if auth_type == "oauth":
            buttons.append([InlineKeyboardButton("🔄 Re-authenticate", callback_data=f"models:reauth:{model_key}")])
        elif auth_type == "api_key":
            buttons.append([InlineKeyboardButton("🔐 Update API Key", callback_data=f"models:apikey:{model_key}")])
        buttons.append([InlineKeyboardButton("⚙️ Parameters", callback_data=f"models:params:{model_key}")])
        buttons.append([InlineKeyboardButton("✏️ Edit Label", callback_data=f"models:edit_label:{model_key}")])
        buttons.append([InlineKeyboardButton("🗑 Delete", callback_data=f"models:delete_confirm:{model_key}")])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="models:main")])

        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _handle_models_callback(self, query, data: str):
        """Handle all models:* callback routing."""
        user = query.from_user

        if data == "models:main":
            await self._models_menu_main(query)

        elif data.startswith("models:detail:"):
            model_key = data.split(":", 2)[2]
            await self._models_menu_detail(query, model_key)

        elif data.startswith("models:set_default:"):
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            await set_config("provider.active_model", model_key)
            await set_config("provider.chat_model", entry["model_id"])
            if self.agent:
                await self.agent.reload_provider()
            await query.answer(f"Default → {entry.get('label', model_key)}")
            await self._models_menu_main(query)

        elif data == "models:ctx":
            # Show context window presets for active model
            models = await get_config("provider.models", [])
            active_key = await get_config("provider.active_model", "gemini-pro")
            current = next((m for m in models if m.get("key") == active_key), None)
            current_ctx = int(current.get("context_window", 0)) if current else 0
            label = current.get("label", active_key) if current else active_key

            presets = [
                (128000, "128K"),
                (200000, "200K"),
                (500000, "500K"),
                (1000000, "1M"),
                (1048576, "1M (exact)"),
            ]
            buttons = []
            for value, display in presets:
                check = " ✓" if value == current_ctx else ""
                buttons.append(InlineKeyboardButton(
                    f"{display}{check}", callback_data=f"models:ctx:{value}"
                ))
            keyboard = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
            keyboard.append([InlineKeyboardButton("✏️ Custom", callback_data="models:ctx:custom")])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="models:ctx:back")])

            ctx_display = f"{current_ctx/1000000:.1f}M" if current_ctx >= 1000000 else f"{current_ctx//1000}K"
            await query.edit_message_text(
                f"📐 <b>Context Window — {label}</b>\n\n"
                f"Current: <b>{ctx_display}</b> ({current_ctx:,} tokens)\n\n"
                f"Select a preset or enter custom value:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        elif data.startswith("models:ctx:"):
            ctx_action = data.split(":", 2)[2]

            if ctx_action == "back":
                # Back to model detail (active model)
                active_key = await get_config("provider.active_model", "gemini-pro")
                await self._models_menu_detail(query, active_key)

            elif ctx_action == "custom":
                self._auth_state[user.id] = {
                    "type": "ctx_window",
                    "step": "input",
                    "chat_id": query.message.chat_id,
                    "message_id": query.message.message_id,
                }
                await query.edit_message_text(
                    "📐 <b>Custom Context Window</b>\n\n"
                    "Send the context window size in tokens (number only).\n"
                    "Examples: <code>1000000</code>, <code>200000</code>, <code>2000000</code>\n\n"
                    "Send /cancel to abort.",
                    parse_mode="HTML",
                )

            else:
                # Apply preset context window value
                try:
                    new_ctx = int(ctx_action)
                except ValueError:
                    await query.answer("Invalid value", show_alert=True)
                    return

                models = await get_config("provider.models", [])
                active_key = await get_config("provider.active_model", "gemini-pro")
                updated = False
                for m in models:
                    if m.get("key") == active_key:
                        m["context_window"] = new_ctx
                        updated = True
                        break

                if updated:
                    await set_config("provider.models", models)
                    if self.agent:
                        await self.agent.reload_provider()

                    ctx_display = f"{new_ctx/1000000:.1f}M" if new_ctx >= 1000000 else f"{new_ctx//1000}K"
                    model_label = next((m.get("label", active_key) for m in models if m.get("key") == active_key), active_key)
                    await query.answer(f"Context window set to {ctx_display}")

                    # Re-render preset menu with updated checkmark
                    presets = [
                        (128000, "128K"),
                        (200000, "200K"),
                        (500000, "500K"),
                        (1000000, "1M"),
                        (1048576, "1M (exact)"),
                    ]
                    buttons = []
                    for value, display in presets:
                        check = " ✓" if value == new_ctx else ""
                        buttons.append(InlineKeyboardButton(
                            f"{display}{check}", callback_data=f"models:ctx:{value}"
                        ))
                    keyboard = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
                    keyboard.append([InlineKeyboardButton("✏️ Custom", callback_data="models:ctx:custom")])
                    keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="models:ctx:back")])

                    await query.edit_message_text(
                        f"📐 <b>Context Window — {model_label}</b>\n\n"
                        f"Current: <b>{ctx_display}</b> ({new_ctx:,} tokens)\n\n"
                        f"Select a preset or enter custom value:",
                        parse_mode="HTML",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )
                else:
                    await query.answer("Model not found", show_alert=True)

        elif data.startswith("models:params:"):
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            driver = entry.get("driver", "")
            params = entry.get("params") or self._DRIVER_DEFAULT_PARAMS.get(driver, {})
            import json as _json
            params_json = _json.dumps(params, indent=2)
            self._auth_state[user.id] = {
                "type": "models_params",
                "model_key": model_key,
                "chat_id": query.message.chat_id,
            }
            await query.edit_message_text(
                f"⚙️ <b>Parameters — {entry.get('label', model_key)}</b>\n\n"
                f"<pre>{params_json}</pre>\n\n"
                f"Be careful, changing parameters can make the AI model work unstable.\n\n"
                f"Send new JSON to update, or /cancel to abort.",
                parse_mode="HTML",
            )

        elif data.startswith("models:edit_label:"):
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            self._auth_state[user.id] = {
                "type": "models_edit_label",
                "model_key": model_key,
                "chat_id": query.message.chat_id,
            }
            await query.edit_message_text(
                f"✏️ <b>Edit Label — {entry.get('label', model_key)}</b>\n\n"
                f"Send the new display label.\n"
                f"Send /cancel to abort.",
                parse_mode="HTML",
            )

        elif data.startswith("models:thinking:"):
            # Show thinking level picker for a specific model
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            current_tb = (entry.get("params") or {}).get("thinking_budget")
            levels = [
                (None, "OFF"),
                (1024, "Low (1K)"),
                (4096, "Medium (4K)"),
                (8192, "High (8K)"),
                (24576, "Max (24K)"),
                (-1, "Dynamic"),
            ]
            buttons = []
            for value, display in levels:
                check = " ✓" if current_tb == value else ""
                # Encode None as "none" in callback
                val_str = "none" if value is None else str(value)
                buttons.append(InlineKeyboardButton(
                    f"{display}{check}", callback_data=f"models:thinking_set:{model_key}:{val_str}"
                ))
            keyboard = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data=f"models:detail:{model_key}")])
            await query.edit_message_text(
                f"💭 <b>Thinking Level — {entry.get('label', model_key)}</b>\n\n"
                f"Current: <b>{self._format_thinking_level(current_tb)}</b>\n\n"
                f"OFF = no thinking budget\n"
                f"Dynamic = model decides (Gemini default)\n\n"
                f"Select a level:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        elif data.startswith("models:thinking_set:"):
            # Apply thinking level to model params
            parts = data.split(":", 3)
            model_key = parts[2]
            val_str = parts[3]
            if val_str == "none":
                new_tb = None
            else:
                new_tb = int(val_str)
            models = await get_config("provider.models", [])
            updated = False
            for m in models:
                if m.get("key") == model_key:
                    if "params" not in m or not m["params"]:
                        driver = m.get("driver", "")
                        m["params"] = self._DRIVER_DEFAULT_PARAMS.get(driver, self._DRIVER_DEFAULT_PARAMS["_default"]).copy()
                    m["params"]["thinking_budget"] = new_tb
                    updated = True
                    break
            if updated:
                await set_config("provider.models", models)
                if self.agent:
                    await self.agent.reload_provider()
                await query.answer(f"Thinking → {self._format_thinking_level(new_tb)}")
            await self._models_menu_detail(query, model_key)

        elif data.startswith("models:reasoning:"):
            # Toggle reasoning_visible for a specific model
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            updated = False
            for m in models:
                if m.get("key") == model_key:
                    current = bool(m.get("reasoning_visible", False))
                    m["reasoning_visible"] = not current
                    updated = True
                    break
            if updated:
                await set_config("provider.models", models)
                # Update active conversation's reasoning_visible
                if self.agent:
                    active_key = await get_config("provider.active_model", "gemini-pro")
                    if model_key == active_key:
                        for conv in self.agent.conversations._active.values():
                            conv.reasoning_visible = not current
                new_state = "ON" if not current else "OFF"
                await query.answer(f"Reasoning → {new_state}")
            await self._models_menu_detail(query, model_key)

        elif data.startswith("models:delete_confirm:"):
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            active_key = await get_config("provider.active_model", "gemini-pro")
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            warn = ""
            if model_key == active_key:
                warn = "\n\n⚠️ This is the current default model. Deleting will switch to the next available."
            if len(models) <= 1:
                await query.answer("Cannot delete the only model", show_alert=True)
                return
            await query.edit_message_text(
                f"🗑 <b>Delete {entry.get('label', model_key)}?</b>{warn}",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("🗑 Yes, Delete", callback_data=f"models:delete:{model_key}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"models:detail:{model_key}"),
                    ],
                ]),
            )

        elif data.startswith("models:delete:"):
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            active_key = await get_config("provider.active_model", "gemini-pro")
            models = [m for m in models if m.get("key") != model_key]
            await set_config("provider.models", models)
            # If deleted was active → switch to first remaining
            if model_key == active_key and models:
                new_default = models[0]
                await set_config("provider.active_model", new_default["key"])
                await set_config("provider.chat_model", new_default["model_id"])
                if self.agent:
                    await self.agent.reload_provider()
            await query.answer("Model deleted")
            await self._models_menu_main(query)

        elif data == "models:add_menu":
            # Show 3 top-level auth categories
            keyboard = [
                [InlineKeyboardButton("Google (OAuth)", callback_data="models:add:google_cca")],
                [InlineKeyboardButton("OpenAI (OAuth)", callback_data="models:add:codex")],
                [InlineKeyboardButton("🔑 AI API Key", callback_data="models:add_apikey")],
                [InlineKeyboardButton("⬅️ Back", callback_data="models:main")],
            ]
            await query.edit_message_text(
                "🤖 <b>Add Model</b>\n\n"
                "Choose authentication method:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        elif data == "models:add_apikey":
            # Sub-menu: pick an API key provider
            providers = [
                ("openai", "OpenAI"),
                ("anthropic", "Anthropic (Claude)"),
                ("groq", "Groq"),
                ("together", "Together AI"),
                ("openrouter", "OpenRouter"),
                ("custom", "Custom (enter URL)"),
            ]
            buttons = [
                [InlineKeyboardButton(label, callback_data=f"models:add_ak:{key}")]
                for key, label in providers
            ]
            buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="models:add_menu")])
            await query.edit_message_text(
                "🔑 <b>Add Model — API Key</b>\n\n"
                "Select provider:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(buttons),
            )

        elif data.startswith("models:add_ak:"):
            # API key provider selected — start multi-step add
            provider_key = data.split(":", 2)[2]
            _ak_providers = {
                "openai": {
                    "label": "OpenAI",
                    "driver": "openai_compat",
                    "base_url": "https://api.openai.com/v1",
                    "credential_key": "credential.openai_api_key",
                    "examples": "gpt-4.1, gpt-4.1-mini, o3, o4-mini",
                },
                "anthropic": {
                    "label": "Anthropic",
                    "driver": "anthropic",
                    "credential_key": "credential.anthropic_api_key",
                    "examples": "claude-sonnet-4-20250514, claude-opus-4-0-20250514",
                },
                "groq": {
                    "label": "Groq",
                    "driver": "openai_compat",
                    "base_url": "https://api.groq.com/openai/v1",
                    "credential_key": "credential.groq_api_key",
                    "examples": "llama-3.3-70b-versatile, qwen/qwen3-32b",
                },
                "together": {
                    "label": "Together AI",
                    "driver": "openai_compat",
                    "base_url": "https://api.together.xyz/v1",
                    "credential_key": "credential.together_api_key",
                    "examples": "meta-llama/Llama-4-Maverick-17B-128E",
                },
                "openrouter": {
                    "label": "OpenRouter",
                    "driver": "openai_compat",
                    "base_url": "https://openrouter.ai/api/v1",
                    "credential_key": "credential.openrouter_api_key",
                    "examples": "google/gemini-2.5-pro, anthropic/claude-sonnet-4",
                },
            }
            meta = _ak_providers.get(provider_key)

            if provider_key == "custom":
                # Custom URL — ask for base_url first
                self._auth_state[user.id] = {
                    "type": "models_add",
                    "driver": "openai_compat",
                    "step": "base_url",
                }
                try:
                    await query.answer()
                except Exception:
                    pass
                await (query._bot or self.app.bot).send_message(
                    chat_id=query.message.chat_id,
                    text=(
                        "🔑 <b>Add Model — Custom API</b>\n\n"
                        "Send the base URL (OpenAI-compatible endpoint).\n"
                        "Example: <code>https://api.example.com/v1</code>\n\n"
                        "Send /cancel to abort."
                    ),
                    parse_mode="HTML",
                )
            elif meta:
                # Known provider — check if API key exists
                from syne.db.models import get_config as _gc
                existing_key = await _gc(meta["credential_key"], None)
                self._auth_state[user.id] = {
                    "type": "models_add",
                    "driver": meta["driver"],
                    "base_url": meta["base_url"],
                    "credential_key": meta["credential_key"],
                    "step": "apikey" if not existing_key else "model_id",
                }
                try:
                    await query.answer()
                except Exception:
                    pass
                chat_id = query.message.chat_id
                if existing_key:
                    masked = str(existing_key)[:8] + "..."
                    await (query._bot or self.app.bot).send_message(
                        chat_id=chat_id,
                        text=(
                            f"🔑 <b>Add Model — {meta['label']}</b>\n\n"
                            f"API Key: <code>{masked}</code> (already stored)\n\n"
                            f"Send the <b>model ID</b> (e.g. <code>{meta['examples'].split(', ')[0]}</code>).\n"
                            f"Available: {meta['examples']}\n\n"
                            f"Send /cancel to abort."
                        ),
                        parse_mode="HTML",
                    )
                else:
                    await (query._bot or self.app.bot).send_message(
                        chat_id=chat_id,
                        text=(
                            f"🔑 <b>Add Model — {meta['label']}</b>\n\n"
                            f"Paste your <b>{meta['label']} API key</b> below.\n"
                            f"It will be saved to the database and <b>NOT</b> stored in chat history.\n\n"
                            f"Send /cancel to abort."
                        ),
                        parse_mode="HTML",
                    )

        elif data.startswith("models:reauth:"):
            # Re-authenticate an OAuth model
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            driver = entry.get("driver", "")
            # Map driver to auth provider key
            _driver_to_auth = {
                "google_cca": "google",
                "codex": "codex",
                "anthropic": "claude",
            }
            auth_key = _driver_to_auth.get(driver)
            if not auth_key:
                await query.answer("This model doesn't use OAuth", show_alert=True)
                return
            providers = await self._get_auth_providers()
            auth_entry = next((p for p in providers if p.get("oauth_driver") == auth_key or p.get("key") == auth_key), None)
            if not auth_entry:
                await query.answer("Auth provider not found", show_alert=True)
                return
            chat_id = query.message.chat_id
            await query.edit_message_text(
                f"🔄 Starting {entry.get('label', model_key)} re-authentication...",
            )
            await self._start_oauth_flow(auth_entry, chat_id, user.id, None)

        elif data.startswith("models:apikey:"):
            # Update API key for a model
            model_key = data.split(":", 2)[2]
            models = await get_config("provider.models", [])
            entry = next((m for m in models if m.get("key") == model_key), None)
            if not entry:
                await query.answer("Model not found", show_alert=True)
                return
            label = entry.get("label", model_key)
            # Use credential_key from model entry first, then fallback
            cred_key = entry.get("credential_key", "")
            if not cred_key:
                _driver_creds = {
                    "openai_compat": "credential.openai_api_key",
                    "groq": "credential.groq_api_key",
                    "together": "credential.together_api_key",
                }
                cred_key = _driver_creds.get(entry.get("driver", ""), "")
            if not cred_key:
                await query.answer("No credential key for this driver", show_alert=True)
                return
            self._auth_state[user.id] = {
                "type": "apikey",
                "provider": entry.get("driver", ""),
                "credential_key": cred_key,
                "label": label,
            }
            await query.edit_message_text(
                f"🔐 <b>{label} API Key</b>\n\n"
                f"Paste your API key below.\n"
                f"It will be saved to the database and <b>NOT</b> stored in chat history.\n\n"
                f"Send /cancel to abort.",
                parse_mode="HTML",
            )

        elif data.startswith("models:add:"):
            # OAuth driver selected — ask for model ID
            driver = data.split(":", 2)[2]
            logger.info(f"Add model (OAuth): driver={driver}, user={user.id}")
            _oauth_meta = {
                "google_cca": ("Google (OAuth)", "gemini-2.5-pro, gemini-2.5-flash, gemini-3-pro-preview"),
                "codex": ("OpenAI (OAuth)", "gpt-5.2, o3-pro"),
            }
            driver_label, examples = _oauth_meta.get(driver, (driver, "model-name"))
            self._auth_state[user.id] = {
                "type": "models_add",
                "driver": driver,
                "step": "model_id",
            }
            try:
                await query.answer()
            except Exception:
                pass
            example_first = examples.split(", ")[0]
            await (query._bot or self.app.bot).send_message(
                chat_id=query.message.chat_id,
                text=(
                    f"🤖 <b>Add Model — {driver_label}</b>\n\n"
                    f"Send the model ID (e.g. <code>{example_first}</code>).\n\n"
                    f"Available: {examples}\n\n"
                    f"Send /cancel to abort."
                ),
                parse_mode="HTML",
            )

    async def _cmd_embedding(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /embedding command — CRUD embedding management menu."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can manage embedding models.")
            return
        await self._embed_menu_main(update)

    async def _embed_menu_main(self, update_or_query):
        """Render CRUD embedding list."""
        models = await get_config("provider.embedding_models", [])
        active_key = await get_config("provider.active_embedding", "together-bge")

        buttons = []
        for m in models:
            key = m.get("key", "")
            label = m.get("label", key)
            prefix = "✅ " if key == active_key else ""
            buttons.append([InlineKeyboardButton(
                f"{prefix}{label}", callback_data=f"embed:detail:{key}"
            )])
        buttons.append([InlineKeyboardButton("➕ Add Embedding", callback_data="embed:add_menu")])

        count = len(models)
        text = f"🧬 <b>Embedding</b> — {count} registered"
        markup = InlineKeyboardMarkup(buttons)

        if hasattr(update_or_query, 'message') and update_or_query.message:
            await update_or_query.message.reply_text(text, parse_mode="HTML", reply_markup=markup)
        else:
            await update_or_query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)

    async def _embed_menu_detail(self, query, embed_key: str):
        """Show embedding detail submenu."""
        models = await get_config("provider.embedding_models", [])
        active_key = await get_config("provider.active_embedding", "together-bge")
        entry = next((m for m in models if m.get("key") == embed_key), None)
        if not entry:
            await query.edit_message_text("❌ Embedding not found.")
            return

        label = entry.get("label", embed_key)
        driver = entry.get("driver", "?")
        model_id = entry.get("model_id", "?")
        dims = entry.get("dimensions", "?")
        cost = entry.get("cost", "?")
        is_active = embed_key == active_key

        text = (
            f"🧬 <b>{label}</b>\n\n"
            f"Driver: <code>{driver}</code> | Model: <code>{model_id}</code>\n"
            f"Dimensions: {dims} | Cost: {cost}"
        )
        if is_active:
            text += "\n✅ <b>Active</b>"

        buttons = []
        if not is_active:
            buttons.append([InlineKeyboardButton("✅ Set Active", callback_data=f"embed:set_active:{embed_key}")])
        buttons.append([InlineKeyboardButton("✏️ Edit Label", callback_data=f"embed:edit_label:{embed_key}")])
        buttons.append([InlineKeyboardButton("🗑 Delete", callback_data=f"embed:delete_confirm:{embed_key}")])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="embed:main")])

        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _handle_embed_callback(self, query, data: str):
        """Handle all embed:* callback routing."""
        user = query.from_user

        if data == "embed:main":
            await self._embed_menu_main(query)

        elif data.startswith("embed:detail:"):
            embed_key = data.split(":", 2)[2]
            await self._embed_menu_detail(query, embed_key)

        elif data.startswith("embed:set_active:"):
            embed_key = data.split(":", 2)[2]
            models = await get_config("provider.embedding_models", [])
            entry = next((m for m in models if m.get("key") == embed_key), None)
            if not entry:
                await query.answer("Embedding not found", show_alert=True)
                return
            # Apply embedding (test + dimension change handling)
            chat_id = query.message.chat_id
            from telegram.ext import ContextTypes
            success, result = await self._apply_embedding(embed_key, chat_id, None)
            if success:
                await query.answer(f"Active → {entry.get('label', embed_key)}")
            else:
                await query.answer(f"Failed: {result[:50]}", show_alert=True)
            await self._embed_menu_main(query)

        elif data.startswith("embed:edit_label:"):
            embed_key = data.split(":", 2)[2]
            models = await get_config("provider.embedding_models", [])
            entry = next((m for m in models if m.get("key") == embed_key), None)
            if not entry:
                await query.answer("Embedding not found", show_alert=True)
                return
            self._auth_state[user.id] = {
                "type": "embed_edit_label",
                "embed_key": embed_key,
                "chat_id": query.message.chat_id,
            }
            await query.edit_message_text(
                f"✏️ <b>Edit Label — {entry.get('label', embed_key)}</b>\n\n"
                f"Send the new display label.\n"
                f"Send /cancel to abort.",
                parse_mode="HTML",
            )

        elif data.startswith("embed:delete_confirm:"):
            embed_key = data.split(":", 2)[2]
            models = await get_config("provider.embedding_models", [])
            entry = next((m for m in models if m.get("key") == embed_key), None)
            active_key = await get_config("provider.active_embedding", "together-bge")
            if not entry:
                await query.answer("Embedding not found", show_alert=True)
                return
            if len(models) <= 1:
                await query.answer("Cannot delete the only embedding", show_alert=True)
                return
            warn = ""
            if embed_key == active_key:
                warn = "\n\n⚠️ This is the active embedding. Deleting will switch to the next available."
            await query.edit_message_text(
                f"🗑 <b>Delete {entry.get('label', embed_key)}?</b>{warn}",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("🗑 Yes, Delete", callback_data=f"embed:delete:{embed_key}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"embed:detail:{embed_key}"),
                    ],
                ]),
            )

        elif data.startswith("embed:delete:"):
            embed_key = data.split(":", 2)[2]
            models = await get_config("provider.embedding_models", [])
            active_key = await get_config("provider.active_embedding", "together-bge")
            models = [m for m in models if m.get("key") != embed_key]
            await set_config("provider.embedding_models", models)
            if embed_key == active_key and models:
                new_active = models[0]
                await set_config("provider.active_embedding", new_active["key"])
                if self.agent:
                    await self.agent.reload_provider()
            await query.answer("Embedding deleted")
            await self._embed_menu_main(query)

        elif data == "embed:add_menu":
            drivers = [
                ("ollama", "Ollama (local, FREE)"),
                ("together", "Together AI (API Key)"),
                ("openai_compat", "OpenAI-compatible (API Key)"),
            ]
            buttons = [
                InlineKeyboardButton(label, callback_data=f"embed:add:{key}")
                for key, label in drivers
            ]
            buttons.append(InlineKeyboardButton("⬅️ Back", callback_data="embed:main"))
            keyboard = [[btn] for btn in buttons]
            await query.edit_message_text(
                "🧬 <b>Add Embedding</b>\n\n"
                "Choose a driver:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        elif data.startswith("embed:add:"):
            driver = data.split(":", 2)[2]
            logger.info(f"Add embedding: driver={driver}, user={user.id}")
            driver_labels = {
                "ollama": "Ollama (local, FREE)",
                "together": "Together AI (API Key)",
                "openai_compat": "OpenAI-compatible (API Key)",
            }
            driver_label = driver_labels.get(driver, driver)
            self._auth_state[user.id] = {
                "type": "embed_add",
                "driver": driver,
                "step": "model_id",
            }
            try:
                await query.answer()
            except Exception:
                pass
            await (query._bot or self.app.bot).send_message(
                chat_id=query.message.chat_id,
                text=(
                    f"🧬 <b>Add Embedding — {driver_label}</b>\n\n"
                    "Send the model ID (e.g. <code>qwen3-embedding:0.6b</code>, <code>BAAI/bge-base-en-v1.5</code>).\n\n"
                    "Send /cancel to abort."
                ),
                parse_mode="HTML",
            )

    async def _apply_embedding(self, embed_key: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, str]:
        """Apply an embedding model from registry — test before switching.
        
        If the new model has different dimensions, all existing memories are
        deleted (incompatible vector spaces cannot be mixed). User is warned.
        
        Args:
            embed_key: Key of embedding model to switch to
            chat_id: Chat ID for sending status messages
            context: Telegram context for sending messages
            
        Returns:
            Tuple of (success, message)
        """
        from ..llm.drivers import test_embedding, get_model_from_list
        from ..db.connection import get_connection
        
        # Get embedding registry
        models = await get_config("provider.embedding_models", [])
        embed_entry = get_model_from_list(models, embed_key)
        
        if not embed_entry:
            return False, f"Embedding '{embed_key}' not found in registry"
        
        # Save current for rollback
        previous_key = await get_config("provider.active_embedding", "together-bge")
        await set_config("provider.previous_embedding", previous_key)
        
        # Check if dimensions will change (requires memory reset)
        current_dims = await get_config("provider.embedding_dimensions", 768)
        new_dims = embed_entry.get("dimensions", 768)
        dims_changed = (current_dims != new_dims)
        
        if dims_changed:
            # Count existing memories
            async with get_connection() as conn:
                row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memory WHERE embedding IS NOT NULL")
                mem_count = row["cnt"] if row else 0
            
            if mem_count > 0:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"⚠️ **Dimension change** ({current_dims} → {new_dims})\n"
                        f"This will delete all {mem_count} existing memories.\n"
                        f"Vectors from different models are incompatible."
                    ),
                    parse_mode="Markdown",
                )
        
        # Send "testing" message
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Testing {embed_entry.get('label', embed_key)}...",
        )
        
        try:
            success, error = await test_embedding(embed_entry, timeout=15)
            
            if success:
                # Reset memory if dimensions changed
                if dims_changed:
                    async with get_connection() as conn:
                        deleted = await conn.execute("DELETE FROM memory WHERE embedding IS NOT NULL")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="🗑️ Memories cleared (incompatible vector dimensions).",
                    )
                
                await set_config("provider.active_embedding", embed_key)
                await set_config("provider.embedding_model", embed_entry.get("model_id", ""))
                await set_config("provider.embedding_dimensions", new_dims)
                await set_config("provider.embedding_driver", embed_entry.get("driver", ""))
                
                # Hot-reload provider (picks up new embedding)
                await self.agent.reload_provider()
                
                return True, embed_entry.get("label", embed_key)
            else:
                return False, f"{embed_entry.get('label', embed_key)} failed: {error}"
                
        except Exception as e:
            return False, f"{embed_entry.get('label', embed_key)} failed: {str(e)[:100]}"

    async def _apply_model(self, model_key: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, str]:
        """Apply a model from registry — test before switching.
        
        Args:
            model_key: Key of model to switch to
            chat_id: Chat ID for sending status messages
            context: Telegram context for sending messages
            
        Returns:
            Tuple of (success, message)
        """
        from ..llm.drivers import create_hybrid_provider, test_model, get_model_from_list
        
        # Get model registry
        models = await get_config("provider.models", [])
        model_entry = get_model_from_list(models, model_key)
        
        if not model_entry:
            return False, f"Model '{model_key}' not found in registry"
        
        # Save current model for rollback
        previous_model_key = await get_config("provider.active_model", "gemini-pro")
        await set_config("provider.previous_model", previous_model_key)
        
        # Send "testing" message
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Testing {model_entry.get('label', model_key)}...",
        )
        
        try:
            # Switch directly — no test call. Model is already registered
            # with a valid driver. Testing wastes a CCA request and often
            # triggers 429 rate limits on Google CCA's tight quotas.
            await set_config("provider.active_model", model_key)
            await set_config("provider.chat_model", model_entry.get("model_id", model_key))
            
            # Hot-reload provider in the running agent (handles context manager + compaction adjustment)
            await self.agent.reload_provider()
            
            return True, model_entry.get("label", model_key)
                
        except Exception as e:
            return False, f"{model_entry.get('label', model_key)} failed: {str(e)[:100]}"

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /restart command — owner only, restarts the Syne process."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can restart Syne.")
            return

        await update.message.reply_text("🔄 Restarting Syne...")
        logger.info(f"Restart requested by {user.id}")

        # Save restart flag so we can notify when back up
        import json, tempfile, os
        restart_flag = os.path.join(tempfile.gettempdir(), "syne_restart_flag.json")
        try:
            with open(restart_flag, "w") as f:
                json.dump({"chat_id": update.effective_chat.id, "user_id": user.id}, f)
        except Exception as e:
            logger.warning(f"Failed to save restart flag: {e}")

        import sys
        # Exit with non-zero code so systemd (Restart=on-failure) will restart us
        # SIGTERM = exit 0 = clean exit = no restart. sys.exit(1) = failure = restart.
        sys.exit(1)

    def _path_id(self, path: str) -> str:
        """Generate a short ID for a path (for callback_data 64-byte limit)."""
        import hashlib
        short = hashlib.md5(path.encode()).hexdigest()[:8]
        self._browse_paths[short] = path
        return short

    # ── Auth (credential management) ───────────────────────────

    # Default auth providers — used when auth.providers not configured in DB.
    # After first init, these are seeded into DB and DB becomes source of truth.
    _DEFAULT_AUTH_PROVIDERS = [
        {"key": "google", "label": "Google (Gemini)", "auth_type": "oauth", "oauth_driver": "google"},
        {"key": "codex", "label": "OpenAI / Codex", "auth_type": "oauth", "oauth_driver": "codex"},
        {"key": "claude", "label": "Claude / Anthropic", "auth_type": "oauth", "oauth_driver": "claude"},
        {"key": "groq", "label": "Groq", "auth_type": "api_key", "credential_key": "credential.groq_api_key"},
        {"key": "together", "label": "Together AI", "auth_type": "api_key", "credential_key": "credential.together_api_key"},
        {"key": "openrouter", "label": "OpenRouter", "auth_type": "api_key", "credential_key": "credential.openrouter_api_key"},
    ]

    async def _get_auth_providers(self) -> list[dict]:
        """Get auth providers from DB, falling back to defaults.
        
        On first call with empty DB, seeds defaults into DB.
        """
        providers = await get_config("auth.providers", None)
        if providers is None:
            # Seed defaults into DB
            await set_config("auth.providers", self._DEFAULT_AUTH_PROVIDERS, "Auth provider registry")
            return self._DEFAULT_AUTH_PROVIDERS
        return providers

    async def _cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel command — abort active operation, auth flow, or running sub-agents."""
        user = update.effective_user
        chat = update.effective_chat
        cancelled = False

        # Cancel active processing task
        task = self._active_tasks.get(chat.id)
        if task and not task.done():
            task.cancel()
            self._active_tasks.pop(chat.id, None)
            cancelled = True

        # Cancel auth flow
        if user.id in self._auth_state:
            self._auth_state.pop(user.id, None)
            cancelled = True

        # Cancel running sub-agents for this chat's session
        subagent_cancelled = 0
        if self.agent.subagents and self.agent.conversations:
            key = f"telegram:{chat.id}"
            conv = self.agent.conversations._active.get(key)
            if conv:
                subagent_cancelled = await self.agent.subagents.cancel_by_session(conv.session_id)
                if subagent_cancelled:
                    cancelled = True

        if cancelled:
            parts = ["✋ Cancelled."]
            if subagent_cancelled:
                parts.append(f"({subagent_cancelled} sub-agent(s) stopped)")
            await update.message.reply_text(" ".join(parts))
        else:
            await update.message.reply_text("Nothing to cancel.")

    async def _cmd_groups(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /groups command — manage group members, aliases, access levels.

        Usage (owner DM only):
            /groups list                          — list all registered groups
            /groups members <group_id_or_name>    — list members in a group
            /groups set <group> <member_id> <level>  — set access (owner/family/public)
            /groups alias <group> <member_id> <name>  — set per-group alias
        """
        user = update.effective_user
        chat = update.effective_chat

        # Owner-only, DM-only
        from ..db.models import get_user
        db_user = await get_user("telegram", str(user.id))
        if not db_user or db_user.get("access_level") != "owner":
            await update.message.reply_text("⚠️ Only the owner can manage groups.")
            return
        if chat.type != "private":
            await update.message.reply_text("⚠️ Use /groups in DM only.")
            return

        args = context.args or []
        if not args:
            # Show interactive menu with group list
            await self._group_menu_main(update)
            return

        subcommand = args[0].lower()

        if subcommand == "add" and len(args) >= 2:
            await self._group_add(update, args[1])
        elif subcommand == "list":
            await self._group_list(update)
        elif subcommand == "members" and len(args) >= 2:
            await self._group_members(update, args[1])
        elif subcommand == "set" and len(args) >= 4:
            await self._group_set_access(update, args[1], args[2], args[3])
        elif subcommand == "alias" and len(args) >= 4:
            alias_name = " ".join(args[3:])  # Allow multi-word aliases
            await self._group_set_alias(update, args[1], args[2], alias_name)
        elif subcommand == "model":
            if len(args) >= 3:
                await self._group_set_model(update, args[1], args[2])
            elif len(args) >= 2:
                await self._group_get_model(update, args[1])
            else:
                await update.message.reply_text("❌ Usage: /groups model <group> [model_key]")
        elif subcommand == "settings" and len(args) >= 2:
            if len(args) == 2:
                # View settings
                await self._group_view_settings(update, args[1])
            elif len(args) >= 4:
                # Set or delete a setting
                key = args[2]
                value = " ".join(args[3:])
                await self._group_set_setting(update, args[1], key, value)
            else:
                await update.message.reply_text("❌ Usage: /groups settings <group> [key] [value|--delete]")
        else:
            await update.message.reply_text("❌ Invalid syntax. Use /groups for help.")

    async def _handle_group_callback(self, query, data: str):
        """Handle interactive group menu callbacks."""
        parts = data.split(":")
        # groups:action:param1:param2...
        action = parts[1] if len(parts) > 1 else ""
        
        # Owner check
        user = query.from_user
        from ..db.models import get_user
        db_user = await get_user("telegram", str(user.id))
        if not db_user or db_user.get("access_level") != "owner":
            await query.answer("⚠️ Owner only", show_alert=True)
            return
        
        if action == "main":
            await self._group_menu_main(None, query=query)
        
        elif action == "view" and len(parts) >= 3:
            group_id = parts[2]
            await self._group_menu_view(query, group_id)
        
        elif action == "model_list" and len(parts) >= 3:
            group_id = parts[2]
            await self._group_menu_model_list(query, group_id)
        
        elif action == "model_set" and len(parts) >= 4:
            group_id = parts[2]
            model_key = parts[3]
            
            from ..db.models import get_group
            group = await get_group("telegram", group_id)
            if not group:
                await query.answer("Group not found", show_alert=True)
                return
            
            settings = group.get("settings") or {}
            
            if model_key == "__default__":
                settings.pop("model", None)
            else:
                settings["model"] = model_key
            
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET settings = $1, updated_at = NOW() WHERE platform = 'telegram' AND platform_group_id = $2",
                    json.dumps(settings), group_id,
                )
            
            # Clear cached conversation
            self._clear_group_conversation(group_id)
            
            label = model_key if model_key != "__default__" else "default"
            await query.answer(f"Model set to {label}")
            
            # Refresh model list view
            await self._group_menu_model_list(query, group_id)
        
        elif action == "members" and len(parts) >= 3:
            group_id = parts[2]
            await self._group_menu_members(query, group_id)
        
        elif action == "settings" and len(parts) >= 3:
            group_id = parts[2]
            await self._group_menu_settings(query, group_id)
        
        elif action == "toggle" and len(parts) >= 3:
            group_id = parts[2]
            from ..db.models import get_group
            group = await get_group("telegram", group_id)
            if not group:
                await query.answer("Group not found", show_alert=True)
                return
            
            new_enabled = not group.get("enabled", True)
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET enabled = $1, updated_at = NOW() WHERE platform = 'telegram' AND platform_group_id = $2",
                    new_enabled, group_id,
                )
            
            status = "enabled" if new_enabled else "disabled"
            await query.answer(f"Group {status}")
            await self._group_menu_view(query, group_id)
        
        elif action == "delete_confirm" and len(parts) >= 3:
            group_id = parts[2]
            from ..db.models import get_group
            group = await get_group("telegram", group_id)
            name = (group or {}).get("name", group_id)
            
            buttons = [
                [
                    InlineKeyboardButton("🗑 Yes, Delete", callback_data=f"groups:delete:{group_id}"),
                    InlineKeyboardButton("❌ Cancel", callback_data=f"groups:view:{group_id}"),
                ]
            ]
            await query.edit_message_text(
                f"⚠️ Delete group <b>{name}</b>?\n\nThis removes all group config and member data.",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(buttons),
            )
        
        elif action == "delete" and len(parts) >= 3:
            group_id = parts[2]
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "DELETE FROM groups WHERE platform = 'telegram' AND platform_group_id = $1",
                    group_id,
                )
            self._clear_group_conversation(group_id)
            await query.answer("Group deleted")
            await self._group_menu_main(None, query=query)
        
        elif action == "member" and len(parts) >= 4:
            group_id = parts[2]
            member_id = parts[3]
            await self._group_menu_member_detail(query, group_id, member_id)
        
        elif action == "member_access" and len(parts) >= 5:
            group_id = parts[2]
            member_id = parts[3]
            new_level = parts[4]
            
            from ..db.models import get_group
            group = await get_group("telegram", group_id)
            if not group:
                await query.answer("Group not found", show_alert=True)
                return
            
            # Check first owner protection
            settings = group.get("settings") or {}
            members = settings.get("members") or {}
            
            if member_id in members:
                members[member_id]["access"] = new_level
            else:
                members[member_id] = {"access": new_level}
            
            settings["members"] = members
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET settings = $1, updated_at = NOW() WHERE platform = 'telegram' AND platform_group_id = $2",
                    json.dumps(settings), group_id,
                )
            
            # Also update global users table if needed
            from ..db.models import get_user, update_user
            db_user = await get_user("telegram", member_id)
            if db_user:
                # Check first owner protection before changing global level
                if db_user.get("access_level") == "owner":
                    from ..db.connection import get_connection as gc
                    async with gc() as conn:
                        first_owner = await conn.fetchrow(
                            "SELECT platform_id FROM users WHERE platform = 'telegram' AND access_level = 'owner' ORDER BY created_at LIMIT 1"
                        )
                    if first_owner and first_owner["platform_id"] == member_id:
                        if new_level != "owner":
                            await query.answer("Cannot demote first owner!", show_alert=True)
                            await self._group_menu_member_detail(query, group_id, member_id)
                            return
                
                await update_user("telegram", member_id, access_level=new_level)
            
            await query.answer(f"Access set to {new_level}")
            await self._group_menu_member_detail(query, group_id, member_id)
        
        elif action == "member_alias_prompt" and len(parts) >= 4:
            group_id = parts[2]
            member_id = parts[3]

            from ..db.models import get_group
            group = await get_group("telegram", group_id)
            members = ((group or {}).get("settings") or {}).get("members") or {}
            info = members.get(member_id, {})
            current = info.get("alias", "none")
            m_name = info.get("name", member_id)

            # Set interactive input state
            self._auth_state[query.from_user.id] = {
                "type": "group_alias",
                "group_id": group_id,
                "member_id": member_id,
                "chat_id": query.message.chat_id,
            }

            text = (
                f"✏️ <b>Set Alias for {m_name}</b>\n\n"
                f"Current alias: {current}\n\n"
                f"Send the new alias, or /cancel to abort."
            )
            buttons = [[InlineKeyboardButton("⬅️ Back", callback_data=f"groups:member:{group_id}:{member_id}")]]
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))
        
        elif action == "add_prompt":
            await query.edit_message_text(
                "➕ <b>Add Group</b>\n\n"
                "Send the group ID to register:\n"
                "<code>/groups add &lt;group_id&gt;</code>\n\n"
                "To find group ID: add the bot to the group, "
                "or forward a message from the group to @userinfobot.",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⬅️ Back", callback_data="groups:main")]
                ]),
            )
        
        else:
            await query.answer("Unknown action")

    async def _group_menu_main(self, update: Update, query=None):
        """Show main group menu — list of groups + add button."""
        from ..db.connection import get_connection
        from ..db.models import get_config
        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT platform_group_id, name, enabled, settings FROM groups WHERE platform = 'telegram' ORDER BY name"
            )

        # Build model label lookup
        models = await get_config("provider.models", [])
        active = await get_config("provider.active_model", "")
        model_map = {m["key"]: m.get("label", m["key"]) for m in models if "key" in m}
        default_label = model_map.get(active, active)

        buttons = []
        for row in rows:
            gid = row["platform_group_id"]
            name = row["name"] or gid
            emoji = "✅" if row["enabled"] else "⛔"
            settings = row["settings"] if isinstance(row["settings"], dict) else {}
            mk = settings.get("model")
            mlabel = model_map.get(mk, mk) if mk else f"Default ({default_label})"
            buttons.append([InlineKeyboardButton(
                f"{emoji} {name}  [{mlabel}]", callback_data=f"groups:view:{gid}"
            )])
        
        buttons.append([InlineKeyboardButton("➕ Add Group", callback_data="groups:add_prompt")])
        
        text = "📋 <b>Group Management</b>\n\nSelect a group to manage:"
        if not rows:
            text = "📋 <b>Group Management</b>\n\nNo groups registered yet."
        
        markup = InlineKeyboardMarkup(buttons)
        if query:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)
        else:
            await update.message.reply_text(text, parse_mode="HTML", reply_markup=markup)

    async def _group_menu_view(self, query, group_id: str):
        """Show group detail menu with action buttons."""
        from ..db.models import get_group
        group = await get_group("telegram", group_id)
        if not group:
            await query.edit_message_text(f"❌ Group {group_id} not found.")
            return
        
        name = group.get("name", group_id)
        settings = group.get("settings") or {}
        model_key = settings.get("model")
        enabled = "✅ Enabled" if group.get("enabled") else "⛔ Disabled"
        mention = "Yes" if group.get("require_mention") else "No"

        if model_key:
            model_display = f"<code>{model_key}</code>"
        else:
            _models = await get_config("provider.models", [])
            _active = await get_config("provider.active_model", "gemini-pro")
            _def_lbl = next((m.get("label", _active) for m in _models if m.get("key") == _active), _active)
            model_display = f"Default ({_def_lbl})"

        text = (
            f"📋 <b>{name}</b>\n\n"
            f"ID: <code>{group_id}</code>\n"
            f"Status: {enabled}\n"
            f"Require mention: {mention}\n"
            f"Model: {model_display}\n"
        )
        
        # Add custom settings
        custom_keys = [k for k in settings if k != "model" and k != "members"]
        if custom_keys:
            text += "\nSettings:\n"
            for k in custom_keys:
                text += f"  • {k}: {settings[k]}\n"
        
        buttons = [
            [InlineKeyboardButton("🤖 Change Model", callback_data=f"groups:model_list:{group_id}")],
            [InlineKeyboardButton("👥 Members", callback_data=f"groups:members:{group_id}")],
            [
                InlineKeyboardButton(
                    "⛔ Disable" if group.get("enabled") else "✅ Enable",
                    callback_data=f"groups:toggle:{group_id}"
                ),
                InlineKeyboardButton("🗑 Delete", callback_data=f"groups:delete_confirm:{group_id}"),
            ],
            [InlineKeyboardButton("⬅️ Back", callback_data="groups:main")],
        ]
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _group_menu_model_list(self, query, group_id: str):
        """Show model selection for a group."""
        from ..db.models import get_config, get_group

        group = await get_group("telegram", group_id)
        current_model = ((group or {}).get("settings") or {}).get("model")

        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        default_label = next((m.get("label", active_key) for m in models if m.get("key") == active_key), active_key)

        buttons = []
        for m in models:
            key = m.get("key", "")
            label = m.get("label", key)
            check = " ✓" if key == current_model else ""
            buttons.append([InlineKeyboardButton(
                f"{label}{check}", callback_data=f"groups:model_set:{group_id}:{key}"
            )])

        # Add "default" option with resolved label
        check = " ✓" if not current_model else ""
        buttons.append([InlineKeyboardButton(
            f"🔄 Default ({default_label}){check}", callback_data=f"groups:model_set:{group_id}:__default__"
        )])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=f"groups:view:{group_id}")])
        
        name = (group or {}).get("name", group_id)
        await query.edit_message_text(
            f"🤖 <b>Select Model for {name}</b>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _group_menu_members(self, query, group_id: str):
        """Show group members list with clickable edit buttons."""
        from ..db.models import get_group
        group = await get_group("telegram", group_id)
        if not group:
            await query.edit_message_text("❌ Group not found.")
            return
        
        name = group.get("name", group_id)
        members = ((group.get("settings") or {}).get("members") or {})
        
        if not members:
            text = f"👥 <b>{name} — Members</b>\n\nNo members collected yet."
            buttons = [[InlineKeyboardButton("⬅️ Back", callback_data=f"groups:view:{group_id}")]]
        else:
            text = f"👥 <b>{name} — Members</b>\n\nTap a member to edit:"
            buttons = []
            for mid, info in sorted(members.items(), key=lambda x: x[1].get("name", "")):
                m_name = info.get("alias") or info.get("name", "Unknown")
                access = info.get("access", "public")
                emoji = {"owner": "👑", "family": "👨‍👩‍👦", "public": "👤"}.get(access, "👤")
                buttons.append([InlineKeyboardButton(
                    f"{emoji} {m_name} — {access}",
                    callback_data=f"groups:member:{group_id}:{mid}",
                )])
            buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=f"groups:view:{group_id}")])
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _group_menu_member_detail(self, query, group_id: str, member_id: str):
        """Show member detail with access level and alias options."""
        from ..db.models import get_group
        group = await get_group("telegram", group_id)
        if not group:
            await query.edit_message_text("❌ Group not found.")
            return
        
        members = ((group.get("settings") or {}).get("members") or {})
        info = members.get(member_id, {})
        m_name = info.get("name", "Unknown")
        alias = info.get("alias", "")
        access = info.get("access", "public")
        username = info.get("username", "")
        
        text = f"👤 <b>Member Detail</b>\n\n"
        text += f"Name: {m_name}\n"
        if username:
            text += f"Username: @{username}\n"
        text += f"ID: <code>{member_id}</code>\n"
        if alias:
            text += f"Alias: {alias}\n"
        text += f"Access: <b>{access}</b>"
        
        # Access level buttons
        levels = ["owner", "family", "public"]
        access_buttons = []
        for lvl in levels:
            check = " ✓" if lvl == access else ""
            access_buttons.append(InlineKeyboardButton(
                f"{lvl}{check}", callback_data=f"groups:member_access:{group_id}:{member_id}:{lvl}"
            ))
        
        buttons = [
            access_buttons,
            [InlineKeyboardButton("✏️ Set Alias", callback_data=f"groups:member_alias_prompt:{group_id}:{member_id}")],
            [InlineKeyboardButton("⬅️ Back", callback_data=f"groups:members:{group_id}")],
        ]
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _group_menu_settings(self, query, group_id: str):
        """Show group settings with edit options."""
        from ..db.models import get_group
        group = await get_group("telegram", group_id)
        if not group:
            await query.edit_message_text("❌ Group not found.")
            return
        
        name = group.get("name", group_id)
        settings = group.get("settings") or {}
        custom = {k: v for k, v in settings.items() if k not in ("members", "model")}
        
        if custom:
            text = f"⚙️ <b>{name} — Settings</b>\n\n"
            for k, v in custom.items():
                text += f"• <code>{k}</code>: {v}\n"
        else:
            text = f"⚙️ <b>{name} — Settings</b>\n\nNo custom settings."
        
        text += "\n\nTo set: <code>/groups settings " + name + " key value</code>"
        
        buttons = [[InlineKeyboardButton("⬅️ Back", callback_data=f"groups:view:{group_id}")]]
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _group_get_model(self, update: Update, group_ref: str):
        """Show current model override for a group."""
        from ..db.models import get_group
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return
        
        model = (group.get("settings") or {}).get("model")
        name = group.get("name", group.get("platform_group_id"))
        if model:
            await update.message.reply_text(f"🤖 {name}: model = `{model}`", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"🤖 {name}: using default model (no override)")

    async def _group_set_model(self, update: Update, group_ref: str, model_key: str):
        """Set model override for a group."""
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return
        
        group_id = group["platform_group_id"]
        name = group.get("name", group_id)
        
        # "default" or "none" clears the override
        if model_key.lower() in ("default", "none", "clear"):
            settings = group.get("settings") or {}
            settings.pop("model", None)
            from ..db.connection import get_connection
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET settings = $1, updated_at = NOW() WHERE platform = 'telegram' AND platform_group_id = $2",
                    json.dumps(settings), group_id,
                )
            # Clear cached conversation so new provider takes effect
            self._clear_group_conversation(group_id)
            await update.message.reply_text(f"✅ {name}: model override cleared (using default)")
            return
        
        # Validate model key exists in registry
        from ..db.models import get_config
        models = await get_config("provider.models", None)
        if models:
            from ..llm.drivers import get_model_from_list
            entry = get_model_from_list(models, model_key)
            if not entry:
                available = ", ".join(f"`{m.get('key')}`" for m in models if m.get("key"))
                await update.message.reply_text(
                    f"❌ Model `{model_key}` not found.\n\nAvailable: {available}",
                    parse_mode="Markdown",
                )
                return
        
        # Save to group settings
        settings = group.get("settings") or {}
        settings["model"] = model_key
        from ..db.connection import get_connection
        async with get_connection() as conn:
            await conn.execute(
                "UPDATE groups SET settings = $1, updated_at = NOW() WHERE platform = 'telegram' AND platform_group_id = $2",
                json.dumps(settings), group_id,
            )
        
        # Clear cached conversation so new provider takes effect
        self._clear_group_conversation(group_id)
        
        label = ""
        if models:
            from ..llm.drivers import get_model_from_list as gm
            e = gm(models, model_key)
            if e:
                label = f" ({e.get('label', '')})"
        
        await update.message.reply_text(f"✅ {name}: model set to `{model_key}`{label}", parse_mode="Markdown")

    def _clear_group_conversation(self, group_id: str):
        """Clear cached conversation for a group so model override takes effect."""
        if not self.agent or not self.agent.conversations:
            return
        key = f"telegram:{group_id}"
        if key in self.agent.conversations._active:
            del self.agent.conversations._active[key]
            logger.info(f"Cleared cached conversation for group {group_id}")

    async def _group_add(self, update: Update, group_id_str: str):
        """Manually register a group by ID."""
        # Normalize group ID
        group_id_str = group_id_str.strip()
        
        # Check if already registered
        from ..db.models import get_group
        existing = await get_group("telegram", group_id_str)
        if existing:
            await update.message.reply_text(f"ℹ️ Group `{group_id_str}` is already registered.", parse_mode="Markdown")
            return
        
        # Try to get group info from Telegram
        group_name = f"Group {group_id_str}"
        try:
            chat = await self.app.bot.get_chat(int(group_id_str))
            if chat.title:
                group_name = chat.title
        except Exception as e:
            logger.warning(f"Could not get chat info for {group_id_str}: {e}")
        
        # Register the group
        from ..db.connection import get_connection
        async with get_connection() as conn:
            await conn.execute(
                """INSERT INTO groups (platform, platform_group_id, name, enabled, require_mention)
                   VALUES ($1, $2, $3, true, true)
                   ON CONFLICT (platform, platform_group_id) DO NOTHING""",
                "telegram", group_id_str, group_name,
            )
        
        await update.message.reply_text(
            f"✅ Group registered:\n\n"
            f"• Name: {group_name}\n"
            f"• ID: `{group_id_str}`\n"
            f"• Mention required: yes\n\n"
            f"Bot will now respond in this group when mentioned.",
            parse_mode="Markdown",
        )

    async def _group_list(self, update: Update):
        """List all registered groups."""
        from ..db.models import list_groups, get_config
        groups = await list_groups(platform="telegram", enabled_only=False)
        if not groups:
            await update.message.reply_text("No groups registered.")
            return

        # Build model label lookup
        models = await get_config("provider.models", [])
        active = await get_config("provider.active_model", "")
        model_map = {m["key"]: m.get("label", m["key"]) for m in models if "key" in m}
        default_label = model_map.get(active, active)

        lines = ["📋 <b>Registered Groups</b>\n"]
        for g in groups:
            status = "✅" if g.get("enabled") else "❌"
            mention = "📢" if not g.get("require_mention") else "🔇"
            settings = g.get("settings", {}) or {}
            member_count = len(settings.get("members", {}))
            mk = settings.get("model")
            mlabel = model_map.get(mk, mk) if mk else f"Default ({default_label})"
            # Custom settings (excluding members and model)
            custom = {k: v for k, v in settings.items() if k not in ("members", "model")}
            settings_str = ""
            if custom:
                settings_str = "\n   ⚙️ " + ", ".join(f"{k}={v}" for k, v in sorted(custom.items()))

            lines.append(
                f"{status} <b>{g['name']}</b>\n"
                f"   ID: <code>{g['platform_group_id']}</code>\n"
                f"   🤖 {mlabel}\n"
                f"   {mention} mention={'required' if g.get('require_mention') else 'optional'}"
                f" | {member_count} members{settings_str}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _group_members(self, update: Update, group_ref: str):
        """List members in a group. group_ref can be ID or partial name."""
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return

        settings = group.get("settings", {}) or {}
        members = settings.get("members", {})

        if not members:
            await update.message.reply_text(
                f"👥 <b>{group['name']}</b> — no members collected yet.\n"
                f"Members are auto-collected when they chat in the group.",
                parse_mode="HTML",
            )
            return

        lines = [f"👥 <b>{group['name']}</b> — {len(members)} members\n"]
        for mid, info in sorted(members.items(), key=lambda x: x[1].get("name", "")):
            name = info.get("name", "?")
            alias = info.get("alias")
            access = info.get("access", "public")
            username = info.get("username")

            # Access level emoji
            access_icon = {"owner": "👑", "family": "👨‍👩‍👧", "public": "👤"}.get(access, "❓")

            alias_str = f' → "<b>{alias}</b>"' if alias else ""
            user_str = f" (@{username})" if username else ""
            lines.append(f"{access_icon} {name}{user_str}{alias_str}\n   ID: <code>{mid}</code> | {access}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _group_set_access(self, update: Update, group_ref: str, member_id: str, level: str):
        """Set a member's access level in a group."""
        level = level.lower()
        if level not in ("owner", "family", "public"):
            await update.message.reply_text("❌ Access level must be: owner, family, or public")
            return

        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return

        from ..db.models import update_group_member
        result = await update_group_member(
            platform="telegram",
            platform_group_id=group["platform_group_id"],
            member_id=member_id,
            access=level,
        )
        if result:
            name = result.get("name", member_id)
            await update.message.reply_text(
                f"✅ <b>{name}</b> ({member_id}) set to <b>{level}</b> in {group['name']}",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("❌ Failed to update member.")

    async def _group_set_alias(self, update: Update, group_ref: str, member_id: str, alias: str):
        """Set a member's alias in a group."""
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return

        from ..db.models import update_group_member
        result = await update_group_member(
            platform="telegram",
            platform_group_id=group["platform_group_id"],
            member_id=member_id,
            alias=alias,
        )
        if result:
            name = result.get("name", member_id)
            await update.message.reply_text(
                f"✅ <b>{name}</b> ({member_id}) will be called \"<b>{alias}</b>\" in {group['name']}",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("❌ Failed to update alias.")

    async def _group_view_settings(self, update: Update, group_ref: str):
        """View group settings (excluding members — those are in /groups members)."""
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return

        settings = group.get("settings", {}) or {}
        # Filter out 'members' — that's managed via /groups members
        display = {k: v for k, v in settings.items() if k != "members"}

        if not display:
            await update.message.reply_text(
                f"⚙️ <b>{group['name']}</b> — no custom settings.\n\n"
                f"Set with: <code>/groups settings {group_ref} key value</code>",
                parse_mode="HTML",
            )
            return

        lines = [f"⚙️ <b>{group['name']}</b> settings:\n"]
        for k, v in sorted(display.items()):
            lines.append(f"  <b>{k}</b>: {v}")
        lines.append(f"\nEdit: <code>/groups settings {group_ref} key value</code>")
        lines.append(f"Delete: <code>/groups settings {group_ref} key --delete</code>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _group_set_setting(self, update: Update, group_ref: str, key: str, value: str):
        """Set or delete a group setting."""
        group = await self._resolve_group(group_ref)
        if not group:
            await update.message.reply_text(f"❌ Group not found: {group_ref}")
            return

        # Protect 'members' key — managed via /groups set and /groups alias
        if key == "members":
            await update.message.reply_text("❌ Use /groups set and /groups alias to manage members.")
            return

        from ..db.connection import get_connection

        if value == "--delete":
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET settings = settings - $1 WHERE id = $2",
                    key, group["id"],
                )
            await update.message.reply_text(
                f"🗑️ Setting <b>{key}</b> removed from {group['name']}",
                parse_mode="HTML",
            )
        else:
            async with get_connection() as conn:
                await conn.execute(
                    "UPDATE groups SET settings = jsonb_set(COALESCE(settings, '{}'), $1, $2) WHERE id = $3",
                    [key], f'"{value}"', group["id"],
                )
            await update.message.reply_text(
                f"✅ <b>{group['name']}</b>: <b>{key}</b> = {value}",
                parse_mode="HTML",
            )

    async def _resolve_group(self, ref: str) -> Optional[dict]:
        """Resolve a group reference — by ID or partial name match."""
        from ..db.models import get_group, list_groups

        # Try exact ID match first (with or without minus)
        group = await get_group("telegram", ref)
        if group:
            return group
        if not ref.startswith("-"):
            group = await get_group("telegram", f"-{ref}")
            if group:
                return group

        # Partial name match
        all_groups = await list_groups(platform="telegram", enabled_only=False)
        ref_lower = ref.lower()
        matches = [g for g in all_groups if ref_lower in (g.get("name") or "").lower()]
        if len(matches) == 1:
            return matches[0]
        return None

    async def _cmd_members(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /members command — manage global user access levels.

        Usage (owner DM only):
            /members list                     — list all registered users
            /members set <id> <level>         — set access level (owner/family/public/blocked)
        """
        user = update.effective_user
        chat = update.effective_chat

        # Owner-only, DM-only
        from ..db.models import get_user
        db_user = await get_user("telegram", str(user.id))
        if not db_user or db_user.get("access_level") != "owner":
            await update.message.reply_text("⚠️ Only the owner can manage members.")
            return
        if chat.type != "private":
            await update.message.reply_text("⚠️ Use /members in DM only.")
            return

        args = context.args or []
        if not args:
            await self._members_menu_main(update)
            return

        subcommand = args[0].lower()

        if subcommand == "list":
            await self._members_list(update)
        elif subcommand == "set" and len(args) >= 3:
            await self._members_set(update, args[1], args[2])
        else:
            await update.message.reply_text("❌ Invalid syntax. Use /members for help.")

    async def _members_menu_main(self, update=None, query=None):
        """Show global members list as interactive menu (blocked users excluded)."""
        from ..db.models import list_users, get_config as _gc
        users = await list_users(platform="telegram")

        active_users = [u for u in users if u.get("access_level") != "blocked"]
        blocked_count = len(users) - len(active_users)

        level_order = {"owner": 0, "family": 1, "public": 2, "approved": 3, "pending": 4}
        active_users.sort(key=lambda u: (level_order.get(u.get("access_level", "public"), 9), u.get("display_name") or u.get("name", "")))

        # Resolve model labels for display
        models = await _gc("provider.models", [])
        active_key = await _gc("provider.active_model", "gemini-pro")
        default_label = next((m.get("label", active_key) for m in models if m.get("key") == active_key), active_key)
        model_lookup = {m.get("key"): m.get("label", m.get("key")) for m in models}

        buttons = []
        for u in active_users:
            access = u.get("access_level", "public")
            icon = {"owner": "👑", "family": "👨‍👩‍👦", "public": "👤", "approved": "✅", "pending": "⏳"}.get(access, "❓")
            name = u.get("display_name") or u.get("name", "?")
            pid = u.get("platform_id", "?")
            # Resolve per-user model or default
            user_model_key = ((u.get("preferences") or {}).get("model"))
            if user_model_key:
                model_short = model_lookup.get(user_model_key, user_model_key)
            else:
                model_short = default_label
            buttons.append([InlineKeyboardButton(
                f"{icon} {name} — {access} | {model_short}",
                callback_data=f"mbr:view:{pid}",
            )])

        if blocked_count > 0:
            buttons.append([InlineKeyboardButton(f"🚫 Blocked Users ({blocked_count})", callback_data="mbr:blocked_list")])

        text = "👥 <b>Global Members</b>\n\nTap a user to manage:" if active_users else "👥 <b>Global Members</b>\n\nNo active users."
        markup = InlineKeyboardMarkup(buttons)

        if query:
            await query.edit_message_text(text, parse_mode="HTML", reply_markup=markup)
        else:
            await update.message.reply_text(text, parse_mode="HTML", reply_markup=markup)

    async def _members_menu_blocked(self, query):
        """Show blocked users submenu."""
        from ..db.models import list_users
        users = await list_users(platform="telegram")
        blocked = [u for u in users if u.get("access_level") == "blocked"]
        blocked.sort(key=lambda u: u.get("display_name") or u.get("name", ""))

        buttons = []
        for u in blocked:
            name = u.get("display_name") or u.get("name", "?")
            pid = u.get("platform_id", "?")
            buttons.append([InlineKeyboardButton(f"🚫 {name}", callback_data=f"mbr:blocked_view:{pid}")])

        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data="mbr:main")])

        text = "🚫 <b>Blocked Users</b>\n\nTap a user to manage:" if blocked else "🚫 <b>Blocked Users</b>\n\nNo blocked users."
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _members_menu_detail(self, query, platform_id: str):
        """Show user detail with access level options."""
        from ..db.models import get_user, get_first_owner
        user = await get_user("telegram", platform_id)
        if not user:
            await query.edit_message_text(f"❌ User {platform_id} not found.")
            return
        
        name = user.get("display_name") or user.get("name", "Unknown")
        access = user.get("access_level", "public")
        
        first_owner = await get_first_owner("telegram")
        is_first_owner = first_owner and first_owner.get("platform_id") == platform_id
        
        user_model = ((user.get("preferences") or {}).get("model"))
        if user_model:
            model_label = f"<code>{user_model}</code>"
        else:
            from ..db.models import get_config as _gc
            _models = await _gc("provider.models", [])
            _active = await _gc("provider.active_model", "gemini-pro")
            _def_lbl = next((m.get("label", _active) for m in _models if m.get("key") == _active), _active)
            model_label = f"<i>Default ({_def_lbl})</i>"
        text = f"👤 <b>{name}</b>\n\nID: <code>{platform_id}</code>\nAccess: <b>{access}</b>\nModel: {model_label}"
        if is_first_owner:
            text += "\n🔒 <i>First owner (immutable)</i>"

        # Access level buttons
        levels = ["owner", "family", "public", "blocked"]
        access_buttons = []
        for lvl in levels:
            check = " ✓" if lvl == access else ""
            cb = f"mbr:set:{platform_id}:{lvl}"
            access_buttons.append(InlineKeyboardButton(f"{lvl}{check}", callback_data=cb))

        buttons = [access_buttons]

        # Contextual action buttons
        caller_id = str(query.from_user.id)
        action_row = []
        if not is_first_owner and access not in ("owner", "pending") and platform_id != caller_id:
            action_row.append(InlineKeyboardButton("🗑 Release", callback_data=f"mbr:release:{platform_id}"))
        if action_row:
            buttons.append(action_row)

        buttons.append([InlineKeyboardButton("🤖 Change Model", callback_data=f"mbr:model_list:{platform_id}")])
        back_target = "mbr:blocked_list" if access == "blocked" else "mbr:main"
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=back_target)])
        
        await query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))

    async def _members_menu_model_list(self, query, platform_id: str):
        """Show model selection for a user."""
        from ..db.models import get_config, get_user
        user = await get_user("telegram", platform_id)
        current_model = ((user or {}).get("preferences") or {}).get("model")

        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        default_label = next((m.get("label", active_key) for m in models if m.get("key") == active_key), active_key)

        buttons = []
        for m in models:
            key = m.get("key", "")
            label = m.get("label", key)
            check = " ✓" if key == current_model else ""
            buttons.append([InlineKeyboardButton(
                f"{label}{check}", callback_data=f"mbr:model_set:{platform_id}:{key}"
            )])

        # Add "default" option with resolved label
        check = " ✓" if not current_model else ""
        buttons.append([InlineKeyboardButton(
            f"🔄 Default ({default_label}){check}", callback_data=f"mbr:model_set:{platform_id}:__default__"
        )])
        buttons.append([InlineKeyboardButton("⬅️ Back", callback_data=f"mbr:view:{platform_id}")])

        name = (user or {}).get("display_name") or (user or {}).get("name", platform_id)
        await query.edit_message_text(
            f"🤖 <b>Select Model for {name}</b>",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _handle_members_callback(self, query, data: str):
        """Handle interactive members menu callbacks."""
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        
        # Owner check
        from ..db.models import get_user
        db_user = await get_user("telegram", str(query.from_user.id))
        if not db_user or db_user.get("access_level") != "owner":
            await query.answer("⚠️ Owner only", show_alert=True)
            return
        
        if action == "main":
            await self._members_menu_main(query=query)

        elif action == "blocked_list":
            await self._members_menu_blocked(query)

        elif action == "blocked_view" and len(parts) >= 3:
            await self._members_menu_detail(query, parts[2])

        elif action == "view" and len(parts) >= 3:
            await self._members_menu_detail(query, parts[2])
        
        elif action == "set" and len(parts) >= 4:
            platform_id = parts[2]
            new_level = parts[3]
            
            from ..db.models import get_user as gu, update_user, get_first_owner
            
            # First owner protection
            first_owner = await get_first_owner("telegram")
            if first_owner and first_owner.get("platform_id") == platform_id and new_level != "owner":
                await query.answer("🔒 First owner cannot be changed!", show_alert=True)
                return
            
            # Self-downgrade protection
            if platform_id == str(query.from_user.id) and new_level != "owner":
                await query.answer("⚠️ Can't downgrade yourself!", show_alert=True)
                return
            
            result = await update_user("telegram", platform_id, access_level=new_level)
            if result:
                await query.answer(f"Access set to {new_level}")
            else:
                await query.answer("Failed to update", show_alert=True)
            
            await self._members_menu_detail(query, platform_id)

        elif action == "unblock" and len(parts) >= 3:
            platform_id = parts[2]
            from ..db.models import get_user as gu_ub, update_user, get_first_owner

            first_owner = await get_first_owner("telegram")
            if first_owner and first_owner.get("platform_id") == platform_id:
                await query.answer("🔒 First owner cannot be changed!", show_alert=True)
                return
            if platform_id == str(query.from_user.id):
                await query.answer("⚠️ Can't modify yourself!", show_alert=True)
                return

            result = await update_user("telegram", platform_id, access_level="pending")
            if result:
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(f"pending_notify:{platform_id}")
                await query.answer("User set to pending")
            else:
                await query.answer("Failed to update", show_alert=True)
            await self._members_menu_blocked(query)

        elif action == "release" and len(parts) >= 3:
            platform_id = parts[2]
            from ..db.models import get_first_owner, delete_user

            first_owner = await get_first_owner("telegram")
            if first_owner and first_owner.get("platform_id") == platform_id:
                await query.answer("🔒 First owner cannot be released!", show_alert=True)
                return
            if platform_id == str(query.from_user.id):
                await query.answer("⚠️ Can't release yourself!", show_alert=True)
                return

            deleted = await delete_user("telegram", platform_id)
            if deleted:
                # Clear cached DM conversation
                if self.agent and self.agent.conversations:
                    key = f"telegram:{platform_id}"
                    if key in self.agent.conversations._active:
                        del self.agent.conversations._active[key]
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(f"pending_notify:{platform_id}")
                await query.answer("User released")
            else:
                await query.answer("Failed to delete user", show_alert=True)
            await self._members_menu_main(query=query)

        elif action == "model_list" and len(parts) >= 3:
            await self._members_menu_model_list(query, parts[2])

        elif action == "model_set" and len(parts) >= 4:
            platform_id = parts[2]
            model_key = parts[3]

            from ..db.models import get_user as gu2, update_user
            user = await gu2("telegram", platform_id)
            if not user:
                await query.answer("User not found", show_alert=True)
                return

            prefs = user.get("preferences") or {}
            if model_key == "__default__":
                prefs.pop("model", None)
            else:
                prefs["model"] = model_key

            result = await update_user("telegram", platform_id, preferences=prefs)
            if not result:
                await query.answer("Failed to update", show_alert=True)
                return

            # Clear cached DM conversation so new provider takes effect
            if self.agent and self.agent.conversations:
                key = f"telegram:{platform_id}"
                if key in self.agent.conversations._active:
                    del self.agent.conversations._active[key]
                    logger.info(f"Cleared cached conversation for user {platform_id}")

            label = model_key if model_key != "__default__" else "default"
            await query.answer(f"Model set to {label}")
            await self._members_menu_detail(query, platform_id)

        else:
            await query.answer("Unknown action")

    async def _members_list(self, update: Update):
        """List all registered users with access levels (blocked excluded)."""
        from ..db.models import list_users
        users = await list_users(platform="telegram")
        if not users:
            await update.message.reply_text("No users registered.")
            return

        active_users = [u for u in users if u.get("access_level") != "blocked"]
        blocked_count = len(users) - len(active_users)

        lines = ["👥 <b>Registered Users</b>\n"]

        # Sort: owner first, then family, then public, then others
        level_order = {"owner": 0, "family": 1, "public": 2, "approved": 3, "pending": 4}
        active_users.sort(key=lambda u: (level_order.get(u.get("access_level", "public"), 9), u.get("name", "")))

        for u in active_users:
            access = u.get("access_level", "public")
            access_icon = {
                "owner": "👑",
                "family": "👨‍👩‍👧",
                "public": "👤",
                "approved": "✅",
                "pending": "⏳",
            }.get(access, "❓")

            name = u.get("display_name") or u.get("name", "?")
            platform_id = u.get("platform_id", "?")
            lines.append(f"{access_icon} <b>{name}</b> — {access}\n   ID: <code>{platform_id}</code>")

        if blocked_count > 0:
            lines.append(f"\n🚫 <i>{blocked_count} blocked user(s) hidden</i>")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _members_set(self, update: Update, member_id: str, level: str):
        """Set a user's global access level."""
        level = level.lower()
        valid_levels = ("owner", "family", "public", "pending", "blocked")
        if level not in valid_levels:
            await update.message.reply_text(
                f"❌ Access level must be one of: {', '.join(valid_levels)}"
            )
            return

        from ..db.models import get_user, update_user, get_first_owner

        # Check user exists
        db_user = await get_user("telegram", member_id)
        if not db_user:
            await update.message.reply_text(
                f"❌ User with ID <code>{member_id}</code> not found.\n"
                f"They need to DM the bot first (or chat in an approved group).",
                parse_mode="HTML",
            )
            return

        # IMMUTABLE: First owner cannot be changed via UI — ever
        first_owner = await get_first_owner("telegram")
        if first_owner and db_user.get("platform_id") == first_owner.get("platform_id"):
            if level != "owner":
                await update.message.reply_text(
                    "🔒 The original owner's access level cannot be changed.\n"
                    "This is a permanent protection — only direct database access can modify it."
                )
                return

        # Prevent any owner from downgrading themselves
        current_user = update.effective_user
        if member_id == str(current_user.id) and level != "owner":
            await update.message.reply_text(
                "⚠️ You can't downgrade your own access level.\n"
                "Another owner would need to do this."
            )
            return

        result = await update_user("telegram", member_id, access_level=level)
        if result:
            name = result.get("display_name") or result.get("name", member_id)
            prev_level = db_user.get("access_level", "?")
            await update.message.reply_text(
                f"✅ <b>{name}</b> (<code>{member_id}</code>)\n"
                f"   {prev_level} → <b>{level}</b>",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("❌ Failed to update access level.")

    async def _handle_auth_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
        """Handle user input during auth flow.
        
        Returns True if the message was handled (credential processed),
        False if it should be treated as normal chat.
        """
        user_id = update.effective_user.id
        state = self._auth_state.get(user_id)
        if not state:
            return False

        chat_id = update.effective_chat.id
        auth_type = state.get("type")
        provider = state.get("provider", "")

        if auth_type == "oauth":
            return await self._process_oauth_input(user_id, chat_id, text, state, context)
        elif auth_type == "apikey":
            return await self._process_apikey_input(user_id, chat_id, text, state, context)
        elif auth_type in ("addmodel", "models_add"):
            return await self._process_addmodel_input(user_id, chat_id, text, state, context)
        elif auth_type in ("addembed", "embed_add"):
            return await self._process_addembed_input(user_id, chat_id, text, state, context)
        elif auth_type == "ctx_window":
            return await self._process_ctx_window_input(user_id, chat_id, text, state, context)
        elif auth_type == "models_params":
            return await self._process_models_params(user_id, chat_id, text, state, context)
        elif auth_type == "models_edit_label":
            return await self._process_models_edit_label(user_id, chat_id, text, state, context)
        elif auth_type == "embed_edit_label":
            return await self._process_embed_edit_label(user_id, chat_id, text, state, context)
        elif auth_type == "eval_add":
            return await self._process_addeval_input(user_id, chat_id, text, state, context)
        elif auth_type == "group_alias":
            return await self._process_group_alias_input(user_id, chat_id, text, state, context)

        return False

    async def _process_oauth_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Process pasted OAuth callback URL."""
        from urllib.parse import urlparse, parse_qs

        text = text.strip()

        # Check if this looks like a callback URL
        if not text.startswith("http"):
            # Not a URL — user might be chatting normally
            bot = context.bot if context else self.app.bot
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ That doesn't look like a callback URL.\n"
                     "Paste the full URL from your browser address bar, or send /cancel to abort.",
            )
            return True  # Still consume the message to prevent history save

        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]

        if not code:
            bot = context.bot if context else self.app.bot
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ No authorization code found in that URL.\n"
                     "Make sure you copy the ENTIRE URL from the browser address bar.",
            )
            return True

        provider = state["provider"]
        bot = context.bot if context else self.app.bot

        await bot.send_message(chat_id=chat_id, text="🔄 Exchanging authorization code...")

        try:
            if provider == "google":
                await self._exchange_google_oauth(code, state)
                await bot.send_message(chat_id=chat_id, text="✅ Google OAuth configured successfully!")
            elif provider == "codex":
                await self._exchange_codex_oauth(code, state)
                await bot.send_message(chat_id=chat_id, text="✅ OpenAI/Codex OAuth configured successfully!")
            elif provider == "claude":
                await self._exchange_claude_oauth(code, state)
                await bot.send_message(chat_id=chat_id, text="✅ Claude OAuth configured successfully!")
            else:
                await bot.send_message(chat_id=chat_id, text=f"❌ Unknown provider: {provider}")
        except Exception as e:
            logger.error(f"OAuth exchange failed for {provider}: {e}", exc_info=True)
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ OAuth exchange failed: {str(e)[:200]}\n\nTry again via /models → Re-authenticate.",
            )
        finally:
            # Always clear auth state
            self._auth_state.pop(user_id, None)

        return True

    async def _process_apikey_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Process pasted API key."""
        text = text.strip()
        credential_key = state.get("credential_key", "")
        label = state.get("label", state.get("provider", "?"))
        bot = context.bot if context else self.app.bot

        if not credential_key:
            await bot.send_message(chat_id=chat_id, text="❌ No credential key configured for this provider.")
            self._auth_state.pop(user_id, None)
            return True

        # Basic validation — API keys are typically 20+ chars alphanumeric
        if len(text) < 10:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ That looks too short for an API key. Paste the full key, or send /cancel to abort.",
            )
            return True

        try:
            await set_config(credential_key, text, f"{label} API key")
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ {label} API key saved.\n\n⚠️ Use /restart to apply.",
            )
            logger.info(f"API key saved for {label} ({credential_key}) by user {user_id}")
        except Exception as e:
            logger.error(f"Failed to save API key for {label}: {e}")
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ Failed to save API key: {str(e)[:200]}",
            )
        finally:
            self._auth_state.pop(user_id, None)

        return True

    # Default context windows per driver (used as suggestion)
    _DEFAULT_CONTEXT_WINDOWS = {
        "google_cca": 1048576,  # 1M
        "codex": 1048576,       # 1M
        "anthropic": 200000,    # 200K
        "openai_compat": 128000,
        "groq": 131072,         # 128K
    }

    # Default auth type per driver
    _DRIVER_AUTH_TYPES = {
        "google_cca": "oauth",
        "codex": "oauth",
        "anthropic": "api_key",
        "openai_compat": "api_key",
        "groq": "api_key",
        "ollama": "none",
        "together": "api_key",
    }

    # Default LLM parameters per driver (all 7 keys)
    _DRIVER_DEFAULT_PARAMS = {
        "google_cca":   {"temperature": 0.7, "max_tokens": None, "thinking_budget": -1,    "top_p": 0.95, "top_k": 40,   "frequency_penalty": None, "presence_penalty": None},
        "codex":        {"temperature": 0.7, "max_tokens": None, "thinking_budget": 10000, "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
        "anthropic":    {"temperature": 0.3, "max_tokens": None, "thinking_budget": 32000, "top_p": 0.99, "top_k": 50,   "frequency_penalty": None, "presence_penalty": None},
        "openai_compat":{"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
        "together":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
        "ollama":       {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 0.9,  "top_k": 40,   "frequency_penalty": None, "presence_penalty": None},
        "_default":     {"temperature": 0.7, "max_tokens": None, "thinking_budget": None,  "top_p": 1.0,  "top_k": None, "frequency_penalty": 0,    "presence_penalty": 0},
    }

    async def _process_addmodel_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Multi-step model addition flow."""
        bot = context.bot if context else self.app.bot
        step = state.get("step", "")
        text = text.strip()

        if step == "base_url":
            # Custom API key provider — user enters base URL
            if not text.startswith("http"):
                await bot.send_message(chat_id=chat_id, text="⚠️ URL must start with http:// or https://", parse_mode="HTML")
                return True
            state["base_url"] = text.rstrip("/")
            state["credential_key"] = "credential.custom_api_key"
            state["step"] = "apikey"
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"Base URL: <code>{text}</code>\n\n"
                    f"Now paste your <b>API key</b>.\n"
                    f"It will be saved to the database and <b>NOT</b> stored in chat history.\n\n"
                    f"Send /cancel to abort."
                ),
                parse_mode="HTML",
            )
            return True

        elif step == "apikey":
            # Save API key, then ask for model ID
            cred_key = state.get("credential_key", "credential.custom_api_key")
            await set_config(cred_key, text)
            state["step"] = "model_id"
            masked = text[:8] + "..."
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔑 API key saved (<code>{masked}</code>)\n\n"
                    f"Now send the <b>model ID</b>.\n\n"
                    f"Send /cancel to abort."
                ),
                parse_mode="HTML",
            )
            return True

        elif step == "model_id":
            state["model_id"] = text
            state["step"] = "label"
            await bot.send_message(
                chat_id=chat_id,
                text=f"Model ID: <code>{text}</code>\n\n"
                     f"Now send a <b>display label</b> (e.g. <code>Gemini 2.5 Flash</code>, <code>GPT-5.2</code>).\n"
                     f"Or send <code>.</code> to use model ID as label.",
                parse_mode="HTML",
            )
            return True

        elif step == "label":
            label = text if text != "." else state["model_id"]
            state["label"] = label
            state["step"] = "context_window"
            driver = state.get("driver", "")
            default_ctx = self._DEFAULT_CONTEXT_WINDOWS.get(driver, 128000)
            await bot.send_message(
                chat_id=chat_id,
                text=f"Label: <b>{label}</b>\n\n"
                     f"Send <b>context window</b> in tokens (number only).\n"
                     f"Or send <code>.</code> for default ({default_ctx:,}).",
                parse_mode="HTML",
            )
            return True

        elif step == "context_window":
            driver = state.get("driver", "")
            if text == ".":
                ctx_window = self._DEFAULT_CONTEXT_WINDOWS.get(driver, 128000)
            else:
                try:
                    ctx_window = int(text.replace(",", "").replace("_", ""))
                except ValueError:
                    await bot.send_message(chat_id=chat_id, text="⚠️ Enter a number, or <code>.</code> for default.", parse_mode="HTML")
                    return True

            # Build model entry and save
            auth_type = self._DRIVER_AUTH_TYPES.get(driver, "api_key")
            entry = {
                "key": state["model_id"].replace("/", "-").replace(".", "-").lower(),
                "label": state["label"],
                "driver": driver,
                "model_id": state["model_id"],
                "auth": auth_type,
                "context_window": ctx_window,
                "params": self._DRIVER_DEFAULT_PARAMS.get(driver, self._DRIVER_DEFAULT_PARAMS["_default"]).copy(),
                "reasoning_visible": False,
            }
            # Include base_url and credential_key for API key models
            if state.get("base_url"):
                entry["base_url"] = state["base_url"]
            if state.get("credential_key"):
                entry["credential_key"] = state["credential_key"]

            # Append to provider.models in DB
            models = await get_config("provider.models", [])
            # Check for duplicate key
            existing = next((m for m in models if m["key"] == entry["key"]), None)
            if existing:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Model <code>{entry['key']}</code> already exists. Remove it first or use a different model ID.",
                    parse_mode="HTML",
                )
                self._auth_state.pop(user_id, None)
                return True

            models.append(entry)
            await set_config("provider.models", models)

            ctx_display = f"{ctx_window/1000000:.1f}M" if ctx_window >= 1000000 else f"{ctx_window//1000}K"
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ <b>Model added:</b>\n\n"
                    f"• Key: <code>{entry['key']}</code>\n"
                    f"• Label: {entry['label']}\n"
                    f"• Driver: {driver}\n"
                    f"• Auth: {auth_type}\n"
                    f"• Context: {ctx_display}\n\n"
                    f"Use /models to switch to it."
                ),
                parse_mode="HTML",
            )
            logger.info(f"Model added: {entry}")
            self._auth_state.pop(user_id, None)
            return True

        return False

    async def _process_ctx_window_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Handle custom context window text input."""
        bot = context.bot if context else self.app.bot
        text = text.strip()

        try:
            new_ctx = int(text.replace(",", "").replace("_", ""))
        except ValueError:
            await bot.send_message(
                chat_id=chat_id,
                text="Enter a number (e.g. <code>1000000</code>), or /cancel to abort.",
                parse_mode="HTML",
            )
            return True

        if new_ctx < 1000:
            await bot.send_message(chat_id=chat_id, text="Value too small. Minimum 1,000 tokens.")
            return True

        # Update registry
        models = await get_config("provider.models", [])
        active_key = await get_config("provider.active_model", "gemini-pro")
        updated = False
        model_label = active_key
        for m in models:
            if m.get("key") == active_key:
                m["context_window"] = new_ctx
                model_label = m.get("label", active_key)
                updated = True
                break

        if updated:
            await set_config("provider.models", models)
            if self.agent:
                await self.agent.reload_provider()

            ctx_display = f"{new_ctx/1000000:.1f}M" if new_ctx >= 1000000 else f"{new_ctx//1000}K"
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ Context window for <b>{model_label}</b> set to <b>{ctx_display}</b> ({new_ctx:,} tokens)",
                parse_mode="HTML",
            )
        else:
            await bot.send_message(chat_id=chat_id, text="Model not found in registry.")

        self._auth_state.pop(user_id, None)
        return True

    async def _process_models_params(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Handle JSON input for editing model LLM parameters."""
        bot = context.bot if context else self.app.bot
        text = text.strip()
        model_key = state.get("model_key", "")

        # Parse JSON
        try:
            params = json.loads(text)
        except json.JSONDecodeError as e:
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ Invalid JSON: {e}\n\nSend valid JSON or /cancel to abort.",
            )
            return True

        if not isinstance(params, dict):
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Must be a JSON object (dict), not a list or value.\n\nSend valid JSON or /cancel to abort.",
            )
            return True

        # Validate known keys
        errors = []
        if "temperature" in params:
            t = params["temperature"]
            if not isinstance(t, (int, float)) or t < 0.0 or t > 2.0:
                errors.append("temperature: must be a number 0.0–2.0")
        if "max_tokens" in params:
            mt = params["max_tokens"]
            if mt is not None and (not isinstance(mt, int) or mt <= 0):
                errors.append("max_tokens: must be a positive integer or null")
        if "thinking_budget" in params:
            tb = params["thinking_budget"]
            if tb is not None and (not isinstance(tb, int) or tb < -1):
                errors.append("thinking_budget: must be an integer >= 0, -1 (dynamic), or null")
        if "top_p" in params:
            tp = params["top_p"]
            if tp is not None and (not isinstance(tp, (int, float)) or tp < 0.0 or tp > 1.0):
                errors.append("top_p: must be a number 0.0–1.0 or null")
        if "top_k" in params:
            tk = params["top_k"]
            if tk is not None and (not isinstance(tk, int) or tk <= 0):
                errors.append("top_k: must be a positive integer or null")
        if "frequency_penalty" in params:
            fp = params["frequency_penalty"]
            if fp is not None and (not isinstance(fp, (int, float)) or fp < -2.0 or fp > 2.0):
                errors.append("frequency_penalty: must be a number -2.0–2.0 or null")
        if "presence_penalty" in params:
            pp = params["presence_penalty"]
            if pp is not None and (not isinstance(pp, (int, float)) or pp < -2.0 or pp > 2.0):
                errors.append("presence_penalty: must be a number -2.0–2.0 or null")

        if errors:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Validation errors:\n• " + "\n• ".join(errors) + "\n\nFix and resend, or /cancel to abort.",
            )
            return True

        # Save to model entry
        models = await get_config("provider.models", [])
        updated = False
        label = model_key
        for m in models:
            if m.get("key") == model_key:
                m["params"] = params
                label = m.get("label", model_key)
                updated = True
                break

        if updated:
            await set_config("provider.models", models)
            if self.agent:
                await self.agent.reload_provider()
            import json as _json
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ Parameters updated for <b>{label}</b>:\n<pre>{_json.dumps(params, indent=2)}</pre>",
                parse_mode="HTML",
            )
        else:
            await bot.send_message(chat_id=chat_id, text="Model not found in registry.")

        self._auth_state.pop(user_id, None)
        return True

    async def _process_models_edit_label(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Handle text input for editing a model label."""
        bot = context.bot if context else self.app.bot
        text = text.strip()
        model_key = state.get("model_key", "")

        models = await get_config("provider.models", [])
        updated = False
        for m in models:
            if m.get("key") == model_key:
                m["label"] = text
                updated = True
                break

        if updated:
            await set_config("provider.models", models)
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ Label updated to <b>{text}</b>",
                parse_mode="HTML",
            )
        else:
            await bot.send_message(chat_id=chat_id, text="❌ Model not found in registry.")

        self._auth_state.pop(user_id, None)
        return True

    async def _process_group_alias_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Handle text input for setting a group member alias."""
        bot = context.bot if context else self.app.bot
        text = text.strip()
        group_id = state.get("group_id", "")
        member_id = state.get("member_id", "")

        from ..db.models import update_group_member
        result = await update_group_member(
            platform="telegram",
            platform_group_id=group_id,
            member_id=member_id,
            alias=text,
        )
        if result:
            name = result.get("name", member_id)
            await bot.send_message(
                chat_id=chat_id,
                text=f'✅ <b>{name}</b> ({member_id}) alias set to "<b>{text}</b>"',
                parse_mode="HTML",
            )
        else:
            await bot.send_message(chat_id=chat_id, text="❌ Failed to update alias.")

        self._auth_state.pop(user_id, None)
        return True

    async def _process_embed_edit_label(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Handle text input for editing an embedding label."""
        bot = context.bot if context else self.app.bot
        text = text.strip()
        embed_key = state.get("embed_key", "")

        models = await get_config("provider.embedding_models", [])
        updated = False
        for m in models:
            if m.get("key") == embed_key:
                m["label"] = text
                updated = True
                break

        if updated:
            await set_config("provider.embedding_models", models)
            await bot.send_message(
                chat_id=chat_id,
                text=f"✅ Embedding label updated to <b>{text}</b>",
                parse_mode="HTML",
            )
        else:
            await bot.send_message(chat_id=chat_id, text="❌ Embedding not found in registry.")

        self._auth_state.pop(user_id, None)
        return True

    async def _process_addembed_input(self, user_id: int, chat_id: int, text: str, state: dict, context) -> bool:
        """Multi-step embedding provider addition flow."""
        bot = context.bot if context else self.app.bot
        step = state.get("step", "")
        text = text.strip()

        if step == "model_id":
            state["model_id"] = text
            state["step"] = "dimensions"
            await bot.send_message(
                chat_id=chat_id,
                text=f"Model ID: <code>{text}</code>\n\n"
                     f"Send <b>embedding dimensions</b> (number only, e.g. <code>768</code>, <code>1024</code>, <code>1536</code>).",
                parse_mode="HTML",
            )
            return True

        elif step == "dimensions":
            try:
                dims = int(text.replace(",", ""))
            except ValueError:
                await bot.send_message(chat_id=chat_id, text="⚠️ Enter a number (e.g. <code>768</code>, <code>1024</code>).", parse_mode="HTML")
                return True

            state["dimensions"] = dims
            driver = state.get("driver", "")

            if driver == "ollama":
                state["step"] = "label"
                state["base_url"] = "http://localhost:11434"
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"Dimensions: {dims}\n\n"
                         f"Send a <b>display label</b> (e.g. <code>Ollama — qwen3 (local)</code>).\n"
                         f"Or send <code>.</code> to auto-generate.",
                    parse_mode="HTML",
                )
            else:
                state["step"] = "label"
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"Dimensions: {dims}\n\n"
                         f"Send a <b>display label</b> (e.g. <code>Together — BGE Base</code>).\n"
                         f"Or send <code>.</code> to auto-generate.",
                    parse_mode="HTML",
                )
            return True

        elif step == "label":
            driver = state.get("driver", "")
            model_id = state.get("model_id", "")
            if text == ".":
                label = f"{driver} — {model_id}"
            else:
                label = text

            auth_type = self._DRIVER_AUTH_TYPES.get(driver, "api_key")
            cost = "FREE (local)" if driver == "ollama" else "API Key"

            entry = {
                "key": model_id.replace("/", "-").replace(".", "-").replace(":", "-").lower(),
                "label": label,
                "driver": driver,
                "model_id": model_id,
                "auth": auth_type,
                "dimensions": state["dimensions"],
                "cost": cost,
            }
            if driver == "ollama":
                entry["base_url"] = state.get("base_url", "http://localhost:11434")

            # Append to provider.embedding_models in DB
            models = await get_config("provider.embedding_models", [])
            existing = next((m for m in models if m["key"] == entry["key"]), None)
            if existing:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Embedding <code>{entry['key']}</code> already exists.",
                    parse_mode="HTML",
                )
                self._auth_state.pop(user_id, None)
                return True

            models.append(entry)
            await set_config("provider.embedding_models", models)

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ <b>Embedding added:</b>\n\n"
                    f"• Key: <code>{entry['key']}</code>\n"
                    f"• Label: {entry['label']}\n"
                    f"• Driver: {driver}\n"
                    f"• Dimensions: {entry['dimensions']}\n"
                    f"• Cost: {cost}\n\n"
                    f"Use /embedding to activate it."
                ),
                parse_mode="HTML",
            )
            logger.info(f"Embedding added: {entry}")
            self._auth_state.pop(user_id, None)
            return True

        return False

    async def _start_oauth_flow(self, provider_entry: dict, chat_id: int, user_id: int, context):
        """Generate OAuth URL and send it to user via Telegram.
        
        Args:
            provider_entry: Dict from auth.providers with key, label, oauth_driver, etc.
        """
        bot = context.bot if context else self.app.bot
        driver = provider_entry.get("oauth_driver", provider_entry.get("key", ""))
        label = provider_entry.get("label", driver)

        # Map oauth_driver → URL generator
        _generators = {
            "google": self._generate_google_oauth_url,
            "codex": self._generate_codex_oauth_url,
            "claude": self._generate_claude_oauth_url,
        }

        generator = _generators.get(driver)
        if not generator:
            await bot.send_message(chat_id=chat_id, text=f"❌ Unknown OAuth driver: {driver}")
            return

        try:
            url, pkce_state = generator()
            self._auth_state[user_id] = {
                "type": "oauth",
                "provider": driver,  # Used by _exchange methods
                "verifier": pkce_state["verifier"],
                "state": pkce_state.get("state", ""),
                "label": label,
            }

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🔐 **{label} OAuth Setup**\n\n"
                    f"1️⃣ Open this URL in your browser:\n\n"
                    f"`{url}`\n\n"
                    f"2️⃣ Sign in and authorize access\n"
                    f'3️⃣ Browser will show "This site can\'t be reached" — that\'s normal!\n'
                    f"4️⃣ Copy the ENTIRE URL from the browser address bar\n"
                    f"5️⃣ Paste it here\n\n"
                    f"⏱️ You have 5 minutes. Send /cancel to abort."
                ),
                parse_mode="Markdown",
            )
            logger.info(f"OAuth flow started for {label} ({driver}) by user {user_id}")

        except Exception as e:
            logger.error(f"Failed to start OAuth flow for {label}: {e}", exc_info=True)
            await bot.send_message(
                chat_id=chat_id,
                text=f"❌ Failed to start OAuth flow: {str(e)[:200]}",
            )
            self._auth_state.pop(user_id, None)

    def _generate_google_oauth_url(self) -> tuple[str, dict]:
        """Generate Google OAuth URL with PKCE."""
        import base64, hashlib, secrets
        from urllib.parse import urlencode

        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

        # Same credentials as google_oauth.py
        client_id = base64.b64decode(
            "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLm"
            "FwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
        ).decode()

        params = urlencode({
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": "http://localhost:8085/oauth2callback",
            "scope": " ".join([
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/userinfo.email",
                "https://www.googleapis.com/auth/userinfo.profile",
            ]),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
            "access_type": "offline",
            "prompt": "consent",
        })
        url = f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
        return url, {"verifier": verifier, "challenge": challenge}

    def _generate_codex_oauth_url(self) -> tuple[str, dict]:
        """Generate Codex/OpenAI OAuth URL with PKCE."""
        import base64, hashlib, secrets
        from urllib.parse import urlencode

        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        state = secrets.token_hex(16)

        params = urlencode({
            "response_type": "code",
            "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
            "redirect_uri": "http://localhost:1455/auth/callback",
            "scope": "openid profile email offline_access",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        })
        url = f"https://auth.openai.com/oauth/authorize?{params}"
        return url, {"verifier": verifier, "challenge": challenge, "state": state}

    def _generate_claude_oauth_url(self) -> tuple[str, dict]:
        """Generate Claude OAuth URL with PKCE."""
        import base64, hashlib, secrets
        from urllib.parse import urlencode

        verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        state = secrets.token_hex(16)

        params = urlencode({
            "response_type": "code",
            "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
            "redirect_uri": "http://localhost:9742/oauth/callback",
            "scope": "user:inference user:profile",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        })
        url = f"https://claude.ai/oauth/authorize?{params}"
        return url, {"verifier": verifier, "challenge": challenge, "state": state}

    async def _exchange_google_oauth(self, code: str, state: dict):
        """Exchange Google auth code for tokens and save to DB."""
        import base64, time
        from ..auth.google_oauth import GoogleCredentials, _discover_project, _get_user_email

        client_id = base64.b64decode(
            "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLm"
            "FwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
        ).decode()
        client_secret = base64.b64decode("R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=").decode()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": "http://localhost:8085/oauth2callback",
                    "code_verifier": state["verifier"],
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        if not token_data.get("refresh_token"):
            raise RuntimeError("No refresh token received. Try again with prompt=consent.")

        access_token = token_data["access_token"]
        expires_at = time.time() + token_data["expires_in"] - 300

        email = await _get_user_email(access_token)
        project_id = await _discover_project(access_token)

        creds = GoogleCredentials(
            access_token=access_token,
            refresh_token=token_data["refresh_token"],
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )
        await creds.save_to_db()
        logger.info(f"Google OAuth credentials saved for {email}")

    async def _exchange_codex_oauth(self, code: str, state: dict):
        """Exchange Codex auth code for tokens and save to DB."""
        import time

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://auth.openai.com/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                    "code": code,
                    "code_verifier": state["verifier"],
                    "redirect_uri": "http://localhost:1455/auth/callback",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        if not access_token or not refresh_token:
            raise RuntimeError("Token exchange failed — missing tokens.")

        expires_at = time.time() + token_data.get("expires_in", 0)

        await set_config("credential.codex_access_token", access_token, "Codex OAuth access token")
        await set_config("credential.codex_refresh_token", refresh_token, "Codex OAuth refresh token")
        await set_config("credential.codex_expires_at", expires_at, "Codex OAuth token expiry")
        logger.info("Codex OAuth credentials saved to DB")

    async def _exchange_claude_oauth(self, code: str, state: dict):
        """Exchange Claude auth code for tokens and save to DB."""
        import time

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://platform.claude.com/v1/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                    "code": code,
                    "code_verifier": state["verifier"],
                    "redirect_uri": "http://localhost:9742/oauth/callback",
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "anthropic-beta": "oauth-2025-04-20",
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        if not access_token or not refresh_token:
            raise RuntimeError("Token exchange failed — missing tokens.")

        expires_at = time.time() + token_data.get("expires_in", 3600) - 300

        # Try to get user profile
        email = None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/me",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "oauth-2025-04-20",
                    },
                )
                if resp.status_code == 200:
                    email = resp.json().get("email") or resp.json().get("name")
        except Exception:
            pass

        from ..db.credentials import set_claude_oauth_credentials
        await set_claude_oauth_credentials(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
        )
        logger.info(f"Claude OAuth credentials saved for {email}")

    async def _get_credential_status(self) -> str:
        """Get formatted status of all credentials from DB provider registry."""
        import time
        from ..db.credentials import (
            get_google_oauth_credentials,
            get_claude_oauth_credentials,
            get_credential,
        )

        providers = await self._get_auth_providers()
        lines = ["📋 **Credential Status**\n"]

        for p in providers:
            key = p.get("key", "")
            label = p.get("label", key)
            auth_type = p.get("auth_type", "")

            if auth_type == "oauth":
                driver = p.get("oauth_driver", key)
                if driver == "google":
                    creds = await get_google_oauth_credentials()
                    if creds and creds.get("refresh_token"):
                        exp = creds.get("expires_at", 0)
                        expired = time.time() >= exp
                        email = creds.get("email", "?")
                        status = "⚠️ expired" if expired else "✅ active"
                        lines.append(f"🔑 **{label}:** {status} ({email})")
                    else:
                        lines.append(f"🔑 **{label}:** ❌ not configured")
                elif driver == "claude":
                    creds = await get_claude_oauth_credentials()
                    if creds and creds.get("refresh_token"):
                        exp = creds.get("expires_at", 0)
                        expired = time.time() >= exp
                        email = creds.get("email", "?")
                        status = "⚠️ expired" if expired else "✅ active"
                        lines.append(f"🔑 **{label}:** {status} ({email})")
                    else:
                        lines.append(f"🔑 **{label}:** ❌ not configured")
                elif driver == "codex":
                    token = await get_credential("credential.codex_access_token")
                    if token:
                        exp = await get_credential("credential.codex_expires_at", 0)
                        expired = time.time() >= float(exp or 0)
                        status = "⚠️ expired" if expired else "✅ active"
                        lines.append(f"🔑 **{label}:** {status}")
                    else:
                        lines.append(f"🔑 **{label}:** ❌ not configured")
                else:
                    lines.append(f"🔑 **{label}:** ❓ unknown driver ({driver})")

            elif auth_type == "api_key":
                cred_key = p.get("credential_key", "")
                if cred_key:
                    val = await get_credential(cred_key)
                    if val:
                        masked = f"{str(val)[:4]}...{str(val)[-4:]}" if len(str(val)) > 8 else "***"
                        lines.append(f"🔐 **{label}:** ✅ ({masked})")
                    else:
                        lines.append(f"🔐 **{label}:** ❌ not configured")
                else:
                    lines.append(f"🔐 **{label}:** ❓ no credential_key")

        return "\n".join(lines)

    @staticmethod
    def _contains_credential(text: str) -> bool:
        """Check if text contains credential patterns.
        
        Used to prevent accidental credential leaks in normal chat.
        Conservative — only catches high-confidence patterns to avoid
        false positives on normal conversation.
        """
        if not text or len(text) < 20:
            return False

        from ..security import _SAFE_REDACT_PATTERNS

        for pattern, _ in _SAFE_REDACT_PATTERNS:
            if pattern.search(text):
                # Double-check: the matched text should look like a standalone credential,
                # not a code snippet or URL discussion
                match = pattern.search(text)
                if match:
                    matched = match.group(0)
                    # Skip if it's clearly a code example or markdown
                    if text.count('\n') > 3:  # Multi-line = likely code/docs
                        continue
                    # Skip if surrounded by backticks (code context)
                    start = max(0, match.start() - 5)
                    end = min(len(text), match.end() + 5)
                    context = text[start:end]
                    if '`' in context:
                        continue
                    return True

        return False

    # ── Browse (directory picker) ──────────────────────────────

    async def _cmd_browse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /browse — interactive directory picker for CLI session sharing."""
        user = update.effective_user
        chat = update.effective_chat

        # Owner-only, DM-only
        if chat.type in ("group", "supergroup"):
            await update.message.reply_text("⚠️ /browse only works in DM.")
            return
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Owner only.")
            return

        # Get Syne project root as default
        import os
        syne_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        current_browse = self._browse_cwd.get(user.id)

        # Start path = current browse dir, or home
        start_path = current_browse or os.path.expanduser("~")

        await self._send_browse_picker(chat.id, user.id, start_path, context)

    async def _send_browse_picker(
        self, chat_id: int, user_id: int, path: str, context, message_id: int | None = None
    ):
        """Send or edit the directory picker inline keyboard."""
        import os

        path = os.path.realpath(path)
        current_browse = self._browse_cwd.get(user_id)

        # List directory entries
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            entries = []

        # Separate dirs and show only dirs (this is a directory picker)
        dirs = []
        for e in entries:
            full = os.path.join(path, e)
            if os.path.isdir(full) and not e.startswith("."):
                dirs.append(e)

        # Build buttons — max 30 dirs shown
        buttons = []

        # Parent directory (if not root)
        if path != "/":
            parent_id = self._path_id(os.path.dirname(path))
            buttons.append([InlineKeyboardButton("📁 ..", callback_data=f"brw:n:{parent_id}")])

        # Directory entries (2 per row)
        row = []
        for d in dirs[:30]:
            label = f"📂 {d}"
            pid = self._path_id(os.path.join(path, d))
            row.append(InlineKeyboardButton(label, callback_data=f"brw:n:{pid}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        # Action buttons
        sel_id = self._path_id(path)
        buttons.append([InlineKeyboardButton("✅ Select this directory", callback_data=f"brw:s:{sel_id}")])
        buttons.append([InlineKeyboardButton(
            "💬 Back to Telegram session" if current_browse else "❌ Cancel",
            callback_data="brw:reset",
        )])

        # Header text — just show current path being browsed
        text = f"📂 `{path}`"
        if current_browse:
            text = f"✅ Active: `{current_browse}`\n\n{text}"

        markup = InlineKeyboardMarkup(buttons)

        if message_id:
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text,
                    parse_mode="Markdown",
                    reply_markup=markup,
                )
                return
            except Exception:
                pass

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )

    # ── Helpers ───────────────────────────────────────────────

    def _format_thinking_level(self, budget) -> str:
        """Format thinking budget as human-readable level."""
        if budget is None:
            return "OFF"
        budget = int(budget)
        if budget == 0:
            return "OFF"
        if budget == -1:
            return "Dynamic"
        levels = {1024: "Low", 4096: "Medium", 8192: "High", 10240: "High", 24576: "Max"}
        level_name = levels.get(budget, "custom")
        return f"{level_name} ({budget:,})"


    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        user = query.from_user

        # Auth check — owner only
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await query.edit_message_text("⚠️ Owner only.")
            return

        if data.startswith("models:"):
            await self._handle_models_callback(query, data)

        elif data.startswith("embed:"):
            await self._handle_embed_callback(query, data)

        elif data.startswith("eval:"):
            await self._handle_eval_callback(query, data)

        elif data.startswith("autocapture:"):
            toggle = data.split(":", 1)[1]
            enabled = toggle == "on"
            if enabled:
                eval_driver = await get_config("memory.evaluator_driver", "ollama")
                if eval_driver == "ollama":
                    eval_model = await get_config("memory.evaluator_model", "qwen3:0.6b")
                    from ..memory.evaluator import check_model_available
                    available = await check_model_available(model=eval_model)
                    if not available:
                        await query.edit_message_text(
                            f"❌ Evaluator model `{eval_model}` not available.\n"
                            f"Run `ollama pull {eval_model}` first, or use /evaluator to configure.",
                            parse_mode="Markdown",
                        )
                        return
            await set_config("memory.auto_capture", enabled)
            buttons = [
                InlineKeyboardButton(f"{'✅ ' if enabled else ''}ON", callback_data="autocapture:on"),
                InlineKeyboardButton(f"{'✅ ' if not enabled else ''}OFF", callback_data="autocapture:off"),
            ]
            msg = "📝 **Auto-capture:** ON ⚠️" if enabled else "📝 **Auto-capture:** OFF"
            await query.edit_message_text(
                msg,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([buttons]),
            )

        elif data.startswith("approve:") or data.startswith("reject:"):
            # User approval/rejection
            parts = data.split(":", 1)
            action_type = parts[0]
            target_user_id = parts[1] if len(parts) > 1 else ""
            
            if not target_user_id:
                await query.edit_message_text("❌ Invalid user ID.")
                return
            
            target_user = await get_user("telegram", target_user_id)
            
            if not target_user:
                await query.edit_message_text(f"❌ User {target_user_id} not found.")
                return
            
            target_name = target_user.get("display_name") or target_user.get("name", "Unknown")
            
            if action_type == "approve":
                await update_user("telegram", target_user_id, access_level="public")
                await query.edit_message_text(
                    f"✅ **{target_name}** (`{target_user_id}`) approved.\n"
                    f"They can now chat with the bot.",
                    parse_mode="Markdown",
                )
                # Notify the user they've been approved
                for attempt in range(3):
                    try:
                        await self.app.bot.send_message(
                            chat_id=int(target_user_id),
                            text="✅ Your access has been approved! You can now send messages.",
                        )
                        break
                    except Exception as e:
                        logger.error(f"Failed to notify approved user {target_user_id} (attempt {attempt + 1}/3): {e}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                
                # Clear pending notification tracker
                pending_key = f"pending_notify:{target_user_id}"
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(pending_key)
            
            else:  # reject
                await update_user("telegram", target_user_id, access_level="blocked")
                await query.edit_message_text(
                    f"🚫 **{target_name}** (`{target_user_id}`) rejected.",
                    parse_mode="Markdown",
                )
                # Notify the user
                for attempt in range(3):
                    try:
                        await self.app.bot.send_message(
                            chat_id=int(target_user_id),
                            text="Sorry, your access request was not approved.",
                        )
                        break
                    except Exception as e:
                        logger.error(f"Failed to notify rejected user {target_user_id} (attempt {attempt + 1}/3): {e}")
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                
                # Clear pending notification tracker
                pending_key = f"pending_notify:{target_user_id}"
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(pending_key)

        elif data.startswith("mbr:"):
            # Interactive global members menu
            await self._handle_members_callback(query, data)

        elif data.startswith("groups:"):
            # Interactive group management menu
            await self._handle_group_callback(query, data)

        elif data.startswith("group_approve:") or data.startswith("group_reject:"):
            # Group approval/rejection
            parts = data.split(":", 1)
            action_type = parts[0]
            group_id = parts[1] if len(parts) > 1 else ""
            
            if not group_id:
                await query.edit_message_text("❌ Invalid group ID.")
                return
            
            if action_type == "group_approve":
                # Register the group in DB
                from ..db.connection import get_connection
                async with get_connection() as conn:
                    # Check if group already exists
                    existing = await conn.fetchrow(
                        "SELECT id FROM groups WHERE platform = 'telegram' AND platform_group_id = $1",
                        group_id,
                    )
                    if not existing:
                        # Try to get actual group title from Telegram
                        group_title = f"Group {group_id}"
                        try:
                            chat_info = await self.app.bot.get_chat(int(group_id))
                            if chat_info and chat_info.title:
                                group_title = chat_info.title
                        except Exception:
                            pass
                        await conn.execute(
                            """INSERT INTO groups (name, platform, platform_group_id, enabled, require_mention, settings)
                               VALUES ($1, 'telegram', $2, true, true, '{}')""",
                            group_title,
                            group_id,
                        )
                
                await query.edit_message_text(
                    f"✅ Group `{group_id}` approved.\n"
                    f"Bot will now respond when mentioned.",
                    parse_mode="Markdown",
                )
                
                # Clear pending tracker
                pending_key = f"pending_group:{group_id}"
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(pending_key)
            
            else:  # group_reject
                # Leave the group
                try:
                    await self.app.bot.leave_chat(int(group_id))
                except Exception as e:
                    logger.warning(f"Failed to leave rejected group {group_id}: {e}")
                
                await query.edit_message_text(
                    f"🚫 Group `{group_id}` rejected. Bot has left the group.",
                    parse_mode="Markdown",
                )
                
                # Clear pending tracker
                pending_key = f"pending_group:{group_id}"
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(pending_key)

        elif data.startswith("brw:"):
            # Browse directory picker callbacks
            import os
            parts = data.split(":", 2)  # brw:action:id
            action = parts[1] if len(parts) > 1 else ""
            path_hash = parts[2] if len(parts) > 2 else ""

            if action == "n":
                # Navigate to directory
                target = self._browse_paths.get(path_hash, "")
                if target and os.path.isdir(target):
                    await self._send_browse_picker(
                        query.message.chat_id, user.id, target, context,
                        message_id=query.message.message_id,
                    )

            elif action == "s":
                # Select directory as working directory
                target = self._browse_paths.get(path_hash, "")
                if target and os.path.isdir(target):
                    self._browse_cwd[user.id] = target
                    await query.edit_message_text(
                        f"📂 **Browse mode active**\n\n"
                        f"Working directory: `{target}`\n"
                        f"Session shared with CLI in this directory.\n\n"
                        f"Use /browse to change or go back to default.",
                        parse_mode="Markdown",
                    )

            elif action == "reset":
                # Back to default Telegram session
                was_active = user.id in self._browse_cwd
                self._browse_cwd.pop(user.id, None)
                if was_active:
                    await query.edit_message_text(
                        "💬 **Back to Telegram session**\n\n"
                        "Messages now use your regular Telegram session.",
                        parse_mode="Markdown",
                    )
                else:
                    await query.edit_message_text(
                        "👌 Cancelled.",
                        parse_mode="Markdown",
                    )

    def _get_display_name(self, user) -> str:
        """Get a display name for a Telegram user."""
        if user.first_name and user.last_name:
            return f"{user.first_name} {user.last_name}"
        return user.first_name or user.username or str(user.id)

    @staticmethod
    def _extract_reply_context_raw(update: Update) -> dict | None:
        """Extract reply context as raw dict for InboundContext.
        
        Returns:
            Dict with 'sender' and 'body' keys, or None.
        """
        if not update.message or not update.message.reply_to_message:
            return None
        
        reply = update.message.reply_to_message
        reply_text = reply.text or reply.caption or ""
        if not reply_text.strip():
            return None
        
        sender = "Unknown"
        if reply.from_user:
            if reply.from_user.is_bot:
                sender = reply.from_user.first_name or "Bot"
            else:
                sender = reply.from_user.first_name or reply.from_user.username or str(reply.from_user.id)
        
        # Truncate
        max_quote = 500
        if len(reply_text) > max_quote:
            reply_text = reply_text[:max_quote] + "…"
        
        return {"sender": sender, "body": reply_text}

    def _is_reply_to_bot(self, update: Update) -> bool:
        """Check if message is a reply to the bot."""
        if update.message and update.message.reply_to_message:
            reply = update.message.reply_to_message
            if reply.from_user and reply.from_user.is_bot:
                return True
        return False

    async def _send_response_with_media(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE = None, reply_to_message_id: int | None = None):
        """Send a response, handling MEDIA: paths as photos/documents.
        
        If response contains 'MEDIA: /path/to/file', send the file as a photo
        (if image) or document, with the remaining text as caption.
        
        Args:
            chat_id: Telegram chat ID
            text: Response text (may contain MEDIA: path)
            context: Telegram context (optional — uses self.app.bot if None)
            reply_to_message_id: Optional message ID to reply/quote to
        
        Returns:
            The sent Message object, or None if send failed.
        """
        import os

        # Get bot instance — from context if available, otherwise from app
        bot = context.bot if context else self.app.bot

        # Use communication sub-core for universal processing
        caption_text, media_path = extract_media(text)
        caption_text = process_outbound(caption_text)

        if media_path and os.path.isfile(media_path):
            try:
                from .formatting import markdown_to_telegram_html

                # Truncate caption for Telegram (max 1024 for photos/documents)
                if len(caption_text) > 1024:
                    caption_text = caption_text[:1020] + "..."

                # Convert caption to HTML (consistent with _send_response)
                caption_html = markdown_to_telegram_html(caption_text) if caption_text else None

                ext = os.path.splitext(media_path)[1].lower()
                sent_msg = None
                reply_params = {"message_id": reply_to_message_id} if reply_to_message_id else None
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    with open(media_path, "rb") as f:
                        try:
                            sent_msg = await bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption_html,
                                parse_mode="HTML" if caption_html else None,
                                reply_parameters=reply_params,
                            )
                        except Exception:
                            # HTML parse failed — retry without parse_mode
                            f.seek(0)
                            sent_msg = await bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption_text or None,
                                reply_parameters=reply_params,
                            )
                else:
                    with open(media_path, "rb") as f:
                        try:
                            sent_msg = await bot.send_document(
                                chat_id=chat_id,
                                document=f,
                                caption=caption_html,
                                parse_mode="HTML" if caption_html else None,
                                reply_parameters=reply_params,
                            )
                        except Exception:
                            f.seek(0)
                            sent_msg = await bot.send_document(
                                chat_id=chat_id,
                                document=f,
                                caption=caption_text or None,
                                reply_parameters=reply_params,
                            )
                logger.info(f"Sent media: {media_path} to {chat_id}")
                return sent_msg
            except Exception as e:
                logger.error(f"Failed to send media {media_path}: {e}")
                # Fall through to send as text
                caption_text = text  # Send full text as fallback

        # No media or media send failed — send as text
        return await self._send_response(chat_id, caption_text, context, reply_to_message_id=reply_to_message_id)

    async def _send_response(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE = None, reply_to_message_id: int | None = None):
        """Send a response, splitting if too long for Telegram.
        
        Converts LLM markdown to Telegram HTML for reliable rendering.
        Falls back to plain text if HTML parsing fails.
        
        Args:
            chat_id: Telegram chat ID
            text: Response text (markdown from LLM)
            context: Telegram context (optional — uses self.app.bot if None)
            reply_to_message_id: Optional message ID to reply/quote to
        
        Returns:
            The last sent Message object, or None if send failed.
        """
        from .formatting import markdown_to_telegram_html

        # Universal outbound processing (path strip + narration strip + cleanup)
        text = process_outbound(text)

        # Guard: empty text after processing — don't send empty message to Telegram
        if not text or not text.strip():
            logger.warning(f"Empty response after outbound processing for chat {chat_id}, skipping send")
            return None
        
        # Get bot instance — from context if available, otherwise from app
        bot = context.bot if context else self.app.bot
        
        # Convert markdown to Telegram HTML (platform-specific formatting)
        html_text = markdown_to_telegram_html(text)
        
        # Split using communication sub-core
        chunks = split_message(html_text, max_length=4096)

        # Only apply reply_to on the FIRST chunk
        reply_params = {"message_id": reply_to_message_id} if reply_to_message_id else None
        last_msg = None
        for i, chunk in enumerate(chunks):
            rp = reply_params if i == 0 else None
            try:
                last_msg = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="HTML",
                    reply_parameters=rp,
                )
            except Exception:
                # HTML parse failed — send as plain text
                try:
                    last_msg = await bot.send_message(
                        chat_id=chat_id,
                        text=text if i == 0 else chunk,  # Use original markdown for first chunk fallback
                        reply_parameters=rp,
                    )
                except Exception:
                    last_msg = await bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        reply_parameters=rp,
                    )

        return last_msg

    async def _deliver_subagent_result(self, message: str, parent_session_id: int):
        """Deliver sub-agent results to the chat that spawned it."""
        # Find chat_id from active conversations by parent_session_id
        # Same pattern as _send_status_message
        for key, conv in self.agent.conversations._active.items():
            if conv.session_id == parent_session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "telegram":
                    chat_id = int(parts[1])
                    try:
                        max_len = 4096
                        if len(message) <= max_len:
                            await self._bot.send_message(chat_id=chat_id, text=message)
                        else:
                            await self._bot.send_message(
                                chat_id=chat_id,
                                text=message[:max_len - 50] + "\n\n_(truncated — result too long)_",
                                parse_mode="Markdown",
                            )
                    except Exception as e:
                        logger.error(f"Failed to deliver sub-agent result to {chat_id}: {e}")
                return
        logger.warning(f"No active chat found for parent session {parent_session_id}")

    async def _send_status_message(self, session_id: int, message: str):
        """Send a status notification to the chat associated with a session."""
        # Find chat_id from active conversations by session_id
        for key, conv in self.agent.conversations._active.items():
            if conv.session_id == session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "telegram":
                    chat_id = int(parts[1])
                    try:
                        await self._bot.send_message(chat_id=chat_id, text=message)
                    except Exception as e:
                        logger.error(f"Failed to send status to {chat_id}: {e}")
                return
        logger.debug(f"No active chat found for session {session_id}")

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in update processing."""
        error = context.error
        # Log with full context for debugging
        if update:
            logger.error(f"Telegram error processing update {type(update).__name__}: {type(error).__name__}: {error}", exc_info=error)
        else:
            logger.error(f"Telegram error (no update): {type(error).__name__}: {error}", exc_info=error)
        # Check if polling is still alive after error
        if self.app and self.app.updater and self.app.updater.running:
            logger.debug("Polling still running after error")
        else:
            logger.critical("POLLING STOPPED after error — bot will not receive new messages!")
