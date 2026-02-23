"""Telegram channel adapter."""

import asyncio
import httpx
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
from ..ratelimit import check_rate_limit

logger = logging.getLogger("syne.telegram")


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

    async def start(self):
        """Start the Telegram bot."""
        self.app = (
            Application.builder()
            .token(self.bot_token)
            .build()
        )

        # Command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        self.app.add_handler(CommandHandler("version", self._cmd_version))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("memory", self._cmd_memory))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("compact", self._cmd_compact))
        self.app.add_handler(CommandHandler("think", self._cmd_think))
        self.app.add_handler(CommandHandler("reasoning", self._cmd_reasoning))
        self.app.add_handler(CommandHandler("autocapture", self._cmd_autocapture))
        self.app.add_handler(CommandHandler("identity", self._cmd_identity))
        self.app.add_handler(CommandHandler("restart", self._cmd_restart))
        self.app.add_handler(CommandHandler("model", self._cmd_model))
        self.app.add_handler(CommandHandler("embedding", self._cmd_embedding))
        self.app.add_handler(CommandHandler("browse", self._cmd_browse))
        # Update commands removed — use `syne update` / `syne updatedev` from CLI

        # Message handler — catch all text messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))

        # Photo handler
        self.app.add_handler(MessageHandler(
            filters.PHOTO & ~filters.LOCATION,
            self._handle_photo,
        ))

        # Voice message handler
        self.app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO,
            self._handle_voice,
        ))

        # Document/file handler
        self.app.add_handler(MessageHandler(
            filters.Document.ALL,
            self._handle_document,
        ))

        # Location handler
        self.app.add_handler(MessageHandler(
            filters.LOCATION | filters.Regex(r'^/location'),
            self._handle_location,
        ))

        # Reaction update handler
        self.app.add_handler(MessageReactionHandler(self._handle_reaction_update))

        # Chat member handler — detect when bot is added to/removed from groups
        self.app.add_handler(ChatMemberHandler(
            self._handle_my_chat_member,
            ChatMemberHandler.MY_CHAT_MEMBER,
        ))

        # Callback query handler (inline button clicks)
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))

        # Error handler
        self.app.add_error_handler(self._handle_error)

        logger.info("Starting Telegram bot...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        # Register bot commands menu (the "/" button in Telegram)
        from telegram import BotCommand
        await self.app.bot.set_my_commands([
            BotCommand("start", "Welcome message"),
            BotCommand("help", "Available commands"),
            BotCommand("version", "Version info"),
            BotCommand("status", "Agent status"),
            BotCommand("memory", "Memory statistics"),
            BotCommand("compact", "Compact conversation history"),
            BotCommand("think", "Set thinking level"),
            BotCommand("reasoning", "Toggle reasoning visibility (on/off)"),
            BotCommand("autocapture", "Toggle auto memory capture (on/off)"),
            BotCommand("clear", "Clear current conversation"),
            BotCommand("identity", "Show agent identity"),
            BotCommand("model", "Switch LLM model (owner only)"),
            BotCommand("embedding", "Switch embedding model (owner only)"),
            BotCommand("restart", "Restart Syne (owner only)"),
            BotCommand("browse", "Browse directories (share session with CLI)"),
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
            # Process the message via agent (DM context, not group)
            response = await self.agent.handle_message(
                platform="telegram",
                chat_id=str(chat_id),
                user_name=user_name,
                user_platform_id=str(chat_id),
                message=payload,
                display_name=user_name,
                is_group=False,
                message_metadata={"scheduled": True},
            )
            
            if response:
                # Parse reply tags (no incoming message for cron)
                response, reply_to = self._parse_reply_tag(response)
                await self._send_response_with_media(chat_id, response, None, reply_to_message_id=reply_to)
        
        except Exception as e:
            logger.error(f"Error processing scheduled message: {e}", exc_info=True)
            try:
                await self.app.bot.send_message(
                    chat_id,
                    f"⚠️ Scheduled task failed: {str(e)[:100]}"
                )
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
        # RATE LIMITING: Check before processing
        # Get user's access level for rate limit check (owner may be exempt)
        # ═══════════════════════════════════════════════════════════════
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        
        allowed, rate_msg = check_rate_limit(str(user.id), access_level)
        if not allowed:
            logger.info(f"Rate limited user {user.id}: {rate_msg}")
            await update.message.reply_text(f"⏱️ {rate_msg}")
            return

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
            db_user = await self._ensure_user(user)
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

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): {text[:100]}")

        # Track incoming message ID for reaction context
        message_id = update.message.message_id
        self._track_message(chat.id, message_id, text[:100])

        # Extract reply/quote context and prepend to message
        reply_context = self._extract_reply_context(update)
        if reply_context:
            text = f"{reply_context}\n\n{text}"

        # Keep typing indicator alive throughout the entire processing
        async with _TypingIndicator(context.bot, chat.id):
            try:
                # Include message_id in metadata for tool context
                metadata = {
                    "message_id": message_id,
                    "chat_id": str(chat.id),
                }

                # Browse mode: route to CLI-compatible session with cwd
                browse_cwd = self._browse_cwd.get(user.id) if not is_group else None
                if browse_cwd:
                    import getpass
                    username = getpass.getuser()
                    cli_chat_id = f"cli:{username}:{browse_cwd}"
                    metadata["cwd"] = browse_cwd

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
                                is_group=False,
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
                        is_group=is_group,
                        message_metadata=metadata,
                    )

                if response:
                    # Check if reasoning visibility is ON — prepend thinking if available
                    reasoning_visible = await get_config("session.reasoning_visible", False)
                    if reasoning_visible:
                        if browse_cwd:
                            import getpass as _gp
                            key = f"cli:cli:{_gp.getuser()}:{browse_cwd}"
                        else:
                            key = f"telegram:{chat.id}"
                        conv = self.agent.conversations._active.get(key)
                        thinking = getattr(conv, '_last_thinking', None) if conv else None
                        if thinking:
                            thinking_block = f"💭 **Thinking:**\n_{thinking[:3000]}_\n\n"
                            response = thinking_block + response

                    # Parse reply and react tags from LLM response
                    response, reply_to = self._parse_reply_tag(response, message_id)
                    response, react_emojis = self._parse_react_tags(response)

                    # Send reaction to incoming message if requested
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, message_id, emoji)

                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    # Track bot's response for reaction context
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:100])

            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                # Provide more useful error context
                error_msg = str(e)
                if "400" in error_msg:
                    await update.message.reply_text("⚠️ LLM request failed (bad request). This may be a conversation format issue — try /clear to start fresh.")
                elif "429" in error_msg:
                    await update.message.reply_text("⚠️ Rate limited. Please wait a moment and try again.")
                elif "401" in error_msg or "403" in error_msg:
                    await update.message.reply_text("⚠️ Authentication error. Owner may need to refresh credentials.")
                else:
                    await update.message.reply_text("Sorry, something went wrong. Please try again.")

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
        
        # Auto-discover/create user from group interaction
        await self._ensure_user(user)
        
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

    async def _ensure_user(self, tg_user) -> dict:
        """Ensure user exists in database, creating if needed."""
        return await get_or_create_user(
            name=tg_user.first_name or tg_user.username or str(tg_user.id),
            platform="telegram",
            platform_id=str(tg_user.id),
            display_name=self._get_display_name(tg_user),
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
        except Exception as e:
            logger.error(f"Failed to notify owner about pending user: {e}")

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
        if new_status in ("member", "administrator") and old_status in ("left", "kicked"):
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
            
            # Don't spam — track notifications
            pending_key = f"pending_group:{chat.id}"
            if not hasattr(self, '_pending_notified'):
                self._pending_notified = set()
            if pending_key in self._pending_notified:
                return
            self._pending_notified.add(pending_key)
            
            buttons = [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"group_approve:{chat.id}"),
                    InlineKeyboardButton("❌ Reject", callback_data=f"group_reject:{chat.id}"),
                ]
            ]
            
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
            except Exception as e:
                logger.error(f"Failed to notify owner about group addition: {e}")
        
        # Bot was removed from group
        elif new_status in ("left", "kicked") and old_status in ("member", "administrator"):
            logger.info(f"Bot removed from group: {chat.title} ({chat.id})")
            # Clean up pending tracker
            pending_key = f"pending_group:{chat.id}"
            if hasattr(self, '_pending_notified'):
                self._pending_notified.discard(pending_key)

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
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        allowed, rate_msg = check_rate_limit(str(user.id), access_level)
        if not allowed:
            await update.message.reply_text(f"⏱️ {rate_msg}")
            return

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

        db_user = await self._ensure_user(user)
        
        # Block pending users
        if not is_group and db_user.get("access_level") == "pending":
            await self._handle_pending_user(update, db_user)
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): [photo] {caption[:100]}")

        # Extract reply/quote context
        reply_context = self._extract_reply_context(update)
        if reply_context:
            caption = f"{reply_context}\n\n{caption}" if caption else reply_context

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
                metadata = {
                    "image": {
                        "mime_type": "image/jpeg",
                        "base64": photo_b64,
                    }
                }

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_text,
                    display_name=self._get_display_name(user),
                    is_group=is_group,
                    message_metadata=metadata,
                )

                if response:
                    response, reply_to = self._parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = self._parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)

            except Exception as e:
                logger.error(f"Error handling photo: {e}", exc_info=True)
                await update.message.reply_text("Sorry, something went wrong processing that photo.")

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages and audio files — transcribe via STT and process as text."""
        if not update.message:
            return

        user = update.effective_user
        chat = update.effective_chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        allowed, rate_msg = check_rate_limit(str(user.id), access_level)
        if not allowed:
            await update.message.reply_text(f"⏱️ {rate_msg}")
            return

        # Group checks — voice in groups requires mention/reply context
        if is_group:
            if not self._is_reply_to_bot(update):
                return  # Ignore voice in groups unless replying to bot

        db_user = await self._ensure_user(user)
        
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

                # Extract reply/quote context for voice replies
                reply_context = self._extract_reply_context(update)

                # Process the transcribed text as a normal message
                # Include transcription indicator so agent knows it's from voice
                user_message = f"[Voice message transcription]: {transcribed_text}"
                if reply_context:
                    user_message = f"{reply_context}\n\n{user_message}"
                
                metadata = {
                    "message_id": update.message.message_id,
                    "voice_transcription": True,
                    "original_text": transcribed_text,
                }

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_message,
                    display_name=self._get_display_name(user),
                    is_group=is_group,
                    message_metadata=metadata,
                )

                if response:
                    response, reply_to = self._parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = self._parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    # Track bot's response message ID
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:50])

            except Exception as e:
                logger.error(f"Error handling voice: {e}", exc_info=True)
                await update.message.reply_text("Sorry, something went wrong processing that voice message.")

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
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        allowed, rate_msg = check_rate_limit(str(user.id), access_level)
        if not allowed:
            await update.message.reply_text(f"⏱️ {rate_msg}")
            return

        # Group checks — documents in groups require mention/reply
        if is_group:
            if caption:
                result = await self._process_group_message(update, context, caption)
                if result is None:
                    return
                caption = result
            elif not self._is_reply_to_bot(update):
                return

        db_user = await self._ensure_user(user)

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
        reply_context = self._extract_reply_context(update)
        if reply_context:
            caption = f"{reply_context}\n\n{caption}" if caption else reply_context

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

                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user.first_name or str(user.id),
                    user_platform_id=str(user.id),
                    message=user_text,
                    display_name=self._get_display_name(user),
                    is_group=is_group,
                    message_metadata=metadata,
                )

                if response:
                    response, reply_to = self._parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = self._parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    sent = await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:50])

            except Exception as e:
                logger.error(f"Error handling document: {e}", exc_info=True)
                await update.message.reply_text("Sorry, something went wrong processing that file.")

    async def _handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle location messages — reverse geocode and pass address to LLM."""
        if not update.message or not update.message.location:
            return

        user = update.message.from_user
        chat = update.message.chat
        is_group = chat.type in ("group", "supergroup")

        # Rate limiting
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        allowed, rate_msg = check_rate_limit(str(user.id), access_level)
        if not allowed:
            await update.message.reply_text(f"⏱️ {rate_msg}")
            return

        # Group checks
        if is_group and not self._is_reply_to_bot(update):
            return  # Ignore location in groups without reply to bot

        db_user = await self._ensure_user(user)

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

        # Reply context
        reply_context = self._extract_reply_context(update)

        user_text = ""
        if reply_context:
            user_text += reply_context + "\n\n"
        if caption:
            user_text += caption + "\n\n"
        user_text += location_text

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): {location_text}")

        metadata = {
            "message_id": update.message.message_id,
            "location": {"latitude": lat, "longitude": lng},
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
                    is_group=is_group,
                    message_metadata=metadata,
                )

                if response:
                    response, reply_to = self._parse_reply_tag(response, update.message.message_id)
                    response, react_emojis = self._parse_react_tags(response)
                    for emoji in react_emojis:
                        await self.send_reaction(chat.id, update.message.message_id, emoji)
                    await self._send_response_with_media(chat.id, response, context, reply_to_message_id=reply_to)

            except Exception as e:
                logger.error(f"Error handling location: {e}", exc_info=True)

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
                metadata = {
                    "message_id": message_id,
                    "chat_id": str(chat.id),
                    "is_reaction": True,
                    "reaction_emojis": added_emojis,
                }
                response = await self.agent.handle_message(
                    platform="telegram",
                    chat_id=str(chat.id),
                    user_name=user_name,
                    user_platform_id=user_id,
                    message=event_text,
                    display_name=user_name,
                    is_group=(chat.type != "private"),
                    message_metadata=metadata,
                )
                if response and response.strip().upper() != "NO_REPLY":
                    response, reply_to = self._parse_reply_tag(response, message_id)
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
/think — Set thinking level (off/low/medium/high/max)
/reasoning — Toggle reasoning visibility (on/off)
/autocapture — Toggle auto memory capture (on/off)
/clear — Clear conversation history
/identity — Show agent identity
/browse — Browse directories (share session with CLI)

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
        thinking_budget = await get_config("session.thinking_budget", None)
        reasoning_visible = await get_config("session.reasoning_visible", False)

        # Context window and driver name from registry
        context_window = active_model_entry.get("context_window", 128000) if active_model_entry else 128000
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
            # Rough token estimate (chars / 3.5)
            est_tokens = int(chars / 3.5)
            max_tokens = context_window
            pct = round(est_tokens / max_tokens * 100)
            
            # Format context window for display
            if max_tokens >= 1000000:
                ctx_display = f"{max_tokens / 1000000:.1f}M"
            else:
                ctx_display = f"{max_tokens // 1000}K"
            
            session_info = (
                f"📋 Messages: {msg_count} | ~{est_tokens:,}/{max_tokens:,} tokens ({pct}%)\n"
                f"📐 Context window: {ctx_display}\n"
                f"🧹 Compactions: {compactions}"
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

        # Browse mode indicator
        browse_cwd = self._browse_cwd.get(update.effective_user.id)
        if browse_cwd:
            status_lines.append(f"📂 Browse: `{browse_cwd}`")

        if session_info:
            status_lines.append("")
            status_lines.append("**Current session:**")
            status_lines.append(session_info)

        await update.message.reply_text("\n".join(status_lines), parse_mode="Markdown")

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

    # Shared thinking level definitions
    THINK_LEVELS = {
        "off": 0,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "max": 24576,
    }

    async def _cmd_think(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /think command — show inline buttons or set directly."""
        user = update.effective_user

        # Only owner can change thinking
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can change thinking settings.")
            return

        # Parse argument
        args = update.message.text.split(maxsplit=1)
        level = args[1].strip().lower() if len(args) > 1 else None

        if level is None:
            # Show current + inline buttons
            saved = await get_config("session.thinking_budget", None)
            current = self._budget_to_level(saved)

            buttons = []
            for name in self.THINK_LEVELS:
                label = f"✅ {name}" if name == current else name
                buttons.append(InlineKeyboardButton(label, callback_data=f"think:{name}"))
            # 3 buttons per row
            keyboard = [buttons[:3], buttons[3:]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                f"💭 **Thinking:** {current}",
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )
            return

        if level not in self.THINK_LEVELS:
            await update.message.reply_text(
                f"❌ Unknown level: `{level}`\nUse: `off`, `low`, `medium`, `high`, `max`",
                parse_mode="Markdown",
            )
            return

        await self._apply_thinking(level)
        budget = self.THINK_LEVELS[level]
        emoji = "💭" if budget > 0 else "🔇"
        await update.message.reply_text(
            f"{emoji} Thinking set to **{level}**" + (f" ({budget} tokens)" if budget > 0 else ""),
            parse_mode="Markdown",
        )

    async def _cmd_reasoning(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reasoning command — toggle with inline buttons."""
        user = update.effective_user

        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can change reasoning settings.")
            return

        args = update.message.text.split(maxsplit=1)
        toggle = args[1].strip().lower() if len(args) > 1 else None

        if toggle is None:
            current = await get_config("session.reasoning_visible", False)
            state = "ON" if current else "OFF"
            buttons = [
                InlineKeyboardButton(f"{'✅ ' if current else ''}ON", callback_data="reasoning:on"),
                InlineKeyboardButton(f"{'✅ ' if not current else ''}OFF", callback_data="reasoning:off"),
            ]
            await update.message.reply_text(
                f"🔍 **Reasoning visibility:** {state}",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([buttons]),
            )
            return

        if toggle not in ("on", "off"):
            await update.message.reply_text(f"❌ Use: `on` or `off`", parse_mode="Markdown")
            return

        visible = toggle == "on"
        await set_config("session.reasoning_visible", visible)
        emoji = "🔍" if visible else "🔇"
        await update.message.reply_text(
            f"{emoji} Reasoning visibility set to **{toggle.upper()}**",
            parse_mode="Markdown",
        )

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

    async def _cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command — switch LLM model with inline buttons.
        
        Uses driver-based model registry from DB (provider.models).
        Tests the model before switching; rolls back on failure.
        """
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can switch models.")
            return

        # Get model registry from DB
        models = await get_config("provider.models", [])
        if not models:
            await update.message.reply_text("⚠️ No models configured. Check provider.models in config.")
            return
        
        # Get current active model
        active_model_key = await get_config("provider.active_model", "gemini-pro")

        # Build buttons from DB entries
        buttons = []
        for model in models:
            key = model.get("key", "")
            label = model.get("label", key)
            is_active = (key == active_model_key)
            btn_label = f"✅ {label}" if is_active else label
            buttons.append(InlineKeyboardButton(btn_label, callback_data=f"model:{key}"))

        # Arrange buttons in rows of 2
        keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
        
        # Get current model label and context window
        current_model = next((m for m in models if m.get("key") == active_model_key), None)
        current_label = current_model.get("label", active_model_key) if current_model else active_model_key
        ctx_window = current_model.get("context_window") if current_model else None
        
        ctx_info = ""
        if ctx_window:
            if ctx_window >= 1000000:
                ctx_info = f"\n📐 Context window: {ctx_window / 1000000:.1f}M tokens"
            else:
                ctx_info = f"\n📐 Context window: {ctx_window // 1000}K tokens"
        
        await update.message.reply_text(
            f"🤖 **Current model:** {current_label}{ctx_info}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    async def _cmd_embedding(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /embedding command — switch embedding model with inline buttons.
        
        Uses driver-based embedding registry from DB (provider.embedding_models).
        Tests the embedding before switching; rolls back on failure.
        """
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Only the owner can switch embedding models.")
            return

        # Get embedding registry from DB
        models = await get_config("provider.embedding_models", [])
        if not models:
            await update.message.reply_text("⚠️ No embedding models configured. Check provider.embedding_models in config.")
            return
        
        # Get current active embedding
        active_key = await get_config("provider.active_embedding", "together-bge")

        # Build buttons from DB entries
        buttons = []
        for model in models:
            key = model.get("key", "")
            label = model.get("label", key)
            cost = model.get("cost", "")
            is_active = (key == active_key)
            btn_label = f"✅ {label}" if is_active else label
            buttons.append(InlineKeyboardButton(btn_label, callback_data=f"embedding:{key}"))

        # Arrange buttons — one per row (labels are long)
        keyboard = [[btn] for btn in buttons]
        
        # Get current model label
        current_model = next((m for m in models if m.get("key") == active_key), None)
        current_label = current_model.get("label", active_key) if current_model else active_key
        current_cost = current_model.get("cost", "") if current_model else ""
        
        await update.message.reply_text(
            f"🧠 **Current embedding:** {current_label}\n💰 Cost: {current_cost}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
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
            # Create provider and test it
            provider = await create_hybrid_provider(model_entry)
            success, error = await test_model(provider, timeout=10)
            
            if success:
                # Test passed — save new model
                await set_config("provider.active_model", model_key)
                
                # Auto-adjust compaction threshold based on context window
                # Default: ~75% of context window in chars (tokens * 3.5)
                ctx_window = model_entry.get("context_window")
                if ctx_window:
                    new_threshold = int(ctx_window * 0.75 * 3.5)
                    await set_config("session.compaction_threshold", new_threshold)
                    logger.info(f"Auto-adjusted compaction threshold to {new_threshold} chars for {ctx_window} token context")
                
                # Hot-reload provider in the running agent
                await self.agent.reload_provider()
                
                return True, model_entry.get("label", model_key)
            else:
                # Test failed — report error
                return False, f"{model_entry.get('label', model_key)} failed: {error}"
                
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

    @staticmethod
    def _format_thinking_level(budget) -> str:
        """Format thinking budget as human-readable level."""
        if budget is None:
            return "default"
        budget = int(budget)
        levels = {0: "off", 1024: "low", 4096: "medium", 8192: "high", 24576: "max"}
        return levels.get(budget, f"{budget} tokens")

    def _budget_to_level(self, budget) -> str:
        """Convert budget value to level name."""
        if budget is None:
            return "default"
        budget = int(budget)
        for name, val in self.THINK_LEVELS.items():
            if budget == val:
                return name
        return f"{budget} tokens"

    async def _apply_thinking(self, level: str):
        """Apply thinking level to DB and all active conversations."""
        budget = self.THINK_LEVELS[level]
        await set_config("session.thinking_budget", budget)
        for k, c in self.agent.conversations._active.items():
            c.thinking_budget = budget

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

        if data.startswith("think:"):
            level = data.split(":", 1)[1]
            if level in self.THINK_LEVELS:
                await self._apply_thinking(level)
                budget = self.THINK_LEVELS[level]
                # Rebuild buttons with new checkmark
                buttons = []
                for name in self.THINK_LEVELS:
                    label = f"✅ {name}" if name == level else name
                    buttons.append(InlineKeyboardButton(label, callback_data=f"think:{name}"))
                keyboard = [buttons[:3], buttons[3:]]
                emoji = "💭" if budget > 0 else "🔇"
                await query.edit_message_text(
                    f"{emoji} **Thinking:** {level}" + (f" ({budget} tokens)" if budget > 0 else ""),
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )

        elif data.startswith("reasoning:"):
            toggle = data.split(":", 1)[1]
            visible = toggle == "on"
            await set_config("session.reasoning_visible", visible)
            buttons = [
                InlineKeyboardButton(f"{'✅ ' if visible else ''}ON", callback_data="reasoning:on"),
                InlineKeyboardButton(f"{'✅ ' if not visible else ''}OFF", callback_data="reasoning:off"),
            ]
            emoji = "🔍" if visible else "🔇"
            await query.edit_message_text(
                f"{emoji} **Reasoning visibility:** {'ON' if visible else 'OFF'}",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([buttons]),
            )

        elif data.startswith("model:"):
            model_key = data.split(":", 1)[1]
            
            # Get model registry
            models = await get_config("provider.models", [])
            model_entry = next((m for m in models if m.get("key") == model_key), None)
            
            if model_entry:
                # Test and apply model
                chat_id = query.message.chat_id
                success, result = await self._apply_model(model_key, chat_id, context)
                
                # Get updated active model
                active_model_key = model_key if success else await get_config("provider.active_model", "gemini-pro")
                
                # Rebuild buttons
                buttons = []
                for m in models:
                    key = m.get("key", "")
                    label = m.get("label", key)
                    is_active = (key == active_model_key)
                    btn_label = f"✅ {label}" if is_active else label
                    buttons.append(InlineKeyboardButton(btn_label, callback_data=f"model:{key}"))
                keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
                
                if success:
                    await query.edit_message_text(
                        f"✅ **Switched to:** {result}\n\n⚠️ Use /restart to fully apply.",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )
                else:
                    # Rollback and show error
                    previous = await get_config("provider.previous_model", "gemini-pro")
                    previous_entry = next((m for m in models if m.get("key") == previous), None)
                    previous_label = previous_entry.get("label", previous) if previous_entry else previous
                    await query.edit_message_text(
                        f"❌ {result}\n\nRolled back to {previous_label}.",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )

        elif data.startswith("embedding:"):
            embed_key = data.split(":", 1)[1]
            
            # Get embedding registry
            models = await get_config("provider.embedding_models", [])
            embed_entry = next((m for m in models if m.get("key") == embed_key), None)
            
            if embed_entry:
                chat_id = query.message.chat_id
                success, result = await self._apply_embedding(embed_key, chat_id, context)
                
                # Get updated active embedding
                active_key = embed_key if success else await get_config("provider.active_embedding", "together-bge")
                
                # Rebuild buttons
                buttons = []
                for m in models:
                    key = m.get("key", "")
                    label = m.get("label", key)
                    is_active = (key == active_key)
                    btn_label = f"✅ {label}" if is_active else label
                    buttons.append(InlineKeyboardButton(btn_label, callback_data=f"embedding:{key}"))
                keyboard = [[btn] for btn in buttons]
                
                if success:
                    current_cost = embed_entry.get("cost", "")
                    await query.edit_message_text(
                        f"✅ **Switched embedding to:** {result}\n💰 Cost: {current_cost}\n\n⚠️ Use /restart to fully apply.",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )
                else:
                    previous = await get_config("provider.previous_embedding", "together-bge")
                    previous_entry = next((m for m in models if m.get("key") == previous), None)
                    previous_label = previous_entry.get("label", previous) if previous_entry else previous
                    await query.edit_message_text(
                        f"❌ {result}\n\nRolled back to {previous_label}.",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                    )

        elif data.startswith("autocapture:"):
            toggle = data.split(":", 1)[1]
            enabled = toggle == "on"
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
                try:
                    await self.app.bot.send_message(
                        chat_id=int(target_user_id),
                        text="✅ Your access has been approved! You can now send messages.",
                    )
                except Exception:
                    pass  # User may have blocked the bot
                
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
                try:
                    await self.app.bot.send_message(
                        chat_id=int(target_user_id),
                        text="Sorry, your access request was not approved.",
                    )
                except Exception:
                    pass
                
                # Clear pending notification tracker
                pending_key = f"pending_notify:{target_user_id}"
                if hasattr(self, '_pending_notified'):
                    self._pending_notified.discard(pending_key)

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
                        await conn.execute(
                            """INSERT INTO groups (name, platform, platform_group_id, enabled, require_mention, settings)
                               VALUES ($1, 'telegram', $2, true, true, '{}')""",
                            f"Group {group_id}",
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
    def _extract_reply_context(update: Update) -> str | None:
        """Extract quoted/replied message content for LLM context.
        
        When a user replies to a message, include the original message text
        so the LLM understands what's being referenced.
        
        Returns:
            Context string to prepend to message, or None if no reply.
        """
        if not update.message or not update.message.reply_to_message:
            return None
        
        reply = update.message.reply_to_message
        if not reply.text and not reply.caption:
            return None  # Skip media-only replies without text
        
        reply_text = reply.text or reply.caption or ""
        if not reply_text.strip():
            return None
        
        # Identify who sent the original message
        sender = "Unknown"
        if reply.from_user:
            if reply.from_user.is_bot:
                sender = reply.from_user.first_name or "Bot"
            else:
                sender = reply.from_user.first_name or reply.from_user.username or str(reply.from_user.id)
        
        # Truncate very long quoted messages
        max_quote = 500
        if len(reply_text) > max_quote:
            reply_text = reply_text[:max_quote] + "…"
        
        return f"[Replying to {sender}: \"{reply_text}\"]"

    def _is_reply_to_bot(self, update: Update) -> bool:
        """Check if message is a reply to the bot."""
        if update.message and update.message.reply_to_message:
            reply = update.message.reply_to_message
            if reply.from_user and reply.from_user.is_bot:
                return True
        return False

    @staticmethod
    def _parse_reply_tag(text: str, incoming_message_id: int | None = None) -> tuple[str, int | None]:
        """Parse [[reply_to_current]] or [[reply_to:<id>]] tags from response text.

        Returns:
            Tuple of (cleaned_text, reply_to_message_id or None)
        """
        import re
        # Match [[reply_to_current]] or [[ reply_to_current ]]
        if re.search(r'\[\[\s*reply_to_current\s*\]\]', text):
            text = re.sub(r'\[\[\s*reply_to_current\s*\]\]', '', text).strip()
            return text, incoming_message_id
        # Match [[reply_to:<id>]] or [[ reply_to: <id> ]]
        m = re.search(r'\[\[\s*reply_to:\s*(\d+)\s*\]\]', text)
        if m:
            text = re.sub(r'\[\[\s*reply_to:\s*\d+\s*\]\]', '', text).strip()
            return text, int(m.group(1))
        return text, None

    @staticmethod
    def _parse_react_tags(text: str) -> tuple[str, list[str]]:
        """Parse [[react:<emoji>]] tags from response text.

        Returns:
            Tuple of (cleaned_text, list_of_emojis_to_react_with)
        """
        import re
        emojis = []
        for m in re.finditer(r'\[\[\s*react:\s*(.+?)\s*\]\]', text):
            emojis.append(m.group(1).strip())
        if emojis:
            text = re.sub(r'\[\[\s*react:\s*.+?\s*\]\]', '', text).strip()
        return text, emojis

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
        from telegram import Message

        # Get bot instance — from context if available, otherwise from app
        bot = context.bot if context else self.app.bot

        # Check for MEDIA: in response
        media_path = None
        caption_text = text

        if "\n\nMEDIA: " in text:
            parts = text.rsplit("\n\nMEDIA: ", 1)
            caption_text = parts[0].strip()
            media_path = parts[1].strip()
        elif text.startswith("MEDIA: "):
            media_path = text[7:].strip()
            caption_text = ""

        # Strip server paths from outgoing messages — NEVER expose internal file locations
        # Catches: "File: /home/syne/...", bare "/home/...", "Saved to: /tmp/..."
        # (?<!\S) ensures we don't match URL paths like https://example.com/home/...
        import re
        _PATH_STRIP_RE = re.compile(
            r'(?:(?:File|Path|Saved to|Output|Lokasi|Location)\s*:?\s*\n?\s*)?(?<!\S)/(?:home|tmp|var|opt|usr)/\S+',
            re.IGNORECASE,
        )
        caption_text = _PATH_STRIP_RE.sub('', caption_text).strip()
        caption_text = re.sub(r'\n{3,}', '\n\n', caption_text).strip()

        if media_path and os.path.isfile(media_path):
            try:
                from ..formatting import markdown_to_telegram_html

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
        import re
        from ..formatting import markdown_to_telegram_html
        
        # Strip server paths from response — NEVER expose internal file locations
        _PATH_STRIP_RE = re.compile(
            r'(?:(?:File|Path|Saved to|Output|Lokasi|Location)\s*:?\s*\n?\s*)?(?<!\S)/(?:home|tmp|var|opt|usr)/\S+',
            re.IGNORECASE,
        )
        text = _PATH_STRIP_RE.sub('', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        # Get bot instance — from context if available, otherwise from app
        bot = context.bot if context else self.app.bot
        
        # Convert markdown to Telegram HTML
        html_text = markdown_to_telegram_html(text)
        
        max_length = 4096
        last_msg = None

        chunks = []
        remaining = html_text
        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            split_at = remaining.rfind("\n", 0, max_length)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, max_length)
            if split_at == -1:
                split_at = max_length

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()

        # Only apply reply_to on the FIRST chunk
        reply_params = {"message_id": reply_to_message_id} if reply_to_message_id else None
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

    async def _deliver_subagent_result(self, message: str):
        """Deliver sub-agent results to the last active Telegram chat."""
        # Find the most recently active conversation
        if not self.agent.conversations._active:
            logger.warning("No active conversation to deliver sub-agent result")
            return

        # Get the last active chat_id
        last_key = max(
            self.agent.conversations._active.keys(),
            key=lambda k: getattr(
                self.agent.conversations._active[k], '_last_activity', 0
            ) if hasattr(self.agent.conversations._active[k], '_last_activity') else 0,
        )

        # Extract chat_id from key (format: "telegram:chat_id")
        parts = last_key.split(":", 1)
        if len(parts) == 2 and parts[0] == "telegram":
            chat_id = int(parts[1])
            try:
                # Split if too long
                max_len = 4096
                if len(message) <= max_len:
                    await self._bot.send_message(chat_id=chat_id, text=message)
                else:
                    # Send first chunk with note
                    await self._bot.send_message(
                        chat_id=chat_id,
                        text=message[:max_len - 50] + "\n\n_(truncated — result too long)_",
                        parse_mode="Markdown",
                    )
            except Exception as e:
                logger.error(f"Failed to deliver sub-agent result to {chat_id}: {e}")

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
        """Handle errors."""
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)
