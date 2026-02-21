"""Telegram channel adapter."""

import asyncio
import logging
import re
from collections import deque
from typing import Optional, Tuple
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ReactionTypeEmoji
from telegram.ext import (
    Application,
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
)
from ..ratelimit import check_rate_limit

logger = logging.getLogger("syne.telegram")


class _TypingIndicator:
    """Keeps sending 'typing' action every 4s until cancelled.

    Usage:
        async with _TypingIndicator(bot, chat_id):
            await long_running_work()
    """

    def __init__(self, bot: Bot, chat_id: int, interval: float = 4.0):
        self._bot = bot
        self._chat_id = chat_id
        self._interval = interval
        self._task: Optional[asyncio.Task] = None

    async def _loop(self):
        try:
            while True:
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
        self.app.add_handler(CommandHandler("forget", self._cmd_forget))
        self.app.add_handler(CommandHandler("compact", self._cmd_compact))
        self.app.add_handler(CommandHandler("think", self._cmd_think))
        self.app.add_handler(CommandHandler("reasoning", self._cmd_reasoning))
        self.app.add_handler(CommandHandler("autocapture", self._cmd_autocapture))
        self.app.add_handler(CommandHandler("identity", self._cmd_identity))
        self.app.add_handler(CommandHandler("restart", self._cmd_restart))
        self.app.add_handler(CommandHandler("model", self._cmd_model))
        self.app.add_handler(CommandHandler("embedding", self._cmd_embedding))
        self.app.add_handler(CommandHandler("browse", self._cmd_browse))
        self.app.add_handler(CommandHandler("update", self._cmd_update))
        self.app.add_handler(CommandHandler("updatedev", self._cmd_updatedev))

        # Message handler — catch all text messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))

        # Photo handler
        self.app.add_handler(MessageHandler(
            filters.PHOTO,
            self._handle_photo,
        ))

        # Voice message handler
        self.app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO,
            self._handle_voice,
        ))

        # Reaction update handler
        self.app.add_handler(MessageReactionHandler(self._handle_reaction_update))

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
            BotCommand("forget", "Clear current conversation"),
            BotCommand("identity", "Show agent identity"),
            BotCommand("model", "Switch LLM model (owner only)"),
            BotCommand("embedding", "Switch embedding model (owner only)"),
            BotCommand("restart", "Restart Syne (owner only)"),
            BotCommand("browse", "Browse directories (share session with CLI)"),
            BotCommand("update", "Check and install updates"),
        ])

        # Wire sub-agent delivery: when a sub-agent completes, send result to the last active chat
        if self.agent.conversations:
            self._bot = self.app.bot
            self.agent.conversations.set_delivery_callback(self._deliver_subagent_result)
            self.agent.conversations.set_status_callback(self._send_status_message)

        # Wire up reactions tool with this channel reference
        from ..tools.reactions import set_telegram_channel
        set_telegram_channel(self)

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
                # Send response back to user
                await self._send_response_with_media(chat_id, response, None)
        
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
                return  # Message filtered out
            text = result

        # Handle DMs - auto-create user
        else:
            await self._ensure_user(user)

        if not text:
            return

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): {text[:100]}")

        # Track incoming message ID for reaction context
        message_id = update.message.message_id
        self._track_message(chat.id, message_id, text[:100])

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
                        response = await self.agent.conversations.handle_message(
                            platform="cli",
                            chat_id=cli_chat_id,
                            user=tg_user,
                            message=text,
                            is_group=False,
                            message_metadata=metadata,
                        )
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

                    sent = await self._send_response_with_media(chat.id, response, context)
                    # Track bot's response for reaction context
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:100])

            except Exception as e:
                logger.error(f"Error handling message: {e}", exc_info=True)
                # Provide more useful error context
                error_msg = str(e)
                if "400" in error_msg:
                    await update.message.reply_text("⚠️ LLM request failed (bad request). This may be a conversation format issue — try /forget to start fresh.")
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

        await self._ensure_user(user)

        logger.info(f"[{chat.type}] {user.first_name} ({user.id}): [photo] {caption[:100]}")

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
                    await self._send_response_with_media(chat.id, response, context)

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

        await self._ensure_user(user)

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

                # Process the transcribed text as a normal message
                # Include transcription indicator so agent knows it's from voice
                user_message = f"[Voice message transcription]: {transcribed_text}"
                
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
                    sent = await self._send_response_with_media(chat.id, response, context)
                    # Track bot's response message ID
                    if sent:
                        self._track_message(chat.id, sent.message_id, response[:50])

            except Exception as e:
                logger.error(f"Error handling voice: {e}", exc_info=True)
                await update.message.reply_text("Sorry, something went wrong processing that voice message.")

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
            event_text = f"[Reaction: {emoji_str} from {user_name} on message '{msg_preview}']"
            logger.info(f"Reaction received: {emoji_str} from {user_id} on msg {message_id}")

            # Only process reactions on bot's own messages or if configured to
            # For now, just log — could pass to agent as system event
            # await self.agent.handle_system_event(event_text, chat_id=str(chat.id))

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
/forget — Clear conversation history
/identity — Show agent identity
/browse — Browse directories (share session with CLI)
/update — Check and install updates

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
        """Handle /memory command — show memory stats."""
        from ..db.connection import get_connection

        async with get_connection() as conn:
            total = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
            by_cat = await conn.fetch("""
                SELECT category, COUNT(*) as c
                FROM memory GROUP BY category ORDER BY c DESC
            """)

        lines = [f"🧠 **Memory: {total['c']} items**\n"]
        for row in by_cat:
            lines.append(f"• {row['category']}: {row['c']}")

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
                keep_recent=20,
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

    async def _cmd_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forget command — archive current session."""
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

        import os
        import signal
        # Send SIGTERM to self — systemd will auto-restart (Restart=always)
        os.kill(os.getpid(), signal.SIGTERM)

    def _path_id(self, path: str) -> str:
        """Generate a short ID for a path (for callback_data 64-byte limit)."""
        import hashlib
        short = hashlib.md5(path.encode()).hexdigest()[:8]
        self._browse_paths[short] = path
        return short

    # ── Update commands ────────────────────────────────────────

    async def _cmd_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /update — check and install updates (version-based)."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Owner only.")
            return

        msg = await update.message.reply_text("🔍 Checking for updates...")
        await self._run_update(msg, context, dev=False)

    async def _cmd_updatedev(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /updatedev — force pull and reinstall (hidden command)."""
        user = update.effective_user
        existing_user = await get_user("telegram", str(user.id))
        access_level = existing_user.get("access_level", "public") if existing_user else "public"
        if access_level != "owner":
            await update.message.reply_text("⚠️ Owner only.")
            return

        msg = await update.message.reply_text("📥 Pulling latest code (dev)...")
        await self._run_update(msg, context, dev=True)

    async def _run_update(self, msg, context, dev: bool = False):
        """Run update directly (git + pip) then self-restart."""
        import asyncio
        import subprocess
        import os

        syne_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        loop = asyncio.get_event_loop()

        # Ensure HOME is set for git credentials
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "HOME": os.path.expanduser("~")}

        def _do():
            """Blocking update logic."""
            from . import __version__ as current_version

            # Git fetch
            try:
                subprocess.run(
                    ["git", "fetch", "origin"], cwd=syne_dir,
                    capture_output=True, text=True, timeout=30, env=env,
                )
            except subprocess.TimeoutExpired:
                return "error", "❌ Git fetch timed out."

            if not dev:
                # Check remote version
                result = subprocess.run(
                    ["git", "show", "origin/main:syne/__init__.py"],
                    cwd=syne_dir, capture_output=True, text=True, env=env,
                )
                remote_version = current_version
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if line.startswith("__version__"):
                            remote_version = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if remote_version == current_version:
                    return "up_to_date", f"✅ Already up to date (v{current_version})"

            # Git pull
            try:
                result = subprocess.run(
                    ["git", "pull"], cwd=syne_dir,
                    capture_output=True, text=True, timeout=60, env=env,
                )
            except subprocess.TimeoutExpired:
                return "error", "❌ Git pull timed out."
            if result.returncode != 0:
                return "error", f"❌ Git pull failed: {result.stderr.strip()}"

            # Pip install
            venv_pip = os.path.join(syne_dir, ".venv", "bin", "pip")
            try:
                result = subprocess.run(
                    [venv_pip, "install", "-e", ".", "-q"],
                    cwd=syne_dir, capture_output=True, text=True, timeout=120, env=env,
                )
            except subprocess.TimeoutExpired:
                return "error", "❌ pip install timed out."
            if result.returncode != 0:
                return "error", f"❌ Install failed: {result.stderr.strip()}"

            label = "dev" if dev else "latest"
            return "restart", f"✅ Updated ({label}). Restarting..."

        status, text = await loop.run_in_executor(None, _do)
        await msg.edit_text(text)

        if status == "restart":
            # Self-restart via SIGTERM — systemd auto-restarts
            import signal
            await asyncio.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

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

    def _is_reply_to_bot(self, update: Update) -> bool:
        """Check if message is a reply to the bot."""
        if update.message and update.message.reply_to_message:
            reply = update.message.reply_to_message
            if reply.from_user and reply.from_user.is_bot:
                return True
        return False

    async def _send_response_with_media(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE = None):
        """Send a response, handling MEDIA: paths as photos/documents.
        
        If response contains 'MEDIA: /path/to/file', send the file as a photo
        (if image) or document, with the remaining text as caption.
        
        Args:
            chat_id: Telegram chat ID
            text: Response text (may contain MEDIA: path)
            context: Telegram context (optional — uses self.app.bot if None)
        
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

        if media_path and os.path.isfile(media_path):
            try:
                # Truncate caption for Telegram (max 1024 for photos)
                if len(caption_text) > 1024:
                    caption_text = caption_text[:1020] + "..."

                ext = os.path.splitext(media_path)[1].lower()
                sent_msg = None
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    with open(media_path, "rb") as f:
                        try:
                            sent_msg = await bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption_text or None,
                                parse_mode="Markdown" if caption_text else None,
                            )
                        except Exception:
                            # Markdown parse failed — retry without parse_mode
                            f.seek(0)
                            sent_msg = await bot.send_photo(
                                chat_id=chat_id,
                                photo=f,
                                caption=caption_text or None,
                            )
                else:
                    with open(media_path, "rb") as f:
                        try:
                            sent_msg = await bot.send_document(
                                chat_id=chat_id,
                                document=f,
                                caption=caption_text or None,
                                parse_mode="Markdown" if caption_text else None,
                            )
                        except Exception:
                            f.seek(0)
                            sent_msg = await bot.send_document(
                                chat_id=chat_id,
                                document=f,
                                caption=caption_text or None,
                            )
                logger.info(f"Sent media: {media_path} to {chat_id}")
                return sent_msg
            except Exception as e:
                logger.error(f"Failed to send media {media_path}: {e}")
                # Fall through to send as text
                caption_text = text  # Send full text as fallback

        # No media or media send failed — send as text
        return await self._send_response(chat_id, caption_text, context)

    async def _send_response(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE = None):
        """Send a response, splitting if too long for Telegram.
        Falls back to plain text if Markdown parsing fails.
        
        Args:
            chat_id: Telegram chat ID
            text: Response text
            context: Telegram context (optional — uses self.app.bot if None)
        
        Returns:
            The last sent Message object, or None if send failed.
        """
        # Get bot instance — from context if available, otherwise from app
        bot = context.bot if context else self.app.bot
        
        max_length = 4096
        last_msg = None

        chunks = []
        remaining = text
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

        for chunk in chunks:
            try:
                last_msg = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                )
            except Exception:
                # Markdown parse failed — send as plain text
                last_msg = await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
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
