"""Telegram channel adapter."""

import logging
import re
from typing import Optional, Tuple
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
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


class TelegramChannel:
    """Telegram bot adapter for Syne."""

    def __init__(self, agent: SyneAgent, bot_token: str):
        self.agent = agent
        self.bot_token = bot_token
        self.app: Optional[Application] = None

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
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("memory", self._cmd_memory))
        self.app.add_handler(CommandHandler("forget", self._cmd_forget))
        self.app.add_handler(CommandHandler("compact", self._cmd_compact))
        self.app.add_handler(CommandHandler("identity", self._cmd_identity))

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
            BotCommand("status", "Agent status"),
            BotCommand("memory", "Memory statistics"),
            BotCommand("compact", "Compact conversation history"),
            BotCommand("forget", "Clear current conversation"),
            BotCommand("identity", "Show agent identity"),
        ])

        # Wire sub-agent delivery: when a sub-agent completes, send result to the last active chat
        if self.agent.conversations:
            self._bot = self.app.bot
            self.agent.conversations.set_delivery_callback(self._deliver_subagent_result)

        logger.info("Telegram bot started.")

    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped.")

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

        # Send typing indicator
        await chat.send_action("typing")

        try:
            response = await self.agent.handle_message(
                platform="telegram",
                chat_id=str(chat.id),
                user_name=user.first_name or user.username or str(user.id),
                user_platform_id=str(user.id),
                message=text,
                display_name=self._get_display_name(user),
                is_group=is_group,  # Pass group flag for security restrictions
            )

            if response:
                await self._send_response_with_media(chat.id, response, context)

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
        await chat.send_action("typing")

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
/status — Show agent status
/memory — Show memory stats
/compact — Compact conversation history
/forget — Clear conversation history
/identity — Show agent identity

Or just send me a message!"""

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        from ..db.connection import get_connection

        async with get_connection() as conn:
            mem_count = await conn.fetchrow("SELECT COUNT(*) as c FROM memory")
            user_count = await conn.fetchrow("SELECT COUNT(*) as c FROM users")
            session_count = await conn.fetchrow(
                "SELECT COUNT(*) as c FROM sessions WHERE status = 'active'"
            )

        status = f"""🧠 **Syne Status**

Provider: {self.agent.provider.name}
Memories: {mem_count['c']}
Users: {user_count['c']}
Active sessions: {session_count['c']}
Tools: {len(self.agent.tools.list_tools('owner'))}"""

        await update.message.reply_text(status, parse_mode="Markdown")

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

    async def _cmd_identity(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /identity command."""
        from ..db.models import get_identity
        identity = await get_identity()

        lines = ["🪪 **Identity**\n"]
        for k, v in identity.items():
            lines.append(f"• **{k}**: {v}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    # ── Helpers ───────────────────────────────────────────────

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

    async def _send_response_with_media(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
        """Send a response, handling MEDIA: paths as photos/documents.
        
        If response contains 'MEDIA: /path/to/file', send the file as a photo
        (if image) or document, with the remaining text as caption.
        """
        import os

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
                if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                    with open(media_path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=caption_text or None,
                            parse_mode="Markdown" if caption_text else None,
                        )
                else:
                    with open(media_path, "rb") as f:
                        await context.bot.send_document(
                            chat_id=chat_id,
                            document=f,
                            caption=caption_text or None,
                        )
                logger.info(f"Sent media: {media_path} to {chat_id}")
                return
            except Exception as e:
                logger.error(f"Failed to send media {media_path}: {e}")
                # Fall through to send as text
                caption_text = text  # Send full text as fallback

        # No media or media send failed — send as text
        await self._send_response(chat_id, caption_text, context)

    async def _send_response(self, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
        """Send a response, splitting if too long for Telegram.
        Falls back to plain text if Markdown parsing fails."""
        max_length = 4096

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
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                )
            except Exception:
                # Markdown parse failed — send as plain text
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                )

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

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)
