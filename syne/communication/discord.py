"""Discord channel adapter.

Discord is a first-class channel alongside Telegram, not an ability: it owns a
persistent gateway websocket, its own inbound event loop and its own approval
state, none of which fit the request/response shape an ability has.

NON-PRIVILEGED BY DESIGN
------------------------
Discord gates full message text behind the privileged MESSAGE_CONTENT intent,
which requires app review once a bot passes 100 servers. This channel runs
WITHOUT it. Verified against discord.py 2.7.1: `Intents.default()` already has
message_content = False, and the Discord docs state an app "will receive empty
values in the content, embeds, attachments and components fields" unless that
intent is granted.

The practical consequence - and it is a feature, not a limitation - is that in a
server Discord only delivers text when the bot is **mentioned, replied to, or
DMed**. That is exactly the interaction model Syne's Telegram groups already
use (require_mention), so nothing is lost; it also means the bot is structurally
incapable of reading server chatter it was not addressed in.

WHAT LIVES HERE vs ELSEWHERE
----------------------------
Shared behaviour is reused, never reimplemented:
  * outbound.process_outbound / extract_all_media / split_message  - text pipeline
  * formatting.markdown_to_discord                                  - Discord markdown
  * inbound.InboundContext / build_user_context_prefix              - per-message context
  * registry.register_channel                                       - cross-channel reach
Only Discord-specific mapping is written here.

Owner approval for unknown DMs and new guilds is deliberately NOT in this file
yet - that gate is delivered over Telegram and is the next phase of work. Until
then an unapproved sender is logged and ignored (fail-closed), never admitted.
"""

import asyncio
import logging
from typing import Any, Optional

import discord

from ..agent import SyneAgent
from .tags import parse_reply_tag, parse_react_tags
from .outbound import (
    extract_all_media,
    process_outbound,
    split_message,
)
from .formatting import escape_markdown, markdown_to_discord
from .registry import (
    CHANNEL_DISCORD,
    CHANNEL_TELEGRAM,
    get_channel,
    register_channel,
    unregister_channel,
)
from ..db.models import (
    create_user,
    get_config,
    get_group,
    get_user,
    update_user,
)

logger = logging.getLogger("syne.discord")


# Discord hard-caps a message's `content` field at 2000 characters (Message
# Resource: "Message contents (up to 2000 characters)"). Telegram's limit is
# 4096, so the shared split_message default is wrong here and must be passed
# explicitly on every send.
DISCORD_MAX_MESSAGE = 2000


class DiscordChannel:
    """Discord bot adapter for Syne."""

    def __init__(self, agent: SyneAgent, bot_token: str):
        self.agent = agent
        self.bot_token = bot_token
        self.client: Optional[discord.Client] = None
        # Gateway task — discord.Client.start() blocks until disconnect, so it
        # is owned as a task rather than awaited inline.
        self._task: Optional[asyncio.Task] = None
        # Deduplicate inbound messages. Discord can redeliver on resume, and a
        # double-processed message means a double LLM call and a double reply.
        self._seen: set[int] = set()
        self._seen_order: list[int] = []
        # Approval prompts already delivered to the owner, keyed "dm:<id>" /
        # "guild:<id>", so a stranger hammering the bot produces exactly one
        # Telegram notification instead of one per message. In-memory by
        # design: after a restart the owner gets one fresh prompt on the next
        # message, which is better than a request silently lost forever.
        self._pending_notified: set[str] = set()

    # ── Lifecycle ────────────────────────────────────────────

    def _build_intents(self) -> discord.Intents:
        """Build the gateway intents.

        Starts from Intents.default() (guilds, guild_messages, dm_messages,
        reactions - all non-privileged) and leaves the three privileged flags
        OFF explicitly rather than by omission, so the choice is visible to the
        next reader and a future edit has to argue with a comment.
        """
        intents = discord.Intents.default()
        intents.message_content = False   # privileged: needs Discord app review
        intents.members = False           # privileged: not needed
        intents.presences = False         # privileged: not needed
        return intents

    async def start(self) -> None:
        """Connect to the Discord gateway.

        Returns once the gateway task is spawned; it does not wait for READY,
        so a slow or failing Discord connection cannot hold up boot of the
        other channels.
        """
        # allowed_mentions is NOT a declared Client param; it travels through
        # **options into ConnectionState (verified: _connection.allowed_mentions
        # is the exact object passed). Left unset, discord defaults to parsing
        # every mention in the payload -- so any @everyone or @role echoed from
        # an LLM reply would really ping a whole server. replied_user stays True
        # so a direct reply still notifies the person being answered.
        self.client = discord.Client(
            intents=self._build_intents(),
            allowed_mentions=discord.AllowedMentions(
                everyone=False, roles=False, users=True, replied_user=True
            ),
        )
        self._register_handlers()

        logger.info("Starting Discord gateway...")
        self._task = asyncio.create_task(self._run_gateway(), name="discord-gateway")

    async def _run_gateway(self) -> None:
        """Run client.start() and classify the ways it can fail."""
        try:
            await self.client.start(self.bot_token)
        except asyncio.CancelledError:
            raise
        except discord.LoginFailure:
            # Bad or revoked token. Not retryable — retrying hammers Discord
            # and can get the IP temporarily banned.
            logger.error(
                "Discord login failed: token rejected. Fix the stored token "
                "(credential.discord_bot_token) and restart."
            )
        except discord.PrivilegedIntentsRequired:
            # Should be unreachable: this channel requests no privileged
            # intents. If it fires, someone enabled one without portal access.
            logger.error(
                "Discord refused the requested privileged intents. This channel "
                "is designed to run without them — check _build_intents()."
            )
        except Exception as e:
            logger.error(f"Discord gateway stopped: {type(e).__name__}: {e}", exc_info=True)

    async def stop(self) -> None:
        """Close the gateway and drop the registry entry."""
        unregister_channel(CHANNEL_DISCORD)
        if self.client and not self.client.is_closed():
            await self.client.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Discord channel stopped.")

    # ── Handlers ─────────────────────────────────────────────

    def _register_handlers(self) -> None:
        """Attach gateway event handlers.

        Registered via client.event (verified present on discord.Client 2.7.1)
        rather than a Bot subclass, because Syne needs no command framework —
        slash commands and prefix parsing are not used.
        """
        client = self.client

        @client.event
        async def on_ready():
            user = client.user
            logger.info(
                "Discord connected as %s (id=%s), %d guild(s)",
                user, getattr(user, "id", "?"), len(client.guilds),
            )
            # Only advertise the channel once it can actually send.
            register_channel(CHANNEL_DISCORD, self)

        @client.event
        async def on_message(message: discord.Message):
            try:
                await self._handle_message(message)
            except Exception as e:
                logger.error(
                    f"Unhandled error in Discord on_message: {type(e).__name__}: {e}",
                    exc_info=True,
                )

    def _is_duplicate(self, message_id: int) -> bool:
        """True if this message id was already processed (bounded memory)."""
        if message_id in self._seen:
            return True
        self._seen.add(message_id)
        self._seen_order.append(message_id)
        if len(self._seen_order) > 500:
            self._seen.discard(self._seen_order.pop(0))
        return False

    async def _handle_message(self, message: discord.Message) -> None:
        """Process one inbound Discord message."""
        # Never react to self or to any other bot — trivially avoids loops.
        if message.author.bot:
            return
        if self.client.user and message.author.id == self.client.user.id:
            return

        if self._is_duplicate(message.id):
            logger.debug("Dropping duplicate Discord message %s", message.id)
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        text = (message.content or "").strip()

        # Without MESSAGE_CONTENT, Discord sends empty content for anything the
        # bot was not addressed in. Empty text is therefore the normal case for
        # ambient server chatter, not an error.
        if not text:
            logger.debug(
                "Discord message %s has empty content (expected without the "
                "MESSAGE_CONTENT intent unless addressed) — ignoring",
                message.id,
            )
            return

        if is_dm:
            handled = await self._authorize_dm(message)
        else:
            handled = await self._authorize_guild(message)
        if not handled:
            return

        # Auto-collect the guild speaker into groups.settings['members'].
        # That registry is what inbound.py:103-108 resolves ctx.sender_access
        # from for a guild message, and what the member menus read. Mirrors
        # telegram.py:962: name/username only, so a manually assigned
        # access/alias is never clobbered (models.py:409), and a new member
        # defaults to "public" (models.py:444).
        #
        # Placed before the mention strip, matching Telegram, which collects
        # before its require_mention check -- a member who was authorized is
        # worth recording even if this particular message yields no prompt.
        if not is_dm and message.guild:
            try:
                from ..db.models import update_group_member
                await update_group_member(
                    platform="discord",
                    platform_group_id=str(message.guild.id),
                    member_id=str(message.author.id),
                    name=(message.author.display_name
                          or message.author.name
                          or str(message.author.id)),
                    username=message.author.name,
                )
            except Exception as e:
                logger.debug("Discord auto-collect member failed: %s", e)

        # Strip the bot mention so the LLM sees a clean prompt.
        if not is_dm:
            text = self._strip_mention(text)
            if not text:
                return

        guild_name = message.guild.name if message.guild else "DM"
        logger.info(
            "[discord/%s] %s (%s): %s",
            guild_name, message.author, message.author.id, text[:100],
        )

        # 👀 read receipt, matching Telegram and WhatsApp. Without the
        # MESSAGE_CONTENT intent Discord only delivers text when the bot
        # is addressed, so reaching this line already means it is for us.
        try:
            await message.add_reaction("👀")
        except Exception as e:
            logger.debug("Discord 👀 receipt failed: %s", e)

        await self._dispatch(message, text, is_dm)

    def _strip_mention(self, text: str) -> str:
        """Remove a leading mention of this bot from the message text.

        Discord delivers mentions as <@id> / <@!id> markup. Only a leading
        mention is removed: one buried mid-sentence is part of what the user
        actually wrote.
        """
        me = self.client.user
        if not me:
            return text.strip()
        for token in (f"<@{me.id}>", f"<@!{me.id}>"):
            if text.startswith(token):
                return text[len(token):].strip()
        return text.strip()

    # ── Authorization ────────────────────────────────────────

    async def _authorize_dm(self, message: discord.Message) -> bool:
        """Decide whether to serve a DM.

        Mirrors the Telegram DM gate: only the exact string "open" admits an
        unknown sender. Any other value — including a typo or an unreadable one
        — falls through to approval, which asks the owner on Telegram. An
        unreadable policy must never be the thing that admits a stranger.

        No row is written while a user is pending, deliberately. The Telegram
        gate persists the user first and then gates on "did this row exist
        before this message" (telegram.py: _existing), so only the FIRST
        message is held and a second one slips through while the owner is
        still deciding. Here the ABSENCE of the row is the pending state, so
        every message before approval is held.
        """
        existing = await get_user("discord", str(message.author.id))
        if existing:
            if existing.get("access_level") == "blocked":
                logger.debug("Ignoring blocked Discord user %s", message.author.id)
                return False
            # A user revoked via /dcmembers Delete keeps their row (and with it
            # their history) but carries preferences.pending_approval: hold
            # them exactly like an unknown sender until the owner approves.
            prefs = existing.get("preferences") or {}
            if prefs.get("pending_approval") and existing.get("access_level") == "public":
                policy = str(await get_config("discord.dm_policy", "approval"))
                if policy != "open":
                    await self._handle_pending_dm(message, policy)
                    return False
                await self._clear_pending_flag(str(message.author.id), prefs)
            return True

        policy = str(await get_config("discord.dm_policy", "approval"))
        if policy != "open":
            await self._handle_pending_dm(message, policy)
            return False

        # create_user() is used here instead of the get-or-create helper: that
        # helper persists NOTHING when is_dm is False (models.py:182-194
        # returns an ephemeral dict), so the admission would not stick and this
        # person would be re-admitted on every message; and when is_dm is True
        # it can auto-promote them to OWNER (models.py:167-181). Creating the
        # row directly as "public" avoids both - the admission sticks, and
        # ownership stays unobtainable over Discord.
        await create_user(
            name=message.author.name,
            platform="discord",
            platform_id=str(message.author.id),
            display_name=message.author.display_name,
            access_level="public",
        )
        logger.info(
            "discord.dm_policy=open: admitted new Discord user %s", message.author.id
        )
        return True

    async def _authorize_guild(self, message: discord.Message) -> bool:
        """Decide whether to serve a guild message.

        Approval is per guild, so a new channel inside an approved guild needs
        no second approval — the `groups` row is keyed on the guild id.
        """
        if not message.guild:
            return False

        group = await get_group("discord", str(message.guild.id))
        if group:
            return bool(group.get("enabled", True))

        policy = str(await get_config("discord.guild_policy", "allowlist"))
        if policy != "open":
            await self._handle_pending_guild(message, policy)
            return False
        logger.info(
            "discord.guild_policy=open: serving unregistered guild %s",
            message.guild.id,
        )
        return True

    # ── Approval (prompt delivered on Telegram) ──────────────

    async def _owner_telegram_id(self) -> Optional[str]:
        """Telegram chat id of the owner, or None if there is not one.

        Approval is asked on Telegram because that is where the owner is, and
        because ownership is deliberately not establishable over Discord.
        """
        from ..db.connection import get_connection

        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT platform_id FROM users "
                "WHERE platform = 'telegram' AND access_level = 'owner' "
                "ORDER BY id LIMIT 1"
            )
        return str(row["platform_id"]) if row else None

    async def _ask_owner(self, key: str, text: str, buttons: list) -> bool:
        """Send one approval prompt to the owner over Telegram.

        `key` dedupes: twenty messages from a stranger must not become twenty
        prompts. The key is marked only once the send actually succeeded, so a
        transient Telegram failure does not permanently swallow the request.
        """
        if key in self._pending_notified:
            return False

        telegram = get_channel(CHANNEL_TELEGRAM)
        if telegram is None:
            logger.warning(
                "Cannot ask for approval (%s): Telegram channel not registered.", key
            )
            return False

        # getattr rather than a direct call: if this is ever run against a
        # telegram.py without the cross-channel surface, degrade to a log
        # instead of raising inside a gateway event handler.
        prompt = getattr(telegram, "send_inline_prompt", None)
        if prompt is None:
            logger.warning(
                "Cannot ask for approval (%s): Telegram channel has no "
                "send_inline_prompt.", key
            )
            return False

        owner_id = await self._owner_telegram_id()
        if not owner_id:
            logger.warning("Cannot ask for approval (%s): no owner on Telegram.", key)
            return False

        ok = await prompt(owner_id, text, buttons)
        if ok:
            self._pending_notified.add(key)
        return ok

    async def _handle_pending_dm(self, message: discord.Message, policy: str) -> None:
        """Hold a DM from an unknown user and ask the owner about them."""
        author = message.author
        logger.info(
            "Discord DM from unknown user %s (%s) held: dm_policy=%r — asking owner.",
            author, author.id, policy,
        )
        asked = await self._ask_owner(
            "dm:%s" % author.id,
            (
                "🔔 **New Discord user wants access:**\n\n"
                "• Name: %s\n"
                "• Handle: %s\n"
                "• ID: `%s`\n\n"
                "Approve or reject?"
            ) % (
                escape_markdown(author.display_name or author.name),
                escape_markdown(author.name),
                author.id,
            ),
            [
                ("✅ Approve", "discord_approve:%s" % author.id),
                ("❌ Reject", "discord_reject:%s" % author.id),
            ],
        )
        # Only acknowledge on the first held message. Saying it every time
        # would turn a dedupe win into a spam loop in the user's DM.
        if asked:
            await self._safe_send(
                message.channel,
                "⏳ Your access is pending approval from the owner. Please wait.",
            )

    async def _handle_pending_guild(self, message: discord.Message, policy: str) -> None:
        """Hold a guild message and ask the owner about the server.

        Nothing is sent into the server itself: announcing presence in a place
        the owner has not approved is exactly what allowlist mode prevents.
        """
        guild = message.guild
        logger.info(
            "Discord message from unapproved guild %r (%s) held: guild_policy=%r "
            "— asking owner.",
            guild.name, guild.id, policy,
        )
        await self._ask_owner(
            "guild:%s" % guild.id,
            (
                "🔔 **Syne was added to a Discord server:**\n\n"
                "• Server: %s\n"
                "• ID: `%s`\n"
                "• Members: %s\n\n"
                "Approve or leave?"
            ) % (
                escape_markdown(guild.name),
                guild.id,
                guild.member_count or "?",
            ),
            [
                ("✅ Approve", "discord_guild_approve:%s" % guild.id),
                ("❌ Reject", "discord_guild_reject:%s" % guild.id),
            ],
        )

    # ── Approval actions (invoked from the Telegram callback) ─

    async def _resolve_user_names(self, user_id: str) -> tuple[str, str]:
        """Best-effort (name, display_name) for a Discord user id."""
        user = self.client.get_user(int(user_id)) if self.client else None
        if user is None and self.client:
            try:
                user = await self.client.fetch_user(int(user_id))
            except Exception as e:
                logger.debug("Discord user %s unresolvable: %s", user_id, e)
        if user is None:
            return str(user_id), str(user_id)
        return user.name, user.display_name or user.name

    async def _dm_user(self, user_id: str, text: str) -> None:
        """Best-effort DM to a Discord user (approval / rejection notice)."""
        if not self.client:
            return
        try:
            user = self.client.get_user(int(user_id))
            if user is None:
                user = await self.client.fetch_user(int(user_id))
            await user.send(text)
        except Exception as e:
            logger.debug("Could not DM Discord user %s: %s", user_id, e)

    async def _clear_pending_flag(self, user_id: str, prefs: Optional[dict] = None) -> None:
        """Drop preferences.pending_approval, preserving every other key."""
        if prefs is None:
            existing = await get_user("discord", str(user_id))
            prefs = (existing or {}).get("preferences") or {}
        if not prefs.get("pending_approval"):
            return
        new_prefs = dict(prefs)
        new_prefs.pop("pending_approval", None)
        try:
            await update_user("discord", str(user_id), preferences=new_prefs)
        except Exception as e:
            logger.error("Failed to clear pending flag for Discord user %s: %s", user_id, e)

    async def approve_user(self, user_id: str) -> tuple[bool, str]:
        """Admit a pending Discord DM user as "public". Returns (ok, label).

        access_level is pinned to "public" and never derived from a count, so
        this path cannot hand out ownership the way the get-or-create helper
        can (models.py promote blocks).
        """
        try:
            name, display = await self._resolve_user_names(user_id)
            existing = await get_user("discord", str(user_id))
            if existing:
                await update_user("discord", str(user_id), access_level="public")
                await self._clear_pending_flag(str(user_id))
            else:
                await create_user(
                    name=name,
                    platform="discord",
                    platform_id=str(user_id),
                    display_name=display,
                    access_level="public",
                )
        except Exception as e:
            logger.error("Approving Discord user %s failed: %s", user_id, e)
            return False, str(user_id)

        self._pending_notified.discard("dm:%s" % user_id)
        await self._dm_user(
            user_id, "✅ Your access has been approved. You can send messages now."
        )
        logger.info("Discord user %s approved by owner", user_id)
        return True, display

    async def reject_user(self, user_id: str) -> tuple[bool, str]:
        """Reject a pending Discord DM user. Returns (ok, label).

        The row IS written here, as "blocked": every later message then costs
        one indexed lookup and dies at the blocked check in _authorize_dm,
        instead of re-entering the gate and re-asking the owner after a
        restart has cleared the in-memory dedupe.
        """
        try:
            name, display = await self._resolve_user_names(user_id)
            existing = await get_user("discord", str(user_id))
            if existing:
                await update_user("discord", str(user_id), access_level="blocked")
                await self._clear_pending_flag(str(user_id))
            else:
                await create_user(
                    name=name,
                    platform="discord",
                    platform_id=str(user_id),
                    display_name=display,
                    access_level="blocked",
                )
        except Exception as e:
            logger.error("Rejecting Discord user %s failed: %s", user_id, e)
            return False, str(user_id)

        self._pending_notified.discard("dm:%s" % user_id)
        await self._dm_user(user_id, "Sorry, your access request was not approved.")
        logger.info("Discord user %s rejected by owner", user_id)
        return True, display

    async def approve_guild(self, guild_id: str) -> tuple[bool, str]:
        """Register a Discord server so the bot may answer in it.

        Upsert rather than insert-if-absent: a server that was approved,
        rejected and re-added must come back enabled, not collide.
        """
        from ..db.connection import get_connection

        guild = self.client.get_guild(int(guild_id)) if self.client else None
        if guild is None and self.client:
            try:
                guild = await self.client.fetch_guild(int(guild_id))
            except Exception as e:
                logger.warning("Discord guild %s unreachable: %s", guild_id, e)
        name = guild.name if guild is not None else "Guild %s" % guild_id

        try:
            require_mention = bool(await get_config("discord.require_mention", True))
            async with get_connection() as conn:
                await conn.execute(
                    """INSERT INTO groups
                           (name, platform, platform_group_id, enabled,
                            require_mention, settings)
                       VALUES ($1, 'discord', $2, true, $3, '{}')
                       ON CONFLICT (platform, platform_group_id) DO UPDATE
                           SET enabled = true, name = EXCLUDED.name""",
                    name, str(guild_id), require_mention,
                )
        except Exception as e:
            logger.error("Approving Discord guild %s failed: %s", guild_id, e)
            return False, name

        self._pending_notified.discard("guild:%s" % guild_id)
        logger.info("Discord guild %r (%s) approved by owner", name, guild_id)
        return True, name

    async def reject_guild(self, guild_id: str) -> tuple[bool, str]:
        """Leave a rejected Discord server. Returns (left, label).

        Uses Guild.leave() (verified present on discord.py 2.7.1). If the
        guild is not in cache there is nothing to leave from - report that
        rather than claiming success.
        """
        guild = self.client.get_guild(int(guild_id)) if self.client else None
        name = guild.name if guild is not None else "Guild %s" % guild_id
        left = False
        if guild is not None:
            try:
                await guild.leave()
                left = True
            except Exception as e:
                logger.error("Failed to leave Discord guild %s: %s", guild_id, e)
        else:
            logger.warning(
                "Cannot leave Discord guild %s: not in cache (already removed?)",
                guild_id,
            )
        self._pending_notified.discard("guild:%s" % guild_id)
        return left, name

    # ── Inbound context ──────────────────────────────────────

    async def _build_inbound(self, message: discord.Message, is_dm: bool) -> Any:
        """Map a Discord message onto the channel-agnostic InboundContext.

        This is the single place Discord-specific shapes are translated, so
        downstream code never learns what platform it is serving.
        """
        from .inbound import InboundContext, load_group_settings

        author = message.author
        reply_sender = None
        reply_body = None
        # message.reference carries the replied-to message; resolved is already
        # populated when Discord shipped it with the event, so the common case
        # costs no extra HTTP call.
        ref = message.reference
        if ref is not None:
            resolved = getattr(ref, "resolved", None)
            if isinstance(resolved, discord.Message):
                reply_sender = str(resolved.author)
                reply_body = (resolved.content or "").strip() or None

        ctx = InboundContext(
            channel="discord",
            platform="discord",
            chat_type="direct" if is_dm else "group",
            conversation_label=author.display_name or author.name,
            group_subject=message.guild.name if message.guild else None,
            # Keyed on the GUILD for group settings, matching _authorize_guild.
            chat_id=str(message.channel.id) if is_dm else str(message.guild.id),
            sender_name=(author.display_name or author.name) if not is_dm else None,
            sender_id=str(author.id) if not is_dm else None,
            sender_username=author.name if not is_dm else None,
            # In a non-privileged setup, receiving text in a guild at all means
            # the bot was addressed.
            was_mentioned=not is_dm,
            has_reply_context=reply_body is not None,
            reply_to_sender=reply_sender,
            reply_to_body=reply_body,
        )
        await load_group_settings(ctx)
        return ctx

    async def _member_model_override(self, message: discord.Message,
                                     is_dm: bool) -> Optional[str]:
        """Per-entry model override from config discord.members.

        Keyed on the DM author's user id, or on the guild id for a server —
        the same keys /discordmembers writes and the gates above read. Returns
        None when there is no entry or no override, which leaves the globally
        active model in place.
        """
        key = str(message.author.id) if is_dm else (
            str(message.guild.id) if message.guild else "")
        if not key:
            return None
        raw = await get_config("discord.members", [])
        if isinstance(raw, str):
            import json as _json
            try:
                raw = _json.loads(raw)
            except Exception:
                return None
        if not isinstance(raw, list):
            return None
        for item in raw:
            if isinstance(item, dict) and str(item.get("id") or "") == key:
                return item.get("model") or None
        return None

    # ── Dispatch ─────────────────────────────────────────────

    async def _dispatch(self, message: discord.Message, text: str, is_dm: bool) -> None:
        """Hand the message to the agent and deliver the reply."""
        from .inbound import build_user_context_prefix

        inbound = await self._build_inbound(message, is_dm)
        original_text = text
        prefix = build_user_context_prefix(inbound)
        if prefix:
            text = f"{prefix}\n\n{text}"

        author = message.author
        # Per-entry model override, if /discordmembers set one for this
        # user or server. Consumed in conversation.py alongside the
        # WhatsApp and remote-node overrides.
        model_override = await self._member_model_override(message, is_dm)
        metadata = {
            "message_id": message.id,
            "chat_id": inbound.chat_id,
            "inbound": inbound,
            "original_text": original_text,
            **({"model_override": model_override} if model_override else {}),
        }

        # typing() is an async context manager on Messageable (verified on
        # discord.py 2.7.1) and keeps the indicator alive for the whole call.
        try:
            async with message.channel.typing():
                response = await self.agent.handle_message(
                    platform="discord",
                    chat_id=inbound.chat_id,
                    user_name=author.name or str(author.id),
                    user_platform_id=str(author.id),
                    message=text,
                    display_name=author.display_name or author.name,
                    message_metadata=metadata,
                    # agent.py:2907 forwards this straight into the DB helper,
                    # whose promote blocks (models.py:126-162, 167-181) raise
                    # the caller to OWNER when owner_count == 0 - so on a fresh
                    # install the first Discord DM would become owner. Pinned
                    # False: ownership is Telegram-only. Correct for both paths
                    # - DM users are already persisted by _authorize_dm, and
                    # guild users are meant to be ephemeral (models.py:183-194).
                    is_dm=False,
                )
        except Exception as e:
            logger.error(f"Discord dispatch failed: {type(e).__name__}: {e}", exc_info=True)
            from .errors import classify_error
            _access = inbound.sender_access
            if not _access:
                try:
                    _u = await get_user("discord", str(author.id))
                except Exception:
                    _u = None
                _access = (_u or {}).get("access_level") or "public"
            _model = ""
            try:
                _conv = self.agent.conversations._active.get(f"discord:{inbound.chat_id}")
                if _conv:
                    _model = getattr(_conv.provider, "model_id", "") or _conv.provider.name
            except Exception:
                pass
            await self._safe_send(
                message.channel, classify_error(e, model=_model, access_level=_access)
            )
            return

        if not response:
            return

        # Reply/react tags are channel-agnostic; parse them the same way
        # Telegram does so behaviour matches across channels.
        # tags.py:16 takes the triggering message id as the 2nd argument and
        # returns it for [[reply_to_current]]. Omitting it silently disables
        # quote-replies, so it is passed explicitly.
        response, reply_to = parse_reply_tag(response, message.id)
        response, reactions = parse_react_tags(response)

        for emoji in reactions[:1]:
            try:
                await message.add_reaction(emoji)
            except Exception as e:
                logger.debug("Discord reaction %r failed: %s", emoji, e)

        await self.send_response(
            message.channel,
            response,
            reply_to=message if reply_to else None,
        )

    # ── Outbound ─────────────────────────────────────────────

    async def send_response(
        self,
        channel: Any,
        text: str,
        reply_to: Optional[discord.Message] = None,
    ) -> None:
        """Send an agent response, applying the shared outbound pipeline."""
        # _dispatch already acted on these tags; sub-agent results and
        # send_message arrive here without a triggering message, so strip
        # any leftover tag instead of showing it.
        text, _ = parse_reply_tag(text or "")
        text, _ = parse_react_tags(text)
        text = process_outbound(text)
        text, media = extract_all_media(text)
        text = markdown_to_discord(text)

        first = True
        if text.strip():
            for chunk in split_message(text, max_length=DISCORD_MAX_MESSAGE):
                if not chunk.strip():
                    continue
                # Only the first chunk quotes the user's message; quoting every
                # chunk of a long answer is noise.
                await self._safe_send(
                    channel, chunk, reply_to=reply_to if first else None
                )
                first = False

        for path in media:
            try:
                await channel.send(file=discord.File(path))
            except Exception as e:
                logger.error("Discord file send failed for %s: %s", path, e)

    async def _safe_send(
        self,
        channel: Any,
        text: str,
        reply_to: Optional[discord.Message] = None,
    ) -> None:
        """Send one message, degrading instead of raising.

        A failed send must never bubble into the event handler: that would
        abort the rest of a multi-chunk reply and leave the user with half an
        answer and no explanation.
        """
        try:
            if reply_to is not None:
                await reply_to.reply(text)
            else:
                await channel.send(text)
        except discord.Forbidden:
            logger.warning(
                "Discord refused the send (missing permission in that channel)."
            )
        except discord.HTTPException as e:
            logger.error("Discord send failed: %s", e)
        except Exception as e:
            logger.error("Discord send failed: %s: %s", type(e).__name__, e)

    # ── Used by the registry / cross-channel callers ─────────

    async def is_group_target(self, chat_id: str) -> bool:
        """True unless chat_id resolves to a DM channel.

        send_message uses this to apply the "groups only from a scheduled
        task" rule to Discord, whose channel ids (unlike Telegram's '-' or
        WhatsApp's '@g.us') carry no hint of what they are. Anything that
        cannot be resolved counts as a group: fail closed.
        """
        if not self.client or self.client.is_closed():
            return True
        try:
            target = self.client.get_channel(int(chat_id))
            if target is None:
                target = await self.client.fetch_channel(int(chat_id))
        except Exception:
            return True
        return not isinstance(target, discord.DMChannel)

    async def send_to_chat(self, chat_id: str, text: str) -> bool:
        """Send a message to a Discord channel id. Returns success.

        This is the entry point other channels reach through the registry, so
        it takes a plain id and never a discord.py object.
        """
        if not self.client or self.client.is_closed():
            logger.warning("send_to_chat called while Discord is not connected")
            return False
        try:
            target = self.client.get_channel(int(chat_id))
            if target is None:
                target = await self.client.fetch_channel(int(chat_id))
        except Exception as e:
            logger.error("Discord channel %s unreachable: %s", chat_id, e)
            return False

        await self.send_response(target, text)
        return True
