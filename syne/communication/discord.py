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
from .formatting import markdown_to_discord
from .registry import CHANNEL_DISCORD, register_channel, unregister_channel
from ..db.models import (
    get_config,
    get_group,
    get_or_create_user,
    get_user,
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
        self.client = discord.Client(intents=self._build_intents())
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

    # ── Authorization (fail-closed until the approval gate lands) ──

    async def _authorize_dm(self, message: discord.Message) -> bool:
        """Decide whether to serve a DM.

        Mirrors the Telegram DM gate: only the exact string "open" admits an
        unknown sender. Any other value — including a typo or an unreadable one
        — falls through to approval, and until the Telegram-delivered approval
        UI exists, approval means "ignore and log". An unreadable policy must
        never be the thing that admits a stranger.
        """
        existing = await get_user("discord", str(message.author.id))
        if existing:
            if existing.get("access_level") == "blocked":
                logger.debug("Ignoring blocked Discord user %s", message.author.id)
                return False
            return True

        policy = str(await get_config("discord.dm_policy", "approval"))
        if policy != "open":
            logger.info(
                "Discord DM from unknown user %s (%s) held: dm_policy=%r. "
                "Approval UI is not wired yet — ignoring.",
                message.author, message.author.id, policy,
            )
            return False

        await get_or_create_user(
            name=message.author.name,
            platform="discord",
            platform_id=str(message.author.id),
            display_name=message.author.display_name,
            # models.py:126 auto-promotes the first DM sender to OWNER when no
            # owner row exists. Ownership must only ever be established over
            # Telegram, so this stays False even though this IS a DM.
            is_dm=False,
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
            logger.info(
                "Discord message from unapproved guild %r (%s) held: "
                "guild_policy=%r. Approval UI is not wired yet — ignoring.",
                message.guild.name, message.guild.id, policy,
            )
            return False
        logger.info(
            "discord.guild_policy=open: serving unregistered guild %s",
            message.guild.id,
        )
        return True

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
        metadata = {
            "message_id": message.id,
            "chat_id": inbound.chat_id,
            "inbound": inbound,
            "original_text": original_text,
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
                    # agent.py:2907 - group senders must NOT be persisted as
                    # users, so the real DM-ness has to be forwarded here.
                    is_dm=is_dm,
                )
        except Exception as e:
            logger.error(f"Discord dispatch failed: {type(e).__name__}: {e}", exc_info=True)
            await self._safe_send(message.channel, "⚠️ Something broke while handling that.")
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
        text = process_outbound(text or "")
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
