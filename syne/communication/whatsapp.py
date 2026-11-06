"""WhatsApp channel adapter.

WhatsApp is a first-class channel alongside Telegram and Discord, not an
ability: it owns a persistent connection, its own inbound event loop and its
own member state, none of which fit the request/response shape an ability has.

WHY NEONIZE AND NOT wacli
-------------------------
The previous implementation (syne/abilities/whatsapp.py) drove the `wacli`
binary as a subprocess. `wacli sync --follow` holds an EXCLUSIVE lock on the
WhatsApp store, so every outbound message and every media download had to:

    stop sync  ->  run `wacli send ...`  ->  start sync again

See the comment at abilities/whatsapp.py:964-967. That makes the listener die
and respawn on every single reply, and any message that arrives inside that
window is simply never seen. It also means a send failure can leave the bridge
with no listener at all.

neonize is a Python package that embeds the whatsmeow Go core as a prebuilt
shared object in its wheel (neonize-linux-amd64.so) — no Go toolchain, no
compile step. One connection lives inside this process and carries BOTH
directions, so sending no longer interrupts receiving. The only manual step
that remains is scanning a QR once to pair.

NO APPROVAL GATE, BY DESIGN
---------------------------
Discord fails closed: an unknown DM or guild is held while the owner is asked
over Telegram (discord.py:302-369). WhatsApp deliberately does NOT do that.
A WhatsApp number is already a human the owner has in their phone, and the
trigger word is the gate: nothing is processed unless the bot is addressed by
name. Unknown senders therefore auto-register as "public" on first triggered
message, exactly as the wacli bridge did.

NO COMMANDS
-----------
There are zero slash commands here. WhatsApp (like Discord) is pure chat; all
management lives on Telegram — /wamembers drives the member list this module
reads, and the leave-group action calls into leave_group() below.

WHAT LIVES HERE vs ELSEWHERE
----------------------------
Shared behaviour is reused, never reimplemented:
  * outbound.process_outbound / extract_all_media / split_message - text pipeline
  * formatting.markdown_to_whatsapp                               - WA markdown
  * inbound.InboundContext / build_user_context_prefix            - per-message context
  * registry.register_channel                                     - cross-channel reach
Only WhatsApp-specific mapping is written here.

THE TRIGGER WORD IS ABSOLUTE
----------------------------
Ported verbatim from the wacli bridge (abilities/whatsapp.py:753-813) so
behaviour does not change under users:
  * text ALWAYS requires the trigger word — in groups AND in DMs, from
    strangers AND from the owner. A group's require_mention flag is read only
    to decide `enabled`; it deliberately does NOT relax the trigger.
  * from_me in a group is always dropped — Syne's own replies come back as
    from_me and would otherwise feed themselves forever
  * from_me in a DM is dropped unless it is the self-chat
  * media in the self-chat needs no trigger (there is no one else to address)

The owner can therefore use the trigger anywhere, self-chat included. The only
self-feeding risk left is the self-chat, where a reply of ours arrives as
from_me in the same chat; two guards cover it without ever restricting the
owner — the echo hash below and the message-id dedup.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
from typing import Any, Optional

from neonize.aioze.client import NewAClient
from neonize.events import (
    ConnectedEv,
    DisconnectedEv,
    LoggedOutEv,
    MessageEv,
    PairStatusEv,
)
from neonize.utils.enum import ChatPresence, ChatPresenceMedia
from neonize.utils.jid import build_jid

from ..agent import SyneAgent
from ..db.models import (
    create_group,
    create_user,
    get_config,
    get_group,
    get_user,
    set_config,
    update_group,
    update_group_member,
)
from .formatting import markdown_to_whatsapp
from .inbound import InboundContext, build_user_context_prefix, load_group_settings
from .outbound import extract_all_media, process_outbound, split_message
from .registry import CHANNEL_WHATSAPP, register_channel, unregister_channel

logger = logging.getLogger("syne.whatsapp")


# WhatsApp accepts long bodies, but 4096 matches Telegram and keeps replies
# readable on a phone. Passed explicitly because split_message's default is a
# Telegram assumption, not a WhatsApp fact.
WHATSAPP_MAX_MESSAGE = 4096


class _ChatPresenceIndicator:
    """Keeps WhatsApp's "typing…" indicator alive while the agent thinks.

    The WhatsApp counterpart of telegram.py's _TypingIndicator. neonize
    exposes send_chat_presence(jid, state, media) (aioze/client.py:2213).
    WhatsApp expires a COMPOSING state by itself after a few seconds, so it
    must be refreshed on a loop; on exit PAUSED is sent so the indicator
    clears at once instead of lingering until that expiry.

    Best-effort by design: every failure is swallowed. A presence update is
    cosmetic and must never take down message handling.

    No Presence.AVAILABLE is broadcast — that would park the account as
    permanently "online" and rewrite its last-seen, which is the owner's
    call, not this channel's. whatsmeow may therefore refuse chat presence
    outright; if the first send fails the loop exits rather than retrying
    for the whole max_duration. The 👀 receipt does not depend on it.

    max_duration caps the loop so a hung LLM call cannot leave it running.
    """

    def __init__(self, channel, chat_jid: str, interval: float = 8.0,
                 max_duration: float = 300.0):
        self._channel = channel
        self._chat_jid = chat_jid
        self._interval = interval
        self._max_duration = max_duration
        self._task: Optional[asyncio.Task] = None

    async def _send(self, state) -> None:
        client = self._channel.client
        if not client:
            return
        await client.send_chat_presence(
            self._channel._to_jid(self._chat_jid),
            state,
            ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
        )

    async def _loop(self) -> None:
        start = time.monotonic()
        try:
            while True:
                if time.monotonic() - start > self._max_duration:
                    logger.warning(
                        "[whatsapp] typing indicator timeout (%ss) for %s",
                        self._max_duration, self._chat_jid,
                    )
                    return
                try:
                    await self._send(ChatPresence.CHAT_PRESENCE_COMPOSING)
                except Exception as e:
                    logger.debug("[whatsapp] chat presence unavailable: %s", e)
                    return
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            await self._send(ChatPresence.CHAT_PRESENCE_PAUSED)
        except Exception as e:
            logger.debug("[whatsapp] chat presence clear failed: %s", e)

# How long a sent message stays "ours" for echo suppression. whatsmeow echoes
# our own sends back as from_me; in a DM self-chat those are not dropped by the
# from_me rules, so they are matched by content instead.
_ECHO_TTL = 60.0

# Bounded dedup memory for inbound message ids.
_SEEN_MAX = 500

# Bounded cache for @lid -> phone number lookups. One entry per contact that
# ever messages us, so a couple of thousand is already generous.
_LID_CACHE_MAX = 2000

# Server parts of a WhatsApp JID.
_SERVER_DM = "s.whatsapp.net"
_SERVER_GROUP = "g.us"
_SERVER_LID = "lid"


def _is_postgres_store(store: str) -> bool:
    """True when neonize will treat this string as a Postgres DSN.

    Mirrors goneonize/main.go getDB(), which picks the driver on exactly this
    prefix test -- `strings.HasPrefix(db, "postgres")` -- and otherwise falls
    back to sqlite3 with the string used as a filename. Both "postgres://" and
    "postgresql://" therefore qualify.
    """
    return store.strip().lower().startswith("postgres")


def _redact_store(store: str) -> str:
    """Strip the password out of a DSN before it reaches the log."""
    if not _is_postgres_store(store):
        return store
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", store)


def _with_sslmode(store: str) -> str:
    """Pin sslmode on a Postgres DSN that does not already carry one.

    whatsmeow opens the store through github.com/lib/pq, whose default is
    sslmode=require. The Syne Postgres normally runs on localhost, or over a
    docker-compose network, with ssl off -- so the very DSN asyncpg accepts
    makes the Go side fail with:
        pq: SSL is not enabled on the server

    The silence is the expensive part. That error is raised inside the task
    connect() spawns, and nothing awaits that task, so the channel logs
    'Starting WhatsApp (neonize) client' and then stops dead: no error, no QR
    callback, no whatsmeow_* tables. It reads like a slow connect rather than
    a failed one.

    'prefer' rather than 'disable': it still negotiates TLS when the server
    offers it and falls back to plaintext when it does not, so this stays
    correct on an install whose Postgres does have SSL. Verified against
    neonize 0.10.0 -- the bare DSN raises, while prefer and disable both
    reach the QR callback.
    """
    if not _is_postgres_store(store) or "sslmode=" in store.lower():
        return store
    sep = "&" if "?" in store else "?"
    return store + sep + "sslmode=prefer"


def _default_store(agent: SyneAgent) -> str:
    """Syne's own PostgreSQL, falling back to a local SQLite file.

    Sharing the main database means one pg_dump captures the WhatsApp pairing
    along with everything else, which is the whole reason for not keeping a
    separate file.

    The fallback exists so the channel still starts if settings cannot be read:
    a file store pairs and works exactly the same, it just sits outside the
    main backup. Note that whatsmeow cannot migrate a pairing between the two
    -- switching stores means scanning the QR once more.
    """
    dsn = getattr(getattr(agent, "settings", None), "database_url", None)
    if dsn and _is_postgres_store(str(dsn)):
        return str(dsn)
    return os.path.expanduser("~/.syne/whatsapp.sqlite3")


class WhatsAppChannel:
    """WhatsApp adapter driving the neonize (whatsmeow) client in-process."""

    def __init__(self, agent: SyneAgent, store: Optional[str] = None):
        self.agent = agent
        # Where whatsmeow keeps the device state and the Signal keys. Defaults
        # to Syne's own database; see _default_store. whatsmeow owns these
        # tables and prefixes every one of them whatsmeow_*, so they cannot
        # collide with Syne's schema -- and because they are whatsmeow's, they
        # are deliberately absent from schema.sql and from `syne db migrate`.
        # Wrapped in _with_sslmode because lib/pq defaults to sslmode=require
        # and the local Postgres refuses it; without this the store never
        # opens and the QR never arrives. See the helper for the full story.
        self.store = _with_sslmode(store or _default_store(agent))
        self.client: Optional[NewAClient] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._running = False
        # Our own number, learned on connect. Needed to tell a self-chat from a
        # DM to someone else.
        self._own_pn: str = ""

        # Inbound dedup: whatsmeow can redeliver on reconnect, and a
        # double-processed message is a double LLM call and a double reply.
        self._seen: set[str] = set()
        self._seen_order: list[str] = []

        # chat+content hash -> time we sent it, for echo suppression.
        self._echo: dict[str, float] = {}

        # lid local-part -> phone number, filled from whatsmeow_lid_map on
        # first miss. Only HITS are cached: a miss can become a hit later once
        # whatsmeow syncs the contact, and caching "" would hide that forever.
        self._lid_pn: dict[str, str] = {}

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> None:
        """Pair/connect and return once the connect task is spawned.

        Does not wait for the connection to come up: a WhatsApp server that is
        slow or an unpaired session must not hold up boot of Telegram.
        """
        # Only a file store needs a directory. Doing this unconditionally would
        # call dirname() on a DSN and try to mkdir something like
        # "postgresql://syne:syne@localhost:5433".
        if not _is_postgres_store(self.store):
            os.makedirs(os.path.dirname(self.store) or ".", exist_ok=True)

        # uuid is passed explicitly because neonize derives it from the store
        # string otherwise (client.py: `uuid or name`). With a DSN that would
        # make the database password both the client identifier and the
        # ContactStore key; a fixed id keeps the credential out of both.
        self.client = NewAClient(name=self.store, uuid="syne")
        self._register_handlers()
        self._running = True

        logger.info(
            "Starting WhatsApp (neonize) client, store: %s",
            _redact_store(self.store),
        )
        # connect() itself creates the long-lived task internally and returns
        # it; hold the reference so stop() can cancel deterministically.
        self._connect_task = await self.client.connect()

    async def stop(self) -> None:
        """Disconnect and drop the registry entry."""
        self._running = False
        unregister_channel(CHANNEL_WHATSAPP)

        if self.agent and self.agent.conversations:
            self.agent.conversations.remove_delivery_callback(
                self._deliver_subagent_result
            )
            self.agent.conversations.remove_status_callback(self._send_status_message)

        if self.client:
            try:
                await self.client.disconnect()
            except Exception as e:
                logger.debug("WhatsApp disconnect raised: %s", e)

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("WhatsApp channel stopped.")

    # ── Handlers ─────────────────────────────────────────────

    def _register_handlers(self) -> None:
        """Attach neonize event handlers.

        neonize dispatches by protobuf event class; the registered set is
        converted to the Go subscriber list at connect() time, so handlers MUST
        be attached before start() calls connect().
        """
        client = self.client

        @client.event(MessageEv)
        async def on_message(_client: NewAClient, ev: MessageEv):
            try:
                await self._handle_message(ev)
            except Exception as e:
                logger.error(
                    "Unhandled error in WhatsApp on_message: %s: %s",
                    type(e).__name__, e, exc_info=True,
                )

        @client.event(ConnectedEv)
        async def on_connected(_client: NewAClient, _ev: ConnectedEv):
            me = getattr(_client, "me", None)
            self._own_pn = self._own_number(me)
            logger.info("WhatsApp connected (own number: %s)", self._own_pn or "?")

            # Only advertise the channel once it can actually send.
            register_channel(CHANNEL_WHATSAPP, self)

            if self.agent and self.agent.conversations:
                self.agent.conversations.add_delivery_callback(
                    self._deliver_subagent_result
                )
                self.agent.conversations.add_status_callback(
                    self._send_status_message
                )

        @client.event(DisconnectedEv)
        async def on_disconnected(_client: NewAClient, _ev: DisconnectedEv):
            # neonize/whatsmeow reconnects on its own; log and keep the channel
            # registered so a transient drop does not surface as "no WhatsApp".
            logger.warning("WhatsApp disconnected (whatsmeow will retry).")

        @client.event(LoggedOutEv)
        async def on_logged_out(_client: NewAClient, _ev: LoggedOutEv):
            logger.error(
                "WhatsApp session logged out — the pairing is gone. Re-pair by "
                "scanning the QR again."
            )
            unregister_channel(CHANNEL_WHATSAPP)

        @client.event(PairStatusEv)
        async def on_pair_status(_client: NewAClient, ev: PairStatusEv):
            logger.info("WhatsApp pair status: %s", ev)

        @client.qr
        async def on_qr(_client: NewAClient, qr: bytes):
            await self._deliver_qr(qr)

    def _own_number(self, me: Any) -> str:
        """Best-effort own phone number from the client's own JID."""
        for attr in ("JID", "jid"):
            jid = getattr(me, attr, None)
            user = getattr(jid, "User", None)
            if user:
                return str(user)
        return ""

    async def _deliver_qr(self, qr: bytes) -> None:
        """Persist the pairing QR where the owner can reach it.

        Written to disk rather than pushed to Telegram: delivering media
        cross-channel is a separate concern and a failure there must not break
        pairing. The path is logged at WARNING so it shows up in `syne logs`.
        """
        try:
            import segno  # bundled with neonize

            out_dir = os.path.expanduser("~/.syne")
            os.makedirs(out_dir, exist_ok=True)
            png_path = os.path.join(out_dir, "whatsapp_qr.png")
            txt = qr.decode("utf-8", errors="replace") if isinstance(qr, bytes) else str(qr)
            segno.make(txt).save(png_path, scale=6, border=2)
            logger.warning(
                "WhatsApp pairing required — scan this QR: %s "
                "(WhatsApp > Linked devices > Link a device)",
                png_path,
            )
        except Exception as e:
            logger.error("Failed to render WhatsApp QR: %s: %s", type(e).__name__, e)

    # ── JID helpers ──────────────────────────────────────────

    @staticmethod
    def _jid_str(jid: Any) -> str:
        """Render a neonize JID as 'user@server', without the device suffix.

        Device-scoped JIDs (628xxx:51@s.whatsapp.net) identify a phone/laptop,
        not a person, so the device part is dropped for identity purposes.
        """
        user = str(getattr(jid, "User", "") or "")
        server = str(getattr(jid, "Server", "") or "")
        if not user:
            return server
        return f"{user}@{server}" if server else user

    @staticmethod
    def _local(jid_or_str: Any) -> str:
        """The bare local part (phone number or group id) of a JID."""
        if isinstance(jid_or_str, str):
            return jid_or_str.split(":", 1)[0].split("@", 1)[0]
        return str(getattr(jid_or_str, "User", "") or "").split(":", 1)[0]

    def _to_jid(self, jid_str: str):
        """Parse 'user@server' back into a neonize JID object.

        neonize's send_image/send_document/leave_group take a JID, not a
        string, so this is required for anything beyond send_message.
        """
        raw = (jid_str or "").strip()
        user, _, server = raw.partition("@")
        user = user.split(":", 1)[0]
        return build_jid(user, server or _SERVER_DM)

    async def _fetch_group_subject(self, chat_jid: str) -> str:
        """The real group subject from WhatsApp, or '' when unavailable.

        The `groups` row does not exist yet the first time a new group is
        seen, so deriving the name from the JID alone labels the group with
        its raw digits ("120363433706421292"). whatsmeow knows the actual
        subject, so ask it. Best-effort by design: a failure here must never
        block an inbound message, hence the broad except and the '' return
        that lets the caller fall back.
        """
        client = getattr(self, "client", None)
        if client is None:
            return ""
        try:
            info = await client.get_group_info(self._to_jid(chat_jid))
            return (info.GroupName.Name or "").strip()
        except Exception as e:
            logger.debug(
                "[whatsapp] get_group_info failed for %s: %s: %s",
                chat_jid, type(e).__name__, e,
            )
            return ""

    async def _resolve_lid_pn(self, jid_like: Any) -> str:
        """Phone number behind an @lid JID, or '' when unknown.

        whatsmeow keeps its own lid->pn mapping in the table
        whatsmeow_lid_map of Syne's database (the store is Postgres, see
        _default_store). That table belongs to whatsmeow and is NOT part of
        schema.sql, so it is simply absent on an install that has never
        paired — every failure here is therefore swallowed and reported as
        "unknown", never raised.

        Needed because MessageSource.SenderAlt is not always populated: a
        self-chat message was observed in production arriving as
        176295656874158@lid with no SenderAlt at all, which registered the
        owner under their lid instead of their phone number.
        """
        raw = jid_like if isinstance(jid_like, str) else self._jid_str(jid_like)
        if not raw or not raw.endswith(f"@{_SERVER_LID}"):
            return ""
        lid = self._local(raw)
        if not lid:
            return ""
        cached = self._lid_pn.get(lid)
        if cached:
            return cached

        try:
            from ..db.connection import get_connection
            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT pn FROM whatsmeow_lid_map WHERE lid = $1", lid
                )
        except Exception as e:
            logger.debug("[whatsapp] lid map unavailable for %s: %s", lid, e)
            return ""

        pn = self._local(str(row["pn"] or "")) if row else ""
        if pn and len(self._lid_pn) < _LID_CACHE_MAX:
            self._lid_pn[lid] = pn
        return pn

    # ── Trigger word (ported verbatim from the wacli bridge) ──

    def _has_trigger_word(self, text: str, trigger: str) -> bool:
        """True if `text` addresses the bot by name.

        An empty trigger means "no trigger configured" and admits everything;
        an empty text never matches a real trigger.
        """
        if not trigger:
            return True
        try:
            return re.search(
                rf"\b{re.escape(trigger)}\b", text or "", re.IGNORECASE
            ) is not None
        except Exception:
            return False

    def _strip_trigger_word(self, text: str, trigger: str) -> str:
        """Strip a leading trigger word (e.g. 'Molt:', 'molt,')."""
        if not trigger:
            return (text or "").strip()
        try:
            return re.sub(
                rf"^\s*{re.escape(trigger)}[,:]?\s*", "", text or "",
                flags=re.IGNORECASE,
            ).strip()
        except Exception:
            return (text or "").strip()

    async def _get_trigger_name(self) -> str:
        """Trigger name from config, falling back to identity (as Telegram)."""
        from ..db.models import get_identity

        trigger = await get_config("whatsapp.trigger_name", None)
        if not trigger:
            trigger = await get_config("telegram.bot_trigger_name", None)
        if trigger:
            return str(trigger)
        identity = await get_identity()
        return identity.get("name", "Syne")

    # ── Member allowlist (config whatsapp.allowed_chat_jids) ──

    async def _load_members(self) -> dict:
        """Load WA members from config. Returns {key: entry}.

        Entry shape: {'jid', 'name', 'model', 'blocked'}. Deliberately the SAME
        config key and shape /wamembers reads and writes, so existing entries
        keep working untouched — there is no migration. Access level is NOT
        stored here: it lives on users.access_level (DM) and
        groups.settings['members'] (group member), and absent means "public".
        """
        import json as _json

        raw = await get_config("whatsapp.allowed_chat_jids", default=[])
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except Exception:
                raw = [x.strip() for x in raw.split(",") if x.strip()]

        by_jid: dict[str, dict] = {}
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    jid = str(item.get("jid") or "").strip()
                    if jid:
                        by_jid[jid] = {
                            "jid": jid,
                            "name": str(item.get("name") or "").strip(),
                            "model": item.get("model") or None,
                            "blocked": bool(item.get("blocked", False)),
                        }
                else:
                    jid = str(item).strip()
                    if jid:
                        by_jid[jid] = {
                            "jid": jid, "name": "", "model": None, "blocked": False,
                        }
        return by_jid

    async def _save_members(self, by_jid: dict) -> None:
        await set_config("whatsapp.allowed_chat_jids", list(by_jid.values()))

    def _match_member(self, by_jid: dict, chat_jid: str, alt_pn: str = "") -> tuple:
        """Find a member entry for this chat. Returns (key, entry) or (None, None).

        Tries the full JID, the bare local part, and — for an @lid chat — the
        phone number resolved for it, which the caller passes as alt_pn:
        MessageSource.SenderAlt when whatsmeow fills it in, otherwise the
        whatsmeow_lid_map lookup done by _resolve_lid_pn. SenderAlt alone is
        NOT enough — it was seen empty on a self-chat @lid message — and this
        last candidate is what makes a member already on the allowlist under
        their phone number match an @lid chat instead of being registered a
        second time under the lid.
        """
        local = self._local(chat_jid)
        candidates = [chat_jid, local]
        if alt_pn:
            candidates += [alt_pn, f"{alt_pn}@{_SERVER_DM}"]
        for cand in candidates:
            if cand and cand in by_jid:
                return cand, by_jid[cand]
        return None, None

    # ── Inbound ──────────────────────────────────────────────

    def _is_duplicate(self, msg_id: str) -> bool:
        """True if this message id was already processed (bounded memory)."""
        if not msg_id:
            return False
        if msg_id in self._seen:
            return True
        self._seen.add(msg_id)
        self._seen_order.append(msg_id)
        if len(self._seen_order) > _SEEN_MAX:
            self._seen.discard(self._seen_order.pop(0))
        return False

    def _content_hash(self, chat_jid: str, text: str) -> str:
        return f"{chat_jid}:{hashlib.md5((text or '').encode()).hexdigest()[:12]}"

    @staticmethod
    def _extract_content(ev: MessageEv) -> tuple[str, str, str]:
        """Pull (text, caption, media_kind) out of a message event.

        Reads the protobuf fields directly instead of trusting Info.MediaType:
        HasField is authoritative about what the message actually carries.
        media_kind is '' for a plain text message.
        """
        msg = ev.Message
        text = ""
        caption = ""
        kind = ""

        # `conversation` carries a plain text body. It sits inside the message
        # oneof, so HasField IS legal on it — verified against the generated
        # stub, whose HasField literal lists "conversation" alongside
        # imageMessage / extendedTextMessage. The except arm is belt-and-braces
        # for a future protobuf where it leaves the oneof and HasField raises.
        try:
            if msg.HasField("conversation"):
                text = msg.conversation or ""
        except (ValueError, AttributeError):
            text = getattr(msg, "conversation", "") or ""

        try:
            if msg.HasField("extendedTextMessage"):
                text = msg.extendedTextMessage.text or text
            if msg.HasField("imageMessage"):
                kind = "image"
                caption = msg.imageMessage.caption or ""
            elif msg.HasField("videoMessage"):
                kind = "video"
                caption = msg.videoMessage.caption or ""
            elif msg.HasField("audioMessage"):
                kind = "audio"
            elif msg.HasField("documentMessage"):
                kind = "document"
                caption = msg.documentMessage.caption or ""
        except Exception:
            pass

        return (text or "").strip(), (caption or "").strip(), kind

    async def _handle_message(self, ev: MessageEv) -> None:
        """Process one inbound WhatsApp message."""
        src = ev.Info.MessageSource
        chat_jid = self._jid_str(src.Chat)
        sender_jid = self._jid_str(src.Sender)
        # whatsmeow USUALLY fills SenderAlt with the real phone number when
        # Sender is a privacy-preserving @lid — but not always: a self-chat
        # message arrived in production as 176295656874158@lid with SenderAlt
        # empty, and the lid then became the member key. The phone number is
        # the canonical identity, so fall back to whatsmeow's own lid map.
        # Both calls no-op on a non-lid JID, so an ordinary DM or group
        # message costs nothing.
        alt_pn = self._local(src.SenderAlt) if src.HasField("SenderAlt") else ""
        if not alt_pn:
            alt_pn = (await self._resolve_lid_pn(sender_jid)
                      or await self._resolve_lid_pn(chat_jid))
        from_me = bool(src.IsFromMe)
        is_group = bool(src.IsGroup)
        msg_id = ev.Info.ID

        text, caption, media_kind = self._extract_content(ev)
        if not text and not caption and not media_kind:
            # Protocol-only payload — e.g. the sender-key distribution that
            # rides along with the FIRST message in a new group. whatsmeow
            # dispatches one event per <enc> child, so that first message
            # arrives twice with the SAME id: once as this empty carrier,
            # once as the real text. Returning BEFORE _is_duplicate is
            # deliberate: marking the id as seen here would make the real
            # text look like a duplicate and silently drop it.
            return

        if self._is_duplicate(msg_id):
            logger.debug("Dropping duplicate WhatsApp message %s", msg_id)
            return

        logger.info(
            "[whatsapp] inbound: from_me=%s group=%s chat=%s media=%s text=%s",
            from_me, is_group, chat_jid, media_kind or "-", (text or caption)[:60],
        )

        # from_me rules — ported verbatim (abilities/whatsapp.py:753-771).
        if from_me:
            if is_group:
                logger.info("[whatsapp] drop: from_me in group (%s)", chat_jid)
                return
            if not sender_jid:
                logger.info("[whatsapp] drop: from_me with no sender")
                return
            if self._local(sender_jid) != self._local(chat_jid):
                logger.info(
                    "[whatsapp] drop: from_me to other contact (%s != %s)",
                    self._local(sender_jid), self._local(chat_jid),
                )
                return

        # Echo suppression: our own replies come back as from_me, and in a
        # self-chat the rules above let them through, so match on content. This
        # is what lets the owner use the trigger in the self-chat without the
        # reply re-triggering the bot.
        probe = text or caption
        if probe:
            h = self._content_hash(chat_jid, probe)
            now = time.time()
            if h in self._echo and (now - self._echo[h]) < _ECHO_TTL:
                del self._echo[h]
                logger.debug("[whatsapp] echo suppressed: %s", probe[:60])
                return

        # The trigger word is MANDATORY everywhere — group and DM alike. The
        # groups row is read ONLY to honour `enabled`; require_mention
        # deliberately does NOT relax the trigger requirement.
        trigger = await self._get_trigger_name()
        group_row = None
        if is_group:
            group_row = await get_group("whatsapp", chat_jid)
            if group_row is not None and not group_row.get("enabled", True):
                logger.info("[whatsapp] drop: group disabled (%s)", chat_jid)
                return

        if media_kind:
            if not is_group and from_me:
                # Self-chat media needs no trigger — there is no one else being
                # addressed in a chat with yourself.
                text = caption or ""
            elif self._has_trigger_word(caption, trigger):
                text = self._strip_trigger_word(caption, trigger)
            else:
                logger.info(
                    "[whatsapp] %s without trigger, skipping (group=%s)",
                    media_kind, is_group,
                )
                return
        else:
            if not self._has_trigger_word(text, trigger):
                logger.info('[whatsapp] no trigger "%s" in: %s', trigger, text[:80])
                return
            text = self._strip_trigger_word(text, trigger)
            if not text:
                return

        # Members: blocked check + auto-register.
        by_jid = await self._load_members()
        push_name = ev.Info.Pushname or ""

        if is_group:
            member_key, member = self._match_member(by_jid, chat_jid)
            chat_local = chat_jid.split("@")[0]
            group_name = ((group_row or {}).get("name") or "").strip()
            # A stored name equal to the JID digits is the old fallback, not a
            # real subject, so treat it as missing and heal it.
            if not group_name or group_name == chat_local:
                group_name = await self._fetch_group_subject(chat_jid) or chat_local
                if (group_row is not None
                        and group_name != (group_row.get("name") or "")):
                    try:
                        await update_group(
                            "whatsapp", chat_jid, name=group_name
                        )
                        logger.info(
                            "[whatsapp] group name resolved: %s (%s)",
                            group_name, chat_jid,
                        )
                    except Exception as e:
                        logger.debug(
                            "[whatsapp] update_group name failed for %s: %s",
                            chat_jid, e,
                        )

            if member and member.get("blocked"):
                logger.info("[whatsapp] blocked group, dropping: %s", chat_jid)
                return
            if not member:
                member = {
                    "jid": chat_jid, "name": group_name,
                    "model": None, "blocked": False,
                }
                by_jid[chat_jid] = member
                await self._save_members(by_jid)
                logger.info(
                    "[whatsapp] auto-registered group: %s (%s)", group_name, chat_jid
                )
            elif ((member.get("name") or "").strip() in ("", chat_local)
                    and group_name != chat_local):
                # Entry registered before the subject was resolvable — /wamembers
                # reads this field, so repair it in place.
                member["name"] = group_name
                by_jid[member_key or chat_jid] = member
                await self._save_members(by_jid)
                logger.info(
                    "[whatsapp] group name healed in allowlist: %s (%s)",
                    group_name, chat_jid,
                )

            # A groups row is what carries enabled / require_mention / members,
            # and what load_group_settings reads for per-member access. Create
            # it on first sight so the group shows up manageable.
            if group_row is None:
                try:
                    group_row = await create_group(
                        platform="whatsapp",
                        platform_group_id=chat_jid,
                        name=group_name,
                        enabled=True,
                        require_mention=True,
                    )
                except Exception as e:
                    logger.debug("[whatsapp] create_group failed: %s", e)

            sender_pn = alt_pn or self._local(sender_jid)
            sender_name = push_name or sender_pn
            conversation_label = member.get("name") or group_name

            # Auto-collect the speaker into groups.settings['members'] — the
            # registry inbound.load_group_settings resolves access/alias from,
            # and the one /wamembers' member submenu lists. Name/username only,
            # so a manually assigned access or alias is never clobbered; a new
            # member defaults to "public" (models.py:444).
            try:
                await update_group_member(
                    platform="whatsapp",
                    platform_group_id=chat_jid,
                    member_id=sender_pn,
                    name=sender_name,
                )
            except Exception as e:
                logger.debug("[whatsapp] auto-collect member failed: %s", e)
        else:
            member_key, member = self._match_member(by_jid, chat_jid, alt_pn)
            chat_local = alt_pn or self._local(chat_jid)
            display = push_name or chat_local

            if member and member.get("blocked"):
                logger.info("[whatsapp] blocked member, dropping: %s", chat_jid)
                return
            if not member:
                reg_key = chat_local or chat_jid
                member = {
                    "jid": reg_key, "name": display,
                    "model": None, "blocked": False,
                }
                by_jid[reg_key] = member
                await self._save_members(by_jid)
                logger.info(
                    "[whatsapp] auto-registered new member: %s (%s)", display, reg_key
                )

            sender_pn = chat_local
            sender_name = member.get("name") or display
            conversation_label = sender_name

            # Persist the DM sender as "public" if they are new. create_user is
            # used directly rather than the get-or-create helper because that
            # helper can promote the first caller to OWNER (models.py:167-181);
            # ownership stays Telegram-only.
            try:
                if not await get_user("whatsapp", sender_pn):
                    await create_user(
                        name=sender_name or sender_pn,
                        platform="whatsapp",
                        platform_id=sender_pn,
                        display_name=sender_name or sender_pn,
                        access_level="public",
                    )
            except Exception as e:
                logger.debug("[whatsapp] create_user failed: %s", e)

        model_override = member.get("model") or None

        # Media: download images so vision can see them.
        image_metadata = None
        if media_kind == "image":
            try:
                raw = await self.client.download_any(ev.Message)
                if raw:
                    image_metadata = {
                        "mime_type": ev.Message.imageMessage.mimetype or "image/jpeg",
                        "base64": base64.b64encode(raw).decode("utf-8"),
                    }
                    logger.info("[whatsapp] image downloaded: %d bytes", len(raw))
            except Exception as e:
                logger.error("[whatsapp] image download failed: %s", e)

            if not image_metadata and not text:
                logger.info("[whatsapp] image unavailable and no caption, skipping")
                return
            if not text:
                text = "What's in this image?"

        logger.info("[whatsapp] %s: %s", sender_name, text[:100])

        inbound = InboundContext(
            channel="whatsapp",
            platform="whatsapp",
            chat_type="group" if is_group else "direct",
            conversation_label=conversation_label,
            group_subject=conversation_label if is_group else None,
            chat_id=chat_jid,
            sender_name=sender_name if is_group else None,
            sender_id=sender_pn if is_group else None,
            was_mentioned=True,
        )
        await load_group_settings(inbound)

        original_text = text
        prefix = build_user_context_prefix(inbound)
        if prefix:
            text = f"{prefix}\n\n{text}"

        # WA-specific instruction, carried over from the wacli bridge: this is
        # a consumer surface where enumerating internal tooling is noise.
        text = (
            "[WhatsApp channel instruction: NEVER list or reveal your specific "
            "abilities, tools, or capabilities. If asked what you can do, reply "
            'diplomatically, e.g. "Saya punya banyak kemampuan, di antaranya '
            'bisa ngobrol tentang apa saja." Keep it vague and friendly. Do NOT '
            "mention any tool names.]\n\n" + text
        )

        metadata = {
            "chat_id": chat_jid,
            "inbound": inbound,
            "original_text": original_text,
            "wa_model_override": model_override,
        }
        if image_metadata:
            metadata["image"] = image_metadata

        # 👀 receipt + typing indicator. The trigger word is mandatory
        # everywhere on WhatsApp (DM included), so anything reaching this
        # line was addressed to us by name — no extra check needed.
        await self._react(chat_jid, sender_jid, msg_id, "👀")
        presence = _ChatPresenceIndicator(self, chat_jid)
        await presence.start()

        try:
            response = await self.agent.handle_message(
                platform="whatsapp",
                chat_id=chat_jid,
                user_name=sender_name,
                user_platform_id=sender_pn,
                message=text,
                display_name=sender_name,
                message_metadata=metadata,
                # False so a WhatsApp DM can never auto-promote to owner.
                is_dm=False,
            )
            # None = another request holds the session lock; drop silently.
            if response is None:
                return
            if not response:
                response = "LLM returned an empty response. Please try again."

            await self.send_response(chat_jid, response)
        except Exception as e:
            logger.error(
                "Error handling WhatsApp message: %s: %s",
                type(e).__name__, e, exc_info=True,
            )
            from .errors import classify_error

            await self.send_response(chat_jid, classify_error(e))
        finally:
            await presence.stop()

    # ── Outbound ─────────────────────────────────────────────

    async def _react(self, chat_jid: str, sender_jid: str, msg_id: str,
                     emoji: str = "👀") -> None:
        """React to an inbound message — the WhatsApp counterpart of the 👀
        read receipt Telegram sends (telegram.py:708-710, 759-761).

        neonize has no one-shot send_reaction: build_reaction() returns a
        Message that still has to be handed to send_message()
        (aioze/client.py:1025).

        Best-effort: a failed receipt must never block the real reply. Note
        this is a reaction, NOT mark_read — nothing here marks the chat as
        read, and Syne never calls neonize's mark_read anywhere.
        """
        client = self.client
        if not client or not msg_id:
            return
        try:
            reaction = await client.build_reaction(
                self._to_jid(chat_jid), self._to_jid(sender_jid), msg_id, emoji,
            )
            await client.send_message(self._to_jid(chat_jid), reaction)
        except Exception as e:
            logger.debug("[whatsapp] reaction %r failed: %s", emoji, e)

    async def send_response(self, chat_jid: str, text: str) -> None:
        """Send an agent response, applying the shared outbound pipeline."""
        if not self.client:
            logger.warning("send_response called while WhatsApp is not connected")
            return

        # MEDIA: paths must be pulled BEFORE process_outbound strips server
        # paths. The first file takes the caption; the rest follow bare.
        text, media = extract_all_media(text or "")
        text = process_outbound(text)
        if not text and not media:
            return
        text = markdown_to_whatsapp(text)

        target = self._to_jid(chat_jid)
        caption_used = False

        for idx, path in enumerate(media):
            cap = text if (idx == 0 and text) else None
            try:
                await self._send_file(target, path, cap)
                if cap:
                    caption_used = True
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.error("WhatsApp file send failed for %s: %s", path, e)

        if text and not caption_used:
            for chunk in split_message(text, max_length=WHATSAPP_MAX_MESSAGE):
                if not chunk.strip():
                    continue
                try:
                    await self.client.send_message(target, chunk)
                    self._echo[self._content_hash(chat_jid, chunk)] = time.time()
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.error("WhatsApp send failed: %s: %s", type(e).__name__, e)

    async def _send_file(self, target: Any, path: str, caption: Optional[str]) -> None:
        """Send one attachment, routed by extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            await self.client.send_image(to=target, file=path, caption=caption)
        else:
            await self.client.send_document(
                to=target,
                file=path,
                caption=caption,
                filename=os.path.basename(path),
            )
        logger.info("[whatsapp] sent file: %s", path)

    # ── Used by the registry / cross-channel callers ─────────

    async def send_to_chat(self, chat_id: str, text: str) -> bool:
        """Send a message to a WhatsApp JID. Returns success.

        Entry point other channels reach through the registry, so it takes a
        plain JID string and never a neonize object.
        """
        if not self.client:
            logger.warning("send_to_chat called while WhatsApp is not connected")
            return False
        try:
            await self.send_response(chat_id, text)
            return True
        except Exception as e:
            logger.error("WhatsApp send_to_chat failed: %s: %s", type(e).__name__, e)
            return False

    async def leave_group(self, group_jid: str) -> tuple[bool, str]:
        """Leave a WhatsApp group. Returns (ok, message).

        Called by the Telegram /wamembers menu. Only leaves the chat; the
        member entry and groups row are the caller's business.
        """
        if not self.client:
            return False, "WhatsApp is not connected."
        if not group_jid.endswith(f"@{_SERVER_GROUP}"):
            return False, f"{group_jid} is not a group JID."
        try:
            await self.client.leave_group(self._to_jid(group_jid))
            logger.info("[whatsapp] left group %s", group_jid)
            return True, "Left the group."
        except Exception as e:
            logger.error("WhatsApp leave_group failed: %s: %s", type(e).__name__, e)
            return False, f"{type(e).__name__}: {e}"

    # ── Sub-agent callbacks ──────────────────────────────────

    async def _deliver_subagent_result(self, message: str, parent_session_id: int):
        """Deliver sub-agent results back to the WhatsApp chat that spawned it."""
        for key, conv in self.agent.conversations._active.items():
            if conv.session_id == parent_session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "whatsapp":
                    await self.send_response(parts[1], message)
                    return

    async def _send_status_message(self, session_id: int, message: str):
        """Send a status notification to the WhatsApp chat for a session."""
        for key, conv in self.agent.conversations._active.items():
            if conv.session_id == session_id:
                parts = key.split(":", 1)
                if len(parts) == 2 and parts[0] == "whatsapp":
                    await self.send_response(parts[1], message)
                    return
