"""Channel registry — process-local lookup of live communication channels.

WHY THIS EXISTS
---------------
Syne's data model has always been multi-channel: `users` is UNIQUE(platform,
platform_id) and `groups` carries a `platform` column. The *runtime*, however,
was not. Each outbound helper kept its own module-level reference to the one
channel that existed:

    tools/send_message.py:15   _telegram_channel = None
    tools/reactions.py:47      _telegram_channel = None
    tools/voice.py:17          _telegram_channel = None
    main.py:35                 _telegram_channel: Optional[TelegramChannel] = None

That works for a single channel and breaks down the moment an action has to
cross channels — e.g. an approval prompt is delivered on Telegram (where the
owner is) but the resulting action (leaving a rejected Discord guild) must be
executed by the Discord channel. There is currently no way to reach channel B
from channel A.

This module is that missing piece: a tiny name -> channel-object map.

DESIGN CONSTRAINTS (deliberate)
-------------------------------
1. No imports of any channel module. The registry must never depend on
   Telegram, Discord, or WhatsApp — it is pure mechanism. This also keeps it
   free of import cycles (channels import the registry, never the reverse) and
   means a missing third-party lib can't break registry import.
2. No side effects at import time. Definitions only.
3. `get_channel()` returns None when absent instead of raising. Every existing
   caller already follows the `if not channel: ...` pattern, so returning None
   lets this slot in without changing anyone's control flow.
4. No approval / notification / routing policy lives here. The registry answers
   exactly one question — "is a channel named X live, and what is its object?"
   Policy belongs to the channel that owns it.

ADDITIVE BY DESIGN
------------------
Registering is opt-in. While no channel registers and no caller looks up, this
module is inert: the existing Telegram and WhatsApp globals keep working
untouched. It adds a capability; it removes nothing.

Not thread-safe by design — Syne runs channels on a single asyncio event loop,
so a plain dict is correct and cheaper than a lock. Do not call these from a
non-loop thread.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("syne.communication.registry")

# Canonical channel names. These intentionally match the `platform` values used
# in the users/groups tables (see schema.sql) so a DB row's platform can be used
# directly as a registry key without translation.
CHANNEL_TELEGRAM = "telegram"
CHANNEL_DISCORD = "discord"
CHANNEL_WHATSAPP = "whatsapp"

# name -> live channel object
_channels: dict[str, Any] = {}


def register_channel(name: str, channel: Any) -> None:
    """Register a live channel object under `name`.

    Called by a channel once it is up and able to send. Re-registering the same
    name replaces the previous object (normal on reconnect) and is logged,
    because a silent swap would be painful to debug.

    Args:
        name: Canonical channel name, e.g. "telegram" / "discord".
        channel: The live channel instance.
    """
    if not name:
        logger.warning("register_channel called with empty name; ignored")
        return

    if name in _channels and _channels[name] is not channel:
        logger.info("Channel %r re-registered (object replaced)", name)
    else:
        logger.info("Channel %r registered", name)

    _channels[name] = channel


def unregister_channel(name: str) -> None:
    """Remove a channel from the registry. Safe to call when absent."""
    if _channels.pop(name, None) is not None:
        logger.info("Channel %r unregistered", name)


def get_channel(name: str) -> Optional[Any]:
    """Return the live channel object for `name`, or None if not registered.

    Returns None rather than raising: callers are expected to degrade
    gracefully, matching the existing `if not channel:` guards.
    """
    return _channels.get(name)


def has_channel(name: str) -> bool:
    """True if a channel is registered under `name`."""
    return name in _channels


def list_channels() -> list[str]:
    """Names of all currently registered channels, sorted. Useful for /status."""
    return sorted(_channels)


def clear_channels() -> None:
    """Drop all registrations. Intended for tests and clean shutdown."""
    _channels.clear()
    logger.debug("Channel registry cleared")
