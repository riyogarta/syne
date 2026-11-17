"""Process-wide maintenance gate (used while /restore replaces the database).

The Telegram restore gate only stopped NEW Telegram updates. WhatsApp,
Discord, remote nodes (gateway) and the scheduler kept running, and turns
that had already started kept writing to a database being dropped and
reloaded. This module is the single switch all of them check:

- ConversationManager.handle_message refuses new turns (every channel and
  the gateway pass through it) and tracks running turns in _turns.
- The scheduler skips its tick.
- begin() sets the flag, cancels every running turn and waits for them to
  finish before the caller starts the restore.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger("syne.maintenance")

NOTICE = "⏳ Database restore in progress — Syne will restart when it finishes."

_reason: Optional[str] = None
_turns: set = set()


def active() -> bool:
    return _reason is not None


def track(task: Optional[asyncio.Task]) -> None:
    if task is not None:
        _turns.add(task)


def untrack(task: Optional[asyncio.Task]) -> None:
    _turns.discard(task)


async def begin(reason: str, timeout: float = 30.0) -> int:
    """Raise the gate, cancel running turns and wait for them (bounded).

    Returns how many turns were cancelled. The calling task itself is never
    cancelled.
    """
    global _reason
    _reason = reason
    me = asyncio.current_task()
    running = [t for t in list(_turns) if t is not me and not t.done()]
    for t in running:
        t.cancel()
    if running:
        done, pending = await asyncio.wait(running, timeout=timeout)
        if pending:
            logger.warning(f"Maintenance: {len(pending)} turn(s) still running after {timeout:.0f}s")
    logger.warning(f"Maintenance ON ({reason}): cancelled {len(running)} running turn(s)")
    return len(running)


def end() -> None:
    global _reason
    if _reason is not None:
        logger.warning(f"Maintenance OFF ({_reason})")
    _reason = None
