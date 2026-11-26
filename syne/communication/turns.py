"""Per-chat registry of the turns a channel is running or has queued.

An edited message replaces its old version: the channel cancels the turn
still working on (or queued with) the old text and processes the new one.
Used by the WhatsApp and Discord channels; Telegram keeps its own
_active_tasks with the same shape.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Hashable, Optional


def edit_marker(message_id) -> str:
    """Line put in front of an edited message's text for the model."""
    return (
        f"[Edited message: this replaces the earlier version of message "
        f"{message_id}. Answer this version.]"
    )


class TurnTracker:
    """{chat: {message_id: task}} in arrival order."""

    def __init__(self) -> None:
        self._turns: Dict[Hashable, Dict[Hashable, asyncio.Task]] = {}

    def register(self, chat: Hashable, message_id: Hashable) -> Optional[asyncio.Task]:
        """Record the current task as the turn of message_id."""
        task = asyncio.current_task()
        if task is not None:
            self._turns.setdefault(chat, {})[message_id] = task
        return task

    def cancel(self, chat: Hashable, message_id: Hashable) -> bool:
        """Cancel the turn of message_id (running or queued). Other turns of
        the chat are left alone. False if there was nothing to cancel."""
        turns = self._turns.get(chat) or {}
        task = turns.get(message_id)
        if task is None or task.done() or task is asyncio.current_task():
            return False
        task.cancel()
        turns.pop(message_id, None)
        return True

    def drop(self, chat: Hashable, message_id: Hashable, task) -> None:
        """Forget a finished turn, unless a newer turn (an edit) took its slot."""
        turns = self._turns.get(chat)
        if not turns:
            return
        if turns.get(message_id) is task:
            turns.pop(message_id, None)
        if not turns:
            self._turns.pop(chat, None)

    def tasks(self, chat: Hashable) -> list:
        return list((self._turns.get(chat) or {}).values())
