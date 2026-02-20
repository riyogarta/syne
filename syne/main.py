"""Syne — Main entry point."""

import asyncio
import logging
import os
from typing import Optional

from .config import load_settings
from .agent import SyneAgent
from .channels.telegram import TelegramChannel
from .scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("syne")

# Global reference for scheduler callback
_telegram_channel: Optional[TelegramChannel] = None


async def _get_telegram_bot_token(settings) -> str | None:
    """Get Telegram bot token from DB (primary) or .env (fallback + migrate).
    
    Priority:
    1. Database (credential.telegram_bot_token)
    2. Environment/settings (.env SYNE_TELEGRAM_BOT_TOKEN) — auto-migrate to DB
    """
    from .db.credentials import get_telegram_bot_token, set_telegram_bot_token, has_credential, CRED_TELEGRAM_BOT_TOKEN
    
    # 1. Try database first
    db_token = await get_telegram_bot_token()
    if db_token:
        logger.debug("Using Telegram bot token from database.")
        return db_token
    
    # 2. Fall back to .env and migrate
    env_token = settings.telegram_bot_token
    if env_token:
        # Auto-migrate to DB
        await set_telegram_bot_token(env_token)
        logger.warning(
            "Bot token migrated from .env to DB. "
            "You can now remove SYNE_TELEGRAM_BOT_TOKEN from .env"
        )
        return env_token
    
    # No token found
    return None


async def _auto_migrate_google_oauth():
    """Auto-migrate Google OAuth credentials from file to DB if needed."""
    from .migrate_credentials import migrate_google_oauth
    try:
        await migrate_google_oauth()
    except Exception as e:
        logger.debug(f"Google OAuth migration check: {e}")


async def _scheduler_callback(task_id: int, payload: str, created_by: int):
    """Callback for scheduler — injects payload as message via Telegram.
    
    Special payloads:
        __syne_update_check__: Check for Syne updates and notify owner.
    
    Args:
        task_id: Scheduled task ID
        payload: Message payload to inject
        created_by: Telegram user ID of task creator
    """
    global _telegram_channel
    
    # ── System tasks ──
    if payload == "__syne_update_check__":
        try:
            from .update_checker import check_and_notify
            message = await check_and_notify()
            if message and _telegram_channel and created_by:
                await _telegram_channel._bot.send_message(
                    chat_id=created_by, text=message, parse_mode="Markdown",
                )
                logger.info(f"Update notification sent to {created_by}")
        except Exception as e:
            logger.error(f"Update check failed: {e}")
        return
    
    if not _telegram_channel:
        logger.warning(f"Scheduler: No Telegram channel for task {task_id}")
        return
    
    # Get the user's DM chat ID (same as user ID for DMs)
    chat_id = created_by
    if not chat_id:
        logger.warning(f"Scheduler: No created_by for task {task_id}")
        return
    
    logger.info(f"Scheduler: Executing task {task_id} for user {chat_id}")
    
    # Process the payload as if the user sent it
    try:
        await _telegram_channel.process_scheduled_message(chat_id, payload)
    except Exception as e:
        logger.error(f"Scheduler: Error executing task {task_id}: {e}", exc_info=True)


async def run():
    """Main run loop."""
    global _telegram_channel
    
    settings = load_settings()
    agent = SyneAgent(settings)
    channels = []
    scheduler = None

    try:
        await agent.start()
        
        # Auto-migrate Google OAuth after DB is ready
        await _auto_migrate_google_oauth()

        # Get Telegram bot token (DB first, then .env with auto-migrate)
        bot_token = await _get_telegram_bot_token(settings)
        
        # Start Telegram if configured
        if bot_token:
            telegram = TelegramChannel(agent, bot_token)
            await telegram.start()
            channels.append(telegram)
            _telegram_channel = telegram  # Store for scheduler callback
            logger.info("Telegram channel active.")
        else:
            logger.warning(
                "No Telegram bot token configured. "
                "Set via DB (credential.telegram_bot_token) or .env (SYNE_TELEGRAM_BOT_TOKEN)."
            )

        # Start scheduler
        scheduler = Scheduler(on_task_execute=_scheduler_callback)
        await scheduler.start()
        logger.info("Scheduler active.")

        # Keep alive
        logger.info("Syne is running. Press Ctrl+C to stop.")
        while agent._running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        # Stop scheduler
        if scheduler:
            await scheduler.stop()
        # Stop channels
        for ch in channels:
            await ch.stop()
        await agent.stop()


def main():
    """Entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
