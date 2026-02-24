"""Scheduler — Background task scheduler with DB persistence.

Runs as an asyncio background task alongside the Telegram bot.
Checks for due tasks every 30 seconds and executes them.

Task types:
- 'once': Execute once at specified time, then disable
- 'interval': Execute every N seconds
- 'cron': Execute based on cron expression (requires croniter)
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Awaitable

logger = logging.getLogger("syne.scheduler")

# Check interval in seconds
_CHECK_INTERVAL = 30


def _parse_cron_next(cron_expr: str, from_time: datetime) -> Optional[datetime]:
    """Parse cron expression and calculate next run time.
    
    Args:
        cron_expr: Cron expression (e.g., "0 9 * * *" for 9 AM daily)
        from_time: Calculate next run from this time
        
    Returns:
        Next run datetime or None if parsing fails
    """
    try:
        from croniter import croniter
        cron = croniter(cron_expr, from_time)
        return cron.get_next(datetime)
    except ImportError:
        logger.error("croniter not installed. Run: pip install croniter")
        return None
    except Exception as e:
        logger.error(f"Invalid cron expression '{cron_expr}': {e}")
        return None


def _calculate_next_run(
    schedule_type: str,
    schedule_value: str,
    from_time: Optional[datetime] = None,
) -> Optional[datetime]:
    """Calculate next run time based on schedule type.
    
    Args:
        schedule_type: 'once', 'interval', or 'cron'
        schedule_value: Value depends on type:
            - once: ISO timestamp
            - interval: seconds as string
            - cron: cron expression
        from_time: Calculate from this time (default: now)
        
    Returns:
        Next run datetime (timezone-aware UTC) or None
    """
    now = from_time or datetime.now(timezone.utc)
    
    if schedule_type == "once":
        # Parse ISO timestamp
        try:
            dt = datetime.fromisoformat(schedule_value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception as e:
            logger.error(f"Invalid ISO timestamp '{schedule_value}': {e}")
            return None
    
    elif schedule_type == "interval":
        # Interval in seconds
        try:
            seconds = int(schedule_value)
            return now + timedelta(seconds=seconds)
        except ValueError:
            logger.error(f"Invalid interval value '{schedule_value}'")
            return None
    
    elif schedule_type == "cron":
        return _parse_cron_next(schedule_value, now)
    
    else:
        logger.error(f"Unknown schedule type: {schedule_type}")
        return None


class Scheduler:
    """Background task scheduler.
    
    Usage:
        scheduler = Scheduler(on_task_execute=my_callback)
        await scheduler.start()
        # ... later ...
        await scheduler.stop()
    """
    
    def __init__(
        self,
        on_task_execute: Callable[[int, str, int], Awaitable[None]],
    ):
        """Initialize scheduler.
        
        Args:
            on_task_execute: Async callback when task executes.
                Args: (task_id, payload, created_by)
                The callback should inject payload as user message.
        """
        self._on_execute = on_task_execute
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the scheduler background task."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")
    
    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_execute()
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
            
            await asyncio.sleep(_CHECK_INTERVAL)
    
    async def _check_and_execute(self):
        """Check for due tasks and execute them."""
        from .db.connection import get_connection
        
        now = datetime.now(timezone.utc)
        
        async with get_connection() as conn:
            # Get all due tasks
            due_tasks = await conn.fetch(
                """
                SELECT id, name, schedule_type, schedule_value, payload, created_by
                FROM scheduled_tasks
                WHERE enabled = true
                  AND next_run <= $1
                ORDER BY next_run ASC
                """,
                now,
            )
            
            for task in due_tasks:
                task_id = task["id"]
                task_name = task["name"]
                schedule_type = task["schedule_type"]
                schedule_value = task["schedule_value"]
                payload = task["payload"]
                created_by = task["created_by"]
                
                logger.info(f"Executing scheduled task: {task_name} (id={task_id})")
                
                try:
                    # Execute the callback
                    await self._on_execute(task_id, payload, created_by)
                    
                    # Update task based on type
                    if schedule_type == "once":
                        # Delete one-time tasks after execution — no reason to keep them
                        await conn.execute(
                            "DELETE FROM scheduled_tasks WHERE id = $1",
                            task_id,
                        )
                        logger.info(f"Deleted one-time task {task_id} ({task_name}) after execution")
                    else:
                        # Calculate next run for interval/cron
                        next_run = _calculate_next_run(schedule_type, schedule_value, now)
                        
                        if next_run:
                            await conn.execute(
                                """
                                UPDATE scheduled_tasks
                                SET last_run = $1,
                                    next_run = $2,
                                    run_count = run_count + 1
                                WHERE id = $3
                                """,
                                now, next_run, task_id,
                            )
                        else:
                            # Failed to calculate next run — disable
                            logger.error(f"Failed to calculate next run for task {task_id}, disabling")
                            await conn.execute(
                                """
                                UPDATE scheduled_tasks
                                SET enabled = false,
                                    last_run = $1,
                                    run_count = run_count + 1
                                WHERE id = $2
                                """,
                                now, task_id,
                            )
                
                except Exception as e:
                    logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
                    # Don't disable on error — will retry next interval


# ═══════════════════════════════════════════════════════════════
# DATABASE OPERATIONS
# ═══════════════════════════════════════════════════════════════

async def create_task(
    name: str,
    schedule_type: str,
    schedule_value: str,
    payload: str,
    created_by: Optional[int] = None,
) -> dict:
    """Create a new scheduled task.
    
    Args:
        name: Task name (for identification)
        schedule_type: 'once', 'interval', or 'cron'
        schedule_value: Value for schedule type
        payload: Message to inject when task runs
        created_by: Telegram user ID of creator
        
    Returns:
        Created task dict or error dict
    """
    from .db.connection import get_connection
    
    # Validate schedule type
    if schedule_type not in ("once", "interval", "cron"):
        return {"success": False, "error": f"Invalid schedule_type: {schedule_type}"}
    
    # Calculate initial next_run
    next_run = _calculate_next_run(schedule_type, schedule_value)
    if not next_run:
        return {"success": False, "error": f"Invalid schedule_value for {schedule_type}: {schedule_value}"}
    
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO scheduled_tasks (name, schedule_type, schedule_value, payload, created_by, next_run)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, name, schedule_type, schedule_value, payload, enabled, next_run, created_at
            """,
            name, schedule_type, schedule_value, payload, created_by, next_run,
        )
    
    return {
        "success": True,
        "task": dict(row),
    }


async def list_tasks(enabled_only: bool = False) -> list[dict]:
    """List all scheduled tasks.
    
    Args:
        enabled_only: If True, only return enabled tasks
        
    Returns:
        List of task dicts
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        if enabled_only:
            rows = await conn.fetch(
                """
                SELECT id, name, schedule_type, schedule_value, payload, enabled,
                       last_run, next_run, run_count, created_by, created_at
                FROM scheduled_tasks
                WHERE enabled = true
                ORDER BY next_run ASC
                """
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, name, schedule_type, schedule_value, payload, enabled,
                       last_run, next_run, run_count, created_by, created_at
                FROM scheduled_tasks
                ORDER BY created_at DESC
                """
            )
    
    return [dict(row) for row in rows]


async def get_task(task_id: int) -> Optional[dict]:
    """Get a task by ID.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task dict or None
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, schedule_type, schedule_value, payload, enabled,
                   last_run, next_run, run_count, created_by, created_at
            FROM scheduled_tasks
            WHERE id = $1
            """,
            task_id,
        )
    
    return dict(row) if row else None


async def delete_task(task_id: int) -> bool:
    """Delete a task by ID.
    
    Args:
        task_id: Task ID
        
    Returns:
        True if deleted, False if not found
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM scheduled_tasks WHERE id = $1",
            task_id,
        )
    
    return "DELETE 1" in result


async def enable_task(task_id: int) -> bool:
    """Enable a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        True if updated, False if not found
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        result = await conn.execute(
            "UPDATE scheduled_tasks SET enabled = true WHERE id = $1",
            task_id,
        )
    
    return "UPDATE 1" in result


async def disable_task(task_id: int) -> bool:
    """Disable a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        True if updated, False if not found
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        result = await conn.execute(
            "UPDATE scheduled_tasks SET enabled = false WHERE id = $1",
            task_id,
        )
    
    return "UPDATE 1" in result


async def delete_task_by_name(name: str) -> bool:
    """Delete a task by name.
    
    Args:
        name: Task name
        
    Returns:
        True if deleted, False if not found
    """
    from .db.connection import get_connection
    
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM scheduled_tasks WHERE name = $1",
            name,
        )
    
    return "DELETE" in result and "DELETE 0" not in result
