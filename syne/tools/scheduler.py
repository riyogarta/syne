"""manage_schedule — Create and manage scheduled tasks (owner-only).

Rule 700: Only the owner can use this tool.

Task types:
- 'once': Run once at a specific time (ISO timestamp)
- 'interval': Run every N seconds
- 'cron': Run based on cron expression (e.g., "0 9 * * *" for 9 AM daily)

When a task executes, its payload is injected as a user message
to the conversation.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("syne.tools.scheduler")

# Current user context — set by conversation layer before tool execution.
# Used to auto-fill created_by when creating tasks.
_current_user_platform_id: Optional[int] = None


def set_current_user(platform_id: Optional[int]) -> None:
    """Set the current user's platform ID for task creation.
    
    Called by conversation layer before executing tools, so that
    manage_schedule can auto-fill created_by without relying on
    the LLM to pass it.
    """
    global _current_user_platform_id
    _current_user_platform_id = platform_id


async def manage_schedule_handler(
    action: str,
    name: str = "",
    task_id: int = 0,
    schedule_type: str = "",
    schedule_value: str = "",
    payload: str = "",
) -> str:
    """Handle manage_schedule tool calls.
    
    Args:
        action: create, list, delete, enable, disable, get
        name: Task name (for create/delete_by_name)
        task_id: Task ID (for delete/enable/disable/get)
        schedule_type: 'once', 'interval', or 'cron' (for create)
        schedule_value: Schedule value (for create):
            - once: ISO timestamp (e.g., "2026-02-20T09:00:00+07:00")
            - interval: seconds (e.g., "3600" for hourly)
            - cron: cron expression (e.g., "0 9 * * *")
        payload: Message to inject when task runs (for create)
        
    Returns:
        Result message
    """
    from ..scheduler import (
        create_task,
        list_tasks,
        get_task,
        delete_task,
        delete_task_by_name,
        enable_task,
        disable_task,
    )
    
    if action == "create":
        if not name:
            return "Error: name is required for create action."
        if not schedule_type:
            return "Error: schedule_type is required (once, interval, cron)."
        if not schedule_value:
            return "Error: schedule_value is required."
        if not payload:
            return "Error: payload is required (message to inject when task runs)."
        
        result = await create_task(
            name=name,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            payload=payload,
            created_by=_current_user_platform_id,
        )
        
        if not result["success"]:
            return f"Error: {result['error']}"
        
        task = result["task"]
        next_run = task.get("next_run")
        if next_run and isinstance(next_run, datetime):
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            next_run_str = str(next_run)
        
        return (
            f"✅ Task created:\n"
            f"- ID: {task['id']}\n"
            f"- Name: {task['name']}\n"
            f"- Type: {task['schedule_type']}\n"
            f"- Schedule: {task['schedule_value']}\n"
            f"- Next run: {next_run_str}\n"
            f"- Payload: {task['payload'][:100]}{'...' if len(task['payload']) > 100 else ''}"
        )
    
    elif action == "list":
        tasks = await list_tasks()
        
        if not tasks:
            return "No scheduled tasks."
        
        lines = ["**Scheduled Tasks:**"]
        for t in tasks:
            status = "✅" if t["enabled"] else "❌"
            next_run = t.get("next_run")
            if next_run and isinstance(next_run, datetime):
                next_run_str = next_run.strftime("%m/%d %H:%M")
            else:
                next_run_str = str(next_run) if next_run else "N/A"
            
            lines.append(
                f"- {status} **{t['name']}** (id={t['id']})\n"
                f"  Type: {t['schedule_type']} | Next: {next_run_str} | Runs: {t['run_count']}"
            )
        
        return "\n".join(lines)
    
    elif action == "get":
        if not task_id and not name:
            return "Error: task_id or name is required for get action."
        
        if task_id:
            task = await get_task(task_id)
        else:
            # Find by name
            tasks = await list_tasks()
            task = next((t for t in tasks if t["name"] == name), None)
        
        if not task:
            return f"Task not found: {task_id or name}"
        
        next_run = task.get("next_run")
        last_run = task.get("last_run")
        created_at = task.get("created_at")
        
        def fmt_dt(dt):
            if dt and isinstance(dt, datetime):
                return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            return str(dt) if dt else "Never"
        
        return (
            f"**Task: {task['name']}** (id={task['id']})\n"
            f"- Status: {'Enabled' if task['enabled'] else 'Disabled'}\n"
            f"- Type: {task['schedule_type']}\n"
            f"- Schedule: {task['schedule_value']}\n"
            f"- Next run: {fmt_dt(next_run)}\n"
            f"- Last run: {fmt_dt(last_run)}\n"
            f"- Run count: {task['run_count']}\n"
            f"- Created: {fmt_dt(created_at)}\n"
            f"- Payload:\n```\n{task['payload']}\n```"
        )
    
    elif action == "delete":
        if not task_id and not name:
            return "Error: task_id or name is required for delete action."
        
        if task_id:
            success = await delete_task(task_id)
            identifier = f"id={task_id}"
        else:
            success = await delete_task_by_name(name)
            identifier = f"name={name}"
        
        if success:
            return f"✅ Task deleted ({identifier})"
        return f"Task not found ({identifier})"
    
    elif action == "enable":
        if not task_id:
            return "Error: task_id is required for enable action."
        
        success = await enable_task(task_id)
        if success:
            return f"✅ Task {task_id} enabled"
        return f"Task not found: {task_id}"
    
    elif action == "disable":
        if not task_id:
            return "Error: task_id is required for disable action."
        
        success = await disable_task(task_id)
        if success:
            return f"✅ Task {task_id} disabled"
        return f"Task not found: {task_id}"
    
    else:
        return f"Unknown action: {action}. Use: create, list, get, delete, enable, disable"


# ── Tool Registration Dict ──

MANAGE_SCHEDULE_TOOL = {
    "name": "manage_schedule",
    "description": (
        "Create and manage scheduled tasks. Tasks execute by injecting payload as user message. "
        "Actions: create (new task), list (all tasks), get (task details), delete, enable, disable. "
        "Types: 'once' (ISO timestamp), 'interval' (seconds), 'cron' (cron expression)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "get", "delete", "enable", "disable"],
                "description": "Action to perform",
            },
            "name": {
                "type": "string",
                "description": "Task name (for create/delete by name)",
            },
            "task_id": {
                "type": "integer",
                "description": "Task ID (for delete/enable/disable/get)",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["once", "interval", "cron"],
                "description": "Type of schedule (for create)",
            },
            "schedule_value": {
                "type": "string",
                "description": (
                    "Schedule value. For 'once': ISO timestamp. "
                    "For 'interval': seconds (e.g., '3600'). "
                    "For 'cron': cron expression (e.g., '0 9 * * *')."
                ),
            },
            "payload": {
                "type": "string",
                "description": "Message to inject when task executes (for create)",
            },
        },
        "required": ["action"],
    },
    "handler": manage_schedule_handler,
    "requires_access_level": "owner",
}
