"""manage_schedule — Create and manage scheduled tasks (family+).

Rule 760z: Owner and family can use this tool.

Task types:
- 'once': Run once at a specific time (ISO timestamp)
- 'interval': Run every N seconds
- 'cron': Run based on cron expression (e.g., "0 9 * * *" for 9 AM daily)

When a task executes, its payload is injected as a user message
to the conversation.

Bulk operations:
- 'bulk_create': Create multiple tasks at once (pass JSON array in 'bulk_tasks')
- 'bulk_delete': Delete multiple tasks by ID range or list (pass 'task_ids' as comma-separated)
"""

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("syne.tools.scheduler")

# Current user context — ContextVar for async safety.
# Set by conversation layer before tool execution, so that
# manage_schedule can auto-fill created_by without relying on the LLM.
_current_user_platform_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "current_user_platform_id", default=None
)


def set_current_user(platform_id: Optional[int]) -> None:
    """Set the current user's platform ID for task creation.

    Uses contextvars — safe for concurrent async contexts.
    """
    _current_user_platform_id.set(platform_id)


async def manage_schedule_handler(
    action: str,
    name: str = "",
    task_id: int = 0,
    schedule_type: str = "",
    schedule_value: str = "",
    payload: str = "",
    end_date: str = "",
    bulk_tasks: str = "",
    task_ids: str = "",
) -> str:
    """Handle manage_schedule tool calls.
    
    Args:
        action: create, list, delete, enable, disable, get, bulk_create, bulk_delete
        name: Task name (for create/delete_by_name)
        task_id: Task ID (for delete/enable/disable/get)
        schedule_type: 'once', 'interval', or 'cron' (for create)
        schedule_value: Schedule value (for create):
            - once: ISO timestamp (e.g., "2026-02-20T09:00:00+07:00")
            - interval: seconds (e.g., "3600" for hourly)
            - cron: cron expression (e.g., "0 9 * * *")
        payload: Message to inject when task runs (for create)
        end_date: Optional ISO timestamp — recurring tasks auto-disable after this date
        bulk_tasks: JSON array of tasks for bulk_create. Each item:
            {"name", "schedule_type", "schedule_value", "payload", "end_date"(optional)}
        task_ids: Comma-separated task IDs for bulk_delete (e.g., "64,65,66,67")
            Also supports ranges: "64-131" or mixed: "64-70,75,80-90"
        
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
        
        # Parse end_date if provided
        parsed_end_date = None
        if end_date:
            try:
                parsed_end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if parsed_end_date.tzinfo is None:
                    parsed_end_date = parsed_end_date.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                return f"Error: invalid end_date format '{end_date}'. Use ISO format (e.g., 2026-03-21T08:00:00+07:00)."
        
        result = await create_task(
            name=name,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            payload=payload,
            created_by=_current_user_platform_id.get(),
            end_date=parsed_end_date,
        )
        
        if not result["success"]:
            return f"Error: {result['error']}"
        
        task = result["task"]
        next_run = task.get("next_run")
        if next_run and isinstance(next_run, datetime):
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            next_run_str = str(next_run)
        
        task_end_date = task.get("end_date")
        end_date_str = ""
        if task_end_date and isinstance(task_end_date, datetime):
            end_date_str = f"\n- End date: {task_end_date.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        
        return (
            f"✅ Task created:\n"
            f"- ID: {task['id']}\n"
            f"- Name: {task['name']}\n"
            f"- Type: {task['schedule_type']}\n"
            f"- Schedule: {task['schedule_value']}\n"
            f"- Next run: {next_run_str}"
            f"{end_date_str}\n"
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
            
            end_date = t.get("end_date")
            end_str = ""
            if end_date and isinstance(end_date, datetime):
                end_str = f" | Until: {end_date.strftime('%m/%d %H:%M')}"
            
            lines.append(
                f"- {status} **{t['name']}** (id={t['id']})\n"
                f"  Type: {t['schedule_type']} | Next: {next_run_str}{end_str} | Runs: {t['run_count']}"
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
        
        end_date = task.get("end_date")
        end_date_line = f"\n- End date: {fmt_dt(end_date)}" if end_date else ""
        
        return (
            f"**Task: {task['name']}** (id={task['id']})\n"
            f"- Status: {'Enabled' if task['enabled'] else 'Disabled'}\n"
            f"- Type: {task['schedule_type']}\n"
            f"- Schedule: {task['schedule_value']}"
            f"{end_date_line}\n"
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
    
    elif action == "bulk_create":
        if not bulk_tasks:
            return "Error: bulk_tasks is required (JSON array of task objects)."
        
        try:
            tasks_list = json.loads(bulk_tasks)
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON in bulk_tasks: {e}"
        
        if not isinstance(tasks_list, list):
            return "Error: bulk_tasks must be a JSON array."
        
        if len(tasks_list) > 500:
            return "Error: maximum 500 tasks per bulk_create."
        
        created = 0
        errors = []
        
        for i, t in enumerate(tasks_list):
            t_name = t.get("name", "")
            t_type = t.get("schedule_type", "")
            t_value = t.get("schedule_value", "")
            t_payload = t.get("payload", "")
            t_end = t.get("end_date", "")
            
            if not all([t_name, t_type, t_value, t_payload]):
                errors.append(f"#{i}: missing required fields (name/schedule_type/schedule_value/payload)")
                continue
            
            parsed_end = None
            if t_end:
                try:
                    parsed_end = datetime.fromisoformat(t_end.replace("Z", "+00:00"))
                    if parsed_end.tzinfo is None:
                        parsed_end = parsed_end.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    errors.append(f"#{i} '{t_name}': invalid end_date '{t_end}'")
                    continue
            
            result = await create_task(
                name=t_name,
                schedule_type=t_type,
                schedule_value=t_value,
                payload=t_payload,
                created_by=_current_user_platform_id.get(),
                end_date=parsed_end,
            )
            
            if result["success"]:
                created += 1
            else:
                errors.append(f"#{i} '{t_name}': {result['error']}")
        
        msg = f"✅ Bulk create: {created}/{len(tasks_list)} tasks created."
        if errors:
            msg += f"\n\n❌ {len(errors)} errors:\n" + "\n".join(errors[:20])
            if len(errors) > 20:
                msg += f"\n... and {len(errors) - 20} more"
        return msg
    
    elif action == "bulk_delete":
        if not task_ids:
            return "Error: task_ids is required (comma-separated IDs or ranges like '64-131')."
        
        # Parse task IDs — support ranges like "64-131" and lists like "64,65,66"
        ids_to_delete = []
        for part in task_ids.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    ids_to_delete.extend(range(int(start.strip()), int(end.strip()) + 1))
                except ValueError:
                    return f"Error: invalid range '{part}'. Use format: '64-131'"
            else:
                try:
                    ids_to_delete.append(int(part))
                except ValueError:
                    return f"Error: invalid task ID '{part}'."
        
        if len(ids_to_delete) > 500:
            return f"Error: too many IDs ({len(ids_to_delete)}). Maximum 500 per bulk_delete."
        
        deleted = 0
        not_found = 0
        for tid in ids_to_delete:
            success = await delete_task(tid)
            if success:
                deleted += 1
            else:
                not_found += 1
        
        return f"✅ Bulk delete: {deleted} tasks deleted, {not_found} not found (out of {len(ids_to_delete)} IDs)."
    
    else:
        return f"Unknown action: {action}. Use: create, list, get, delete, enable, disable, bulk_create, bulk_delete"


# ── Tool Registration Dict ──

MANAGE_SCHEDULE_TOOL = {
    "name": "manage_schedule",
    "description": (
        "Create and manage scheduled tasks. Tasks execute by injecting payload as user message. "
        "Actions: create, list, get, delete, enable, disable, bulk_create, bulk_delete. "
        "Types: 'once' (ISO timestamp), 'interval' (seconds), 'cron' (cron expression). "
        "Optional end_date for recurring tasks — auto-disables after the date passes. "
        "Use bulk_create with a JSON array to create many tasks in one call. "
        "Use bulk_delete with comma-separated IDs or ranges (e.g., '64-131') to delete many at once."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "get", "delete", "enable", "disable", "bulk_create", "bulk_delete"],
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
            "end_date": {
                "type": "string",
                "description": (
                    "Optional ISO timestamp for recurring tasks. "
                    "Task auto-disables after this date. "
                    "NULL/empty = no end date (runs forever or once). "
                    "Example: '2026-03-21T08:00:00+07:00'"
                ),
            },
            "bulk_tasks": {
                "type": "string",
                "description": (
                    "JSON array of task objects for bulk_create. "
                    "Each: {\"name\", \"schedule_type\", \"schedule_value\", \"payload\", \"end_date\"(optional)}. "
                    "Max 500 tasks per call."
                ),
            },
            "task_ids": {
                "type": "string",
                "description": (
                    "Comma-separated task IDs for bulk_delete. "
                    "Supports ranges: '64-131' or mixed: '64-70,75,80-90'. Max 500."
                ),
            },
        },
        "required": ["action"],
    },
    "handler": manage_schedule_handler,
    "requires_access_level": "family",
}
