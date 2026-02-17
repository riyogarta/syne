"""Sub-agent system — spawn isolated background sessions for parallel work.

Sub-agents are a CORE capability (not an ability). They enable:
- Parallel processing while main session continues chatting
- Heavy tasks in background (docs, analysis, code gen)
- Isolation: sub-agent failures don't affect main session

Guard rails:
- Max concurrent sub-agents (default: 2)
- Timeout per sub-agent (default: 5 min)
- No nesting: sub-agents cannot spawn sub-agents
- Owner can disable entirely via config
- SECURITY: Sub-agents always run with PUBLIC access level (no owner tools)
"""

import asyncio
import logging
import uuid
from typing import Optional, Callable, Awaitable

from .db.connection import get_connection
from .db.models import get_config
from .llm.provider import LLMProvider, ChatMessage
from .security import get_subagent_access_level, filter_tools_for_subagent

logger = logging.getLogger("syne.subagent")


class SubAgentManager:
    """Manages sub-agent lifecycle: spawn, monitor, deliver results."""

    def __init__(self, provider: LLMProvider, system_prompt: str):
        self.provider = provider
        self.system_prompt = system_prompt
        self._active_runs: dict[str, asyncio.Task] = {}
        self._on_complete: Optional[Callable[[str, str, str], Awaitable[None]]] = None

    def set_completion_callback(self, callback: Callable[[str, str, str], Awaitable[None]]):
        """Set callback for when sub-agent completes.
        
        callback(run_id: str, status: str, result_or_error: str)
        """
        self._on_complete = callback

    async def is_enabled(self) -> bool:
        """Check if sub-agents are enabled."""
        return await get_config("subagents.enabled", True)

    async def max_concurrent(self) -> int:
        """Get max concurrent sub-agents."""
        return await get_config("subagents.max_concurrent", 2)

    async def timeout_seconds(self) -> int:
        """Get sub-agent timeout."""
        return await get_config("subagents.timeout_seconds", 300)

    @property
    def active_count(self) -> int:
        """Number of currently running sub-agents."""
        # Clean up finished tasks
        finished = [rid for rid, task in self._active_runs.items() if task.done()]
        for rid in finished:
            del self._active_runs[rid]
        return len(self._active_runs)

    async def spawn(
        self,
        task: str,
        parent_session_id: int,
        context: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict:
        """Spawn a new sub-agent.
        
        Args:
            task: Task description for the sub-agent
            parent_session_id: Parent session ID for tracking
            context: Optional context to provide (e.g., recent conversation summary)
            model: Optional model override
            
        Returns:
            dict with run_id, status, message
        """
        # Check if enabled
        if not await self.is_enabled():
            return {
                "success": False,
                "error": "Sub-agents are disabled. Owner can enable via config.",
            }

        # Check max concurrent
        max_conc = await self.max_concurrent()
        if self.active_count >= max_conc:
            return {
                "success": False,
                "error": f"Max concurrent sub-agents reached ({max_conc}). Wait for one to complete.",
            }

        # Create run record
        run_id = str(uuid.uuid4())
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO subagent_runs (run_id, parent_session_id, task, model)
                VALUES ($1, $2, $3, $4)
            """, run_id, parent_session_id, task, model)

        # Start the sub-agent task
        timeout = await self.timeout_seconds()
        async_task = asyncio.create_task(
            self._run_subagent(run_id, task, context, model, timeout)
        )
        self._active_runs[run_id] = async_task

        logger.info(f"Sub-agent spawned: run_id={run_id}, task='{task[:80]}'")

        return {
            "success": True,
            "run_id": run_id,
            "message": f"Sub-agent spawned. I'll notify you when it completes.",
        }

    async def _run_subagent(
        self,
        run_id: str,
        task: str,
        context: Optional[str],
        model: Optional[str],
        timeout: int,
    ):
        """Execute a sub-agent task in the background."""
        try:
            result = await asyncio.wait_for(
                self._execute_task(run_id, task, context, model),
                timeout=timeout,
            )
            await self._complete_run(run_id, "completed", result=result)

        except asyncio.TimeoutError:
            error_msg = f"Sub-agent timed out after {timeout}s"
            logger.warning(f"Sub-agent {run_id}: {error_msg}")
            await self._complete_run(run_id, "failed", error=error_msg)

        except Exception as e:
            error_msg = f"Sub-agent error: {str(e)}"
            logger.error(f"Sub-agent {run_id}: {error_msg}")
            await self._complete_run(run_id, "failed", error=error_msg)

    async def _execute_task(
        self,
        run_id: str,
        task: str,
        context: Optional[str],
        model: Optional[str],
    ) -> str:
        """Execute the actual LLM call for a sub-agent task."""
        messages = []

        # ═══════════════════════════════════════════════════════════════
        # SECURITY: Sub-agents are "workers" — they inherit owner privileges
        # for doing actual work (exec, memory, abilities) but CANNOT modify
        # Syne's configuration or policies.
        # Config/management tools are filtered via filter_tools_for_subagent()
        # ═══════════════════════════════════════════════════════════════
        access_level = get_subagent_access_level()  # Returns "owner" (but tools are filtered)
        logger.debug(f"Sub-agent {run_id[:8]} running with access_level={access_level} (config tools blocked)")

        # System prompt for sub-agent — worker privileges
        subagent_prompt = (
            f"{self.system_prompt}\n\n"
            "# SUB-AGENT CONTEXT\n"
            "You are running as a SUB-AGENT in a background session.\n\n"
            "## Task Guidelines:\n"
            "- Complete the task thoroughly.\n"
            "- Be concise but complete in your response.\n"
            "- You CANNOT spawn other sub-agents.\n"
            "- You CANNOT interact with the user directly.\n"
            "- Your result will be delivered to the main session.\n\n"
            "## YOUR CAPABILITIES (Worker Privileges):\n"
            "You CAN use:\n"
            "- exec — execute shell commands (your main tool for getting work done)\n"
            "- memory_search — search stored memories\n"
            "- memory_store — save important information\n"
            "- All abilities — web_search, maps, image_gen, image_analysis, etc.\n\n"
            "You CANNOT use (config/management tools blocked):\n"
            "- update_config, update_soul, update_ability\n"
            "- manage_group, manage_user\n"
            "- spawn_subagent (no nesting)\n\n"
            "Analogy: You're a worker who can do tasks, but cannot change company policy.\n"
        )
        messages.append(ChatMessage(role="system", content=subagent_prompt))

        # Add context if provided
        if context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context from main session:\n{context}",
            ))

        # Task
        messages.append(ChatMessage(role="user", content=task))

        # Call LLM
        response = await self.provider.chat(
            messages=messages,
            temperature=0.3,  # Lower temp for focused task execution
        )

        # Track tokens
        input_tokens = getattr(response, "input_tokens", 0) or 0
        output_tokens = getattr(response, "output_tokens", 0) or 0

        async with get_connection() as conn:
            await conn.execute("""
                UPDATE subagent_runs
                SET input_tokens = $1, output_tokens = $2
                WHERE run_id = $3
            """, input_tokens, output_tokens, run_id)

        return response.content

    async def _complete_run(
        self,
        run_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Mark a run as completed/failed and notify."""
        async with get_connection() as conn:
            await conn.execute("""
                UPDATE subagent_runs
                SET status = $1, result = $2, error = $3, completed_at = NOW()
                WHERE run_id = $4
            """, status, result, error, run_id)

        # Clean up from active runs
        if run_id in self._active_runs:
            del self._active_runs[run_id]

        # Notify via callback
        if self._on_complete:
            output = result if status == "completed" else f"Error: {error}"
            try:
                await self._on_complete(run_id, status, output)
            except Exception as e:
                logger.error(f"Completion callback failed for {run_id}: {e}")

        logger.info(f"Sub-agent {run_id}: {status}")

    async def get_run(self, run_id: str) -> Optional[dict]:
        """Get a sub-agent run by ID."""
        async with get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT run_id, task, status, result, error, model,
                       started_at, completed_at, input_tokens, output_tokens
                FROM subagent_runs
                WHERE run_id = $1
            """, run_id)
            return dict(row) if row else None

    async def list_active(self) -> list[dict]:
        """List all active sub-agent runs."""
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT run_id, task, status, started_at, input_tokens, output_tokens
                FROM subagent_runs
                WHERE status = 'running'
                ORDER BY started_at DESC
            """)
            return [dict(row) for row in rows]

    async def cancel(self, run_id: str) -> bool:
        """Cancel a running sub-agent."""
        if run_id in self._active_runs:
            self._active_runs[run_id].cancel()
            await self._complete_run(run_id, "cancelled", error="Cancelled by user")
            return True
        return False

    async def cancel_all(self):
        """Cancel all running sub-agents."""
        for run_id in list(self._active_runs.keys()):
            await self.cancel(run_id)
