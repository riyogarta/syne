"""Sub-agent system — spawn isolated background sessions for parallel work.

Sub-agents are a CORE capability (not an ability). They enable:
- Parallel processing while main session continues chatting
- Heavy tasks in background (docs, analysis, code gen)
- Isolation: sub-agent failures don't affect main session
- Auto-spawn: main agent can proactively delegate complex tasks

Guard rails:
- Max concurrent sub-agents (default: 2)
- Timeout per sub-agent (default: 5 min)
- No nesting: sub-agents cannot spawn sub-agents
- Owner can disable entirely via config
- SECURITY: Sub-agents inherit owner tools but config/management tools blocked
"""

import asyncio
import json
import logging
import uuid
from typing import Optional, Callable, Awaitable

from .db.connection import get_connection
from .db.models import get_config
from .llm.provider import LLMProvider, ChatMessage, ChatResponse
from .security import get_subagent_access_level, filter_tools_for_subagent

logger = logging.getLogger("syne.subagent")


class SubAgentManager:
    """Manages sub-agent lifecycle: spawn, monitor, deliver results."""

    def __init__(self, provider: LLMProvider, system_prompt: str):
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = None          # ToolRegistry — set by agent after init
        self.abilities = None      # AbilityRegistry — set by agent after init
        self._active_runs: dict[str, asyncio.Task] = {}
        self._on_complete: Optional[Callable[[str, str, str, int], Awaitable[None]]] = None

    def set_completion_callback(self, callback: Callable[[str, str, str, int], Awaitable[None]]):
        """Set callback for when sub-agent completes.

        callback(run_id: str, status: str, result_or_error: str, parent_session_id: int)
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
        return await get_config("subagents.timeout_seconds", 900)

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
        """Execute a sub-agent task with full tool-calling loop.
        
        Sub-agents can use tools (exec, memory, web, abilities) just like
        the main agent, enabling them to do real work autonomously.
        Config/management tools are filtered out for safety.
        """
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
            "You CAN use all available tools:\n"
            "- exec — execute shell commands (your main tool for getting work done)\n"
            "- memory_search / memory_store — search and save information\n"
            "- web_search / web_fetch — search the web and fetch pages\n"
            "- file_read / file_write — read and write files\n"
            "- read_source — read Syne's own source code\n"
            "- All enabled abilities (image_gen, image_analysis, maps, etc.)\n\n"
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

        # Build tool schemas (filtered for sub-agent safety)
        tool_schemas = self._get_tool_schemas(access_level)

        # Get max tool rounds from config
        max_rounds = int(await get_config("session.max_tool_rounds", 25))
        # Sub-agents get same limit as main session
        
        total_input_tokens = 0
        total_output_tokens = 0

        # Initial LLM call
        response = await self.provider.chat(
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.3,
        )
        total_input_tokens += getattr(response, "input_tokens", 0) or 0
        total_output_tokens += getattr(response, "output_tokens", 0) or 0

        # Tool-calling loop — same pattern as conversation._handle_tool_calls
        for round_num in range(max_rounds):
            if not response.tool_calls:
                break

            # Add assistant message with tool calls
            messages.append(ChatMessage(
                role="assistant",
                content=response.content or "",
                metadata={"tool_calls": response.tool_calls},
            ))

            # Execute each tool call
            for tool_call in response.tool_calls:
                # Handle both normalized format (name/args) and OpenAI raw (function.name/arguments)
                if "function" in tool_call:
                    func = tool_call["function"]
                    name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                else:
                    name = tool_call.get("name", "")
                    args = tool_call.get("args", {})
                    if isinstance(args, str):
                        args = json.loads(args)

                tool_call_id = tool_call.get("id")
                logger.info(f"Sub-agent {run_id[:8]} tool call (round {round_num + 1}): {name}({args})")

                # Execute tool or ability
                result = await self._execute_tool(name, args, access_level)

                tool_meta = {"tool_name": name}
                if tool_call_id:
                    tool_meta["tool_call_id"] = tool_call_id
                messages.append(ChatMessage(role="tool", content=str(result), metadata=tool_meta))

            # Get next response — may contain more tool calls
            response = await self.provider.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=0.3,
            )
            total_input_tokens += getattr(response, "input_tokens", 0) or 0
            total_output_tokens += getattr(response, "output_tokens", 0) or 0
        else:
            # Loop exhausted — force a final text response
            if response.tool_calls:
                logger.warning(f"Sub-agent {run_id[:8]} hit tool round limit ({max_rounds})")
                messages.append(ChatMessage(
                    role="system",
                    content=f"STOP. You have used {max_rounds} tool rounds. Summarize what you've done and what remains.",
                ))
                response = await self.provider.chat(
                    messages=messages,
                    tools=None,
                    temperature=0.3,
                )
                total_input_tokens += getattr(response, "input_tokens", 0) or 0
                total_output_tokens += getattr(response, "output_tokens", 0) or 0

        # Track total tokens
        async with get_connection() as conn:
            await conn.execute("""
                UPDATE subagent_runs
                SET input_tokens = $1, output_tokens = $2
                WHERE run_id = $3
            """, total_input_tokens, total_output_tokens, run_id)

        return response.content

    def _get_tool_schemas(self, access_level: str) -> list[dict]:
        """Get tool schemas available to sub-agents (filtered)."""
        schemas = []
        if self.tools:
            schemas.extend(self.tools.to_openai_schema(access_level))
        if self.abilities:
            schemas.extend(self.abilities.to_openai_schema(access_level))
        # Filter out config/management tools
        return filter_tools_for_subagent(schemas)

    async def _execute_tool(self, name: str, args: dict, access_level: str) -> str:
        """Execute a tool or ability by name, with sub-agent safety checks."""
        from .security import is_tool_allowed_for_subagent

        if not is_tool_allowed_for_subagent(name):
            return f"Error: Tool '{name}' is not available to sub-agents."

        # Try core tools first
        if self.tools and self.tools.get(name):
            return await self.tools.execute(name, args, access_level)

        # Try abilities
        if self.abilities and self.abilities.get(name):
            ability_context = {
                "user_id": 0,  # Sub-agent doesn't have user context
                "session_id": 0,
                "access_level": access_level,
                "config": self.abilities.get(name).config or {},
                "_registry": self.abilities,  # For call_ability() support
            }
            ability_result = await self.abilities.execute(name, args, ability_context)
            if ability_result.get("success"):
                return ability_result.get("result", "Done.")
            return f"Error: {ability_result.get('error', 'Unknown error')}"

        return f"Error: Unknown tool or ability '{name}'"

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

            # Fetch parent_session_id for routing the result
            row = await conn.fetchrow(
                "SELECT parent_session_id FROM subagent_runs WHERE run_id = $1",
                run_id,
            )
            parent_session_id = row["parent_session_id"] if row else 0

        # Clean up from active runs
        if run_id in self._active_runs:
            del self._active_runs[run_id]

        # Notify via callback
        if self._on_complete:
            output = result if status == "completed" else f"Error: {error}"
            try:
                await self._on_complete(run_id, status, output, parent_session_id)
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

    async def cleanup_stale_runs(self):
        """Mark any 'running' DB records as failed (stale from previous bot run)."""
        async with get_connection() as conn:
            result = await conn.execute("""
                UPDATE subagent_runs
                SET status = 'failed', error = 'Bot restarted', completed_at = NOW()
                WHERE status = 'running'
            """)
            # result is "UPDATE N" — extract count
            count = int(result.split()[-1]) if result else 0
            if count:
                logger.info(f"Cleaned up {count} stale sub-agent run(s)")
