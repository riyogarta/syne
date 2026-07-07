"""Sub-agent system — spawn isolated background sessions for parallel work.

Sub-agents are a CORE capability (not an ability). They enable:
- Parallel processing while main session continues chatting
- Heavy tasks in background (docs, analysis, code gen)
- Isolation: sub-agent failures don't affect main session
- Auto-spawn: main agent can proactively delegate complex tasks

Guard rails:
- Max concurrent sub-agents (default: 2)
- Timeout per sub-agent (default: 24 hours)
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


class SubAgentIncomplete(Exception):
    """Raised when a sub-agent reaches max_rounds before finishing the task.

    This is a FAILURE mode distinct from a clean error/timeout: the work was
    cut off mid-flight. It carries the partial result so it can still be
    persisted and delivered to the user (not discarded).
    """

    def __init__(self, partial_result: str):
        self.partial_result = partial_result
        super().__init__("Sub-agent reached max rounds before completing the task")


class SubAgentManager:
    """Manages sub-agent lifecycle: spawn, monitor, deliver results."""

    def __init__(self, provider: LLMProvider, system_prompt: str):
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = None          # ToolRegistry — set by agent after init
        self.abilities = None      # AbilityRegistry — set by agent after init
        self._active_runs: dict[str, asyncio.Task] = {}
        self._on_complete: Optional[Callable[[str, str, str, int], Awaitable[None]]] = None
        self._on_start: Optional[Callable[[str, str, int], Awaitable[None]]] = None

    def set_completion_callback(self, callback: Callable[[str, str, str, int], Awaitable[None]]):
        """Set callback for when sub-agent completes.

        callback(run_id: str, status: str, result_or_error: str, parent_session_id: int)
        """
        self._on_complete = callback

    def set_start_callback(self, callback: Callable[..., Awaitable[None]]):
        """Set callback for when a sub-agent starts.

        callback(run_id: str, task: str, parent_session_id: int, resumed_from: Optional[str])
        """
        self._on_start = callback

    async def is_enabled(self) -> bool:
        """Check if sub-agents are enabled."""
        return await get_config("subagents.enabled", True)

    async def max_concurrent(self) -> int:
        """Get max concurrent sub-agents."""
        return await get_config("subagents.max_concurrent", 2)

    async def timeout_seconds(self) -> int:
        """Get sub-agent timeout."""
        return await get_config("subagents.timeout_seconds", 172800)

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
        provider: Optional[LLMProvider] = None,
        resume_from: Optional[str] = None,
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

        # Resume handling: continue an incomplete/failed run from its partial result
        resumed_from = None
        if resume_from:
            # Accept truncated ids (e.g. 8-char prefix shown in notifications)
            try:
                resolved = await self._resolve_run_id(resume_from)
            except ValueError as e:
                return {"success": False, "error": str(e)}
            if not resolved:
                return {
                    "success": False,
                    "error": f"Cannot resume: no sub-agent found with run_id {resume_from}.",
                }
            resume_from = resolved
            prev = await self.get_run(resume_from)
            if not prev:
                return {
                    "success": False,
                    "error": f"Cannot resume: no sub-agent found with run_id {resume_from}.",
                }
            if prev["status"] not in ("incomplete", "failed"):
                return {
                    "success": False,
                    "error": f"Cannot resume run {resume_from}: status is '{prev['status']}' "
                             f"(only 'incomplete' or 'failed' runs can be resumed).",
                }
            resumed_from = resume_from
            # Keep the original task for continuity; if caller gave a new task, append it.
            original_task = prev["task"]
            partial = prev.get("result") or prev.get("error") or "(no partial output recorded)"
            resume_preamble = (
                "\u2550\u2550 CONTINUING A PREVIOUS TASK \u2550\u2550\n"
                f"Original task: {original_task}\n\n"
                "Progress so far (partial output from the previous run):\n"
                f"{partial}\n"
                "\u2500\u2500\u2500\u2500\u2500\n"
                "Continue from this point until the task is fully complete. "
                "Do NOT restart from scratch \u2014 build on the progress above."
            )
            context = (resume_preamble + "\n\n" + context) if context else resume_preamble
            if not task or not task.strip():
                task = original_task

        # Create run record
        run_id = str(uuid.uuid4())
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO subagent_runs (run_id, parent_session_id, task, model)
                VALUES ($1, $2, $3, $4)
            """, run_id, parent_session_id, task, model)

        # Start the sub-agent task
        # Use parent conversation's provider if given, else fall back to default
        effective_provider = provider or self.provider
        timeout = await self.timeout_seconds()
        async_task = asyncio.create_task(
            self._run_subagent(run_id, task, context, model, timeout, effective_provider)
        )
        self._active_runs[run_id] = async_task

        logger.info(f"Sub-agent spawned: run_id={run_id}, task='{task[:80]}'")

        # Notify that the sub-agent has started (best-effort, non-blocking)
        if self._on_start:
            try:
                await self._on_start(run_id, task, parent_session_id, resumed_from)
            except Exception as e:
                logger.error(f"Start callback failed for {run_id}: {e}")

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
        provider: Optional[LLMProvider] = None,
    ):
        """Execute a sub-agent task in the background."""
        try:
            result = await asyncio.wait_for(
                self._execute_task(run_id, task, context, model, provider),
                timeout=timeout,
            )
            await self._complete_run(run_id, "completed", result=result)

        except SubAgentIncomplete as e:
            logger.warning(f"Sub-agent {run_id}: incomplete (reached max rounds)")
            await self._complete_run(run_id, "incomplete", result=e.partial_result)

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
        provider: Optional[LLMProvider] = None,
    ) -> str:
        """Execute a sub-agent task with full tool-calling loop.

        Sub-agents can use tools (exec, memory, web, abilities) just like
        the main agent, enabling them to do real work autonomously.
        Config/management tools are filtered out for safety.
        """
        # Use parent conversation's provider (inherits group/user model override)
        llm = provider or self.provider
        logger.info(f"Sub-agent {run_id[:8]} using provider: {llm.name}")

        messages = []

        # ═══════════════════════════════════════════════════════════════
        # SECURITY: Sub-agents are "workers" — they inherit owner privileges
        # for doing actual work (exec, memory, abilities) but CANNOT modify
        # Syne's configuration or policies.
        # Config/management tools are filtered via filter_tools_for_subagent()
        # ═══════════════════════════════════════════════════════════════
        access_level = get_subagent_access_level()  # Returns "owner" (but tools are filtered)
        logger.debug(f"Sub-agent {run_id[:8]} running with access_level={access_level} (config tools blocked)")

        # Budget (fetched early so it can be injected into the prompt)
        max_rounds = await get_config("subagents.max_rounds", 30)

        # System prompt for sub-agent — worker privileges
        subagent_prompt = (
            f"{self.system_prompt}\n\n"
            "# SUB-AGENT CONTEXT\n"
            "You are running as a SUB-AGENT in a background session.\n\n"
            "## Task Guidelines:\n"
            "- START WORKING IMMEDIATELY. Do NOT research, read source code, test, or plan first.\n"
            "- Execute the task directly using the tools available to you.\n"
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
            "## IMPORTANT CONSTRAINTS:\n"
            "- You have a LIMITED number of tool call rounds. Use them wisely.\n"
            "- NEVER use 'sleep' inside exec to wait/monitor — it wastes rounds.\n"
            "- For long tasks: launch as background script (nohup) then STOP. "
            "Do NOT poll/monitor progress — the user can check manually.\n"
            "- Prefer doing the actual work directly over writing scripts to do it later.\n"
            f"\n## ROUND BUDGET: You have {max_rounds} tool call round(s) total.\n"
            "- If the task genuinely needs MORE rounds than your budget and you cannot\n"
            "  finish, do NOT pretend it is done. Report the partial progress and end your\n"
            "  final message with the exact marker on its own line: [[INCOMPLETE]]\n"
            "- Only use [[INCOMPLETE]] when work genuinely remains. If you finished, do NOT use it.\n"
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

        total_input_tokens = 0
        total_output_tokens = 0

        # ── Factual tool call tracking ──
        # Track every tool call and its success/failure so the completion
        # report contains verifiable data, not LLM-generated claims.
        tool_call_log: list[dict] = []  # [{name, success, error?}, ...]
        tool_call_counts: dict[str, dict] = {}  # name -> {total, success, failed}

        # Initial LLM call
        response = await llm.chat(
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            temperature=0.3,
        )
        total_input_tokens += getattr(response, "input_tokens", 0) or 0
        total_output_tokens += getattr(response, "output_tokens", 0) or 0

        # Tool-calling loop with round limit and inter-round delay
        # (max_rounds already fetched above for the prompt)
        round_delay = await get_config("subagents.round_delay", 2.0)  # seconds between rounds
        round_num = 0
        forced_stop = False
        while response.tool_calls:
            # Hard round limit — force stop to prevent runaway sub-agents
            if round_num >= max_rounds:
                logger.warning(f"Sub-agent {run_id[:8]}: hit max rounds ({max_rounds}), forcing stop")
                messages.append(ChatMessage(
                    role="system",
                    content=(
                        f"STOP. You have reached the maximum of {max_rounds} tool call rounds. "
                        "Summarize what you have accomplished so far. If the task requires more work, "
                        "describe what remains so the user can continue manually or spawn another sub-agent."
                    ),
                ))
                forced_stop = True
                break

            # Warn at 80% of limit
            if round_num == int(max_rounds * 0.8):
                messages.append(ChatMessage(
                    role="system",
                    content=(
                        f"WARNING: You have used {round_num}/{max_rounds} rounds. "
                        "Wrap up your work soon. If the task needs a long-running process, "
                        "launch it as a background script (nohup) and finish — do NOT monitor "
                        "it with sleep loops, that wastes rounds."
                    ),
                ))

            # Inter-round delay — yield CPU/API to main conversation
            if round_num > 0 and round_delay > 0:
                await asyncio.sleep(round_delay)

            # Add assistant message with tool calls
            messages.append(ChatMessage(
                role="assistant",
                content=response.content or "",
                metadata={"tool_calls": response.tool_calls},
            ))

            # Parse all tool calls
            parsed = []
            for tool_call in response.tool_calls:
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
                logger.info(f"Sub-agent {run_id[:8]} tool call (round {round_num + 1}/{max_rounds}): {name}({args})")
                parsed.append((name, args, tool_call_id))

            # Execute tools in parallel.
            tasks = [self._execute_tool(n, a, access_level) for n, a, _ in parsed]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, raw in enumerate(raw_results):
                name, args, tool_call_id = parsed[i]
                is_error = isinstance(raw, Exception) or (isinstance(raw, str) and raw.startswith("Error:"))
                result = str(raw) if not isinstance(raw, Exception) else f"Error: {raw}"

                # Track this call
                success = not is_error
                tool_call_log.append({"name": name, "success": success})
                if name not in tool_call_counts:
                    tool_call_counts[name] = {"total": 0, "success": 0, "failed": 0}
                tool_call_counts[name]["total"] += 1
                tool_call_counts[name]["success" if success else "failed"] += 1

                tool_meta = {"tool_name": name}
                if tool_call_id:
                    tool_meta["tool_call_id"] = tool_call_id
                messages.append(ChatMessage(role="tool", content=result, metadata=tool_meta))

            # Get next response — may contain more tool calls
            response = await llm.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=0.3,
            )
            total_input_tokens += getattr(response, "input_tokens", 0) or 0
            total_output_tokens += getattr(response, "output_tokens", 0) or 0
            round_num += 1

        # If forced stop, get final summary response (no tools)
        if forced_stop:
            response = await llm.chat(
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

        # ── Build factual execution report ──
        # This is appended to the result so the completion callback delivers
        # verifiable data to the user, not just the LLM's narrative.
        llm_narrative = response.content or ""
        # Sub-agent may voluntarily signal it ran out of budget without being
        # engine-forced (e.g. it knew rounds were too few and gave up cleanly).
        # Treat that marker as authoritative for INCOMPLETE status.
        marker_incomplete = "[[INCOMPLETE]]" in llm_narrative
        if marker_incomplete:
            llm_narrative = llm_narrative.replace("[[INCOMPLETE]]", "").rstrip()
        is_incomplete = forced_stop or marker_incomplete
        report_lines = ["\n\n--- EXECUTION REPORT (auto-generated, not editable by AI) ---"]
        _round_flag = ' (FORCED STOP)' if forced_stop else (' (INCOMPLETE)' if marker_incomplete else '')
        report_lines.append(f"Tool call rounds: {round_num}/{max_rounds}{_round_flag}")
        report_lines.append(f"Total tool calls: {len(tool_call_log)}")
        if tool_call_counts:
            for tname, counts in sorted(tool_call_counts.items()):
                status = f"{counts['success']} ok"
                if counts['failed']:
                    status += f", {counts['failed']} failed"
                report_lines.append(f"  {tname}: {counts['total']} calls ({status})")
        else:
            report_lines.append("  (no tools were called)")
        report_lines.append(f"Tokens: {total_input_tokens} in / {total_output_tokens} out")
        report_lines.append("--- END REPORT ---")

        result = llm_narrative + "\n".join(report_lines)
        # INCOMPLETE if the engine forced a stop OR the sub-agent itself signalled
        # it ran out of budget with work remaining. Either way it's resumable.
        if is_incomplete:
            raise SubAgentIncomplete(result)
        return result

    def _get_tool_schemas(self, access_level: str) -> list[dict]:
        """Get tool schemas available to sub-agents (filtered)."""
        schemas = []
        if self.tools:
            schemas.extend(self.tools.to_openai_schema(access_level))
        if self.abilities:
            schemas.extend(self.abilities.to_openai_schema(access_level))
        # Filter out config/management tools
        return filter_tools_for_subagent(schemas)

    async def _execute_tool(
        self, name: str, args: dict, access_level: str,
    ) -> str:
        """Execute a tool or ability by name, with sub-agent safety checks."""
        from .security import is_tool_allowed_for_subagent

        if not is_tool_allowed_for_subagent(name):
            return f"Error: Tool '{name}' is not available to sub-agents."

        # ── Shell guard for sub-agent exec ────────────────────────────────
        # Sub-agents run HEADLESS — there is no human to answer a consent
        # Yes/No. So the guard runs here with source="subagent" and BOTH
        # HARD_DENY and CONSENT stop the command cold (fail-closed): a
        # command that would need approval simply cannot proceed, because no
        # approval is possible. The sub-agent reports the work as unfinished.
        #
        # This also closes the long-standing gap where sub-agent exec ran with
        # NO safety check at all (the stale consent.py comment claiming
        # "sub-agent already runs check_command_safety" was false — it never
        # did). Now it does, via the same shell_guard as the main agent.
        if name in ("exec", "shell"):
            from .shell_guard import analyze, Verdict
            from .shell_exec import run_shell, Outcome
            from .db.connection import get_pool
            _cmd = (args or {}).get("command", "")
            try:
                _pool = get_pool()
            except Exception:
                _pool = None
            # consent_enabled is irrelevant for sub-agents: even with the gate
            # globally off, a CONSENT verdict must NOT silently run headless.
            # We pass consent_enabled=True + approved=False so CONSENT surfaces
            # as NEEDS_CONSENT, which we convert into a hard stop below.
            _res = await run_shell(
                _cmd, source="subagent", cwd=None,
                timeout=int((args or {}).get("timeout", 30) or 30),
                db_pool=_pool, consent_enabled=True, approved=False,
            )
            if _res.outcome == Outcome.RAN:
                return _res.output
            if _res.outcome == Outcome.DRY_RUN:
                _v = _res.verdict.value.upper() if _res.verdict else "?"
                return f"DRY-RUN (not executed) — verdict: {_v}. Reason: {_res.reason}"
            if _res.outcome == Outcome.DENIED:
                return (
                    f"Error: shell guard hard-denied this command — {_res.reason}. "
                    "Cannot run and cannot be approved. Reporting task as unfinished."
                )
            # NEEDS_CONSENT → headless stop, task incomplete.
            raise SubAgentIncomplete(
                f"Command blocked pending owner consent (no interactive approval "
                f"possible in a sub-agent): `{_cmd}`. Reason: {_res.reason}. "
                "Task stopped — owner must run this manually or pre-approve."
            )

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
            output = result if status in ("completed", "incomplete") else f"Error: {error}"
            try:
                await self._on_complete(run_id, status, output, parent_session_id)
            except Exception as e:
                logger.error(f"Completion callback failed for {run_id}: {e}")

        logger.info(f"Sub-agent {run_id}: {status}")

    async def _resolve_run_id(self, run_id: str) -> Optional[str]:
        """Resolve a possibly-truncated run_id to the full UUID.

        The run_id column is UUID; notifications display only the first 8 chars
        for readability. A truncated value would fail a strict UUID match, so we
        accept a prefix and expand it via the DB. Prevents the class of bug where
        a human (or the agent) copies the shortened id shown on screen.

        Returns the full run_id string, or None if no match.
        Raises ValueError if the prefix is ambiguous (matches more than one run).
        """
        rid = (run_id or "").strip()
        if not rid:
            return None
        async with get_connection() as conn:
            # Full UUID -> exact match
            if len(rid) == 36:
                row = await conn.fetchrow(
                    "SELECT run_id::text AS run_id FROM subagent_runs "
                    "WHERE run_id = $1::uuid",
                    rid,
                )
                return row["run_id"] if row else None
            # Truncated prefix (e.g. 8-char id from a notification)
            rows = await conn.fetch(
                "SELECT run_id::text AS run_id FROM subagent_runs "
                "WHERE run_id::text LIKE $1 ORDER BY started_at DESC",
                rid + "%",
            )
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError(
                    f"Ambiguous run_id prefix '{rid}' matches {len(rows)} runs. "
                    "Provide more characters or the full UUID."
                )
            return rows[0]["run_id"]

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

    async def cancel_by_session(self, parent_session_id: int) -> int:
        """Cancel all running sub-agents spawned from a specific session.

        Returns number of cancelled runs.
        """
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT run_id FROM subagent_runs
                WHERE parent_session_id = $1 AND status = 'running'
            """, parent_session_id)

        count = 0
        for row in rows:
            if await self.cancel(row["run_id"]):
                count += 1
        return count

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
