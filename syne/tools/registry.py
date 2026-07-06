"""Tool registry — register and manage agent capabilities."""

import logging
from typing import Callable, Optional
from dataclasses import dataclass, field

from ..security import check_tool_access, TOOL_PERMISSIONS, TOOL_OPERATIONS

logger = logging.getLogger("syne.tools.registry")


@dataclass
class ToolResult:
    """Structured result from tool execution."""
    content: str
    ok: bool = True
    retryable: bool = False
    error_type: str = ""  # "timeout", "rate_limit", "not_found", "permission", "unknown"

    def __str__(self):
        return self.content


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON Schema for parameters
    handler: Callable                   # async function to execute
    permission: int = 0o700             # Linux-style 3-digit octal (owner/family/public)
    operation: str = "x"                # rwx op type: "r" read, "w" write, "x" action
    enabled: bool = True
    scrub_level: str = "aggressive"     # "aggressive" | "safe" | "none"
    # aggressive: full regex scrub (Cookie, PEM, querystring, etc.)
    # safe: high-confidence patterns only (JWT, sk-*, bot tokens, etc.)
    #        Won't corrupt regex/code in output
    # none: tool has its own dedicated scrubber (e.g. exec)


class ToolRegistry:
    """Manages available tools for the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._approval_callback = None  # Optional async callback for interactive approval
        self._schema_cache: dict[str, list[dict]] = {}  # access_level → schemas

    def set_approval_callback(self, callback):
        """Set approval callback for tools that need user confirmation.

        callback(tool_name: str, args: dict) -> tuple[bool, str]
        Returns (approved, reason).
        """
        self._approval_callback = callback

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        permission: int = 0o700,
        scrub_level: str = "aggressive",
        operation: str = None,
    ):
        """Register a new tool.

        Args:
            permission: Linux-style 3-digit octal (owner/family/public).
                Each digit encodes rwx: r=4(read), w=2(write), x=1(action).
                Example: 0o750 = owner(rwx), family(r-x), public(---).
            scrub_level: Credential scrubbing level for tool output.
                "aggressive" — full regex scrub (default, safest)
                "safe" — high-confidence patterns only (for code/content output)
                "none" — tool has its own dedicated scrubber
        """
        if operation is None:
            operation = TOOL_OPERATIONS.get(name, "x")  # unknown -> strictest
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            permission=permission,
            operation=operation,
            scrub_level=scrub_level,
        )
        self._schema_cache.clear()

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)
        self._schema_cache.clear()

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, access_level: str = "public") -> list[Tool]:
        """List available tools for a given access level.

        A tool is included if its permission digit for the access level is > 0.
        """
        if access_level == "blocked":
            return []

        return [
            tool for tool in self._tools.values()
            if tool.enabled and check_tool_access(tool.name, access_level, tool.permission, tool.operation)[0]
        ]

    def to_openai_schema(self, access_level: str = "public") -> list[dict]:
        """Convert available tools to OpenAI function calling format (cached)."""
        if access_level in self._schema_cache:
            return self._schema_cache[access_level]
        tools = self.list_tools(access_level)
        result = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        self._schema_cache[access_level] = result
        return result

    async def execute(
        self,
        name: str,
        arguments: dict,
        access_level: str = "public",
        scheduled: bool = False,
        provider=None,
        conv=None,
    ) -> "ToolResult":
        """Execute a tool by name. Returns ToolResult with error classification.

        `conv` (Conversation) is used only for the consent gate: it carries the
        pending-consent state and the reference back to the agent (with the
        ConsentStore). Callers without a conversation (tests, background jobs)
        can pass None — the gate falls back accordingly.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult(f"Error: Tool '{name}' not found.", ok=False, error_type="not_found")

        if not tool.enabled:
            return ToolResult(f"Error: Tool '{name}' is disabled.", ok=False, error_type="permission")

        # Permission check (rwx class + op bit)
        allowed, reason = check_tool_access(name, access_level, tool.permission, tool.operation)
        if not allowed:
            logger.warning(f"Permission denied: tool={name}, access_level={access_level}, perm={oct(tool.permission)}")
            return ToolResult(f"Error: {reason}", ok=False, error_type="permission")

        # Interactive approval (e.g., CLI file write confirmation)
        if self._approval_callback:
            try:
                approved, reason = await self._approval_callback(name, arguments)
                if not approved:
                    msg = f"Cancelled: {reason}" if reason else "Cancelled by user."
                    return ToolResult(msg, ok=False, error_type="permission")
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        # ─── Consent gate (generic, op=x + destructive op=w) ───────────────
        # Single choke point: any op=x tool and any op=w tool listed in
        # DESTRUCTIVE_TOOLS goes through consent.check_and_hold. Held → return
        # the prompt as a tool result so the LLM shows it to the owner.
        from ..consent import check_and_hold as _consent_gate
        _agent = None
        if conv is not None and getattr(conv, "_mgr", None) is not None:
            _agent = getattr(conv._mgr, "_agent", None)
        gate_action, gate_prompt = await _consent_gate(
            conv=conv, agent=_agent,
            tool_name=name, args=arguments, op=tool.operation,
            scheduled=scheduled,
        )
        if gate_action == "held":
            # Not a permission error and not retryable — a legitimate user-action
            # prompt. Use ok=True so the LLM surfaces it as the tool's normal
            # output rather than triggering error-recovery / retry logic.
            return ToolResult(gate_prompt or "", ok=True)

        try:
            call_args = {**arguments}
            if scheduled and name in ("send_message",):
                call_args["_scheduled"] = True
            if provider and name in ("spawn_subagent",):
                call_args["_provider"] = provider
            result = await tool.handler(**call_args)
            return ToolResult(str(result), ok=True)
        except TimeoutError as e:
            logger.error(f"Timeout executing tool '{name}': {e}")
            return ToolResult(f"Error executing '{name}': {e}", ok=False, retryable=True, error_type="timeout")
        except Exception as e:
            # Classify HTTP errors as retryable for 429/5xx
            retryable = False
            error_type = "unknown"
            try:
                import httpx
                if isinstance(e, httpx.HTTPStatusError):
                    code = e.response.status_code
                    if code == 429 or code >= 500:
                        retryable = True
                        error_type = "rate_limit" if code == 429 else "timeout"
            except ImportError:
                pass
            logger.error(f"Error executing tool '{name}': {e}")
            return ToolResult(f"Error executing '{name}': {e}", ok=False, retryable=retryable, error_type=error_type)
