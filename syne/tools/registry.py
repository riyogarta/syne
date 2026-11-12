"""Tool registry — register and manage agent capabilities."""

import logging
import time
from typing import Callable, Optional
from dataclasses import dataclass, field

from ..security import check_tool_access, TOOL_PERMISSIONS

logger = logging.getLogger("syne.tools.registry")

# Steps slower than this are logged so a stalled tool shows where it hung.
_SLOW_STEP_S = 10


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
    # Linux-style 3-digit octal (owner/family/public). Each digit's r/w/x
    # bits declare BOTH what that class can invoke AND how (read/additive-
    # write/destructive-action). The x bit is what triggers the consent
    # gate — see security.needs_consent.
    permission: int = 0o700
    enabled: bool = True
    hidden: bool = False                # callable but omitted from LLM schema
                                        # (used for deprecated aliases, e.g. exec->shell)
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
        hidden: bool = False,
        operation: str = None,  # deprecated no-op, kept for source compat
    ):
        """Register a new tool.

        Args:
            permission: Linux-style 3-digit octal (owner/family/public).
                Each digit encodes rwx: r=4 (read), w=2 (write), x=1 (action).
                Example: 0o110 — owner + family may perform the action; the
                x bit in either digit triggers the consent gate for that
                class of caller.
            scrub_level: Credential scrubbing level for tool output.
                "aggressive" — full regex scrub (default, safest)
                "safe" — high-confidence patterns only (for code/content output)
                "none" — tool has its own dedicated scrubber
            operation: DEPRECATED and ignored. The octal is now the sole
                source of truth for both access class AND op semantics.
        """
        del operation  # explicit no-op
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            permission=permission,
            scrub_level=scrub_level,
            hidden=hidden,
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
            if tool.enabled and not tool.hidden
            and check_tool_access(tool.name, access_level, tool.permission)[0]
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
        history_scope: Optional[dict] = None,
    ) -> "ToolResult":
        """Execute a tool by name. Returns ToolResult with error classification.

        `conv` (Conversation) is used only for the consent gate: it carries the
        pending-consent state and the reference back to the agent (with the
        ConsentStore). Callers without a conversation (tests, background jobs)
        can pass None — the gate falls back accordingly.
        """
        _t_start = time.monotonic()
        tool = self.get(name)
        if not tool:
            return ToolResult(f"Error: Tool '{name}' not found.", ok=False, error_type="not_found")

        if not tool.enabled:
            return ToolResult(f"Error: Tool '{name}' is disabled.", ok=False, error_type="permission")

        # Permission check — caller's own class digit must be non-zero.
        allowed, reason = check_tool_access(name, access_level, tool.permission)
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

        # ─── Consent gate ───────────────────────────────────────────────────
        # Single choke point: if the caller's own class digit has the x bit
        # set, the consent gate fires. No auxiliary op mapping needed — the
        # octal itself is the source of truth. Held → return prompt as an
        # ok=True result so the LLM surfaces the "balas ya" message to owner.
        from ..consent import check_and_hold as _consent_gate
        _agent = None
        if conv is not None and getattr(conv, "_mgr", None) is not None:
            _agent = getattr(conv._mgr, "_agent", None)
        gate_action, gate_prompt = await _consent_gate(
            conv=conv, agent=_agent,
            tool_name=name, args=arguments,
            access_level=access_level, permission=tool.permission,
            scheduled=scheduled,
        )
        if gate_action == "held":
            # Not a permission error and not retryable — a legitimate user-action
            # prompt. Use ok=True so the LLM surfaces it as the tool's normal
            # output rather than triggering error-recovery / retry logic.
            return ToolResult(gate_prompt or "", ok=True)

        _t_gate = time.monotonic() - _t_start
        if _t_gate > _SLOW_STEP_S:
            logger.warning(f"Tool '{name}': pre-handler (approval/consent) took {_t_gate:.1f}s")
        _t_handler = time.monotonic()
        try:
            call_args = {**arguments}
            if name in ("history_search", "history_expand"):
                # Internal knobs of history.auto_recall, never the model's.
                for _k in ("_exclude_session", "_query_vector"):
                    call_args.pop(_k, None)
                # Visibility comes from the live chat, never from the model:
                # anything it put under _scope is overwritten here. Sub-agents
                # pass the scope captured from their parent chat.
                from .history import build_history_scope
                call_args["_scope"] = (
                    history_scope if history_scope is not None
                    else build_history_scope(conv, access_level)
                )
            if name == "spawn_subagent":
                from .history import build_history_scope
                call_args["_history_scope"] = build_history_scope(conv, access_level)
            if scheduled and name in ("send_message",):
                call_args["_scheduled"] = True
            if name == "send_message":
                # Cross-channel limits depend on who is asking (owner vs
                # family), which the handler cannot see on its own.
                call_args["_access_level"] = access_level
            # Deterministic relay attribution: for a non-scheduled send_message
            # whose destination is a DIFFERENT chat than the current user (a
            # relay to someone else), auto-fill relay_from with the sender's
            # display name so the recipient always sees "Pesan dari <sender>:".
            # This does not depend on the LLM remembering the param.
            if name == "send_message" and not scheduled and conv is not None:
                try:
                    sender_user = getattr(conv, "user", None) or {}
                    sender_pid = str(sender_user.get("platform_id", "") or "")
                    dest = str(call_args.get("chat_id", "") or "")
                    # Compare BARE ids. A WhatsApp JID carries an @server suffix
                    # that the users row never has, so an unstripped compare
                    # makes a send to the sender's own WhatsApp look like a
                    # relay and prepends a bogus "Pesan dari <sender>:".
                    dest_bare = dest.split("@", 1)[0]
                    # Relay = destination is a real chat, different from sender,
                    # and not a group (group sends are scheduled-only anyway).
                    if (
                        dest
                        and sender_pid
                        and dest_bare != sender_pid
                        and not dest.startswith("-")
                        and not dest.endswith("@g.us")
                        and not str(call_args.get("relay_from", "")).strip()
                    ):
                        sender_name = (
                            sender_user.get("display_name")
                            or sender_user.get("name")
                            or ""
                        )
                        if sender_name:
                            call_args["relay_from"] = sender_name
                except Exception:
                    pass  # never block a send over attribution
            if provider and name in ("spawn_subagent",):
                call_args["_provider"] = provider
            result = await tool.handler(**call_args)
            _el = time.monotonic() - _t_handler
            if _el > _SLOW_STEP_S:
                logger.warning(f"Tool '{name}': handler took {_el:.1f}s")
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
