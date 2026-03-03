"""Tool registry — register and manage agent capabilities."""

import logging
from typing import Callable, Optional
from dataclasses import dataclass, field

from ..security import check_tool_access, TOOL_PERMISSIONS

logger = logging.getLogger("syne.tools.registry")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                    # JSON Schema for parameters
    handler: Callable                   # async function to execute
    permission: int = 0o700             # Linux-style 3-digit octal (owner/family/public)
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
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            permission=permission,
            scrub_level=scrub_level,
        )

    def unregister(self, name: str):
        """Remove a tool."""
        self._tools.pop(name, None)

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
            if tool.enabled and check_tool_access(tool.name, access_level, tool.permission)[0]
        ]

    def to_openai_schema(self, access_level: str = "public") -> list[dict]:
        """Convert available tools to OpenAI function calling format."""
        tools = self.list_tools(access_level)
        return [
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

    async def execute(self, name: str, arguments: dict, access_level: str = "public", scheduled: bool = False, provider=None) -> str:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."

        if not tool.enabled:
            return f"Error: Tool '{name}' is disabled."

        # Permission check
        allowed, reason = check_tool_access(name, access_level, tool.permission)
        if not allowed:
            logger.warning(f"Permission denied: tool={name}, access_level={access_level}, perm={oct(tool.permission)}")
            return f"Error: {reason}"

        # Interactive approval (e.g., CLI file write confirmation)
        if self._approval_callback:
            try:
                approved, reason = await self._approval_callback(name, arguments)
                if not approved:
                    return f"Cancelled: {reason}" if reason else "Cancelled by user."
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        try:
            call_args = {**arguments}
            if scheduled and name in ("send_message",):
                call_args["_scheduled"] = True
            if provider and name in ("spawn_subagent",):
                call_args["_provider"] = provider
            result = await tool.handler(**call_args)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            return f"Error executing '{name}': {e}"
