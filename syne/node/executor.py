"""Node-side tool executor — runs tools locally on the remote node."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess

logger = logging.getLogger("syne.node.executor")


async def execute_tool(request_id: str, tool_name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool locally on the node.

    Args:
        request_id: Unique request ID from gateway
        tool_name: Tool name (exec, file_read, file_write, read_source)
        args: Tool arguments

    Returns:
        Tuple of (result_string, success_bool)
    """
    try:
        if tool_name == "exec":
            return await _exec(args)
        elif tool_name == "file_read":
            return await _file_read(args)
        elif tool_name == "file_write":
            return await _file_write(args)
        elif tool_name == "read_source":
            return await _read_source(args)
        else:
            return f"Unknown node tool: {tool_name}", False
    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return f"Error: {e}", False


async def _exec(args: dict) -> tuple[str, bool]:
    """Execute a shell command."""
    command = args.get("command", "")
    if not command:
        return "Error: No command provided", False

    timeout = min(int(args.get("timeout", 120)), 300)
    cwd = args.get("cwd") or os.getcwd()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"

        return output, proc.returncode == 0

    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: Command timed out after {timeout}s", False


async def _file_read(args: dict) -> tuple[str, bool]:
    """Read a file from the local filesystem."""
    path = args.get("path", "")
    if not path:
        return "Error: No path provided", False

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        return f"Error: File not found: {path}", False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return content, True
    except Exception as e:
        return f"Error reading {path}: {e}", False


async def _file_write(args: dict) -> tuple[str, bool]:
    """Write content to a file on the local filesystem."""
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "Error: No path provided", False

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}", True
    except Exception as e:
        return f"Error writing {path}: {e}", False


async def _read_source(args: dict) -> tuple[str, bool]:
    """Read source code — supports tree, read, and search actions."""
    action = args.get("action", "read")
    path = args.get("path", ".")

    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if action == "tree":
        return await _source_tree(path, args)
    elif action == "read":
        return await _file_read({"path": path})
    elif action == "search":
        return await _source_search(path, args)
    else:
        return f"Unknown read_source action: {action}", False


async def _source_tree(path: str, args: dict) -> tuple[str, bool]:
    """List directory tree."""
    max_depth = int(args.get("max_depth", 3))

    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}", False

    lines = []
    _walk_tree(path, lines, prefix="", depth=0, max_depth=max_depth)

    return "\n".join(lines), True


def _walk_tree(path: str, lines: list, prefix: str, depth: int, max_depth: int):
    """Recursively build directory tree."""
    if depth > max_depth:
        return

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        lines.append(f"{prefix}[permission denied]")
        return

    # Filter out common noise
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".ruff_cache"}
    entries = [e for e in entries if e not in skip]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        full_path = os.path.join(path, entry)

        if os.path.isdir(full_path):
            lines.append(f"{prefix}{connector}{entry}/")
            next_prefix = prefix + ("    " if is_last else "│   ")
            _walk_tree(full_path, lines, next_prefix, depth + 1, max_depth)
        else:
            lines.append(f"{prefix}{connector}{entry}")


async def _source_search(path: str, args: dict) -> tuple[str, bool]:
    """Search for pattern in files."""
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: No search pattern provided", False

    try:
        proc = await asyncio.create_subprocess_exec(
            "grep", "-rn", "--include=*.py", "-l", pattern, path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        result = stdout.decode("utf-8", errors="replace").strip()
        return result or "(no matches)", True
    except FileNotFoundError:
        # grep not available, fallback
        return "Error: grep not available on this system", False
    except asyncio.TimeoutError:
        return "Error: Search timed out", False
