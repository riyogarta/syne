"""File Operations — Read and write files (owner-only).

Rule 700: Only the owner can use these tools.

Security restrictions for file_write:
- CAN write to CWD (working directory) or descendants
- CAN write to syne/abilities/ (for dynamic ability creation)
- CANNOT write to syne/ core directories (engine, tools, channels, db, llm, security)

This follows the self-edit pattern: abilities are editable, core is not.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from ..db.models import get_config

logger = logging.getLogger("syne.tools.file_ops")

# Project root (syne/ parent directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Syne core directories that CANNOT be written to
_SYNE_CORE_DIRS = frozenset({
    "syne/engine",
    "syne/tools",
    "syne/channels",
    "syne/db",
    "syne/llm",
    "syne/security",
    "syne/auth",
    "syne/memory",
})

# Allowed to write even inside syne/
_SYNE_WRITABLE_PATHS = frozenset({
    "syne/abilities",
})

# Workspace directory — set by agent at startup for non-CLI channels.
# When set, file_write resolves relative paths here instead of process CWD.
_workspace_dir: Optional[str] = None


def set_workspace(workspace_path: str) -> None:
    """Set the workspace directory for file_write resolution.
    
    Called by SyneAgent at startup. When set, relative paths in file_write
    resolve to workspace/ instead of os.getcwd().
    """
    global _workspace_dir
    _workspace_dir = workspace_path


# Default max read size (100KB)
_DEFAULT_MAX_READ_SIZE = 100 * 1024  # 100KB

# Max lines per read (same as read_source)
_MAX_LINES = 500


def _is_path_under(path: Path, parent: Path) -> bool:
    """Check if path is under parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_write_allowed(path: Path, cwd: Path) -> tuple[bool, str]:
    """Check if writing to this path is allowed.
    
    Rules:
    1. Path must be absolute or resolved
    2. Path must be inside CWD or syne/abilities/
    3. Path must NOT be inside syne/ core directories
    
    Args:
        path: Absolute path to check
        cwd: Current working directory (context)
        
    Returns:
        Tuple of (allowed: bool, reason: str)
    """
    resolved = path.resolve()
    
    # Check if it's inside workspace/ (always writable)
    workspace_path = _PROJECT_ROOT / "workspace"
    if _is_path_under(resolved, workspace_path):
        return True, ""

    # Check if it's inside CWD
    is_in_cwd = _is_path_under(resolved, cwd)
    
    # Check if it's inside syne/abilities/
    is_in_abilities = False
    for writable in _SYNE_WRITABLE_PATHS:
        writable_path = _PROJECT_ROOT / writable
        if _is_path_under(resolved, writable_path):
            is_in_abilities = True
            break
    
    # Check if it's inside syne/ core (blocked)
    is_in_core = False
    for core_dir in _SYNE_CORE_DIRS:
        core_path = _PROJECT_ROOT / core_dir
        if _is_path_under(resolved, core_path):
            is_in_core = True
            break
    
    # Also check direct syne/ files (not in abilities)
    is_direct_syne_file = False
    if _is_path_under(resolved, _PROJECT_ROOT / "syne"):
        if not is_in_abilities and resolved.parent == (_PROJECT_ROOT / "syne"):
            # Direct file in syne/ (like syne/agent.py)
            is_direct_syne_file = True
    
    # Decision
    if is_in_core:
        return False, f"Cannot write to Syne core directory. Path is inside protected area."
    
    if is_direct_syne_file:
        return False, f"Cannot write directly to syne/. Only syne/abilities/ is writable."
    
    if is_in_abilities:
        return True, ""
    
    if is_in_cwd:
        return True, ""
    
    # Not in CWD and not in abilities
    return False, f"Path must be inside working directory or syne/abilities/."


def _read_env_redacted(file_path: Path) -> str:
    """Read .env file with all values redacted.
    
    Shows KEY=*** for each variable so the agent can diagnose
    which variables are set without seeing actual credentials.
    """
    import re
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except Exception as e:
        return f"Error reading {file_path.name}: {e}"
    
    redacted = []
    for line in lines:
        stripped = line.strip()
        # Preserve empty lines and comments
        if not stripped or stripped.startswith("#"):
            redacted.append(stripped)
            continue
        # KEY=VALUE → KEY=***
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', stripped)
        if match:
            key = match.group(1)
            val = match.group(2)
            # Show value length hint for debugging
            clean_val = val.strip().strip("'\"")
            if clean_val:
                redacted.append(f"{key}=***({len(clean_val)} chars)")
            else:
                redacted.append(f"{key}=(empty)")
        else:
            redacted.append("# (non-standard line redacted)")
    
    header = f"📄 {file_path.name} (values redacted for security):\n"
    return header + "\n".join(redacted)


async def file_read_handler(
    path: str,
    offset: int = 1,
    limit: int = _MAX_LINES,
) -> str:
    """Read file contents.
    
    Args:
        path: Path to the file (absolute or relative to CWD)
        offset: Line number to start from (1-indexed)
        limit: Maximum number of lines to read
        
    Returns:
        File contents or error message
    """
    if not path:
        return "Error: path is required."
    
    # Get max read size from config
    max_read_size = await get_config("file_ops.max_read_size", _DEFAULT_MAX_READ_SIZE)
    
    # Resolve path
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    
    file_path = file_path.resolve()
    
    # Block sensitive files — credentials must never reach the LLM
    _BLOCKED_FILENAMES = {".env", ".env.local", ".env.production", ".env.development"}
    _BLOCKED_PATTERNS = {"secrets", ".pem", ".key", "id_rsa", "id_ed25519"}
    fname = file_path.name.lower()
    if fname in _BLOCKED_FILENAMES:
        # Allow reading key names only (values redacted) for self-diagnosis
        return _read_env_redacted(file_path)
    if any(p in fname for p in _BLOCKED_PATTERNS):
        return f"Error: Access denied — {file_path.name} contains credentials and cannot be read."
    
    if not file_path.exists():
        return f"Error: File not found: {path}"
    
    if not file_path.is_file():
        return f"Error: Not a file: {path}"
    
    # Check file size
    file_size = file_path.stat().st_size
    if file_size > max_read_size:
        return (
            f"Error: File too large ({file_size:,} bytes). "
            f"Max allowed: {max_read_size:,} bytes. "
            f"Increase via update_config(key='file_ops.max_read_size', value='{max_read_size * 2}')"
        )
    
    # Read file
    offset = max(1, offset)
    limit = min(limit, _MAX_LINES)
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"
    
    total = len(all_lines)
    
    if total == 0:
        return f"📄 {path} (empty file)"
    
    start_idx = offset - 1  # 1-indexed to 0-indexed
    end_idx = min(start_idx + limit, total)
    
    if start_idx >= total:
        return f"Error: offset {offset} exceeds file length ({total} lines)."
    
    selected = all_lines[start_idx:end_idx]
    
    header = f"📄 {path} (lines {offset}-{end_idx} of {total})"
    if end_idx < total:
        header += f" — use offset={end_idx + 1} to continue"
    
    # Add line numbers
    numbered = []
    for i, line in enumerate(selected, start=offset):
        numbered.append(f"{i:4d} | {line.rstrip()}")
    
    return header + "\n" + "\n".join(numbered)


async def file_write_handler(
    path: str,
    content: str,
    workdir: str = "",
) -> str:
    """Write content to a file.
    
    Security: Only allows writing to:
    - Files inside the working directory (CWD)
    - Files inside syne/abilities/ (for dynamic ability creation)
    
    Will auto-create parent directories.
    
    Args:
        path: Path to the file (absolute or relative to workdir/CWD)
        content: Content to write
        workdir: Working directory context (defaults to CWD)
        
    Returns:
        Success message or error
    """
    if not path:
        return "Error: path is required."
    
    if content is None:
        return "Error: content is required (use empty string for empty file)."
    
    # Determine working directory
    # Priority: explicit workdir > workspace (Telegram) > process CWD (CLI)
    if workdir:
        cwd = Path(workdir).resolve()
    elif _workspace_dir:
        cwd = Path(_workspace_dir).resolve()
    else:
        cwd = Path.cwd()  # CLI mode — user's actual directory
    
    # Resolve path
    file_path = Path(path)
    if not file_path.is_absolute():
        # Special case: syne/abilities/ paths always resolve to project root
        # (self-edit capability), not to workspace
        path_str = str(file_path)
        if path_str.startswith("syne/abilities/") or path_str.startswith("syne/abilities\\"):
            file_path = _PROJECT_ROOT / file_path
        else:
            file_path = cwd / file_path
    
    file_path = file_path.resolve()
    
    # Security check
    allowed, reason = _check_write_allowed(file_path, cwd)
    if not allowed:
        logger.warning(f"file_write blocked: {file_path} — {reason}")
        return f"Error: {reason}"
    
    # Create parent directories
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return f"Error: Permission denied creating directory: {file_path.parent}"
    except Exception as e:
        return f"Error creating directories: {e}"
    
    # Write file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except PermissionError:
        return f"Error: Permission denied writing to: {path}"
    except Exception as e:
        return f"Error writing file: {e}"
    
    size = len(content.encode("utf-8"))
    logger.info(f"file_write: {file_path} ({size} bytes)")
    
    return f"✅ File written: {path} ({size:,} bytes)"


# ── Tool Registration Dicts ──

FILE_READ_TOOL = {
    "name": "file_read",
    "description": (
        "Read contents of a file. "
        "Supports offset/limit for reading large files in chunks. "
        "Max file size configurable via file_ops.max_read_size (default 100KB)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to CWD)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default 1)",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of lines to read (default/max {_MAX_LINES})",
            },
        },
        "required": ["path"],
    },
    "handler": file_read_handler,
    "requires_access_level": "owner",
}

FILE_WRITE_TOOL = {
    "name": "file_write",
    "description": (
        "Write content to a file (create or overwrite). "
        "Auto-creates parent directories. "
        "SECURITY: Only allows writing to CWD or syne/abilities/. "
        "Cannot write to syne/ core directories (engine, tools, channels, db, llm, security)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to workdir/CWD)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory context (defaults to CWD)",
            },
        },
        "required": ["path", "content"],
    },
    "handler": file_write_handler,
    "requires_access_level": "owner",
}
