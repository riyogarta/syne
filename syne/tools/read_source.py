"""read_source — Read-only access to Syne source code.

Allows Syne to read its own source code for:
- Self-healing: understanding how core works to diagnose bugs
- Bug reporting: reading relevant code to draft GitHub issues
- Feature proposals: understanding architecture to suggest improvements
- Ability development: reading base classes and examples

STRICTLY READ-ONLY. Writing is handled by exec (abilities only)
or blocked entirely (core files).
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("syne.tools.read_source")

# Project root (syne/ parent directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Allowed directories (relative to project root)
_ALLOWED_PATHS = [
    "syne/",
    "tests/",
    "README.md",
    "INSTALL.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "docker-compose.yml",
    "requirements.txt",
]

# Blocked patterns (never expose)
_BLOCKED_PATTERNS = [
    ".env",
    "secrets",
    "__pycache__",
    ".pyc",
    ".git/",
    "node_modules",
]

# Max lines per read — high enough to read most files in 1-2 calls
# (largest file agent.py ~1500 lines, security.py ~660 lines)
_MAX_LINES = 500


def _is_allowed_path(rel_path: str) -> bool:
    """Check if a relative path is within allowed scope."""
    # Normalize
    rel_path = rel_path.lstrip("./")

    # Block dangerous patterns
    for pattern in _BLOCKED_PATTERNS:
        if pattern in rel_path:
            return False

    # Must be within allowed paths
    for allowed in _ALLOWED_PATHS:
        if rel_path.startswith(allowed) or rel_path == allowed.rstrip("/"):
            return True

    return False


def _resolve_path(path: str) -> Optional[Path]:
    """Resolve and validate a source path."""
    # Normalize the input
    clean = path.strip().lstrip("./")

    # Prevent path traversal
    if ".." in clean:
        return None

    full_path = _PROJECT_ROOT / clean

    # Must resolve within project root
    try:
        resolved = full_path.resolve()
        if not str(resolved).startswith(str(_PROJECT_ROOT)):
            return None
    except (OSError, ValueError):
        return None

    if not _is_allowed_path(clean):
        return None

    return resolved


async def _handle_read_source(
    action: str = "read",
    path: str = "",
    offset: int = 1,
    limit: int = _MAX_LINES,
    pattern: str = "",
) -> str:
    """Handler for read_source tool."""

    if action == "tree":
        return _action_tree(path or "syne/")

    if action == "read":
        if not path:
            return "Error: 'path' is required. Example: path='syne/agent.py'"
        return _action_read(path, offset, limit)

    if action == "search":
        if not pattern:
            return "Error: 'pattern' is required for search."
        return _action_search(path or "syne/", pattern)

    return f"Unknown action: {action}. Use 'tree', 'read', or 'search'."


def _action_tree(path: str) -> str:
    """List directory tree."""
    resolved = _resolve_path(path)
    if not resolved:
        return f"Error: path '{path}' is not accessible."

    if resolved.is_file():
        size = resolved.stat().st_size
        return f"📄 {path} ({size:,} bytes)"

    if not resolved.is_dir():
        return f"Error: '{path}' not found."

    lines = [f"📁 {path}"]
    count = 0
    max_entries = 200

    for item in sorted(resolved.rglob("*")):
        if count >= max_entries:
            lines.append(f"  ... (truncated, {max_entries}+ entries)")
            break

        rel = item.relative_to(_PROJECT_ROOT)
        rel_str = str(rel)

        # Skip blocked
        skip = False
        for bp in _BLOCKED_PATTERNS:
            if bp in rel_str:
                skip = True
                break
        if skip:
            continue

        indent = "  " * (len(rel.parts) - len(Path(path).parts))
        if item.is_dir():
            lines.append(f"{indent}📁 {item.name}/")
        else:
            size = item.stat().st_size
            lines.append(f"{indent}📄 {item.name} ({size:,} bytes)")
        count += 1

    return "\n".join(lines)


def _action_read(path: str, offset: int, limit: int) -> str:
    """Read file contents with offset/limit."""
    resolved = _resolve_path(path)
    if not resolved:
        return f"Error: path '{path}' is not accessible."

    if not resolved.is_file():
        return f"Error: '{path}' is not a file."

    # Cap limit
    limit = min(limit, _MAX_LINES)
    offset = max(1, offset)

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        return f"Error reading '{path}': {e}"

    total = len(all_lines)
    start_idx = offset - 1  # 1-indexed to 0-indexed
    end_idx = min(start_idx + limit, total)

    if start_idx >= total:
        return f"Error: offset {offset} exceeds file length ({total} lines)."

    selected = all_lines[start_idx:end_idx]

    header = f"📄 {path} (lines {offset}-{end_idx} of {total})"
    if end_idx < total:
        header += f" — use offset={end_idx + 1} to continue"

    numbered = []
    for i, line in enumerate(selected, start=offset):
        numbered.append(f"{i:4d} | {line.rstrip()}")

    return header + "\n" + "\n".join(numbered)


def _action_search(path: str, pattern: str) -> str:
    """Search for pattern in source files (grep-like)."""
    resolved = _resolve_path(path)
    if not resolved:
        return f"Error: path '{path}' is not accessible."

    import re
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: invalid pattern '{pattern}': {e}"

    results = []
    max_results = 50

    if resolved.is_file():
        files = [resolved]
    else:
        files = sorted(resolved.rglob("*.py"))

    for fpath in files:
        rel = fpath.relative_to(_PROJECT_ROOT)
        rel_str = str(rel)

        # Skip blocked
        skip = False
        for bp in _BLOCKED_PATTERNS:
            if bp in rel_str:
                skip = True
                break
        if skip:
            continue

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append(f"{rel_str}:{lineno}: {line.rstrip()[:200]}")
                        if len(results) >= max_results:
                            break
        except Exception:
            continue

        if len(results) >= max_results:
            break

    if not results:
        return f"No matches for '{pattern}' in {path}."

    header = f"🔍 {len(results)} matches for '{pattern}' in {path}"
    if len(results) >= max_results:
        header += f" (truncated at {max_results})"

    return header + "\n" + "\n".join(results)


# ── Tool Registration Dict ──

READ_SOURCE_TOOL = {
    "name": "read_source",
    "description": (
        "Read Syne's own source code (READ-ONLY). "
        "Actions: 'tree' (list files/dirs), 'read' (read file with line numbers), "
        "'search' (grep pattern in source). "
        "Covers: syne/ (all core, tools, channels, db, llm, abilities), tests/, README.md, etc. "
        "Use for: self-healing diagnosis, bug reporting, feature proposals, understanding architecture."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tree", "read", "search"],
                "description": "tree=list files, read=read file contents, search=grep pattern",
            },
            "path": {
                "type": "string",
                "description": "File or directory path relative to project root (e.g. 'syne/agent.py', 'syne/security.py')",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default 1)",
            },
            "limit": {
                "type": "integer",
                "description": f"Max lines to read (default/max {_MAX_LINES}). Most files fit in 1-2 reads.",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (for 'search' action)",
            },
        },
        "required": ["action"],
    },
    "handler": _handle_read_source,
    "requires_access_level": "owner",
}
