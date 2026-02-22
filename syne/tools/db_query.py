"""db_query — Read-only database introspection for self-diagnosis.

Allows Syne to query its own PostgreSQL database for:
- Checking config values
- Inspecting memory entries
- Viewing session/message stats
- Diagnosing issues

STRICTLY READ-ONLY. Only SELECT queries are allowed.
Credential columns are redacted in output.
"""

import logging
import os
import re

logger = logging.getLogger("syne.tools.db_query")

# Columns whose values should be redacted in output
_REDACT_COLUMNS = frozenset({
    "api_key", "token", "secret", "password", "access_token",
    "refresh_token", "bot_token", "credentials",
})

# Max rows to return
_MAX_ROWS = 50

# Max output chars
_MAX_OUTPUT = 8000


def _is_read_only(sql: str) -> bool:
    """Check if SQL is a read-only query."""
    cleaned = sql.strip().rstrip(";").strip()
    # Remove leading comments
    cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL).strip()

    first_word = cleaned.split()[0].upper() if cleaned.split() else ""

    # Only allow SELECT and common read-only commands
    allowed = {"SELECT", "EXPLAIN", "SHOW", "\\DT", "\\D"}
    return first_word in allowed


def _redact_row(columns: list[str], row: tuple) -> tuple:
    """Redact sensitive column values."""
    result = []
    for col, val in zip(columns, row):
        col_lower = col.lower()
        if any(rc in col_lower for rc in _REDACT_COLUMNS):
            if val and str(val).strip():
                result.append(f"***({len(str(val))} chars)")
            else:
                result.append("(empty)")
        else:
            result.append(val)
    return tuple(result)


async def db_query_handler(sql: str) -> str:
    """Execute a read-only SQL query against the Syne database.

    Args:
        sql: SQL SELECT query to execute

    Returns:
        Query results as formatted text, or error message
    """
    if not sql or not sql.strip():
        return "Error: sql is required."

    # Validate read-only
    if not _is_read_only(sql):
        return "Error: Only SELECT queries are allowed. This tool is read-only."

    # Get connection string
    dsn = os.environ.get("SYNE_DATABASE_URL")
    if not dsn:
        return "Error: SYNE_DATABASE_URL not set in environment."

    try:
        import asyncpg
    except ImportError:
        return "Error: asyncpg not available."

    conn = None
    try:
        conn = await asyncpg.connect(dsn)

        # Execute with row limit
        limited_sql = sql.strip().rstrip(";")

        # Add LIMIT if not present
        if "LIMIT" not in limited_sql.upper():
            limited_sql += f" LIMIT {_MAX_ROWS}"

        rows = await conn.fetch(limited_sql)

        if not rows:
            return "Query returned 0 rows."

        # Get column names
        columns = list(rows[0].keys())

        # Redact sensitive columns
        clean_rows = []
        for row in rows:
            clean_rows.append(_redact_row(columns, tuple(row.values())))

        # Format as table
        lines = []
        lines.append(" | ".join(columns))
        lines.append("-+-".join("-" * max(len(c), 5) for c in columns))

        for row in clean_rows:
            vals = []
            for v in row:
                s = str(v) if v is not None else "NULL"
                if len(s) > 120:
                    s = s[:117] + "..."
                vals.append(s)
            lines.append(" | ".join(vals))

        result = f"({len(clean_rows)} rows)\n\n" + "\n".join(lines)

        if len(result) > _MAX_OUTPUT:
            result = result[:_MAX_OUTPUT] + "\n\n[... truncated]"

        return result

    except asyncpg.exceptions.PostgresSyntaxError as e:
        return f"SQL syntax error: {e}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if conn:
            await conn.close()


DB_QUERY_TOOL = {
    "name": "db_query",
    "description": (
        "Execute a read-only SQL SELECT query against your PostgreSQL database. "
        "Use this to inspect your own config, memories, sessions, messages, abilities, "
        "users, rules, and other tables. Only SELECT queries are allowed. "
        "Credential columns are automatically redacted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL SELECT query to execute (read-only)",
            },
        },
        "required": ["sql"],
    },
    "handler": db_query_handler,
    "requires_access_level": "owner",
}
