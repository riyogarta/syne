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

# Columns whose NAMES should trigger redaction in output
_REDACT_COLUMNS = frozenset({
    "api_key", "token", "secret", "password", "access_token",
    "refresh_token", "bot_token", "credentials",
})

# Value-level secret detectors — redact a value REGARDLESS of its column name
# (defeats "SELECT bot_token AS x" aliasing bypass).
_SECRET_VALUE_PATTERNS = (
    # Telegram bot token: <digits>:<35-char base64url>
    re.compile(r'^\d{6,}:[A-Za-z0-9_-]{30,}$'),
    # OpenAI / Anthropic / generic sk- keys
    re.compile(r'^(sk|pk|rk)-[A-Za-z0-9_-]{16,}$'),
    # Google / AIza keys
    re.compile(r'^AIza[A-Za-z0-9_-]{20,}$'),
    # GitHub tokens
    re.compile(r'^gh[posru]_[A-Za-z0-9]{20,}$'),
    # postgres/redis connection strings
    re.compile(r'^[a-z]+://[^:@/]+:[^@/]+@'),
)


def _looks_secret(val) -> bool:
    """Heuristic: does this VALUE look like a credential/token?

    Catches secrets even when the column is aliased to an innocent name.
    Strategy: known token prefixes/formats, OR long high-entropy opaque
    strings (no spaces, mostly base64/hex alphabet).
    """
    if val is None:
        return False
    s = str(val).strip()
    if len(s) < 16:
        return False
    for pat in _SECRET_VALUE_PATTERNS:
        if pat.match(s):
            return True
    # Long opaque single-token strings: no whitespace, base64url/hex alphabet,
    # length >= 32. Typical of API keys / hashes / tokens.
    if len(s) >= 32 and " " not in s and "\n" not in s:
        if re.fullmatch(r'[A-Za-z0-9_\-+/=.:]+', s):
            # crude entropy gate: needs a mix of cases or digits to avoid
            # flagging long words / URLs paths.
            has_digit = any(c.isdigit() for c in s)
            has_alpha = any(c.isalpha() for c in s)
            if has_digit and has_alpha:
                return True
    return False

# Max rows to return
_MAX_ROWS = 50

# Max output chars
_MAX_OUTPUT = 8000

# Hard per-statement timeout (ms) — stops pg_sleep()/volatile-func DoS
_STATEMENT_TIMEOUT_MS = 5000


_DANGEROUS_KEYWORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "UPSERT", "GRANT", "REVOKE",
    "COPY", "EXECUTE", "CALL", "DO", "SET", "RESET",
    "VACUUM", "REINDEX", "CLUSTER", "COMMENT", "LOCK",
    "NOTIFY", "LISTEN", "UNLISTEN", "LOAD", "SECURITY",
    "REASSIGN", "DISCARD", "REFRESH", "IMPORT",
})


def _is_read_only(sql: str) -> bool:
    """Check if SQL is a read-only query.

    Two-layer check:
    1. First keyword must be SELECT, EXPLAIN, or SHOW
    2. Entire query must NOT contain any dangerous keywords (INSERT, DROP, etc.)
       This catches embedded writes like: SELECT 1; DROP TABLE users
       or subquery tricks like: SELECT * FROM (DELETE FROM users RETURNING *)
    """
    cleaned = sql.strip().rstrip(";").strip()
    # Remove comments
    cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE).strip()
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL).strip()

    if not cleaned:
        return False

    # Layer 1: first keyword must be read-only
    first_word = cleaned.split()[0].upper()
    allowed_start = {"SELECT", "EXPLAIN", "SHOW", "\\DT", "\\D"}
    if first_word not in allowed_start:
        return False

    # Layer 2: scan ALL tokens for dangerous keywords
    # Tokenize: split on whitespace and common delimiters, uppercase
    tokens = set(re.findall(r'[A-Za-z_]+', cleaned.upper()))
    found = tokens & _DANGEROUS_KEYWORDS
    if found:
        logger.warning(f"db_query blocked — dangerous keywords found: {found}")
        return False

    return True


def _redact_row(columns: list[str], row: tuple) -> tuple:
    """Redact sensitive values.

    Two independent triggers (either one redacts):
    1. Column NAME matches a known-sensitive name (api_key, token, ...).
    2. VALUE itself looks like a secret (token format or long high-entropy
       opaque string) — this defeats column aliasing such as
       ``SELECT bot_token AS x FROM users``.
    """
    result = []
    for col, val in zip(columns, row):
        col_lower = col.lower()
        name_hit = any(rc in col_lower for rc in _REDACT_COLUMNS)
        value_hit = _looks_secret(val)
        if name_hit or value_hit:
            if val is not None and str(val).strip():
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

        # Enforce read-only at the DATABASE level (defense-in-depth beyond the
        # keyword blocklist). A READ ONLY transaction rejects any write AND
        # blocks volatile side-effecting functions from mutating state. Also
        # bound execution time so pg_sleep()-style DoS can't hang the tool.
        async with conn.transaction(readonly=True):
            await conn.execute(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}")
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
    "description": "Execute a read-only SQL SELECT query on the database.",
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
    "permission": 0o700,
}
