"""Compaction — summarize old conversations to save context window.

Modeled after OpenClaw's compaction system: structured summaries with
iterative updates and anti-hallucination rules.
"""

import logging
from typing import Optional
from .db.connection import get_connection
from .db.models import get_config
from .llm.provider import LLMProvider, ChatMessage

logger = logging.getLogger("syne.compaction")

# ── Initial summary prompt (no previous summary exists) ─────

COMPACTION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

CRITICAL: Do NOT attribute assistant suggestions as user preferences — only include what the user actually said or confirmed.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact names, identifiers, and error messages."""

# ── Update prompt (previous summary exists, merge new info) ──

UPDATE_COMPACTION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- If something is no longer relevant, you may remove it

CRITICAL: Do NOT attribute assistant suggestions as user preferences — only include what the user actually said or confirmed.

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work — update based on progress]

### Blocked
- [Current blockers — remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact names, identifiers, and error messages."""


def _serialize_messages(rows: list) -> str:
    """Serialize message rows into text for summarization.
    
    Wraps in [User]/[Assistant]/[Tool]/[System] labels so the model
    treats it as a transcript, not a conversation to continue.
    """
    parts = []
    for row in rows:
        role = row["role"]
        content = row["content"]
        # No truncation — let the summarizer see full messages
        # (OpenClaw sends all messages to summarizer without truncation)

        if role == "user":
            parts.append(f"[User]: {content}")
        elif role == "assistant":
            parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            parts.append(f"[Tool result]: {content}")
        elif role == "system":
            # Skip system prompts, but include compaction summaries
            if "compaction_summary" in str(row.get("metadata", "")):
                parts.append(f"[Previous summary]: {content}")
            # Otherwise skip (system prompt is noise for summarization)
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


async def get_session_stats(session_id: int) -> dict:
    """Get session statistics for compaction decisions.
    
    Returns:
        Dict with message_count, total_chars, oldest_message, newest_message
    """
    async with get_connection() as conn:
        row = await conn.fetchrow("""
            SELECT 
                COUNT(*) as message_count,
                COALESCE(SUM(LENGTH(content)), 0) as total_chars,
                MIN(created_at) as oldest_message,
                MAX(created_at) as newest_message
            FROM messages
            WHERE session_id = $1
        """, session_id)
        return {
            "message_count": row["message_count"],
            "total_chars": row["total_chars"],
            "oldest_message": row["oldest_message"],
            "newest_message": row["newest_message"],
        }


async def compact_session(
    session_id: int,
    provider: LLMProvider,
    keep_recent: int | None = None,
) -> Optional[dict]:
    """Compact a session by summarizing old messages.
    
    Uses OpenClaw-style structured summaries with iterative updates:
    - First compaction: generates full structured summary
    - Subsequent compactions: merges new messages into existing summary
    
    Keeps the most recent `keep_recent` messages and summarizes the rest.
    If keep_recent is None, reads from config (default: 40).
    Returns dict with summary and stats, or None if no compaction needed.
    """
    if keep_recent is None:
        keep_recent = await get_config("session.compaction_keep_recent", 40)
    async with get_connection() as conn:
        # Get pre-compaction stats
        pre_stats = await get_session_stats(session_id)
        total = pre_stats["message_count"]

        if total <= keep_recent + 5:
            return None  # Not enough to compact

        # Check for previous compaction summary
        prev_summary_row = await conn.fetchrow("""
            SELECT content FROM messages
            WHERE session_id = $1 AND metadata @> '{"type": "compaction_summary"}'::jsonb
            ORDER BY created_at DESC LIMIT 1
        """, session_id)

        previous_summary = None
        if prev_summary_row:
            content = prev_summary_row["content"]
            # Strip the "# Previous Conversation Summary\n" header if present
            if content.startswith("# Previous Conversation Summary\n"):
                previous_summary = content[len("# Previous Conversation Summary\n"):]
            else:
                previous_summary = content

        # Get messages to summarize (everything except recent)
        to_summarize = total - keep_recent
        old_rows = await conn.fetch("""
            SELECT id, role, content, metadata FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, to_summarize)

        if not old_rows:
            return None

        # Serialize conversation for summarization
        summarized_chars = sum(len(r["content"]) for r in old_rows)
        conv_text = _serialize_messages(old_rows)

        # No cap — send full conversation to summarizer
        # Context limit of the summarizer model is the natural constraint

        logger.info(
            f"Compacting session {session_id}: {to_summarize} messages ({summarized_chars} chars) → summary"
            f"{' (iterative update)' if previous_summary else ' (initial)'}"
        )

        # Build summarization prompt
        if previous_summary:
            # Iterative update: merge new messages into existing summary
            prompt_text = (
                f"<conversation>\n{conv_text}\n</conversation>\n\n"
                f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
                f"{UPDATE_COMPACTION_PROMPT}"
            )
        else:
            # First compaction: generate initial structured summary
            prompt_text = (
                f"<conversation>\n{conv_text}\n</conversation>\n\n"
                f"{COMPACTION_PROMPT}"
            )

        summary_response = await provider.chat(
            messages=[
                ChatMessage(role="user", content=prompt_text),
            ],
            temperature=0.1,
            max_tokens=16384,
        )

        summary = summary_response.content

        # Delete old messages
        old_ids = [r["id"] for r in old_rows]
        await conn.execute("DELETE FROM messages WHERE id = ANY($1)", old_ids)

        # Insert summary as first message in the session
        await conn.execute("""
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES ($1, 'system', $2, '{"type": "compaction_summary"}'::jsonb, 
                    (SELECT COALESCE(MIN(created_at), NOW()) FROM messages WHERE session_id = $1))
        """, session_id, f"# Previous Conversation Summary\n{summary}")

        # Update session record
        new_count = keep_recent + 1  # recent + summary
        await conn.execute("""
            UPDATE sessions
            SET summary = $2, message_count = $3, updated_at = NOW()
            WHERE id = $1
        """, session_id, summary, new_count)

        # Post-compaction stats
        post_stats = await get_session_stats(session_id)

        result = {
            "summary": summary,
            "messages_before": total,
            "messages_after": new_count,
            "messages_summarized": to_summarize,
            "chars_before": pre_stats["total_chars"],
            "chars_after": post_stats["total_chars"],
            "summary_length": len(summary),
            "is_update": previous_summary is not None,
        }

        logger.info(
            f"Compaction done: {to_summarize} messages ({summarized_chars} chars) → "
            f"{len(summary)} char summary. Session: {total} → {new_count} messages"
        )
        return result


async def should_compact(session_id: int, threshold: int = 100) -> bool:
    """Check if a session needs compaction based on message count."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT message_count FROM sessions WHERE id = $1",
            session_id,
        )
        if row:
            return row["message_count"] >= threshold
    return False


async def should_compact_by_chars(session_id: int, char_threshold: int = 80000) -> bool:
    """Check if a session needs compaction based on total character count."""
    stats = await get_session_stats(session_id)
    return stats["total_chars"] >= char_threshold


async def auto_compact_check(
    session_id: int,
    provider: LLMProvider,
) -> Optional[dict]:
    """Check and compact if needed using config thresholds. Returns result dict or None."""
    # Get thresholds from config
    msg_threshold = await get_config("session.max_messages", 100)
    char_threshold = await get_config("session.compaction_threshold", 80000)

    needs_compact = (
        await should_compact(session_id, msg_threshold)
        or await should_compact_by_chars(session_id, char_threshold)
    )

    if needs_compact:
        return await compact_session(session_id, provider)  # keep_recent from config
    return None
