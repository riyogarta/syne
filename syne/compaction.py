"""Compaction — summarize old conversations to save context window.

Modeled after OpenClaw's compaction system: structured summaries with
iterative updates and anti-hallucination rules.
"""

import logging
import re
from typing import Optional
from .db.connection import get_connection
from .db.models import get_config
from .llm.provider import LLMProvider, ChatMessage
from .context import DEFAULT_CHARS_PER_TOKEN

logger = logging.getLogger("syne.compaction")

PROMPT_OVERHEAD_TOKENS = 4_096  # space for compaction prompt template + tags
OUTPUT_TOKENS = 16_384  # max_tokens for summary output
# Compaction uses a hardcoded conservative ratio (not the user-configurable one)
# because tokenizer efficiency varies wildly: English ~4 chars/token, but mixed
# Indonesian/code/JSON can be as low as 2.5. Using 3.0 as safe middle ground.
COMPACTION_CHARS_PER_TOKEN = 3.0

_BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{200,}={0,2}')
_MAX_TOOL_RESULT_FOR_SUMMARY = 2_000

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

        if role == "tool":
            # Strip base64 blobs — waste of summarizer tokens
            content = _BASE64_PATTERN.sub("[base64 data removed]", content)
            # Cap long tool results for summarization
            if len(content) > _MAX_TOOL_RESULT_FOR_SUMMARY:
                content = content[:_MAX_TOOL_RESULT_FOR_SUMMARY] + "\n[... truncated for summary]"

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


def _build_preservation_context(messages: list) -> str:
    """Build preservation context from recent messages for pre-compact injection.

    Extracts key signals from the messages that will be KEPT (not summarized)
    so the summarizer knows what is currently active and can prioritize
    relevant context in its summary.

    Returns a string (max ~500 chars) or empty string if nothing useful.
    """
    if not messages:
        return ""

    parts = []

    # Last user message (what user is currently asking about)
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
        if role == "user" and content:
            parts.append(f"Last user request: {content[:150]}")
            break

    # Last 3 tool names used (what work is in progress)
    tool_names = []
    for msg in reversed(messages):
        if len(tool_names) >= 3:
            break
        meta = getattr(msg, "metadata", None) or (msg.get("metadata") if isinstance(msg, dict) else None)
        if meta and isinstance(meta, dict) and meta.get("tool_name"):
            tool_names.append(meta["tool_name"])
    if tool_names:
        parts.append(f"Recent tools: {', '.join(reversed(tool_names))}")

    # Check for in-progress patterns (assistant with tool_calls but no final text response yet)
    has_pending = False
    for msg in reversed(messages[:5]):
        meta = getattr(msg, "metadata", None) or (msg.get("metadata") if isinstance(msg, dict) else None)
        if meta and isinstance(meta, dict) and meta.get("tool_calls"):
            has_pending = True
            break
    if has_pending:
        parts.append("Status: tool execution in progress")

    result = "\n".join(parts)
    return result[:500]


async def compact_session(
    session_id: int,
    provider: LLMProvider,
    keep_recent: int | None = None,
    recent_context: str = "",
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> Optional[dict]:
    """Compact a session by summarizing old messages.

    Uses OpenClaw-style structured summaries with iterative updates:
    - First compaction: generates full structured summary
    - Subsequent compactions: merges new messages into existing summary

    Keeps the most recent `keep_recent` messages and summarizes the rest.
    If keep_recent is None, reads from config (default: 40).
    recent_context: preservation context from recent (kept) messages,
        injected into the summarization prompt so the summarizer
        prioritizes context relevant to ongoing activity.
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

        # Cap input based on provider's actual context window.
        # Use conservative COMPACTION_CHARS_PER_TOKEN (3.0) instead of user-configurable
        # chars_per_token because tokenizer efficiency varies: English ~4, mixed ~2.5-3.
        cpt = COMPACTION_CHARS_PER_TOKEN
        available_tokens = provider.context_window - PROMPT_OVERHEAD_TOKENS - OUTPUT_TOKENS

        # Subtract space used by previous_summary + preservation context
        extra_chars = 0
        if previous_summary:
            extra_chars += len(previous_summary) + 100  # tags + header
        if recent_context:
            extra_chars += len(recent_context) + 200  # tags + note text
        available_tokens -= int(extra_chars / cpt)

        max_input_chars = int(available_tokens * cpt)
        if max_input_chars < 1000:
            max_input_chars = 1000  # absolute minimum
        if len(conv_text) > max_input_chars:
            conv_text = conv_text[-max_input_chars:]
            logger.info(
                f"Truncated summarizer input to {max_input_chars} chars "
                f"(from {summarized_chars}, available_tokens={available_tokens}, "
                f"context_window={provider.context_window}, cpt={cpt})"
            )

        logger.info(
            f"Compacting session {session_id}: {to_summarize} messages ({summarized_chars} chars) → summary"
            f"{' (iterative update)' if previous_summary else ' (initial)'}"
        )

        # Build summarization prompt
        preservation_block = ""
        if recent_context:
            preservation_block = (
                f"\n\n<recent-activity>\n{recent_context}\n</recent-activity>\n\n"
                "Note: The above shows what is currently active in the conversation. "
                "Prioritize preserving context relevant to these ongoing activities."
            )

        if previous_summary:
            prompt_text = (
                f"<conversation>\n{conv_text}\n</conversation>\n\n"
                f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
                f"{UPDATE_COMPACTION_PROMPT}{preservation_block}"
            )
        else:
            prompt_text = (
                f"<conversation>\n{conv_text}\n</conversation>\n\n"
                f"{COMPACTION_PROMPT}{preservation_block}"
            )

        summary_response = await provider.chat(
            messages=[
                ChatMessage(role="user", content=prompt_text),
            ],
            temperature=0.1,
            max_tokens=16384,
            thinking_budget=0,  # No thinking needed for summarization
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


async def should_compact_by_chars(session_id: int, char_threshold: int = 150000) -> bool:
    """Check if a session needs compaction based on total character count."""
    stats = await get_session_stats(session_id)
    return stats["total_chars"] >= char_threshold
