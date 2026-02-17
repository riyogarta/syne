"""Compaction — summarize old conversations to save context window."""

import logging
from typing import Optional
from .db.connection import get_connection
from .db.models import get_config
from .llm.provider import LLMProvider, ChatMessage

logger = logging.getLogger("syne.compaction")

COMPACTION_PROMPT = """Summarize this conversation into a concise summary that preserves:
1. Key decisions made
2. Important facts learned about the user
3. Tasks completed or in progress
4. Any commitments or promises made
5. Critical context needed for future conversations

Rules:
- Be factual. Only include what was explicitly stated or confirmed by the user.
- Do NOT include assistant suggestions that weren't confirmed.
- Do NOT include greetings, small talk, or filler.
- Do NOT make assumptions about user preferences.
- If the user corrected something, use the corrected version.
- Group related items under clear topic headings.

Format as bullet points grouped by topic."""


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
    keep_recent: int = 20,
) -> Optional[dict]:
    """Compact a session by summarizing old messages.
    
    Keeps the most recent `keep_recent` messages and summarizes the rest.
    Returns dict with summary and stats, or None if no compaction needed.
    """
    async with get_connection() as conn:
        # Get pre-compaction stats
        pre_stats = await get_session_stats(session_id)
        total = pre_stats["message_count"]

        if total <= keep_recent + 5:
            return None  # Not enough to compact

        # Get messages to summarize (everything except recent)
        to_summarize = total - keep_recent
        old_rows = await conn.fetch("""
            SELECT id, role, content FROM messages
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, to_summarize)

        if not old_rows:
            return None

        # Build conversation text for summarization
        conv_lines = []
        summarized_chars = 0
        for row in old_rows:
            role = row["role"].upper()
            content = row["content"]
            summarized_chars += len(content)
            # Truncate very long individual messages
            if len(content) > 500:
                content = content[:500] + "..."
            conv_lines.append(f"{role}: {content}")

        conv_text = "\n\n".join(conv_lines)

        # Cap input to summarizer
        if len(conv_text) > 30000:
            conv_text = conv_text[:30000] + "\n\n[...truncated...]"

        logger.info(f"Compacting session {session_id}: {to_summarize} messages ({summarized_chars} chars) → summary")

        # Summarize via LLM
        summary_response = await provider.chat(
            messages=[
                ChatMessage(role="system", content=COMPACTION_PROMPT),
                ChatMessage(role="user", content=conv_text),
            ],
            temperature=0.1,
            max_tokens=2000,
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
    keep_recent: int = 20,
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
        return await compact_session(session_id, provider, keep_recent)
    return None
