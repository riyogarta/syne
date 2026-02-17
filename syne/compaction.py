"""Compaction — summarize old conversations to save context window."""

import logging
from typing import Optional
from .db.connection import get_connection
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

Format as bullet points grouped by topic."""


async def compact_session(
    session_id: int,
    provider: LLMProvider,
    keep_recent: int = 20,
) -> Optional[str]:
    """Compact a session by summarizing old messages.
    
    Keeps the most recent `keep_recent` messages and summarizes the rest.
    Returns the summary text, or None if no compaction needed.
    """
    async with get_connection() as conn:
        # Count messages
        row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM messages WHERE session_id = $1",
            session_id,
        )
        total = row["count"]

        if total <= keep_recent + 10:
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

        # Build conversation text
        conv_lines = []
        for row in old_rows:
            role = row["role"].upper()
            content = row["content"]
            # Truncate very long messages in the summary input
            if len(content) > 500:
                content = content[:500] + "..."
            conv_lines.append(f"{role}: {content}")

        conv_text = "\n\n".join(conv_lines)

        # Don't send too much to summarizer
        if len(conv_text) > 30000:
            conv_text = conv_text[:30000] + "\n\n[...truncated...]"

        logger.info(f"Compacting session {session_id}: {to_summarize} messages → summary")

        # Summarize
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

        # Insert summary as first message
        await conn.execute("""
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES ($1, 'system', $2, '{"type": "compaction_summary"}'::jsonb, 
                    (SELECT MIN(created_at) FROM messages WHERE session_id = $1))
        """, session_id, f"# Previous Conversation Summary\n{summary}")

        # Update session
        new_count = keep_recent + 1  # recent + summary
        await conn.execute("""
            UPDATE sessions
            SET summary = $2, message_count = $3, updated_at = NOW()
            WHERE id = $1
        """, session_id, summary, new_count)

        logger.info(f"Compaction done: {to_summarize} messages → {len(summary)} chars summary")
        return summary


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


async def auto_compact_check(
    session_id: int,
    provider: LLMProvider,
    message_threshold: int = 100,
    keep_recent: int = 20,
) -> Optional[str]:
    """Check and compact if needed. Returns summary or None."""
    if await should_compact(session_id, message_threshold):
        return await compact_session(session_id, provider, keep_recent)
    return None
