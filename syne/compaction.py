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

# ── Initial summary prompt (no previous summary exists) ─────

COMPACTION_PROMPT = """The messages above are a conversation to summarize. Create a detailed summary that another LLM will use to continue the conversation seamlessly — the user should NOT notice any loss of context.

CRITICAL RULES:
- Do NOT attribute assistant suggestions as user preferences — only include what the user actually said or confirmed.
- Preserve ALL specific details: names, numbers, dates, data, preferences, facts mentioned.
- Include the TONE and FLOW of the conversation, not just tasks.

CRITICAL — BUG / FAILURE / COMPLAINT NARRATIVES:
Debugging state is EPHEMERAL. If the transcript contains bug reports, failure
claims, "output tidak sampai", "continuation putus", "tombol tidak muncul",
"muter-muter", or similar, apply this rule:
- If a later message in the transcript shows the issue was FIXED, RESOLVED, or
  DEMONSTRATED WORKING → record it in `## Resolved Issues` with the fix and
  do NOT restate it as an active problem anywhere else.
- If the issue was ABANDONED (user gave up, changed direction, or turned the
  feature off) → record in `## Resolved Issues` as "abandoned" — do NOT carry
  it forward as an open bug.
- ONLY put issues in `## Open Issues` when the transcript shows the user
  ended the conversation still actively wanting them fixed AND no fix has
  been demonstrated yet.
- NEVER preserve frustration expressions ("kzl", "goblok", "capek",
  exclamation floods) as facts. They describe the moment, not durable state.

The next-turn LLM MUST NOT wake up believing a previously-fixed or abandoned
bug is still active. Preserving stale bug narratives as if they were current
is the #1 cause of Syne hallucinating that tools are broken when they work.

Use this format:

## Conversation Summary
[Detailed narrative of what was discussed, in chronological order. Include what the user said, what was agreed upon, what was debated. Be specific — "user asked to embed Al Hadist" not "user requested data processing".]

## Key Facts & Data
- [Specific facts, names, numbers, preferences mentioned by the user]
- [Any data the user provided or referenced]

## Decisions Made
- [What was decided and why — include context]

## Current State
- What was completed
- What is in progress
- What needs to be done next

## Resolved Issues
- [Issues that were fixed / abandoned during this window, with brief resolution note. Future turns must treat these as CLOSED.]

## Open Issues
- [ONLY issues the user was still actively pursuing when the window ended. Empty section is fine and expected.]

## Important Context
- [Anything needed to continue naturally — ongoing topics, user's mood/preferences, unresolved questions]

Be DETAILED. A longer summary that preserves context is better than a short one that loses it."""

# ── Update prompt (previous summary exists, merge new info) ──

UPDATE_COMPACTION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the summary with new information. RULES:
- PRESERVE all existing information — do NOT drop details from previous summary
- ADD new conversations, facts, decisions from the new messages
- If a topic evolved, UPDATE it — don't just append
- Only remove information that is explicitly contradicted or no longer relevant

CRITICAL: Do NOT attribute assistant suggestions as user preferences — only include what the user actually said or confirmed.

CRITICAL — BUG STATE MIGRATION:
When updating the summary, actively re-evaluate the `## Open Issues` section:
- If the NEW messages show a previously-open issue was fixed, verified working,
  or abandoned → MOVE it from Open Issues to `## Resolved Issues` with a
  one-line resolution note. Do NOT leave it in Open Issues.
- If the NEW messages show a stale bug narrative was actually a hallucination
  (assistant claimed a tool failed but the tool_result was actually present)
  → MOVE it to Resolved Issues as "resolved: was hallucination, tool worked".
- NEVER copy an Open Issue forward to the new summary unchanged unless the
  NEW messages show the user still actively pursuing it AND no fix has
  landed. Stale open bugs poison future turns.

Use this format:

## Conversation Summary
[Merge previous summary narrative with new conversations. Keep chronological flow.]

## Key Facts & Data
- [Preserve ALL previous facts, add new ones]

## Decisions Made
- [Preserve previous decisions, add new ones with context]

## Current State
- What was completed
- What is in progress
- What needs to be done next

## Resolved Issues
- [Everything previously resolved, PLUS any Open Issues that got resolved/abandoned in the new window]

## Open Issues
- [ONLY issues still active at the end of the new window. Empty section is fine and expected — do not fill it with stale carryovers.]

## Important Context
- [Preserve and update ongoing context]

Be DETAILED. A longer summary that preserves context is better than a short one that loses it."""


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
            # Base64 already stripped above — no additional truncation

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
            WHERE session_id = $1 AND status = 'active'
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
            WHERE session_id = $1 AND status = 'active'
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, to_summarize)

        if not old_rows:
            return None

        # Calculate max input chars from provider's context window
        cpt = COMPACTION_CHARS_PER_TOKEN
        available_tokens = provider.context_window - PROMPT_OVERHEAD_TOKENS - OUTPUT_TOKENS

        # Subtract space used by previous_summary + preservation context
        extra_chars = 0
        if previous_summary:
            extra_chars += len(previous_summary) + 100  # tags + header
        if recent_context:
            extra_chars += len(recent_context) + 200  # tags + note text
        available_tokens -= int(extra_chars / cpt)

        max_input_chars = max(1000, int(available_tokens * cpt))

        # Adaptive reduction: if messages exceed context, reduce by 4 from
        # oldest until they fit. Remaining messages stay in DB for next compact.
        while len(old_rows) > 4:
            conv_text = _serialize_messages(old_rows)
            if len(conv_text) <= max_input_chars:
                break
            old_rows = old_rows[4:]  # drop 4 oldest
            logger.info(f"Compact adaptive reduction: {len(old_rows)} msgs, {len(conv_text)} chars (max {max_input_chars})")
        else:
            conv_text = _serialize_messages(old_rows)

        summarized_chars = sum(len(r["content"]) for r in old_rows)

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

        # Overlap band: keep the last N% of the summarized batch ALSO as raw
        # (active) messages, so the window transitions smoothly from the
        # condensed summary into verbatim pre-compaction context before the
        # kept recent tail. These overlap messages are ALSO covered by the
        # summary (dual presence = the bridge). Percentage-based, not a fixed
        # count. Default 0% = legacy behavior (archive the entire batch).
        overlap_pct = await get_config("session.compaction_overlap_percent", 0)
        try:
            overlap_pct = float(overlap_pct)
        except (TypeError, ValueError):
            overlap_pct = 0.0
        overlap_pct = max(0.0, min(100.0, overlap_pct))

        overlap_n = int(len(old_rows) * overlap_pct / 100)
        # Never retain the whole batch — at least 1 message must be archived,
        # otherwise nothing is actually compacted.
        overlap_n = min(overlap_n, len(old_rows) - 1)

        # Char-budget guard: the retained raw overlap must never exceed half the
        # summarizer input budget. If the tail is heavy (long messages), shrink
        # overlap_n from the oldest side until it fits. This guarantees the
        # overlap can't blow past the context window regardless of the % chosen.
        if overlap_n > 0:
            budget = max_input_chars * 0.5
            while overlap_n > 0:
                tail = old_rows[len(old_rows) - overlap_n:]
                if sum(len(r["content"]) for r in tail) <= budget:
                    break
                overlap_n -= 1

        archive_rows = old_rows[:len(old_rows) - overlap_n] if overlap_n else old_rows

        # Soft-archive old messages (NEVER delete — retained in DB for semantic
        # search & recovery). They are excluded from context (load_history filters
        # status='active') and from future compaction (queries above filter active).
        # The last `overlap_n` messages are intentionally left active as the bridge.
        old_ids = [r["id"] for r in archive_rows]
        if old_ids:
            await conn.execute(
                "UPDATE messages SET status = 'compacted' WHERE id = ANY($1)", old_ids
            )

        # Insert summary as first message in the session
        await conn.execute("""
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES ($1, 'system', $2, '{"type": "compaction_summary"}'::jsonb, 
                    (SELECT COALESCE(MIN(created_at), NOW()) FROM messages WHERE session_id = $1))
        """, session_id, f"# Previous Conversation Summary\n{summary}")

        # Update session record
        new_count = keep_recent + 1 + overlap_n  # recent + summary + overlap bridge
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
