"""Memory evaluator — decide what's worth remembering."""

import logging
from typing import Optional
import httpx
from ..llm.provider import LLMProvider, ChatMessage

logger = logging.getLogger("syne.memory.evaluator")

EVALUATE_PROMPT = """You are a memory evaluator. Analyze the user's message and determine if it contains information worth storing as a long-term memory.

STORE when the user states:
- Personal facts (name, age, job, location, family)
- Preferences (likes, dislikes, habits)
- Important events or milestones
- Decisions or commitments
- Lessons learned
- Configuration or technical setup notes
- Health information
- Relationships (friends, family, colleagues)

DO NOT STORE:
- Casual greetings ("hi", "thanks", "ok")
- Questions without new information ("what time is it?")
- Temporary/transient info ("I'm going to the store now")
- Assistant suggestions (only store what USER confirms)
- Things that are already common knowledge
- Vague statements without concrete info
- Commands or instructions to the assistant ("check this", "do that")
- Task-level requests ("make a PDF", "fix this bug", "update the code")
- File names, code fixes, or debugging details (these are session-specific, not long-term facts)
- Technical troubleshooting steps ("run sudo apt install X", "restart the service")
- One-time confirmations ("that works now", "send_file confirmed working")
- Scheduler/cron task details (times, reminders) — these belong in the scheduler, not memory

IMPORTANT — conflict resolution:
When the user states something that UPDATES previous info (e.g. "I moved to Bandung" when we stored "lives in Jakarta"), extract the LATEST fact. The storage engine will automatically find and update the old memory. Just extract the new content accurately.

Reply with EXACTLY one line:
- SKIP
- STORE|category|importance|content

Categories: fact, preference, event, lesson, decision, health, relationship, config
Importance: 0.3 (low) to 0.9 (critical)

Examples:
User: "I'm diabetic and take Metformin daily"
→ STORE|health|0.8|User has diabetes and takes Metformin daily

User: "Hi, how are you?"
→ SKIP

User: "I prefer dark mode in all apps"
→ STORE|preference|0.5|User prefers dark mode in all applications

User: "My wife's name is Yuli, she's a lecturer"
→ STORE|relationship|0.7|User's wife is named Yuli and works as a lecturer

User: "Can you check the weather?"
→ SKIP

User: "I moved to Bandung last month"
→ STORE|fact|0.7|User moved to and now lives in Bandung

User: "Actually I switched from Metformin to Januvia"
→ STORE|health|0.8|User switched diabetes medication from Metformin to Januvia

User: "Make that PDF landscape with a comparison table"
→ SKIP

User: "The filename is website_screenshot.py"
→ SKIP

User: "Run sudo apt install ca-certificates"
→ SKIP

User: "Send me a reminder at 3:17 for sahur"
→ SKIP"""


async def evaluate_message(
    provider: LLMProvider,
    user_message: str,
    assistant_response: str = "",
) -> Optional[dict]:
    """Evaluate if a user message contains info worth storing.
    
    Returns dict with {category, importance, content} or None if SKIP.
    """
    # Quick filters — skip obviously non-memorable messages
    stripped = user_message.strip().lower()
    
    # Too short
    if len(stripped) < 5:
        return None

    # Common non-memorable patterns
    skip_patterns = [
        "ok", "oke", "okay", "thanks", "thank you", "terima kasih",
        "makasih", "hi", "halo", "hello", "hey", "lanjut", "next",
        "yes", "no", "ya", "tidak", "gak", "nggak", "yep", "nope",
        "good", "nice", "cool", "bagus", "sip",
    ]
    if stripped in skip_patterns:
        return None

    # Starts with question words only (no info)
    question_only = stripped.startswith(("apa ", "what ", "how ", "gimana ", "kapan ", "when ",
                                          "where ", "dimana ", "siapa ", "who ", "kenapa ", "why "))
    if question_only and len(stripped) < 30:
        return None

    # Use LLM for nuanced evaluation
    try:
        response = await provider.chat(
            messages=[
                ChatMessage(role="system", content=EVALUATE_PROMPT),
                ChatMessage(role="user", content=f"User message: \"{user_message}\""),
            ],
            temperature=0.1,
            max_tokens=None,
        )

        result = response.content.strip()

        if result.startswith("SKIP"):
            return None

        if result.startswith("STORE|"):
            parts = result.split("|", 3)
            if len(parts) == 4:
                _, category, importance_str, content = parts
                try:
                    importance = float(importance_str.strip())
                    importance = max(0.1, min(1.0, importance))
                except ValueError:
                    importance = 0.5

                return {
                    "category": category.strip(),
                    "importance": importance,
                    "content": content.strip(),
                }

        logger.warning(f"Unexpected evaluator response: {result}")
        return None

    except Exception as e:
        logger.error(f"Memory evaluation failed: {e}")
        return None


async def evaluate_message_ollama(
    user_message: str,
    model: str = "qwen3:0.6b",
    base_url: str = "http://localhost:11434",
) -> Optional[dict]:
    """Evaluate if a user message contains info worth storing, using Ollama locally.

    Same logic as evaluate_message() but calls Ollama /api/chat directly,
    avoiding main LLM provider rate limits.

    Returns dict with {category, importance, content} or None if SKIP.
    """
    # Quick filters — same as evaluate_message()
    stripped = user_message.strip().lower()
    if len(stripped) < 5:
        return None

    skip_patterns = [
        "ok", "oke", "okay", "thanks", "thank you", "terima kasih",
        "makasih", "hi", "halo", "hello", "hey", "lanjut", "next",
        "yes", "no", "ya", "tidak", "gak", "nggak", "yep", "nope",
        "good", "nice", "cool", "bagus", "sip",
    ]
    if stripped in skip_patterns:
        return None

    question_only = stripped.startswith(("apa ", "what ", "how ", "gimana ", "kapan ", "when ",
                                          "where ", "dimana ", "siapa ", "who ", "kenapa ", "why "))
    if question_only and len(stripped) < 30:
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": EVALUATE_PROMPT},
                        {"role": "user", "content": f'User message: "{user_message}"'},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        result = data.get("message", {}).get("content", "").strip()
        # qwen3 with /think — strip thinking tags if present
        if "<think>" in result:
            import re
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()

        if result.startswith("SKIP"):
            return None

        if result.startswith("STORE|"):
            parts = result.split("|", 3)
            if len(parts) == 4:
                _, category, importance_str, content = parts
                try:
                    importance = float(importance_str.strip())
                    importance = max(0.1, min(1.0, importance))
                except ValueError:
                    importance = 0.5
                return {
                    "category": category.strip(),
                    "importance": importance,
                    "content": content.strip(),
                }

        logger.warning(f"Unexpected Ollama evaluator response: {result[:200]}")
        return None

    except Exception as e:
        logger.error(f"Ollama memory evaluation failed: {type(e).__name__}: {e}")
        return None


async def check_model_available(
    model: str = "qwen3:0.6b",
    base_url: str = "http://localhost:11434",
) -> bool:
    """Check if an Ollama model is available locally."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        model_base = model.split(":")[0]
        for m in data.get("models", []):
            if model_base in m.get("name", ""):
                return True
        return False
    except Exception:
        return False


def _is_explicit_remember(message: str) -> bool:
    """Detect if user explicitly asked to remember/store something."""
    import re
    lower = message.lower().strip()
    patterns = [
        r"\b(ingat|inget|catat|simpan|remember|memorize|store|save)\b.*\b(ini|this|it|itu)\b",
        r"\b(ini|this)\b.*\b(ingat|inget|catat|simpan|remember|memorize)\b",
        r"^(ingat|inget|catat|simpan|remember|memorize)[:\s]",
        r"\bjangan (lupa|lupakan)\b",
        r"\b(don.?t forget)\b",
    ]
    return any(re.search(p, lower) for p in patterns)


async def evaluate_and_store(
    provider: LLMProvider,
    memory_engine,
    user_message: str,
    user_id: Optional[int] = None,
    evaluator_driver: str = "provider",
    evaluator_model: str = "qwen3:0.6b",
) -> Optional[int]:
    """Evaluate a message and store if worthy. Returns memory ID or None.

    Args:
        evaluator_driver: "ollama" (local, no rate limit) or "provider" (main LLM)
        evaluator_model: Ollama model name when driver == "ollama"
    """
    result = None
    if evaluator_driver == "ollama":
        result = await evaluate_message_ollama(user_message, model=evaluator_model)
        if result is None:
            # Could be a quick-filter SKIP (not an error) — only fallback on error
            # We check by running quick filters ourselves
            logger.debug("Ollama evaluator returned None (SKIP or error)")
        # No fallback to provider — if Ollama says SKIP, trust it
    else:
        result = await evaluate_message(provider, user_message)

    if not result:
        return None

    # Detect explicit "remember this" → permanent memory
    permanent = _is_explicit_remember(user_message)
    
    mem_id = await memory_engine.store_if_new(
        content=result["content"],
        category=result["category"],
        source="user_confirmed",
        user_id=user_id,
        importance=result["importance"],
        permanent=permanent,
    )

    if mem_id:
        logger.info(f"Stored memory #{mem_id}: [{result['category']}] {result['content'][:80]}")
    else:
        logger.debug(f"Duplicate memory skipped: {result['content'][:80]}")

    return mem_id
