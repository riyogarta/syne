"""Hard-rule compliance checker for final assistant responses.

Runs a cheap second-pass LLM (the same evaluator model used by memory
auto-capture) against a draft response and every hard rule from the rules
table. Returns a small verdict object the conversation loop consumes to
either send the draft, prepend a warning, or force a rewrite.

Design notes
------------

* Rule CONTENT stays in the DB (`rules` table). This module is generic — it
  knows nothing rule-specific, only that hard rules exist and the checker
  should verdict against them.
* Wire format is line-based (`CLEAN` or `VIOLATED|CODE1,CODE2|reason`), not
  JSON. Small quantised models (qwen3:0.6b in the default Ollama setup) are
  noticeably more reliable at emitting a single anchored line than a
  well-formed JSON object; the parser here is tolerant of leading /
  trailing junk and `<think>...</think>` blocks (qwen3 emits those).
* Fail-open: ANY error (network, parse failure, timeout, no rules) resolves
  to `ERROR`. The caller degrades gracefully — the response still reaches
  the user, but with a `⚠️ [Rule checker unavailable ...]` warning
  prepended so the owner sees the response wasn't gated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

from .llm.provider import ChatMessage, LLMProvider

logger = logging.getLogger("syne.rule_checker")


class VerdictState(str, Enum):
    CLEAN = "clean"      # draft passes; send it
    VIOLATED = "violated"  # draft breaks 1+ hard rules; regenerate
    ERROR = "error"      # checker itself failed; fail-open with warning


@dataclass
class CheckResult:
    state: VerdictState
    violated: list[str] = field(default_factory=list)  # rule codes
    reason: str = ""


_CHECKER_PROMPT_TEMPLATE = """You are a strict but fair compliance checker. Your ONLY job is to decide whether the DRAFT RESPONSE below violates any of the HARD RULES.

HARD RULES:
{rules_block}

USER MESSAGE (for context — do NOT judge this, only the draft):
{user_message}

DRAFT RESPONSE:
{draft}

Reply with EXACTLY ONE line, no preamble, no explanation before or after:
  CLEAN
      — if the draft complies with every hard rule above
  VIOLATED|CODE1,CODE2|short reason
      — if one or more rules are broken. List the rule CODES separated by commas, then a short one-line reason.

Judge only rule violations. Do NOT flag style, tone, grammar, or anything the rules do not cover. Be strict but do not invent violations."""


def _strip_think(text: str) -> str:
    """qwen3 with /think emits <think>...</think>. Drop it."""
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_verdict_line(raw: str) -> CheckResult:
    """Parse the compact line format into a CheckResult.

    Tolerant: accepts the verdict anywhere in the output, ignores surrounding
    prose. If nothing parses, returns ERROR — safer than pretending CLEAN.
    """
    text = _strip_think(raw or "").strip()
    if not text:
        return CheckResult(VerdictState.ERROR, reason="empty checker output")

    # CLEAN case: exact word anywhere on its own line
    for line in text.splitlines():
        s = line.strip().upper()
        if s == "CLEAN":
            return CheckResult(VerdictState.CLEAN)

    # VIOLATED case
    m = re.search(r"VIOLATED\s*\|\s*([A-Z0-9_,\s]+)\s*\|\s*(.+?)\s*$",
                  text, flags=re.MULTILINE | re.IGNORECASE)
    if m:
        codes = [c.strip().upper() for c in m.group(1).split(",") if c.strip()]
        reason = m.group(2).strip()
        if codes:
            return CheckResult(VerdictState.VIOLATED, violated=codes, reason=reason)

    # Some models write just "VIOLATED" then a bulleted list on next lines.
    # Fallback: if VIOLATED appears anywhere and we can pluck any rule-code
    # tokens (uppercase words with digits/underscores), accept.
    if "VIOLATED" in text.upper():
        codes = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)
        # Drop the literal word "VIOLATED" itself
        codes = [c for c in codes if c != "VIOLATED"]
        if codes:
            return CheckResult(
                VerdictState.VIOLATED,
                violated=list(dict.fromkeys(codes)),  # de-dup preserving order
                reason=text.strip().splitlines()[-1][:200],
            )

    logger.warning(f"Rule checker: unparseable output ({text[:200]!r})")
    return CheckResult(VerdictState.ERROR, reason="unparseable checker output")


def _build_prompt(draft: str, user_message: str, hard_rules: list[dict]) -> str:
    """Compact prompt for the checker LLM. Keeps rule content in DB shape."""
    rules_lines = []
    for r in hard_rules:
        code = r.get("code", "?")
        name = r.get("name", "?")
        desc = r.get("description", "")
        rules_lines.append(f"- [{code}] {name}: {desc}")
    rules_block = "\n".join(rules_lines)

    # Truncate to keep the checker request cheap. Real DATE_VERIFY-style
    # violations always sit in the first paragraph or two; long tail rarely
    # matters. If it does, we can raise these.
    return _CHECKER_PROMPT_TEMPLATE.format(
        rules_block=rules_block,
        user_message=(user_message or "").strip()[:1500],
        draft=(draft or "").strip()[:4000],
    )


async def _check_via_ollama(
    prompt: str, model: str, base_url: str = "http://localhost:11434",
    timeout: float = 30.0,
) -> str:
    """Direct HTTP to Ollama /api/chat — same shape as evaluate_message_ollama."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("message", {}).get("content", "").strip()


async def _check_via_provider(prompt: str, provider: LLMProvider) -> str:
    """Fallback: use the main chat provider. Same provider that runs Molt —
    accepts a small overhead vs Ollama's dedicated cheap model."""
    response = await provider.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        thinking_budget=0,
    )
    return (response.content or "").strip()


async def check_response(
    draft: str,
    user_message: str,
    hard_rules: list[dict],
    evaluator_driver: str = "ollama",
    evaluator_model: str = "qwen3:0.6b",
    provider: Optional[LLMProvider] = None,
) -> CheckResult:
    """Judge a draft response against every hard rule.

    Args:
        draft: The assistant's proposed final text.
        user_message: The user's turn that produced this draft (for context).
        hard_rules: List of hard-rule dicts (from get_rules() filtered to
            severity == 'hard'). Each has keys: code, name, description.
        evaluator_driver: 'ollama' (local, cheap) or 'provider' (main LLM).
            The caller decides this based on both security.rule_checker_driver
            and memory.evaluator_driver — the checker itself does not read
            config or do auto-fallback. If the chosen driver fails, we
            return ERROR and the caller prepends a warning to the response.
        evaluator_model: Ollama model name when driver == 'ollama'.
        provider: Main LLM provider — required when driver == 'provider'.

    Returns:
        CheckResult with state CLEAN / VIOLATED / ERROR.
        ERROR is fail-open: the caller must send the draft with a warning.
    """
    # Nothing to check — pass through as CLEAN. No rules == nothing to
    # violate; empty draft is handled at the caller (skipped before we
    # get here), but be defensive.
    if not hard_rules:
        return CheckResult(VerdictState.CLEAN)
    if not (draft and draft.strip()):
        return CheckResult(VerdictState.CLEAN)

    prompt = _build_prompt(draft, user_message, hard_rules)

    try:
        if evaluator_driver == "ollama":
            raw = await _check_via_ollama(prompt, model=evaluator_model)
        else:
            if provider is None:
                return CheckResult(
                    VerdictState.ERROR,
                    reason="provider driver but no provider passed",
                )
            raw = await _check_via_provider(prompt, provider)
    except Exception as e:
        logger.warning(
            f"Rule checker LLM call failed: {type(e).__name__}: {e}"
        )
        return CheckResult(
            VerdictState.ERROR,
            reason=f"checker call failed: {type(e).__name__}",
        )

    verdict = _parse_verdict_line(raw or "")
    if verdict.state == VerdictState.VIOLATED:
        # Filter to codes we actually know about — hallucinated codes get dropped
        known_codes = {r.get("code", "").upper() for r in hard_rules}
        verdict.violated = [c for c in verdict.violated if c in known_codes]
        if not verdict.violated:
            # Model said VIOLATED but every code is unknown — treat as ERROR
            # so caller fails-open with a warning rather than a phantom block.
            logger.warning(f"Rule checker returned only unknown codes; raw={raw[:200]!r}")
            return CheckResult(VerdictState.ERROR, reason="unknown codes from checker")

    return verdict
