"""run_shell() — the single chokepoint for ALL shell execution in Syne.

Design (memory 54510): there is exactly ONE place that spawns a subprocess.
Every shell need — the `exec`/`shell` tool, sub-agents, and the internal
ability shell-outs (pdf/office/screenshot/whatsapp/node) — routes through
here. No 'little exec' scattered around. This makes the security parser
(shell_guard) impossible to bypass: you cannot run a command without passing
its gate.

Separation of concerns:
  * shell_guard.analyze()  — pure decision (ALLOW/CONSENT/HARD_DENY). No I/O.
  * run_shell()            — orchestration: load runtime allowlist from DB,
                             call analyze(), enforce the verdict, record
                             candidates, spawn subprocess, redact output.

The `source` argument encodes provenance and controls whether the guard runs:
  * "llm" / "subagent"  — UNTRUSTED origin. Guard RUNS. This is the whole
                          point: LLM/sub-agent-issued commands are gated.
  * "internal"          — a hardcoded ability shell-out (fixed argv, e.g.
                          `libreoffice --convert`). Guard SKIPPED: the command
                          is author-controlled, not attacker-influenced.
  * "startup"           — boot-time spawn (e.g. wa bridge) with no conv to
                          prompt. Guard SKIPPED.

FAIL-CLOSED everywhere: if the guard errors, if the allowlist load fails, if
anything is uncertain for an llm/subagent source → HARD_DENY. The only paths
that reach a subprocess are ALLOW (clean) and CONSENT-approved.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .shell_guard import analyze, Verdict

logger = logging.getLogger("syne.shell_exec")


class Outcome(str, Enum):
    RAN = "ran"                # executed; `output` holds the result
    DENIED = "denied"          # HARD_DENY; never executed; `reason` explains
    NEEDS_CONSENT = "consent"  # caller must gate Yes/No then re-invoke with approved=True
    ALLOWED = "allowed"        # check_only=True: guard cleared; caller executes elsewhere (e.g. remote node)


# Sources whose commands are author-controlled (fixed argv) and therefore
# bypass the guard. Everything else is treated as untrusted → guard runs.
_TRUSTED_SOURCES = frozenset({"internal", "startup"})


@dataclass
class ShellResult:
    outcome: Outcome
    output: str = ""
    reason: str = ""
    verdict: Optional[Verdict] = None
    candidates: list[str] | None = None


async def _load_runtime_allowlist(db_pool) -> set[str]:
    """Fetch approved binaries from shell_allowlist. Fail-closed: on ANY error
    return empty set (so only the hardcoded DEFAULT_ALLOWLIST floor applies —
    never widen the gate because of a DB hiccup)."""
    if db_pool is None:
        return set()
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT bin_name FROM shell_allowlist")
        return {r["bin_name"] for r in rows if r["bin_name"]}
    except Exception as e:
        logger.warning(f"allowlist load failed (fail-closed to floor): {e}")
        return set()


async def _load_runtime_denylist(db_pool) -> tuple[set[str], list[str]]:
    """Fetch (binary_denies, pattern_denies) from shell_denylist. Fail-safe:
    on error return empty — a DB hiccup must not silently DROP a deny (that
    would widen access), but it also cannot invent one. We log loudly."""
    if db_pool is None:
        return set(), []
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT entry, kind FROM shell_denylist")
        bins = {r["entry"] for r in rows if r["kind"] == "binary" and r["entry"]}
        pats = [r["entry"] for r in rows if r["kind"] == "pattern" and r["entry"]]
        return bins, pats
    except Exception as e:
        logger.warning(f"denylist load failed: {e}")
        return set(), []


async def _record_candidates(db_pool, candidates: list[str], sample: str, context: str) -> None:
    """Upsert unknown binaries into shell_allowlist_candidates for owner review.
    Best-effort: a failure here must never change the security decision."""
    if db_pool is None or not candidates:
        return
    try:
        async with db_pool.acquire() as conn:
            for b in candidates:
                await conn.execute(
                    """INSERT INTO shell_allowlist_candidates
                           (bin_name, sample_command, context)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (bin_name) DO UPDATE
                         SET seen_count = shell_allowlist_candidates.seen_count + 1,
                             last_seen_at = now(),
                             sample_command = EXCLUDED.sample_command""",
                    b, sample[:500], (context or "")[:200],
                )
    except Exception as e:
        logger.warning(f"candidate record failed (non-fatal): {e}")


async def _bump_hits(db_pool, binaries: set[str]) -> None:
    """Increment hit counter for allowlisted binaries actually used (feeds the
    'promote to hardcoded floor' decision later). Best-effort."""
    if db_pool is None or not binaries:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE shell_allowlist SET hits = hits + 1 WHERE bin_name = ANY($1)",
                list(binaries),
            )
    except Exception:
        pass


async def run_shell(
    command: str,
    *,
    source: str,
    cwd: Optional[str] = None,
    timeout: int = 30,
    db_pool=None,
    consent_enabled: bool = True,
    approved: bool = False,
    output_max: int = 4000,
    redact_fn=None,
    check_only: bool = False,
) -> ShellResult:
    """Execute a shell command through the single chokepoint.

    Args:
        command: raw shell command.
        source: "llm"/"subagent" (guarded) or "internal"/"startup" (trusted).
        cwd: working directory.
        timeout: seconds.
        db_pool: asyncpg pool for allowlist/candidates (None → floor-only).
        consent_enabled: mirrors security.consent_enabled. When False, a
            CONSENT verdict degrades to ALLOW (owner deliberately disabled the
            gate). HARD_DENY is UNAFFECTED — it always blocks.
        approved: True when the caller already obtained Yes for this exact
            command (consent granted). Lets a CONSENT verdict proceed.
        output_max: truncate stdout to this many chars.
        redact_fn: optional callable(str)->str to mask secrets in output.
        check_only: run the guard but skip _spawn(). On success return
            Outcome.ALLOWED so the caller can execute the command elsewhere
            (e.g. forward to a remote node over WebSocket). HARD_DENY /
            NEEDS_CONSENT behave exactly as in the normal path — the caller
            still sees the same verdicts and must handle them the same way.

    Returns:
        ShellResult(outcome, output/reason, verdict, candidates).
    """
    if not command or not isinstance(command, str) or not command.strip():
        return ShellResult(Outcome.DENIED, reason="empty command", verdict=Verdict.HARD_DENY)

    # ── Trusted source (hardcoded argv): skip the guard, execute directly. ──
    if source in _TRUSTED_SOURCES:
        return await _spawn(command, cwd, timeout, output_max, redact_fn)

    # ── Untrusted source (llm/subagent): the guard is mandatory. ──
    try:
        extra = await _load_runtime_allowlist(db_pool)
        deny_bins, deny_pats = await _load_runtime_denylist(db_pool)
        result = analyze(command, extra_allow=extra,
                         extra_deny_bins=deny_bins, extra_deny_patterns=deny_pats)
    except Exception as e:
        # Guard itself failed → fail-closed, never execute.
        logger.error(f"shell_guard raised (fail-closed to DENY): {e}")
        return ShellResult(Outcome.DENIED, reason=f"guard error (fail-closed): {e}",
                           verdict=Verdict.HARD_DENY)

    if result.verdict == Verdict.HARD_DENY:
        await _record_candidates(db_pool, result.candidates or [], command, source)
        return ShellResult(Outcome.DENIED, reason=result.reason,
                           verdict=Verdict.HARD_DENY, candidates=result.candidates)

    if result.verdict == Verdict.CONSENT:
        if consent_enabled and not approved:
            return ShellResult(Outcome.NEEDS_CONSENT, reason=result.reason,
                               verdict=Verdict.CONSENT, candidates=result.candidates)
        # consent gate globally off, OR already approved → proceed.
        logger.info(
            f"CONSENT verdict proceeding ({'approved' if approved else 'gate off'}): {result.reason[:120]}"
        )

    # ── ALLOW (or degraded/approved CONSENT) → execute. ──
    if check_only:
        # Caller runs the command themselves (e.g. remote node). Guard passed.
        return ShellResult(Outcome.ALLOWED, verdict=result.verdict)
    return await _spawn(command, cwd, timeout, output_max, redact_fn)


async def _spawn(command, cwd, timeout, output_max, redact_fn) -> ShellResult:
    """The ONLY subprocess spawn in Syne. Everything funnels here."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        parts = []
        if stdout:
            out = stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
            if out:
                parts.append(f"stdout:\n{out[:output_max]}")
        if stderr:
            err = stderr.decode("utf-8", errors="replace").replace("\x00", "").strip()
            if err:
                parts.append(f"stderr:\n{err[:output_max // 2]}")
        parts.append(f"exit_code: {proc.returncode}")
        output = "\n".join(parts) if parts else f"exit_code: {proc.returncode}"
        if redact_fn:
            try:
                output = redact_fn(output)
            except Exception:
                pass
        return ShellResult(Outcome.RAN, output=output)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return ShellResult(Outcome.RAN, output=f"Error: Command timed out after {timeout}s")
    except Exception as e:
        logger.error(f"spawn error: {e}")
        return ShellResult(Outcome.RAN, output=f"Error: {e}")
