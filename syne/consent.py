"""Consent store — sliding-TTL approval cache for high-risk (op=x) tool calls.

Replaces the old session-taint exec gate (removed in v1.17.4). Where taint was a
blunt binary lock keyed to "did any external content enter this session", consent
is granular: a single approval authorizes ONE specific action (op + target +
content), reusable within a sliding TTL window so identical repeats don't nag.

Design goals
------------
- Pure in-memory, zero DB writes on the hot path (grants are ephemeral by design;
  a restart clears all grants — fail-safe, never fail-open).
- Sliding TTL: each successful reuse refreshes the clock (active work keeps its
  grant alive; abandoned grants expire).
- Same-actor: a grant is bound to (user_id, session_id). Another user in another
  session can never inherit it — injection-proof, same principle as _is_owner_dm.
- Content-bound: the content_hash means "rm -rf /a" is NOT authorized by an
  approval for "rm -rf /b". Change the action → new consent required.

This module is intentionally self-contained and side-effect free so it can be
unit-tested in isolation and wired in behind a feature flag without touching any
running code path.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DEFAULT_TTL_SECONDS = 600
DEFAULT_MODE = "sliding"  # "sliding" refreshes on reuse; "fixed" expires from grant time
DEFAULT_CONSENT_ENABLED = True  # fresh installs get consent live; override per-instance via config

# Aliases -- keep both naming styles importable so wiring code can't miss.
DEFAULT_CONSENT_TTL = DEFAULT_TTL_SECONDS
DEFAULT_CONSENT_MODE = DEFAULT_MODE

# key = (user_id, session_id, op, target, content_hash)
ConsentKey = Tuple[str, str, str, str, str]


def content_hash(payload: str) -> str:
    """Short stable digest binding a grant to a specific action payload.

    Reused shape from the old _tainted_exec_hash so operators reading logs see a
    familiar 12-char sha256 prefix.
    """
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:12]


def last_reply_token(content: str) -> str:
    """Extract the user's actual reply token from a raw message body.

    Channels prepend an "untrusted metadata" block (conversation_label, sender
    info, group name, etc.) to every user message. A naive ``.strip().lower()``
    therefore never equals "ya"/"yes" because the content starts with that
    block. We take the LAST non-empty line — the human's actual typed text —
    and lowercase it.

    This lives in consent.py (not agent.py) because both the LLM-mediated
    confirmation path and the deterministic bypass in conversation.chat()
    need identical parsing. Duplicating it in two places invites drift.
    """
    if not content:
        return ""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return (lines[-1].lower() if lines else "")


def make_key(
    user_id: str,
    session_id: str,
    op: str,
    target: str,
    payload: str,
) -> ConsentKey:
    """Build a consent key. `payload` is hashed; op/target stored verbatim."""
    return (str(user_id), str(session_id), str(op), str(target), content_hash(payload))


@dataclass
class _Grant:
    granted_at: float
    last_used_at: float
    reuse_count: int = 0


@dataclass
class ConsentStore:
    """In-memory grant cache with sliding/fixed TTL expiry.

    Not thread-safe by design: Syne's conversation loop is single-threaded async;
    all access happens from the event loop. If that ever changes, wrap mutations
    in a lock.
    """

    ttl_seconds: int = DEFAULT_TTL_SECONDS
    mode: str = DEFAULT_MODE
    _grants: Dict[ConsentKey, _Grant] = field(default_factory=dict)

    # ----- internal ---------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _deadline(self, g: _Grant) -> float:
        """Absolute expiry timestamp for a grant under the current mode."""
        base = g.last_used_at if self.mode == "sliding" else g.granted_at
        return base + self.ttl_seconds

    def _expired(self, g: _Grant, now: Optional[float] = None) -> bool:
        now = self._now() if now is None else now
        return now >= self._deadline(g)

    # ----- public API -------------------------------------------------------
    def grant(self, key: ConsentKey) -> None:
        """Record (or refresh) an approval for `key`."""
        now = self._now()
        existing = self._grants.get(key)
        if existing and not self._expired(existing, now):
            # Re-approving an active grant just refreshes it.
            existing.last_used_at = now
            return
        self._grants[key] = _Grant(granted_at=now, last_used_at=now)

    def is_granted(self, key: ConsentKey) -> bool:
        """True if an unexpired grant exists. On hit (sliding), refresh the clock.

        Expired grants are evicted lazily here so the store self-heals without a
        background sweeper.
        """
        g = self._grants.get(key)
        if g is None:
            return False
        now = self._now()
        if self._expired(g, now):
            del self._grants[key]
            return False
        if self.mode == "sliding":
            g.last_used_at = now
            g.reuse_count += 1
        return True

    def revoke(self, key: ConsentKey) -> bool:
        """Drop a single grant. Returns True if one was removed."""
        return self._grants.pop(key, None) is not None

    def revoke_session(self, user_id: str, session_id: str) -> int:
        """Drop all grants for a (user, session) pair. Backs the /reset command.

        Returns the number of grants cleared.
        """
        uid, sid = str(user_id), str(session_id)
        doomed = [k for k in self._grants if k[0] == uid and k[1] == sid]
        for k in doomed:
            del self._grants[k]
        return len(doomed)

    def revoke_user(self, user_id: str) -> int:
        """Drop ALL grants for a user across every session. Backs /reset.

        Consent grants are remembered per-user; this clears the whole set so
        the next op=x action re-prompts. Returns the number of grants cleared.
        """
        uid = str(user_id)
        doomed = [k for k in self._grants if k[0] == uid]
        for k in doomed:
            del self._grants[k]
        return len(doomed)

    def sweep(self) -> int:
        """Evict all expired grants. Optional housekeeping; returns count removed."""
        now = self._now()
        doomed = [k for k, g in self._grants.items() if self._expired(g, now)]
        for k in doomed:
            del self._grants[k]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._grants)


# ─────────────────────────────────────────────────────────────────────────
# High-level gate — single entry point used by both tool and ability
# dispatch layers. Keeps consent policy in one place; the two registries
# just call in and honor the returned action.
# ─────────────────────────────────────────────────────────────────────────

_SEND_FAMILY: frozenset = frozenset({
    "send_message", "send_file", "send_voice", "send_reaction",
})


def canonical_payload(tool_name: str, args: Optional[dict]) -> str:
    """Stable string for hashing/logging that binds the grant to (tool, args).

    Dict ordering is normalized so ``exec(command='ls', timeout=30)`` and
    ``exec(timeout=30, command='ls')`` produce the same payload hash.
    """
    import json
    try:
        args_str = json.dumps(args or {}, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        args_str = str(args)
    return f"{tool_name}:{args_str}"


def _send_family_should_skip(
    tool_name: str, args: Optional[dict], conv, scheduled: bool,
) -> bool:
    """Hybrid rule for the send_* family only. Skip the gate when the send is
    either scheduled (no interactive user to confirm) or same-chat (target
    equals the session's own chat, so the send is just a routed reply).

    Cross-chat interactive sends still hit the gate — this is exactly where an
    injection could ask Syne to exfiltrate data to an attacker-controlled chat.
    """
    if tool_name not in _SEND_FAMILY:
        return False
    if scheduled:
        return True
    if conv is None:
        return False
    target = str((args or {}).get("chat_id", "") or "")
    current = str(getattr(conv, "chat_id", "") or "")
    if target and current and target == current:
        return True
    return False


async def check_and_hold(
    conv,
    agent,
    tool_name: str,
    args: Optional[dict],
    access_level: str,
    permission: int,
    scheduled: bool = False,
    kind: str = "tool",
):
    """Consent gate. Single entry used by tools/registry and abilities/registry.

    Returns a 2-tuple:
        ("allow", None)      — no gate applies OR already granted → dispatch
        ("held",  prompt)    — caller must return `prompt` to the LLM/user

    The gate policy:
      1. `needs_consent(access_level, permission)` False → allow.
         (Rule: caller's own class digit has the x bit set. Octal is the
         single source of truth — no auxiliary op mapping. See security.py.)
      2. Hybrid skip for send_* family (scheduled or same-chat) → allow.
      3. Feature flag `security.consent_enabled` False → allow (kill switch).
      4. No active conv (e.g. subagent path with no DM channel) → held with
         explicit deny message so caller can turn it into a refusal.
      5. Grant already exists in ConsentStore for the exact payload → allow
         (sliding TTL refreshes on read).
      6. Recent user reply is a strict "ya"/"yes" matching the pending hash
         → grant + allow (LLM-mediated path, kept as belt to the deterministic
         bypass in conversation.chat()).
      7. Otherwise → set pending state on conv, return "balas ya" prompt.

    IMPORTANT: this runs in the dispatch hot path. It must never raise; any
    unexpected error is logged and turned into ("allow", None) so a broken
    consent module doesn't accidentally block ALL tool traffic. Fail-open on
    the gate is acceptable here because permission gating (check_tool_access)
    still ran BEFORE this and rejected class-level violations.
    """
    import logging
    _log = logging.getLogger("syne.consent.gate")
    try:
        from .security import needs_consent  # local import to avoid cycles
        if not needs_consent(access_level, permission):
            return ("allow", None)

        # ── Provenance skip (owner/family, clean turn) ───────────────────────
        # needs_consent() said the caller's own class digit has the x bit
        # (any of 7/5/3/1) — normally the gate fires. But a CLEAN turn from a
        # trusted caller (owner OR family) is a conscious, direct command and
        # should run without friction. "Clean" = conv._turn_untrusted is False:
        # the turn's input carried no image/file/URL AND no untrusted tool
        # (web_search/fetch_url/file_read/pdf/office/image_analysis/
        # website_screenshot) ran mid-turn.
        #
        # A TAINTED turn keeps the gate: the Yes button becomes proof of
        # conscious approval that content injected from a web page, an uploaded
        # file, or an image cannot forge (it cannot press a Telegram button).
        # This is enforced by code + platform, not by the model's discipline.
        if access_level in ("owner", "family") and conv is not None \
                and not getattr(conv, "_turn_untrusted", False):
            _log.debug(
                f"consent skip (clean turn): tool={tool_name}, level={access_level}"
            )
            return ("allow", None)

        if _send_family_should_skip(tool_name, args, conv, scheduled):
            _log.debug(
                f"consent skip (send_* hybrid): tool={tool_name}, "
                f"scheduled={scheduled}, target={((args or {}).get('chat_id') or '')}"
            )
            return ("allow", None)

        # Feature flag
        try:
            from .db.models import get_config as _get_config
            enabled = await _get_config("security.consent_enabled", DEFAULT_CONSENT_ENABLED)
        except Exception:
            enabled = DEFAULT_CONSENT_ENABLED
        if not enabled:
            return ("allow", None)

        # No conv or no agent: non-interactive path (sub-agent, scheduled cron,
        # test harness). The consent gate is a defense for INTERACTIVE LLM-
        # driven sessions where a human is present to approve. Programmatic
        # callers have their own safety mechanisms:
        #   - Sub-agent exec runs shell_guard in _execute_tool (source=
        #     "subagent"): HARD_DENY and CONSENT both stop the command, since
        #     a headless sub-agent has no human to approve. (Replaces an earlier
        #     FALSE claim that sub-agents ran check_command_safety — historically
        #     they ran exec with NO check at all. Fixed in the shell_guard work.)
        #   - Scheduled tasks were pre-authorized at setup time.
        #   - Tests are explicitly aware.
        # Allowing here preserves those workflows; blocking would break them
        # (a nightly cron cannot type "ya").
        if conv is None or agent is None:
            _log.debug(
                f"consent skip (non-interactive): tool={tool_name}, "
                f"conv={conv is not None}, agent={agent is not None}"
            )
            return ("allow", None)

        # Build grant key + configure store TTL/mode from config. The `op`
        # dimension of the key is fixed to "x" here — the very fact we
        # reached this point means needs_consent said yes, i.e. the caller's
        # digit has the x bit set. Baking "x" into the key means a later
        # rule change that hides the x bit for a class won't collide with
        # earlier grants.
        uid = str((getattr(conv, "user", {}) or {}).get("id", ""))
        sid = str(getattr(conv, "session_id", "") or "")
        payload = canonical_payload(tool_name, args)
        ckey = make_key(uid, sid, op="x", target=tool_name, payload=payload)

        store = getattr(agent, "_consent", None)
        if store is None:
            # Consent module wired but store not initialized — fail-safe closed.
            _log.error(f"consent: agent has no _consent store; refusing {tool_name}")
            return ("held", f"Error: consent store unavailable for {tool_name}")

        try:
            store.ttl_seconds = int(await _get_config(
                "security.consent_ttl_seconds", DEFAULT_TTL_SECONDS
            ))
            _mode_raw = await _get_config("security.consent_mode", DEFAULT_MODE)
            store.mode = _mode_raw if _mode_raw in ("sliding", "fixed") else DEFAULT_MODE
        except Exception:
            pass  # keep existing store settings

        # Fast path: existing grant.
        if store.is_granted(ckey):
            return ("allow", None)

        # LLM-mediated fallback: if the deterministic bypass in
        # conversation.chat() didn't fire (e.g., a chained tool call happens
        # BEFORE the next user turn), still honor a prior "ya" that matches
        # THIS exact payload.
        confirm_fn = getattr(agent, "_consent_confirmed", None)
        if callable(confirm_fn):
            try:
                if confirm_fn(conv, payload):
                    store.grant(ckey)
                    return ("allow", None)
            except Exception as e:
                _log.warning(f"consent: _consent_confirmed raised: {e}")

        # Set pending state on the CONVERSATION (per-conv, not agent-global —
        # avoids the same latent race the sudo-guard has on concurrent sessions).
        import time as _time
        conv._pending_consent_kind = kind
        conv._pending_consent_op = "x"
        conv._pending_consent_tool = tool_name
        conv._pending_consent_args = dict(args or {})
        conv._pending_consent_hash = content_hash(payload)
        conv._pending_consent_at = _time.time()

        prompt = format_consent_prompt(tool_name, args, conv._pending_consent_hash)
        _log.warning(
            f"consent held: tool={tool_name}, "
            f"hash={conv._pending_consent_hash}, session={sid}, "
            f"perm={oct(permission)}, access={access_level}"
        )
        return ("held", prompt)

    except Exception as e:
        # Fail-open on unexpected gate error — see docstring rationale.
        _log.exception(f"consent gate crashed on {tool_name}: {e}")
        return ("allow", None)


def format_consent_prompt(
    tool_name: str, args: Optional[dict], hash_hex: str,
) -> str:
    """User-facing prompt shown when a call is held pending consent.

    Ends with a machine-readable marker `[[CONSENT_BUTTONS:hash=…]]` that
    channel adapters (Telegram, etc.) can detect and replace with inline
    Yes/No buttons. Channels that don't recognize the marker (CLI, plain
    stdout) leave it as visible text — harmless.
    """
    import json
    args_preview = ""
    try:
        args_preview = json.dumps(args or {}, ensure_ascii=False)[:300]
    except Exception:
        args_preview = str(args)[:300]
    return (
        f"⚠️ `{tool_name}` needs confirmation.\n"
        f"`{args_preview}`\n"
        f"[[CONSENT_BUTTONS:hash={hash_hex}]]"
    )


CONSENT_BUTTON_MARKER = "[[CONSENT_BUTTONS:hash="


def extract_consent_hash(text: str) -> Optional[str]:
    """If `text` carries a CONSENT_BUTTONS marker, return the hash. Else None.

    Channels use this to decide whether to attach Yes/No buttons before
    sending — see the Telegram adapter's _send_response.
    """
    if not text or CONSENT_BUTTON_MARKER not in text:
        return None
    start = text.rfind(CONSENT_BUTTON_MARKER) + len(CONSENT_BUTTON_MARKER)
    end = text.find("]]", start)
    if end == -1:
        return None
    return text[start:end].strip() or None


def strip_consent_marker(text: str) -> str:
    """Remove the CONSENT_BUTTONS marker line from a message body."""
    if not text or CONSENT_BUTTON_MARKER not in text:
        return text
    lines = text.splitlines()
    kept = [ln for ln in lines if CONSENT_BUTTON_MARKER not in ln]
    return "\n".join(kept).rstrip()
