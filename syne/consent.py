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

# key = (user_id, session_id, op, target, content_hash)
ConsentKey = Tuple[str, str, str, str, str]


def content_hash(payload: str) -> str:
    """Short stable digest binding a grant to a specific action payload.

    Reused shape from the old _tainted_exec_hash so operators reading logs see a
    familiar 12-char sha256 prefix.
    """
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:12]


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

    def sweep(self) -> int:
        """Evict all expired grants. Optional housekeeping; returns count removed."""
        now = self._now()
        doomed = [k for k, g in self._grants.items() if self._expired(g, now)]
        for k in doomed:
            del self._grants[k]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._grants)
