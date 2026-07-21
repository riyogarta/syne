"""Tests for syne.consent — ConsentStore + content-hash + key helpers.

Rewritten as proper pytest tests. The previous file was a runnable script
with module-level side effects (sys.exit at load time + a hardcoded
sys.path.insert to a machine-specific /home path) that caused
`pytest tests/` to INTERNALERROR before any test ran. That in turn masked
all other test results for anyone who cloned the repo.
"""

import time

import pytest

from syne.consent import ConsentStore, content_hash, make_key


# ---------------------------------------------------------------------------
# Distinct-payload key fixtures — the 4 canonical variants used across tests
# ---------------------------------------------------------------------------


@pytest.fixture
def keys():
    return {
        "base":         make_key("u1", "s1", "x", "exec", "rm -rf /a"),
        "diff_payload": make_key("u1", "s1", "x", "exec", "rm -rf /b"),
        "diff_user":    make_key("u2", "s1", "x", "exec", "rm -rf /a"),
        "diff_session": make_key("u1", "s2", "x", "exec", "rm -rf /a"),
    }


# ---------------------------------------------------------------------------
# content_hash — deterministic, distinguishes different bodies
# ---------------------------------------------------------------------------


class TestContentHash:

    def test_stable_across_calls(self):
        assert content_hash("rm -rf /a") == content_hash("rm -rf /a")

    def test_distinguishes_different_payloads(self):
        assert content_hash("a") != content_hash("b")


# ---------------------------------------------------------------------------
# ConsentStore — grant/revoke/is_granted state machine
# ---------------------------------------------------------------------------


class TestConsentStoreGrant:

    def test_cold_is_not_granted(self, keys):
        s = ConsentStore(ttl_seconds=600)
        assert not s.is_granted(keys["base"])

    def test_after_grant_is_granted(self, keys):
        s = ConsentStore(ttl_seconds=600)
        s.grant(keys["base"])
        assert s.is_granted(keys["base"])

    def test_grant_is_content_bound(self, keys):
        """A different payload with the same actor is NOT granted by proxy."""
        s = ConsentStore(ttl_seconds=600)
        s.grant(keys["base"])
        assert not s.is_granted(keys["diff_payload"])

    def test_grant_is_user_scoped(self, keys):
        """Another user with the same payload is NOT granted by proxy."""
        s = ConsentStore(ttl_seconds=600)
        s.grant(keys["base"])
        assert not s.is_granted(keys["diff_user"])

    def test_grant_is_session_scoped(self, keys):
        """A different session with the same payload is NOT granted by proxy."""
        s = ConsentStore(ttl_seconds=600)
        s.grant(keys["base"])
        assert not s.is_granted(keys["diff_session"])

    def test_regrant_does_not_duplicate(self, keys):
        s = ConsentStore()
        s.grant(keys["base"])
        s.grant(keys["base"])
        assert len(s) == 1


# ---------------------------------------------------------------------------
# TTL behaviour — fixed vs sliding
# ---------------------------------------------------------------------------


class TestConsentStoreTtl:

    def test_fixed_mode_expires_after_ttl(self, keys):
        s = ConsentStore(ttl_seconds=1, mode="fixed")
        s.grant(keys["base"])
        assert s.is_granted(keys["base"])
        time.sleep(1.1)
        assert not s.is_granted(keys["base"])

    def test_sliding_mode_refreshes_on_read(self, keys):
        s = ConsentStore(ttl_seconds=2, mode="sliding")
        s.grant(keys["base"])
        time.sleep(1.2)
        # First is_granted mid-window — sliding mode refreshes.
        assert s.is_granted(keys["base"])
        time.sleep(1.2)
        # Would be dead in fixed mode; alive here thanks to the refresh.
        assert s.is_granted(keys["base"])


# ---------------------------------------------------------------------------
# revoke / revoke_session / sweep
# ---------------------------------------------------------------------------


class TestConsentStoreRevoke:

    def test_revoke_removes_present(self, keys):
        s = ConsentStore()
        s.grant(keys["base"])
        assert s.revoke(keys["base"]) is True
        assert not s.is_granted(keys["base"])

    def test_revoke_missing_returns_false(self, keys):
        s = ConsentStore()
        assert s.revoke(keys["base"]) is False

    def test_revoke_session_clears_only_matching(self):
        s = ConsentStore()
        s.grant(make_key("u1", "s1", "x", "exec", "a"))
        s.grant(make_key("u1", "s1", "x", "exec", "b"))
        s.grant(make_key("u1", "s2", "x", "exec", "c"))  # different session
        cleared = s.revoke_session("u1", "s1")
        assert cleared == 2
        assert len(s) == 1  # the s2 grant survives

    def test_sweep_evicts_expired(self):
        s = ConsentStore(ttl_seconds=1, mode="fixed")
        s.grant(make_key("u1", "s1", "x", "exec", "a"))
        time.sleep(1.1)
        assert s.sweep() == 1
        assert len(s) == 0
