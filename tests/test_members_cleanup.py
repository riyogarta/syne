from datetime import datetime, timedelta, timezone

from syne.members_cleanup import pick_idle

NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)


def _row(level="public", days_ago=90, prefs=None):
    return {
        "access_level": level,
        "preferences": prefs or {},
        "last_active": NOW - timedelta(days=days_ago) if days_ago is not None else None,
    }


def test_idle_public_is_picked():
    assert len(pick_idle([_row()], NOW, 60)) == 1


def test_recent_is_kept():
    assert pick_idle([_row(days_ago=59)], NOW, 60) == []


def test_owner_family_blocked_never_picked():
    rows = [_row("owner"), _row("family"), _row("blocked")]
    assert pick_idle(rows, NOW, 60) == []


def test_already_pending_skipped():
    assert pick_idle([_row(prefs={"pending_approval": "revoked"})], NOW, 60) == []


def test_unknown_activity_skipped():
    assert pick_idle([_row(days_ago=None)], NOW, 60) == []


def test_disabled():
    assert pick_idle([_row()], NOW, 0) == []
