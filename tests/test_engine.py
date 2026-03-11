"""Tests for syne.memory.engine — _source_priority and _detect_conflicts."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from syne.memory.engine import MemoryEngine


def _make_memory(
    id: int,
    content: str,
    category: str = "fact",
    source: str = "system",
    importance: float = 0.5,
    created_at: datetime = None,
    similarity: float = 0.8,
) -> dict:
    """Helper to build a mock memory dict matching recall output."""
    return {
        "id": id,
        "content": content,
        "category": category,
        "source": source,
        "importance": importance,
        "created_at": created_at or datetime(2025, 1, 1, tzinfo=timezone.utc),
        "similarity": similarity,
    }


class TestSourcePriority:
    """Tests for MemoryEngine._source_priority."""

    @pytest.fixture
    def engine(self, mock_provider):
        return MemoryEngine(mock_provider)

    def test_user_confirmed(self, engine):
        assert engine._source_priority("user_confirmed") == 3

    def test_observed(self, engine):
        assert engine._source_priority("observed") == 2

    def test_system(self, engine):
        assert engine._source_priority("system") == 1

    def test_auto_captured(self, engine):
        assert engine._source_priority("auto_captured") == 1

    def test_unknown_source(self, engine):
        assert engine._source_priority("unknown") == 0

    def test_empty_string(self, engine):
        assert engine._source_priority("") == 0

    def test_arbitrary_string(self, engine):
        assert engine._source_priority("something_else") == 0

    def test_system_equals_auto_captured(self, engine):
        assert engine._source_priority("system") == engine._source_priority("auto_captured")

    def test_user_confirmed_highest(self, engine):
        all_sources = ["user_confirmed", "observed", "system", "auto_captured", "unknown"]
        priorities = [engine._source_priority(s) for s in all_sources]
        assert max(priorities) == engine._source_priority("user_confirmed")


class TestDetectConflicts:
    """Tests for MemoryEngine._detect_conflicts."""

    @pytest.fixture
    def engine(self, mock_provider):
        return MemoryEngine(mock_provider)

    def test_empty_list(self, engine):
        result = engine._detect_conflicts([])
        assert result == []

    def test_single_result(self, engine):
        mem = _make_memory(1, "Alice likes cats")
        result = engine._detect_conflicts([mem])
        assert len(result) == 1
        assert "_conflict_status" not in result[0]

    def test_no_conflict_different_categories(self, engine):
        """Two memories in different categories should not conflict."""
        mem_a = _make_memory(1, "Alice likes cats", category="preference", similarity=0.9)
        mem_b = _make_memory(2, "Bob works at Google", category="fact", similarity=0.9)
        result = engine._detect_conflicts([mem_a, mem_b])
        assert "_conflict_status" not in result[0]
        assert "_conflict_status" not in result[1]

    def test_no_conflict_same_content(self, engine):
        """Two memories with identical content should not conflict."""
        mem_a = _make_memory(1, "Alice likes cats", category="fact", similarity=0.9)
        mem_b = _make_memory(2, "Alice likes cats", category="fact", similarity=0.85)
        result = engine._detect_conflicts([mem_a, mem_b])
        assert "_conflict_status" not in result[0]
        assert "_conflict_status" not in result[1]

    def test_no_conflict_low_similarity(self, engine):
        """Memories with similarity below 0.5 should not conflict."""
        mem_a = _make_memory(1, "Alice likes cats", category="fact", similarity=0.4)
        mem_b = _make_memory(2, "Alice likes dogs", category="fact", similarity=0.6)
        result = engine._detect_conflicts([mem_a, mem_b])
        assert "_conflict_status" not in result[0]
        assert "_conflict_status" not in result[1]

    def test_conflict_same_category_different_content(self, engine):
        """Two memories in same category with high similarity but different content -> conflict."""
        mem_a = _make_memory(
            1, "Alice's favorite color is blue",
            category="preference", source="system", similarity=0.9,
        )
        mem_b = _make_memory(
            2, "Alice's favorite color is red",
            category="preference", source="system", similarity=0.85,
        )
        result = engine._detect_conflicts([mem_a, mem_b])
        statuses = {r["id"]: r.get("_conflict_status") for r in result}
        assert "authoritative" in statuses.values()
        assert "conflicted" in statuses.values()

    def test_conflict_winner_by_source_priority(self, engine):
        """Higher source priority wins the conflict."""
        mem_user = _make_memory(
            1, "Alice's favorite color is blue",
            category="preference", source="user_confirmed", similarity=0.9,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        mem_auto = _make_memory(
            2, "Alice's favorite color is red",
            category="preference", source="auto_captured", similarity=0.85,
            created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),  # newer but lower priority
        )
        result = engine._detect_conflicts([mem_user, mem_auto])
        by_id = {r["id"]: r for r in result}
        assert by_id[1]["_conflict_status"] == "authoritative"
        assert by_id[2]["_conflict_status"] == "conflicted"
        assert by_id[2]["_conflicts_with"] == 1

    def test_conflict_winner_by_recency(self, engine):
        """Same source priority -> newer memory wins."""
        older = datetime(2025, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2025, 6, 1, tzinfo=timezone.utc)
        mem_old = _make_memory(
            1, "Alice works at Google",
            category="fact", source="observed", similarity=0.9,
            created_at=older,
        )
        mem_new = _make_memory(
            2, "Alice works at Meta",
            category="fact", source="observed", similarity=0.85,
            created_at=newer,
        )
        result = engine._detect_conflicts([mem_old, mem_new])
        by_id = {r["id"]: r for r in result}
        assert by_id[2]["_conflict_status"] == "authoritative"
        assert by_id[1]["_conflict_status"] == "conflicted"
        assert by_id[1]["_conflicts_with"] == 2

    def test_conflict_loser_references_winner(self, engine):
        """The conflicted memory should reference the authoritative one via _conflicts_with."""
        mem_a = _make_memory(
            10, "Temperature is 25C",
            category="fact", source="user_confirmed", similarity=0.9,
        )
        mem_b = _make_memory(
            20, "Temperature is 30C",
            category="fact", source="auto_captured", similarity=0.8,
        )
        result = engine._detect_conflicts([mem_a, mem_b])
        by_id = {r["id"]: r for r in result}
        assert by_id[20]["_conflicts_with"] == 10

    def test_returns_same_list_object(self, engine):
        """_detect_conflicts should return the same list (mutated in place)."""
        mems = [
            _make_memory(1, "A", similarity=0.9),
            _make_memory(2, "B", similarity=0.9),
        ]
        result = engine._detect_conflicts(mems)
        assert result is mems

    def test_no_conflict_when_only_one_above_threshold(self, engine):
        """If only one memory has similarity > 0.5 in a category, no conflict."""
        mem_a = _make_memory(1, "Alice likes cats", category="fact", similarity=0.9)
        mem_b = _make_memory(2, "Alice likes dogs", category="fact", similarity=0.3)
        result = engine._detect_conflicts([mem_a, mem_b])
        assert "_conflict_status" not in result[0]
        assert "_conflict_status" not in result[1]
