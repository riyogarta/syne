"""Tests for the history_search + history_expand tools.

These lock the *argument-validation and shaping* behaviour so downstream
changes to the regex / SQL / provider layers can't silently break the tool
contract exposed to the LLM. DB round-trips are mocked so tests remain fast
and run without a live PostgreSQL.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from syne.tools import history as H

# The handlers now refuse to run without a scope. The owner in a DM is the
# unrestricted case, so the pre-existing shape tests keep their meaning.
OWNER_DM = {
    "access": "owner", "platform": "telegram", "chat_id": "3478983",
    "is_group": False, "sender_pid": "3478983", "session_id": 1,
}


# ---------------------------------------------------------------------------
# history_search: argument validation
# ---------------------------------------------------------------------------


class TestHistorySearchValidation:

    async def test_empty_query_returns_error(self):
        result = await H.history_search_handler(_scope=OWNER_DM, query="")
        assert result.startswith("Error:")
        assert "query is required" in result

    async def test_whitespace_query_returns_error(self):
        result = await H.history_search_handler(_scope=OWNER_DM, query="   ")
        assert result.startswith("Error:")

    async def test_bad_sort_by_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(_scope=OWNER_DM, query="anything", sort_by="cosmic")
        assert result.startswith("Error:")
        assert "sort_by" in result

    async def test_bad_since_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(_scope=OWNER_DM, query="x", since="yesterday")
        assert result.startswith("Error:")
        assert "since" in result.lower()

    async def test_bad_until_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(_scope=OWNER_DM, 
            query="x", since="2026-07-01", until="tomorrow"
        )
        assert result.startswith("Error:")
        assert "until" in result.lower()

    async def test_no_provider_returns_clear_error(self):
        H.set_embedding_provider(None)
        result = await H.history_search_handler(_scope=OWNER_DM, query="anything")
        assert result.startswith("Error:")
        assert "embedding provider" in result.lower()

    async def test_provider_returns_empty_vector(self):
        prov = MagicMock()
        # embed() itself is awaitable → AsyncMock; return value has .vector = []
        prov.embed = AsyncMock(return_value=MagicMock(vector=[]))
        H.set_embedding_provider(prov)
        result = await H.history_search_handler(_scope=OWNER_DM, query="anything")
        assert result.startswith("Error:")
        assert "empty vector" in result

    async def test_provider_raises_returns_error(self):
        prov = MagicMock()
        prov.embed = AsyncMock(side_effect=RuntimeError("ollama down"))
        H.set_embedding_provider(prov)
        result = await H.history_search_handler(_scope=OWNER_DM, query="anything")
        assert result.startswith("Error:")
        assert "ollama down" in result


# ---------------------------------------------------------------------------
# history_search: shape of the successful response
# ---------------------------------------------------------------------------


class TestHistorySearchShape:

    def _make_row(self, id_: int, session_id: int, similarity: float, content: str):
        """Postgres row look-alike (dict access)."""
        from datetime import datetime, timezone
        return {
            "id": id_,
            "session_id": session_id,
            "content": content,
            "created_at": datetime(2026, 7, 12, 10, 30, tzinfo=timezone.utc),
            "similarity": similarity,
            "chat_id": "test-chat",
        }

    async def test_no_hits_returns_friendly_hint(self):
        prov = MagicMock()
        prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1, 0.2, 0.3]))
        H.set_embedding_provider(prov)

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            result = await H.history_search_handler(_scope=OWNER_DM, query="something obscure")

        assert "No matches" in result
        assert "broader" in result or "wider" in result or "different" in result

    async def test_hits_are_shaped_as_expected(self):
        prov = MagicMock()
        prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1, 0.2, 0.3]))
        H.set_embedding_provider(prov)

        rows = [
            self._make_row(101, 5, 0.87, "berbicara soal decay memory kemarin"),
            self._make_row(202, 7, 0.79, "Aku pikir cap-nya perlu dinaikkan"),
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            result = await H.history_search_handler(_scope=OWNER_DM, 
                query="decay memory", limit=5, sort_by="similarity"
            )

        payload = json.loads(result)
        assert payload["sort_by"] == "similarity"
        assert payload["limit"] == 5
        assert payload["returned"] == 2
        assert payload["next_offset"] == 2
        assert len(payload["hits"]) == 2
        # anchor_id + preview + similarity all present
        for hit in payload["hits"]:
            assert "anchor_id" in hit
            assert "preview" in hit
            assert "similarity" in hit
            assert "created_at" in hit
        # First hit ordered by similarity DESC (cosmetic — order preserved from SQL)
        assert payload["hits"][0]["anchor_id"] == 101

    async def test_limit_clamped_to_50(self):
        prov = MagicMock()
        prov.embed = AsyncMock(return_value=MagicMock(vector=[0.1]))
        H.set_embedding_provider(prov)

        captured_params = {}

        async def fetch_capture(sql, *params):
            captured_params["params"] = params
            return []

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=fetch_capture)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            await H.history_search_handler(_scope=OWNER_DM, query="x", limit=9999)

        # Last two params of the SELECT are limit + offset (see handler SQL).
        params = captured_params["params"]
        assert params[-2] == 50, "limit must be clamped to 50"
        assert params[-1] == 0


# ---------------------------------------------------------------------------
# history_expand: argument validation
# ---------------------------------------------------------------------------


class TestHistoryExpandValidation:

    async def test_empty_anchor_ids_returns_error(self):
        result = await H.history_expand_handler(_scope=OWNER_DM, anchor_ids=[])
        assert result.startswith("Error:")
        assert "anchor_ids" in result

    async def test_non_integer_anchor_ids_returns_error(self):
        result = await H.history_expand_handler(_scope=OWNER_DM, anchor_ids=["not-a-number"])
        assert result.startswith("Error:")
        assert "integers" in result.lower()

    async def test_too_many_anchor_ids_returns_error(self):
        # Cap is 20 — 21 must reject.
        result = await H.history_expand_handler(_scope=OWNER_DM, anchor_ids=list(range(1, 22)))
        assert result.startswith("Error:")
        assert "at most 20" in result

    async def test_context_before_is_capped(self):
        """Rogue context_before shouldn't overwhelm the response — clamped at 20."""
        # No DB — validation happens before any query in successful path, but
        # we can at least verify the handler doesn't crash on absurd bounds.
        # A missing anchor still returns a response; the important thing is no
        # exception + no dump of 500 turns.
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])  # anchor lookup empty → missing list
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            result = await H.history_expand_handler(_scope=OWNER_DM, 
                anchor_ids=[42], context_before=99999, context_after=99999
            )

        payload = json.loads(result)
        # Missing anchor because DB returned empty → present in missing list.
        assert 42 in payload.get("missing_anchor_ids", [])


# ---------------------------------------------------------------------------
# history_expand: window shaping (conversation only, no tool results)
# ---------------------------------------------------------------------------


class TestHistoryExpandWindow:

    async def test_returns_conversation_only(self):
        from datetime import datetime, timedelta, timezone
        base = datetime(2026, 11, 7, 5, 0, tzinfo=timezone.utc)

        def t(i):
            return base + timedelta(seconds=i)

        calls = {}

        async def fake_fetch(sql, *params, timeout=None):
            if "history_expand:emb" in sql:
                return []
            if "id = ANY" in sql:
                return [{"id": 100, "session_id": 51, "created_at": t(100)}]
            if "history_expand:before" in sql:
                calls["before"] = params
                return [{"id": 99, "created_at": t(99)}, {"id": 97, "created_at": t(97)}]
            if "history_expand:after" in sql:
                calls["after"] = params
                return [{"id": 101, "created_at": t(101)}]  # fewer than asked
            if "history_expand:window" in sql:
                calls["window"] = (sql, params)
                return [
                    {"id": 97, "role": "user", "content": "q1", "created_at": t(97)},
                    {"id": 99, "role": "assistant", "content": "a1", "created_at": t(99)},
                    {"id": 100, "role": "user", "content": "anchor", "created_at": t(100)},
                    {"id": 101, "role": "assistant", "content": "a2", "created_at": t(101)},
                ]
            raise AssertionError(f"unexpected SQL: {sql}")

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=fake_fetch)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            result = await H.history_expand_handler(_scope=OWNER_DM, 
                anchor_ids=[100], context_before=2, context_after=3
            )

        # LIMITs passed are the requested counts
        assert calls["before"][-1] == 2
        assert calls["after"][-1] == 3
        # Lower edge = 2nd row before; upper side open (only 1 of 3 found)
        sql, params = calls["window"]
        assert params == (51, t(97), 97)
        assert "<=" not in sql
        # Never numbers the whole session again; tool rows filtered in SQL
        assert "ROW_NUMBER" not in sql
        assert H._CONVO_FILTER in sql

        turns = json.loads(result)["blocks"][0]["turns"]
        assert [x["id"] for x in turns] == [97, 99, 100, 101]
        assert [x["id"] for x in turns if x["is_anchor"]] == [100]
        assert all(x["role"] in ("user", "assistant") for x in turns)

    async def test_zero_context_bounds_on_anchor(self):
        from datetime import datetime, timezone
        at = datetime(2026, 11, 7, 5, 0, tzinfo=timezone.utc)
        seen = {}

        async def fake_fetch(sql, *params, timeout=None):
            if "id = ANY" in sql:
                return [{"id": 7, "session_id": 3, "created_at": at}]
            if "history_expand:before" in sql or "history_expand:after" in sql:
                return []
            seen["params"] = params
            return [{"id": 7, "role": "user", "content": "only", "created_at": at}]

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=fake_fetch)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            result = await H.history_expand_handler(_scope=OWNER_DM, 
                anchor_ids=[7], context_before=0, context_after=0
            )
        assert seen["params"] == (3, at, 7, at, 7)
        assert [x["id"] for x in json.loads(result)["blocks"][0]["turns"]] == [7]


# ---------------------------------------------------------------------------
# history_expand: adaptive radius (pure function)
# ---------------------------------------------------------------------------


def _rows(spec, base=None, step=60):
    """spec: list of (id, role); created_at step seconds apart."""
    from datetime import datetime, timedelta, timezone
    base = base or datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
    return [
        {"id": i, "role": role, "created_at": base + timedelta(seconds=k * step)}
        for k, (i, role) in enumerate(spec)
    ]


SPEC = [(1, "user"), (2, "assistant"), (3, "user"), (4, "assistant"),
        (5, "user"), (6, "assistant"), (7, "user"), (8, "assistant"),
        (9, "user"), (10, "assistant")]


class TestAdaptiveBounds:

    def test_no_embeddings_keeps_whole_window(self):
        rows = _rows(SPEC)
        lo, hi, sb, sa = H.adaptive_bounds(
            rows, 4, {}, threshold=0.5, gap_seconds=7200, block=2,
            edge_before="session_edge", edge_after="max_radius")
        assert (lo, hi) == (0, 9)
        assert (sb, sa) == ("session_edge", "max_radius")

    def test_topic_shift_stops_both_sides(self):
        rows = _rows(SPEC)
        A, B = [1.0, 0.0], [0.0, 1.0]
        vecs = {1: B, 3: A, 5: A, 7: A, 9: B}
        lo, hi, sb, sa = H.adaptive_bounds(
            rows, 4, vecs, threshold=0.5, gap_seconds=0, block=1)
        # anchor exchange = rows 4..5 (user 5); keeps users 3,5,7
        assert (lo, hi) == (2, 7)
        assert (sb, sa) == ("topic_shift", "topic_shift")

    def test_time_gap_stops(self):
        from datetime import timedelta
        rows = _rows(SPEC)
        for r in rows[6:]:
            r["created_at"] += timedelta(hours=5)
        lo, hi, sb, sa = H.adaptive_bounds(
            rows, 4, {}, threshold=0.5, gap_seconds=7200, block=2)
        assert hi == 5 and sa == "time_gap"
        assert lo == 0

    def test_anchor_exchange_always_kept(self):
        rows = _rows(SPEC)
        A, B = [1.0, 0.0], [0.0, 1.0]
        vecs = {1: B, 3: B, 5: A, 7: B, 9: B}
        lo, hi, _, _ = H.adaptive_bounds(
            rows, 4, vecs, threshold=0.5, gap_seconds=0, block=1)
        assert (lo, hi) == (4, 5)

    def test_leading_assistant_rows_grouped(self):
        rows = _rows([(1, "assistant"), (2, "user"), (3, "assistant")])
        lo, hi, _, _ = H.adaptive_bounds(
            rows, 1, {}, threshold=0.5, gap_seconds=0, block=2)
        assert (lo, hi) == (0, 2)

    def test_parse_vec(self):
        assert H._parse_vec("[0.5,-1,2e-1]") == [0.5, -1.0, 0.2]
        assert H._parse_vec(None) is None
        assert H._parse_vec("[]") is None


# ---------------------------------------------------------------------------
# history_expand: dedup (merge overlapping windows, flag in-context turns)
# ---------------------------------------------------------------------------


class TestExpandDedup:

    def _piece(self, aid, ids, sid=51):
        rows = [dict(r, content=f"c{r['id']}") for r in _rows([(i, "user" if i % 2 else "assistant") for i in ids])]
        # make created_at follow id so different pieces line up
        from datetime import datetime, timedelta, timezone
        base = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
        for r in rows:
            r["created_at"] = base + timedelta(seconds=r["id"])
        return {"anchor_id": aid, "session_id": sid, "rows": rows,
                "radius": {"adaptive": True}}

    def test_overlapping_windows_merge(self):
        blocks = H._build_blocks(
            [self._piece(5, [3, 4, 5, 6, 7]), self._piece(7, [6, 7, 8, 9])],
            set(), True, 10, 10)
        assert len(blocks) == 1
        b = blocks[0]
        assert [t["id"] for t in b["turns"]] == [3, 4, 5, 6, 7, 8, 9]
        assert b["anchor_ids"] == [5, 7]
        assert sorted(t["id"] for t in b["turns"] if t["is_anchor"]) == [5, 7]
        assert set(b["radius"]) == {"5", "7"}

    def test_disjoint_windows_stay_apart(self):
        blocks = H._build_blocks(
            [self._piece(3, [1, 2, 3]), self._piece(9, [8, 9, 10])],
            set(), True, 10, 10)
        assert [b["anchor_id"] for b in blocks] == [3, 9]

    def test_other_session_never_merges(self):
        blocks = H._build_blocks(
            [self._piece(3, [1, 2, 3], sid=1), self._piece(3_0, [2, 3], sid=2)],
            set(), True, 10, 10)
        assert len(blocks) == 2

    def test_in_context_turns_flagged_and_blank(self):
        blocks = H._build_blocks([self._piece(5, [4, 5, 6])], {6}, True, 10, 10)
        t6 = blocks[0]["turns"][-1]
        assert t6["already_in_context"] is True and t6["content"] == ""
        assert "already_in_context" not in blocks[0]["turns"][0]

    def test_dedup_off_keeps_overlap(self):
        blocks = H._build_blocks(
            [self._piece(5, [4, 5, 6]), self._piece(6, [5, 6, 7])],
            set(), False, 10, 10)
        assert len(blocks) == 2

    async def test_handler_flags_window_rows(self):
        """End to end: duplicate anchor ids collapse, in-context ids flagged."""
        from datetime import datetime, timedelta, timezone
        base = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)

        def t(i):
            return base + timedelta(seconds=i)

        async def fake_fetch(sql, *params, timeout=None):
            if "history_expand:emb" in sql:
                return []
            if "history_expand:in_context" in sql:
                assert params == (51, 2)
                return [{"id": 12}, {"id": 11}]
            if "id = ANY" in sql:
                return [{"id": 10, "session_id": 51, "created_at": t(10)}]
            if "history_expand:before" in sql or "history_expand:after" in sql:
                return []
            if "history_expand:window" in sql:
                return [
                    {"id": 10, "role": "user", "content": "anchor", "created_at": t(10)},
                    {"id": 11, "role": "assistant", "content": "a", "created_at": t(11)},
                ]
            raise AssertionError(sql)

        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=fake_fetch)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=False)
        scope = dict(OWNER_DM, session_id=51, window_size=2)

        with patch("syne.tools.history.get_connection", return_value=ctx):
            out = json.loads(await H.history_expand_handler(
                _scope=scope, anchor_ids=[10, 10]))
        assert len(out["blocks"]) == 1
        turns = out["blocks"][0]["turns"]
        assert [x["id"] for x in turns] == [10, 11]
        assert turns[1]["already_in_context"] is True
        assert "note" in out


# ---------------------------------------------------------------------------
# Tool schemas — the shape the LLM registration relies on
# ---------------------------------------------------------------------------


class TestToolSchemas:

    def test_history_search_tool_schema_required_fields(self):
        t = H.HISTORY_SEARCH_TOOL
        assert t["name"] == "history_search"
        assert t["permission"] == 0o444  # owner + family (widened from 0o400 in v1.18.36)
        assert "query" in t["parameters"]["properties"]
        assert t["parameters"]["required"] == ["query"]
        assert callable(t["handler"])

    def test_history_expand_tool_schema_required_fields(self):
        t = H.HISTORY_EXPAND_TOOL
        assert t["name"] == "history_expand"
        assert t["permission"] == 0o444  # owner + family (widened from 0o400 in v1.18.36)
        assert "anchor_ids" in t["parameters"]["properties"]
        assert t["parameters"]["required"] == ["anchor_ids"]
        assert callable(t["handler"])


# ---------------------------------------------------------------------------
# _preview helper — cheap unit test on the shape guarantee
# ---------------------------------------------------------------------------


class TestPreviewHelper:

    def test_preview_collapses_whitespace(self):
        assert H._preview("a   b\n\nc") == "a b c"

    def test_preview_short_text_untouched(self):
        assert H._preview("hello") == "hello"

    def test_preview_long_text_gets_ellipsis(self):
        long = "x" * 200
        out = H._preview(long, n=50)
        assert len(out) == 50
        assert out.endswith("…")

    def test_preview_empty_stays_empty(self):
        assert H._preview("") == ""
        assert H._preview(None) == ""
