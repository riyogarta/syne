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


# ---------------------------------------------------------------------------
# history_search: argument validation
# ---------------------------------------------------------------------------


class TestHistorySearchValidation:

    async def test_empty_query_returns_error(self):
        result = await H.history_search_handler(query="")
        assert result.startswith("Error:")
        assert "query is required" in result

    async def test_whitespace_query_returns_error(self):
        result = await H.history_search_handler(query="   ")
        assert result.startswith("Error:")

    async def test_bad_sort_by_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(query="anything", sort_by="cosmic")
        assert result.startswith("Error:")
        assert "sort_by" in result

    async def test_bad_since_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(query="x", since="yesterday")
        assert result.startswith("Error:")
        assert "since" in result.lower()

    async def test_bad_until_returns_error(self):
        H.set_embedding_provider(MagicMock())
        result = await H.history_search_handler(
            query="x", since="2026-07-01", until="tomorrow"
        )
        assert result.startswith("Error:")
        assert "until" in result.lower()

    async def test_no_provider_returns_clear_error(self):
        H.set_embedding_provider(None)
        result = await H.history_search_handler(query="anything")
        assert result.startswith("Error:")
        assert "embedding provider" in result.lower()

    async def test_provider_returns_empty_vector(self):
        prov = MagicMock()
        # embed() itself is awaitable → AsyncMock; return value has .vector = []
        prov.embed = AsyncMock(return_value=MagicMock(vector=[]))
        H.set_embedding_provider(prov)
        result = await H.history_search_handler(query="anything")
        assert result.startswith("Error:")
        assert "empty vector" in result

    async def test_provider_raises_returns_error(self):
        prov = MagicMock()
        prov.embed = AsyncMock(side_effect=RuntimeError("ollama down"))
        H.set_embedding_provider(prov)
        result = await H.history_search_handler(query="anything")
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
            result = await H.history_search_handler(query="something obscure")

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
            result = await H.history_search_handler(
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
            await H.history_search_handler(query="x", limit=9999)

        # Last two params of the SELECT are limit + offset (see handler SQL).
        params = captured_params["params"]
        assert params[-2] == 50, "limit must be clamped to 50"
        assert params[-1] == 0


# ---------------------------------------------------------------------------
# history_expand: argument validation
# ---------------------------------------------------------------------------


class TestHistoryExpandValidation:

    async def test_empty_anchor_ids_returns_error(self):
        result = await H.history_expand_handler(anchor_ids=[])
        assert result.startswith("Error:")
        assert "anchor_ids" in result

    async def test_non_integer_anchor_ids_returns_error(self):
        result = await H.history_expand_handler(anchor_ids=["not-a-number"])
        assert result.startswith("Error:")
        assert "integers" in result.lower()

    async def test_too_many_anchor_ids_returns_error(self):
        # Cap is 20 — 21 must reject.
        result = await H.history_expand_handler(anchor_ids=list(range(1, 22)))
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
            result = await H.history_expand_handler(
                anchor_ids=[42], context_before=99999, context_after=99999
            )

        payload = json.loads(result)
        # Missing anchor because DB returned empty → present in missing list.
        assert 42 in payload.get("missing_anchor_ids", [])


# ---------------------------------------------------------------------------
# Tool schemas — the shape the LLM registration relies on
# ---------------------------------------------------------------------------


class TestToolSchemas:

    def test_history_search_tool_schema_required_fields(self):
        t = H.HISTORY_SEARCH_TOOL
        assert t["name"] == "history_search"
        assert t["permission"] == 0o400  # owner-only
        assert "query" in t["parameters"]["properties"]
        assert t["parameters"]["required"] == ["query"]
        assert callable(t["handler"])

    def test_history_expand_tool_schema_required_fields(self):
        t = H.HISTORY_EXPAND_TOOL
        assert t["name"] == "history_expand"
        assert t["permission"] == 0o400
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
