"""Tests for syne.conversation — pure functions and ConversationManager basics."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock

from syne.conversation import (
    _resolve_tz,
    _fmt_offset,
    _fmt_components,
    _apply_template,
    ConversationManager,
)


# ---------------------------------------------------------------------------
# _resolve_tz
# ---------------------------------------------------------------------------

class TestResolveTz:

    def test_valid_timezone(self):
        tz, name = _resolve_tz("Asia/Jakarta")
        assert name == "Asia/Jakarta"
        # Should produce a usable tzinfo
        dt = datetime(2024, 1, 1, tzinfo=tz)
        assert dt.tzinfo is not None

    def test_utc_explicit(self):
        tz, name = _resolve_tz("UTC")
        assert name == "UTC"

    def test_invalid_timezone_falls_back_to_utc(self):
        tz, name = _resolve_tz("Not/A/Timezone")
        assert name == "UTC"

    def test_none_falls_back_to_utc(self):
        tz, name = _resolve_tz(None)
        assert name == "UTC"

    def test_empty_string_falls_back_to_utc(self):
        tz, name = _resolve_tz("")
        assert name == "UTC"

    def test_whitespace_only_falls_back_to_utc(self):
        tz, name = _resolve_tz("   ")
        assert name == "UTC"

    def test_non_string_falls_back_to_utc(self):
        tz, name = _resolve_tz(12345)
        assert name == "UTC"


# ---------------------------------------------------------------------------
# _fmt_offset
# ---------------------------------------------------------------------------

class TestFmtOffset:

    def test_positive_offset(self):
        td = timedelta(hours=7)
        assert _fmt_offset(td) == "+07:00"

    def test_negative_offset(self):
        td = timedelta(hours=-5)
        assert _fmt_offset(td) == "-05:00"

    def test_zero_offset(self):
        td = timedelta(0)
        assert _fmt_offset(td) == "+00:00"

    def test_fractional_hour_offset(self):
        td = timedelta(hours=5, minutes=30)
        assert _fmt_offset(td) == "+05:30"

    def test_negative_fractional_offset(self):
        td = timedelta(hours=-9, minutes=-45)
        assert _fmt_offset(td) == "-09:45"

    def test_none_offset(self):
        assert _fmt_offset(None) == "+00:00"


# ---------------------------------------------------------------------------
# _fmt_components
# ---------------------------------------------------------------------------

class TestFmtComponents:

    @pytest.fixture
    def sample_dt(self):
        return datetime(2024, 3, 15, 14, 30, 45)  # Friday, March 15, 2024

    def test_indonesian_locale(self, sample_dt):
        result = _fmt_components(sample_dt, "id")
        assert result["day_name"] == "Jumat"
        assert result["month_name"] == "Maret"
        assert result["day"] == 15
        assert result["month"] == 3
        assert result["year"] == 2024
        assert result["hour"] == 14
        assert result["minute"] == 30
        assert result["second"] == 45

    def test_english_locale(self, sample_dt):
        result = _fmt_components(sample_dt, "en")
        assert result["day_name"] == "Friday"
        assert result["month_name"] == "March"

    def test_none_locale_defaults_to_indonesian(self, sample_dt):
        result = _fmt_components(sample_dt, None)
        assert result["day_name"] == "Jumat"

    def test_all_keys_present(self, sample_dt):
        result = _fmt_components(sample_dt, "id")
        expected_keys = {
            "day_name", "month_name", "day", "month", "year",
            "hour", "minute", "second", "date", "time", "full", "iso",
        }
        assert set(result.keys()) == expected_keys

    def test_date_string_format(self, sample_dt):
        result = _fmt_components(sample_dt, "id")
        assert result["date"] == "15 Maret 2024"

    def test_time_string_format(self, sample_dt):
        result = _fmt_components(sample_dt, "en")
        assert result["time"] == "14:30:45"

    def test_full_string_format(self, sample_dt):
        result = _fmt_components(sample_dt, "en")
        assert result["full"] == "Friday, 15 March 2024 14:30:45"

    def test_monday_index(self):
        # 2024-01-01 is a Monday
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = _fmt_components(dt, "id")
        assert result["day_name"] == "Senin"

    def test_sunday_index(self):
        # 2024-01-07 is a Sunday
        dt = datetime(2024, 1, 7, 0, 0, 0)
        result = _fmt_components(dt, "en")
        assert result["day_name"] == "Sunday"


# ---------------------------------------------------------------------------
# _apply_template
# ---------------------------------------------------------------------------

class TestApplyTemplate:

    def test_valid_template(self):
        components = {"day_name": "Friday", "date": "15 March 2024", "time": "14:30:00"}
        result = _apply_template("{day_name}, {date} {time}", components)
        assert result == "Friday, 15 March 2024 14:30:00"

    def test_missing_keys_fallback(self):
        components = {"full": "Friday, 15 March 2024 14:30:00", "iso": "2024-03-15T14:30:00"}
        result = _apply_template("{nonexistent_key}", components)
        assert result == "Friday, 15 March 2024 14:30:00"

    def test_none_template(self):
        components = {"full": "Friday, 15 March 2024 14:30:00"}
        result = _apply_template(None, components)
        # None template → (None or '').format(**components) → '' (empty string)
        assert result == ""

    def test_empty_template(self):
        components = {"full": "test"}
        result = _apply_template("", components)
        assert result == ""

    def test_partial_template(self):
        components = {"day_name": "Friday", "time": "14:30:00"}
        result = _apply_template("Today is {day_name}", components)
        assert result == "Today is Friday"


# ---------------------------------------------------------------------------
# ConversationManager._session_key
# ---------------------------------------------------------------------------

class TestConversationManagerSessionKey:

    def _make_manager(self):
        return ConversationManager(
            provider=MagicMock(),
            memory=MagicMock(),
            tools=MagicMock(),
            abilities=None,
            context_mgr=MagicMock(),
            subagents=None,
        )

    def test_session_key_format(self):
        mgr = self._make_manager()
        key = mgr._session_key("telegram", "12345")
        assert key == "telegram:12345"

    def test_session_key_different_platforms(self):
        mgr = self._make_manager()
        k1 = mgr._session_key("telegram", "100")
        k2 = mgr._session_key("whatsapp", "100")
        assert k1 != k2

    def test_session_key_different_chats(self):
        mgr = self._make_manager()
        k1 = mgr._session_key("telegram", "100")
        k2 = mgr._session_key("telegram", "200")
        assert k1 != k2


# ---------------------------------------------------------------------------
# ConversationManager callback methods
# ---------------------------------------------------------------------------

class TestConversationManagerCallbacks:

    def _make_manager(self):
        return ConversationManager(
            provider=MagicMock(),
            memory=MagicMock(),
            tools=MagicMock(),
            abilities=None,
            context_mgr=MagicMock(),
            subagents=None,
        )

    def test_add_delivery_callback(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_delivery_callback(cb)
        assert cb in mgr._delivery_callbacks

    def test_add_delivery_callback_no_duplicates(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_delivery_callback(cb)
        mgr.add_delivery_callback(cb)
        assert mgr._delivery_callbacks.count(cb) == 1

    def test_remove_delivery_callback(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_delivery_callback(cb)
        mgr.remove_delivery_callback(cb)
        assert cb not in mgr._delivery_callbacks

    def test_remove_delivery_callback_not_present(self):
        mgr = self._make_manager()
        cb = MagicMock()
        # Should not raise
        mgr.remove_delivery_callback(cb)

    def test_add_status_callback(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_status_callback(cb)
        assert cb in mgr._status_callbacks

    def test_add_status_callback_no_duplicates(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_status_callback(cb)
        mgr.add_status_callback(cb)
        assert mgr._status_callbacks.count(cb) == 1

    def test_remove_status_callback(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.add_status_callback(cb)
        mgr.remove_status_callback(cb)
        assert cb not in mgr._status_callbacks

    def test_remove_status_callback_not_present(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.remove_status_callback(cb)

    def test_set_tool_callback(self):
        mgr = self._make_manager()
        cb = MagicMock()
        mgr.set_tool_callback(cb)
        assert mgr._tool_callback is cb

    def test_set_tool_callback_replaces(self):
        mgr = self._make_manager()
        cb1 = MagicMock()
        cb2 = MagicMock()
        mgr.set_tool_callback(cb1)
        mgr.set_tool_callback(cb2)
        assert mgr._tool_callback is cb2

    def test_set_stream_callbacks(self):
        mgr = self._make_manager()
        cbs = MagicMock()
        mgr.set_stream_callbacks(cbs)
        assert mgr._stream_callbacks is cbs

    def test_set_stream_callbacks_none(self):
        mgr = self._make_manager()
        mgr.set_stream_callbacks(MagicMock())
        mgr.set_stream_callbacks(None)
        assert mgr._stream_callbacks is None
