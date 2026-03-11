"""Tests for syne.scheduler module — schedule calculation, task types."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, AsyncMock

from syne.scheduler import (
    _get_tz,
    _parse_cron_next,
    _calculate_next_run,
    Scheduler,
)


# ── Timezone resolution ──────────────────────────────────────────────


class TestGetTz:
    def test_utc(self):
        tz = _get_tz("UTC")
        assert tz is not None

    def test_known_timezone(self):
        tz = _get_tz("Asia/Jakarta")
        assert tz is not None

    def test_invalid_timezone_falls_back_to_utc(self):
        tz = _get_tz("Invalid/Nonexistent")
        assert tz is not None  # Should fallback to UTC


# ── Cron next run ────────────────────────────────────────────────────


class TestParseCronNext:
    def test_valid_cron(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _parse_cron_next("0 9 * * *", base)
        assert result is not None
        assert result > base

    def test_every_minute(self):
        base = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
        result = _parse_cron_next("* * * * *", base)
        assert result is not None
        # Next minute
        expected = base + timedelta(minutes=1)
        assert result.minute == expected.minute

    def test_invalid_cron_returns_none(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = _parse_cron_next("invalid cron expression", base)
        assert result is None

    def test_cron_with_timezone(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        tz = _get_tz("Asia/Jakarta")
        result = _parse_cron_next("0 9 * * *", base, tz=tz)
        assert result is not None
        # Should be converted back to UTC
        assert result.tzinfo is not None


# ── Calculate next run ───────────────────────────────────────────────


class TestCalculateNextRun:
    def test_once_type_valid_iso(self):
        result = _calculate_next_run("once", "2026-12-25T09:00:00+00:00")
        assert result is not None
        assert result.year == 2026
        assert result.month == 12

    def test_once_type_with_z_suffix(self):
        result = _calculate_next_run("once", "2026-12-25T09:00:00Z")
        assert result is not None

    def test_once_type_invalid(self):
        result = _calculate_next_run("once", "not a date")
        assert result is None

    def test_interval_type(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _calculate_next_run("interval", "3600", from_time=base)
        assert result is not None
        assert result == base + timedelta(seconds=3600)

    def test_interval_invalid(self):
        result = _calculate_next_run("interval", "not_a_number")
        assert result is None

    def test_cron_type(self):
        base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = _calculate_next_run("cron", "0 9 * * *", from_time=base)
        assert result is not None
        assert result > base

    def test_unknown_type(self):
        result = _calculate_next_run("weekly", "monday")
        assert result is None

    def test_default_from_time_is_now(self):
        result = _calculate_next_run("interval", "60")
        assert result is not None
        # Should be roughly 60s from now
        diff = (result - datetime.now(timezone.utc)).total_seconds()
        assert 55 < diff < 65


# ── Scheduler lifecycle ──────────────────────────────────────────────


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        callback = AsyncMock()
        scheduler = Scheduler(on_task_execute=callback)

        # Mock the run loop so it doesn't actually run
        with patch.object(scheduler, "_run_loop", new_callable=AsyncMock):
            await scheduler.start()
            assert scheduler._running is True
            await scheduler.stop()
            assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_double_start_warns(self, caplog):
        import logging
        callback = AsyncMock()
        scheduler = Scheduler(on_task_execute=callback)

        with patch.object(scheduler, "_run_loop", new_callable=AsyncMock):
            await scheduler.start()
            with caplog.at_level(logging.WARNING, logger="syne.scheduler"):
                await scheduler.start()  # Should warn
            assert "already running" in caplog.text
            await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        callback = AsyncMock()
        scheduler = Scheduler(on_task_execute=callback)

        with patch.object(scheduler, "_run_loop", new_callable=AsyncMock):
            await scheduler.start()
            assert scheduler._task is not None
            await scheduler.stop()
