"""Tests for scheduler system."""

import pytest
from datetime import datetime, timezone, timedelta

from syne.scheduler import (
    _calculate_next_run,
    create_task,
    list_tasks,
    get_task,
    delete_task,
    enable_task,
    disable_task,
    delete_task_by_name,
)


class TestCalculateNextRun:
    """Tests for _calculate_next_run helper function."""

    def test_once_type_valid(self):
        """Test 'once' schedule type with valid ISO timestamp."""
        iso_time = "2026-03-01T09:00:00+07:00"
        next_run = _calculate_next_run("once", iso_time)
        
        assert next_run is not None
        assert next_run.year == 2026
        assert next_run.month == 3
        assert next_run.day == 1

    def test_once_type_invalid(self):
        """Test 'once' schedule type with invalid timestamp."""
        result = _calculate_next_run("once", "not-a-date")
        assert result is None

    def test_once_type_utc(self):
        """Test 'once' schedule type with UTC timestamp."""
        iso_time = "2026-06-15T12:00:00Z"
        next_run = _calculate_next_run("once", iso_time)
        
        assert next_run is not None
        assert next_run.tzinfo is not None  # Should be timezone-aware

    def test_interval_type_valid(self):
        """Test 'interval' schedule type."""
        now = datetime.now(timezone.utc)
        next_run = _calculate_next_run("interval", "3600", from_time=now)
        
        assert next_run is not None
        # Should be approximately 1 hour from now
        diff = (next_run - now).total_seconds()
        assert 3590 < diff < 3610  # Allow small tolerance

    def test_interval_type_invalid(self):
        """Test 'interval' schedule type with invalid value."""
        result = _calculate_next_run("interval", "not-a-number")
        assert result is None

    def test_interval_zero(self):
        """Test 'interval' schedule type with zero interval."""
        now = datetime.now(timezone.utc)
        next_run = _calculate_next_run("interval", "0", from_time=now)
        
        assert next_run is not None
        # Should be approximately now
        diff = abs((next_run - now).total_seconds())
        assert diff < 1

    def test_cron_type_valid(self):
        """Test 'cron' schedule type with valid expression."""
        # Skip if croniter not installed
        try:
            from croniter import croniter
        except ImportError:
            pytest.skip("croniter not installed")
        
        now = datetime.now(timezone.utc)
        # Every hour at minute 0
        next_run = _calculate_next_run("cron", "0 * * * *", from_time=now)
        
        assert next_run is not None
        assert next_run > now
        assert next_run.minute == 0

    def test_cron_type_invalid(self):
        """Test 'cron' schedule type with invalid expression."""
        result = _calculate_next_run("cron", "invalid cron expression")
        assert result is None

    def test_unknown_type(self):
        """Test unknown schedule type."""
        result = _calculate_next_run("unknown", "some value")
        assert result is None


class TestTaskCRUD:
    """Tests for task create/read/update/delete operations."""

    @pytest.fixture
    async def clean_tasks(self, db_pool):
        """Clean up scheduled_tasks table before/after test."""
        from syne.db.connection import get_connection
        
        async with get_connection() as conn:
            await conn.execute("DELETE FROM scheduled_tasks WHERE name LIKE 'test_%'")
        
        yield
        
        async with get_connection() as conn:
            await conn.execute("DELETE FROM scheduled_tasks WHERE name LIKE 'test_%'")

    @pytest.mark.asyncio
    async def test_create_task_once(self, db_pool, clean_tasks):
        """Test creating a 'once' scheduled task."""
        result = await create_task(
            name="test_once_task",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00+07:00",
            payload="Hello from scheduled task!",
            created_by=12345,
        )
        
        assert result["success"]
        task = result["task"]
        assert task["name"] == "test_once_task"
        assert task["schedule_type"] == "once"
        assert task["payload"] == "Hello from scheduled task!"
        assert task["next_run"] is not None

    @pytest.mark.asyncio
    async def test_create_task_interval(self, db_pool, clean_tasks):
        """Test creating an 'interval' scheduled task."""
        result = await create_task(
            name="test_interval_task",
            schedule_type="interval",
            schedule_value="3600",  # Every hour
            payload="Hourly check",
            created_by=12345,
        )
        
        assert result["success"]
        task = result["task"]
        assert task["schedule_type"] == "interval"
        assert task["schedule_value"] == "3600"

    @pytest.mark.asyncio
    async def test_create_task_invalid_type(self, db_pool, clean_tasks):
        """Test creating task with invalid schedule type."""
        result = await create_task(
            name="test_invalid",
            schedule_type="invalid_type",
            schedule_value="something",
            payload="Test",
        )
        
        assert not result["success"]
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_task_invalid_value(self, db_pool, clean_tasks):
        """Test creating task with invalid schedule value."""
        result = await create_task(
            name="test_invalid_value",
            schedule_type="once",
            schedule_value="not-a-valid-date",
            payload="Test",
        )
        
        assert not result["success"]
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tasks(self, db_pool, clean_tasks):
        """Test listing tasks."""
        # Create some tasks
        await create_task(
            name="test_list_1",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Task 1",
        )
        await create_task(
            name="test_list_2",
            schedule_type="interval",
            schedule_value="60",
            payload="Task 2",
        )
        
        tasks = await list_tasks()
        
        # Should have at least our 2 test tasks
        test_tasks = [t for t in tasks if t["name"].startswith("test_list_")]
        assert len(test_tasks) == 2

    @pytest.mark.asyncio
    async def test_list_tasks_enabled_only(self, db_pool, clean_tasks):
        """Test listing only enabled tasks."""
        # Create enabled task
        await create_task(
            name="test_enabled",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Enabled",
        )
        
        # Create and disable another task
        result = await create_task(
            name="test_disabled",
            schedule_type="once",
            schedule_value="2026-03-02T09:00:00Z",
            payload="Disabled",
        )
        await disable_task(result["task"]["id"])
        
        enabled_tasks = await list_tasks(enabled_only=True)
        test_enabled = [t for t in enabled_tasks if t["name"].startswith("test_")]
        
        # Only enabled task should be listed
        assert all(t["enabled"] for t in test_enabled)
        assert "test_enabled" in [t["name"] for t in test_enabled]
        assert "test_disabled" not in [t["name"] for t in test_enabled]

    @pytest.mark.asyncio
    async def test_get_task(self, db_pool, clean_tasks):
        """Test getting a task by ID."""
        result = await create_task(
            name="test_get",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Get me",
        )
        
        task_id = result["task"]["id"]
        task = await get_task(task_id)
        
        assert task is not None
        assert task["name"] == "test_get"
        assert task["payload"] == "Get me"

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, db_pool, clean_tasks):
        """Test getting a non-existent task."""
        task = await get_task(999999)
        assert task is None

    @pytest.mark.asyncio
    async def test_delete_task(self, db_pool, clean_tasks):
        """Test deleting a task by ID."""
        result = await create_task(
            name="test_delete",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Delete me",
        )
        task_id = result["task"]["id"]
        
        success = await delete_task(task_id)
        assert success
        
        # Verify deleted
        task = await get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_delete_task_not_found(self, db_pool, clean_tasks):
        """Test deleting a non-existent task."""
        success = await delete_task(999999)
        assert not success

    @pytest.mark.asyncio
    async def test_delete_task_by_name(self, db_pool, clean_tasks):
        """Test deleting a task by name."""
        await create_task(
            name="test_delete_by_name",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Delete by name",
        )
        
        success = await delete_task_by_name("test_delete_by_name")
        assert success
        
        # Verify deleted
        tasks = await list_tasks()
        names = [t["name"] for t in tasks]
        assert "test_delete_by_name" not in names

    @pytest.mark.asyncio
    async def test_enable_disable_task(self, db_pool, clean_tasks):
        """Test enabling and disabling a task."""
        result = await create_task(
            name="test_toggle",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Toggle me",
        )
        task_id = result["task"]["id"]
        
        # Task should be enabled by default
        task = await get_task(task_id)
        assert task["enabled"]
        
        # Disable
        success = await disable_task(task_id)
        assert success
        task = await get_task(task_id)
        assert not task["enabled"]
        
        # Enable
        success = await enable_task(task_id)
        assert success
        task = await get_task(task_id)
        assert task["enabled"]


class TestManageScheduleTool:
    """Tests for manage_schedule tool handler."""

    @pytest.fixture
    async def clean_tasks(self, db_pool):
        """Clean up scheduled_tasks table before/after test."""
        from syne.db.connection import get_connection
        
        async with get_connection() as conn:
            await conn.execute("DELETE FROM scheduled_tasks WHERE name LIKE 'tool_test_%'")
        
        yield
        
        async with get_connection() as conn:
            await conn.execute("DELETE FROM scheduled_tasks WHERE name LIKE 'tool_test_%'")

    @pytest.mark.asyncio
    async def test_tool_create(self, db_pool, clean_tasks):
        """Test manage_schedule tool create action."""
        from syne.tools.scheduler import manage_schedule_handler
        
        result = await manage_schedule_handler(
            action="create",
            name="tool_test_task",
            schedule_type="interval",
            schedule_value="3600",
            payload="Test payload",
        )
        
        assert "Task created" in result
        assert "tool_test_task" in result

    @pytest.mark.asyncio
    async def test_tool_list(self, db_pool, clean_tasks):
        """Test manage_schedule tool list action."""
        from syne.tools.scheduler import manage_schedule_handler
        
        # Create a task first
        await manage_schedule_handler(
            action="create",
            name="tool_test_list_task",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Test",
        )
        
        result = await manage_schedule_handler(action="list")
        
        assert "Scheduled Tasks" in result
        assert "tool_test_list_task" in result

    @pytest.mark.asyncio
    async def test_tool_create_missing_fields(self, db_pool, clean_tasks):
        """Test manage_schedule tool with missing required fields."""
        from syne.tools.scheduler import manage_schedule_handler
        
        # Missing name
        result = await manage_schedule_handler(
            action="create",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Test",
        )
        assert "Error" in result
        assert "name" in result.lower()
        
        # Missing schedule_type
        result = await manage_schedule_handler(
            action="create",
            name="test",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Test",
        )
        assert "Error" in result
        assert "schedule_type" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_delete(self, db_pool, clean_tasks):
        """Test manage_schedule tool delete action."""
        from syne.tools.scheduler import manage_schedule_handler
        
        # Create a task
        await create_task(
            name="tool_test_delete",
            schedule_type="once",
            schedule_value="2026-03-01T09:00:00Z",
            payload="Delete me",
        )
        
        result = await manage_schedule_handler(
            action="delete",
            name="tool_test_delete",
        )
        
        assert "deleted" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_unknown_action(self, db_pool, clean_tasks):
        """Test manage_schedule tool with unknown action."""
        from syne.tools.scheduler import manage_schedule_handler
        
        result = await manage_schedule_handler(action="unknown")
        
        assert "Unknown action" in result
