"""v1.23.27: a scheduled task that ran right before /restore began is not
run again when the restore fails (review v1.23.16-19 #2 partial)."""
from datetime import datetime, timedelta, timezone

import pytest

from syne import maintenance as MT
from syne import scheduler as SCH
from syne.scheduler import Scheduler


class _Conn:
    def __init__(self, tasks):
        self.tasks = tasks
        self.writes = []

    async def fetch(self, sql, now):
        return [t for t in self.tasks.values() if t["enabled"] and t["next_run"] <= now]

    async def execute(self, sql, *args):
        self.writes.append((sql, args))
        t = self.tasks.get(args[-1])
        if t is None:
            return
        if "enabled = false" in sql:
            t["enabled"] = False
        elif "next_run = $2" in sql:
            t["next_run"] = args[1]


def _task(tid, kind, value, next_run):
    return {"id": tid, "name": f"t{tid}", "schedule_type": kind, "schedule_value": value,
            "payload": "p", "created_by": 1, "end_date": None, "next_run": next_run,
            "target_chat_id": "x", "target_chat_type": "private", "enabled": True}


@pytest.fixture
def env(monkeypatch):
    from syne.db import connection, models
    tasks = {}
    conn = _Conn(tasks)

    class _CM:
        async def __aenter__(self):
            return conn

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(connection, "get_connection", lambda: _CM())

    async def cfg(key, default=None):
        return default
    monkeypatch.setattr(models, "get_config", cfg)

    async def tz():
        return timezone.utc
    monkeypatch.setattr(SCH, "_get_system_tz", tz)
    monkeypatch.setattr(MT, "_reason", None)
    monkeypatch.setattr(MT, "_turns", set())
    return tasks, conn


async def test_once_task_not_repeated_after_failed_restore(env):
    tasks, conn = env
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    tasks[1] = _task(1, "once", past.isoformat(), past)
    ran = []

    async def on_exec(*a):
        ran.append(a[0])
        MT._reason = "restore"          # /restore begins as the turn ends
    sch = Scheduler(on_exec)
    await sch._check_and_execute()
    assert ran == [1] and conn.writes == [] and tasks[1]["enabled"]
    await sch._check_and_execute()      # gate still up: nothing written
    assert conn.writes == [] and 1 in sch._unrecorded
    MT._reason = None                   # restore failed, gate lifted
    await sch._check_and_execute()
    assert ran == [1] and not tasks[1]["enabled"] and sch._unrecorded == {}


async def test_interval_task_moves_on_after_failed_restore(env):
    tasks, conn = env
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    tasks[2] = _task(2, "interval", "3600", past)
    ran = []

    async def on_exec(*a):
        ran.append(a[0])
        MT._reason = "restore"
    sch = Scheduler(on_exec)
    await sch._check_and_execute()
    MT._reason = None
    await sch._check_and_execute()
    assert ran == [2] and tasks[2]["enabled"]
    assert tasks[2]["next_run"] > datetime.now(timezone.utc) + timedelta(minutes=59)


async def test_cancelled_run_is_retried(env):
    """A run cancelled by begin() did not finish: it runs on a later tick."""
    tasks, conn = env
    past = datetime.now(timezone.utc) - timedelta(seconds=5)
    tasks[3] = _task(3, "once", past.isoformat(), past)
    calls = []

    async def on_exec(*a):
        calls.append(a[0])
    sch = Scheduler(on_exec)
    MT._reason = "restore"
    await sch._check_and_execute()
    assert calls == [] and sch._unrecorded == {}
    MT._reason = None
    await sch._check_and_execute()
    assert calls == [3] and not tasks[3]["enabled"]
