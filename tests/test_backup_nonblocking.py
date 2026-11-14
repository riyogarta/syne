"""/backup and /restore must not block the event loop (1.22.60).

run_backup / run_restore wait on pg_dump|gzip (or psql) synchronously; called
directly inside an async handler they froze every channel for ~70s.
"""

import inspect

from syne.communication.telegram import TelegramChannel


def _src(name):
    return inspect.getsource(getattr(TelegramChannel, name))


def test_backup_runs_in_thread():
    src = _src("_cmd_backup")
    assert "await asyncio.to_thread(run_backup)" in src
    assert "= run_backup()" not in src


def test_restore_runs_in_thread():
    src = inspect.getsource(TelegramChannel)
    assert "await asyncio.to_thread(run_restore, filepath)" in src
    assert "= run_restore(filepath)" not in src
