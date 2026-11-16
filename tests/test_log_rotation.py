"""syne.log rotation (Riyo, 16 Nov 2026): 50 MB active file, 5 gzip backups,
old plain backups gzipped once under the same number (syne/log_rotation.py)."""

import gzip
import logging
import os
from pathlib import Path

from syne import log_rotation as LR


def _logger(handler, name):
    lg = logging.getLogger(name)
    lg.handlers[:] = [handler]
    lg.propagate = False
    lg.setLevel(logging.INFO)
    return lg


def _read_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return f.read()


def test_defaults_50mb_5_backups():
    assert LR.MAX_BYTES == 50 * 1024 * 1024
    assert LR.BACKUP_COUNT == 5


def test_rotation_writes_gz_backups_and_drops_the_oldest(tmp_path):
    base = str(tmp_path / "syne.log")
    h = LR.build_file_handler(base, max_bytes=300, backup_count=2)
    lg = _logger(h, "t_rot_drop")
    try:
        for i in range(40):
            lg.info("line %03d %s", i, "x" * 40)
    finally:
        h.close()
    names = sorted(os.listdir(tmp_path))
    assert names == ["syne.log", "syne.log.1.gz", "syne.log.2.gz"]
    newer, older = _read_gz(base + ".1.gz"), _read_gz(base + ".2.gz")
    assert "line" in newer and "line" in older
    # .1 holds later lines than .2
    first = lambda s: int(s.split("line ")[1][:3])
    assert first(newer) > first(older)


def test_legacy_plain_backups_gzipped_under_same_number(tmp_path):
    base = tmp_path / "syne.log"
    for i in (1, 2, 3):
        p = Path(f"{base}.{i}")
        p.write_text(f"old backup {i}\n" * 50, encoding="utf-8")
        os.utime(p, (1_000_000 + i, 1_000_000 + i))
    made = LR.compress_legacy_backups(str(base), 5)
    assert made == [f"{base}.{i}.gz" for i in (1, 2, 3)]
    for i in (1, 2, 3):
        assert not Path(f"{base}.{i}").exists()
        gz = f"{base}.{i}.gz"
        assert _read_gz(gz).startswith(f"old backup {i}")
        assert int(os.stat(gz).st_mtime) == 1_000_000 + i
    assert not list(tmp_path.glob("*.tmp"))
    assert LR.compress_legacy_backups(str(base), 5) == []


def test_legacy_backups_then_rotate_out_in_order(tmp_path):
    base = str(tmp_path / "syne.log")
    for i in (1, 2):
        Path(f"{base}.{i}").write_text(f"legacy {i}\n", encoding="utf-8")
    h = LR.build_file_handler(base, max_bytes=200, backup_count=2)
    lg = _logger(h, "t_rot_legacy")
    try:
        lg.info("y" * 150)    # fits in the empty active file
        lg.info("z" * 150)    # one rollover: legacy 1 -> .2.gz, legacy 2 dropped
    finally:
        h.close()
    assert _read_gz(base + ".2.gz") == "legacy 1\n"
    assert "y" * 150 in _read_gz(base + ".1.gz")
    assert "z" * 150 in Path(base).read_text(encoding="utf-8")


def test_existing_gz_is_not_overwritten(tmp_path):
    base = str(tmp_path / "syne.log")
    Path(base + ".1").write_text("plain\n", encoding="utf-8")
    with gzip.open(base + ".1.gz", "wt", encoding="utf-8") as f:
        f.write("already\n")
    assert LR.compress_legacy_backups(base, 5) == []
    assert _read_gz(base + ".1.gz") == "already\n"
    assert Path(base + ".1").exists()


def test_gzip_failure_keeps_the_log_as_plain_file(tmp_path, monkeypatch):
    src = tmp_path / "syne.log"
    src.write_text("keep me\n", encoding="utf-8")
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(LR, "_gzip_file", boom)
    LR.gz_rotator(str(src), str(tmp_path / "syne.log.1.gz"))
    assert (tmp_path / "syne.log.1").read_text(encoding="utf-8") == "keep me\n"
    assert not src.exists()


def test_main_uses_gz_handler():
    src = (Path(LR.__file__).parent / "main.py").read_text()
    assert "build_file_handler(_log_file)" in src
    assert "RotatingFileHandler(" not in src
