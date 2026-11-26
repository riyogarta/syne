"""v1.23.49: database backups use bzip2 -9 (.sql.bz2).

Older .sql.gz and plain .sql backups stay listable and restorable.
"""
import bz2
import gzip
import os
import subprocess

import pytest

import syne.cli.cmd_backup as CB


def test_new_backups_are_bz2():
    assert CB.BACKUP_EXT == ".sql.bz2"
    assert CB._COMPRESS_CMD == ["bzip2", "-9"]


def test_decompress_cmd_by_extension():
    assert CB._decompress_cmd("a.sql.bz2") == ["bunzip2", "-c", "a.sql.bz2"]
    assert CB._decompress_cmd("a.sql.gz") == ["gunzip", "-c", "a.sql.gz"]
    assert CB._decompress_cmd("a.sql") is None


def test_list_backups_includes_all_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(CB, "_get_backup_dir", lambda: str(tmp_path))
    for name in ("a.sql.bz2", "b.sql.gz", "c.sql", "d.txt", "e.bz2"):
        (tmp_path / name).write_bytes(b"x")
    names = sorted(os.path.basename(p) for p, _, _ in CB._list_backups())
    assert names == ["a.sql.bz2", "b.sql.gz", "c.sql"]


@pytest.mark.skipif(not all(os.path.exists(f"/usr/bin/{b}") for b in ("bzip2", "bunzip2")),
                    reason="bzip2 not installed")
def test_compress_and_decompress_round_trip(tmp_path):
    sql = b"CREATE TABLE t (v vector);\n" + b"[0.0123,-0.0456]\n" * 1000
    out = tmp_path / "x.sql.bz2"
    with open(out, "wb") as f:
        subprocess.run(CB._COMPRESS_CMD, input=sql, stdout=f, check=True)
    assert bz2.decompress(out.read_bytes()) == sql
    back = subprocess.run(CB._decompress_cmd(str(out)), capture_output=True, check=True)
    assert back.stdout == sql


def test_old_gz_still_decompressed(tmp_path):
    p = tmp_path / "old.sql.gz"
    p.write_bytes(gzip.compress(b"SELECT 1;\n"))
    back = subprocess.run(CB._decompress_cmd(str(p)), capture_output=True, check=True)
    assert back.stdout == b"SELECT 1;\n"


class _P:
    """Fake Popen: pg_dump writes nothing, compressor exits with a code."""
    def __init__(self, rc, err=b""):
        self.rc, self.err = rc, err

    def __call__(self, cmd, stdin=None, stdout=None, stderr=None):
        rc, err = self.rc.get(cmd[0], 0), self.err
        return type("Proc", (), {
            "stdout": type("S", (), {"close": lambda s: None})(),
            "stderr": type("E", (), {"read": lambda s: err})(),
            "returncode": rc,
            "wait": lambda s: rc,
        })()


def test_dump_reports_compressor_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(CB.subprocess, "Popen", _P({"bzip2": 2}, b"disk full"))
    ok, err = CB._dump_to(str(tmp_path / "o.sql.bz2"), "u", "d")
    assert not ok and "bzip2 failed" in err and "disk full" in err


def test_dump_reports_pg_dump_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(CB.subprocess, "Popen", _P({"docker": 1}, b"no db"))
    ok, err = CB._dump_to(str(tmp_path / "o.sql.bz2"), "u", "d")
    assert not ok and err.startswith("pg_dump failed")
