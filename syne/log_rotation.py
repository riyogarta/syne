"""syne.log rotation: 50 MB per file, 5 gzip backups (Riyo, 16 Nov 2026).

Active file: ~/.log-syne/syne.log (plain text, written live).
Backups:     syne.log.1.gz (newest) .. syne.log.5.gz (oldest). On the next
             rotation the oldest is deleted and the rest shift up by one.

RotatingFileHandler still does the shifting; only its namer (adds ".gz") and
rotator (gzip instead of a plain rename) are replaced, the hook documented in
the Python logging cookbook. Backups are read with zcat / zgrep.

Plain backups left by the old 5 MB x 3 setup (syne.log.1 .. .3) are gzipped
once at start under the same number (syne.log.1 -> syne.log.1.gz), so they
take part in the normal rotation and drop off when their turn comes.
"""

import gzip
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler

MAX_BYTES = 50 * 1024 * 1024   # 50 MB active file (~18 days at ~2.7 MB/day)
BACKUP_COUNT = 5               # syne.log.1.gz .. syne.log.5.gz
COMPRESS_LEVEL = 6             # ~9x smaller on syne.log, ~0.1 s per 5 MB


def gz_namer(name: str) -> str:
    """Backup file name: syne.log.N -> syne.log.N.gz."""
    return name + ".gz"


def _gzip_file(source: str, dest: str) -> None:
    """Gzip source into dest (via a temp file), keep its mtime, remove source."""
    tmp = dest + ".tmp"
    try:
        with open(source, "rb") as f_in, gzip.open(
            tmp, "wb", compresslevel=COMPRESS_LEVEL
        ) as f_out:
            shutil.copyfileobj(f_in, f_out, 1024 * 1024)
        st = os.stat(source)
        os.utime(tmp, (st.st_atime, st.st_mtime))
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.remove(source)


def gz_rotator(source: str, dest: str) -> None:
    """Rotator for RotatingFileHandler: gzip the full log into the .gz backup.

    If gzip fails (e.g. disk full) the log is not lost: it is renamed to the
    plain name (dest without .gz), and compress_legacy_backups() gzips it on
    the next start.
    """
    try:
        _gzip_file(source, dest)
    except Exception as e:
        plain = dest[:-3] if dest.endswith(".gz") else dest
        try:
            os.replace(source, plain)
        except OSError:
            pass
        print(f"syne log rotation: gzip failed ({e}); kept plain {plain}", file=sys.stderr)


def compress_legacy_backups(base_filename: str, backup_count: int = BACKUP_COUNT) -> list:
    """Gzip plain backups syne.log.1 .. .N under the same number. Idempotent.

    A number that already has a .gz is left alone. Returns the .gz paths made.
    """
    made = []
    for i in range(1, backup_count + 1):
        plain = f"{base_filename}.{i}"
        gz = plain + ".gz"
        if os.path.isfile(plain) and not os.path.exists(gz):
            try:
                _gzip_file(plain, gz)
                made.append(gz)
            except Exception as e:
                print(f"syne log rotation: could not gzip {plain} ({e})", file=sys.stderr)
    return made


def build_file_handler(
    path: str, max_bytes: int = MAX_BYTES, backup_count: int = BACKUP_COUNT,
) -> RotatingFileHandler:
    """RotatingFileHandler for syne.log with gzip backups."""
    compress_legacy_backups(path, backup_count)
    handler = RotatingFileHandler(
        path, encoding="utf-8", maxBytes=max_bytes, backupCount=backup_count,
    )
    handler.namer = gz_namer
    handler.rotator = gz_rotator
    return handler
