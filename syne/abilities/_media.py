"""Shared image resolution helpers for document-producing abilities.

Used by `pdf.py` (ReportLab), `office.py` (python-docx / python-pptx) so the
sourcing + security rules for embedded images live in exactly one place.

Supported image sources
-----------------------
1. Workspace-relative or bare filename  — ``meter.jpg``, ``uploads/meter.jpg``
2. HTTP(S) URL                          — ``https://example.com/chart.png``
3. Data URI                             — ``data:image/png;base64,iVBORw0...``

Security model
--------------
- Local paths are ``realpath``-ed and MUST resolve inside ``workspace/``.
  This blocks ``../../etc/passwd`` style traversal from leaking host files
  into a generated document.
- URLs go through the same SSRF validator the ``fetch_url`` tool uses
  (``syne.security.is_url_safe_async``): localhost, private/link-local ranges
  and cloud-metadata endpoints are rejected. Redirects are NOT followed
  automatically; each hop is re-validated.
- Every download / decode is size-capped (``MAX_IMAGE_BYTES``).
- Payloads are magic-byte sniffed; anything that is not a real raster image
  is rejected even if the extension claims otherwise.

Failure is always soft: resolution returns ``(None, reason)`` and the caller
renders a small placeholder instead of aborting the whole document.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
from pathlib import Path

logger = logging.getLogger("syne.ability.media")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKSPACE = _PROJECT_ROOT / "workspace"

# Hard cap for a single embedded image (decoded bytes).
MAX_IMAGE_BYTES = 12 * 1024 * 1024

# Markdown image line:  ![alt](src)  with optional {width=60%} suffix.
_IMAGE_LINE_RE = re.compile(
    r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)"
    r"(?:\s*\{(?P<attrs>[^}]*)\})?$"
)

_WIDTH_RE = re.compile(r"width\s*=\s*([0-9]*\.?[0-9]+)\s*(%|px|in|cm)?", re.I)

_DATA_URI_RE = re.compile(r"^data:image/([a-z0-9.+-]+);base64,", re.I)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_image_line(line: str) -> dict | None:
    """Parse a markdown image line.

    Returns ``{"alt": str, "src": str, "width": (value, unit) | None}``
    or ``None`` when the line is not a standalone image.
    """
    m = _IMAGE_LINE_RE.match((line or "").strip())
    if not m:
        return None

    width = None
    attrs = m.group("attrs") or ""
    wm = _WIDTH_RE.search(attrs)
    if wm:
        try:
            width = (float(wm.group(1)), (wm.group(2) or "px").lower())
        except ValueError:
            width = None

    src = m.group("src").strip()
    if not src:
        return None
    return {"alt": (m.group("alt") or "").strip(), "src": src, "width": width}


def is_image_line(line: str) -> bool:
    """True when the whole line is a standalone markdown image."""
    return parse_image_line(line) is not None


def collect_image_srcs(text: str) -> list[str]:
    """Return every distinct image src referenced by standalone image lines."""
    out: list[str] = []
    seen: set[str] = set()
    for ln in (text or "").splitlines():
        info = parse_image_line(ln)
        if info and info["src"] not in seen:
            seen.add(info["src"])
            out.append(info["src"])
    return out


# ---------------------------------------------------------------------------
# Format sniffing
# ---------------------------------------------------------------------------

def sniff_image_ext(data: bytes) -> str:
    """Return a file extension based on magic bytes, or '' if not an image."""
    if len(data) < 12:
        return ""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return ""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _cache_dir() -> Path:
    d = _WORKSPACE / "temp" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_local(src: str) -> tuple[Path | None, str]:
    """Resolve a local image reference, confined to workspace/."""
    p = Path(src)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(_WORKSPACE / src)
        candidates.append(_PROJECT_ROOT / src)
        if "/" not in src and "\\" not in src:
            candidates.append(_WORKSPACE / "uploads" / src)
            candidates.append(_WORKSPACE / "outputs" / src)
            candidates.append(_WORKSPACE / "temp" / src)

    for c in candidates:
        try:
            rp = c.resolve()
        except OSError:
            continue
        if not rp.is_file():
            continue
        if not _is_under(rp, _WORKSPACE):
            return None, f"path outside workspace is not allowed: {src}"
        return rp, ""

    # Bare filename — search recursively (session_* subdirs live under outputs/).
    if "/" not in src and "\\" not in src:
        for base in (_WORKSPACE / "uploads", _WORKSPACE / "outputs", _WORKSPACE / "temp"):
            if not base.is_dir():
                continue
            try:
                for f in base.rglob(src):
                    if f.is_file() and _is_under(f, _WORKSPACE):
                        return f.resolve(), ""
            except OSError:
                continue

    return None, f"image not found in workspace: {src}"


def _store_bytes(data: bytes, hint: str) -> tuple[Path | None, str]:
    """Validate + persist raw image bytes into the workspace image cache."""
    if not data:
        return None, "empty image payload"
    if len(data) > MAX_IMAGE_BYTES:
        return None, f"image too large ({len(data)} bytes, max {MAX_IMAGE_BYTES})"

    ext = sniff_image_ext(data)
    if not ext:
        return None, "payload is not a recognised image format"

    digest = hashlib.sha256(data).hexdigest()[:16]
    out = _cache_dir() / f"{digest}{ext}"
    if not out.exists():
        try:
            out.write_bytes(data)
        except OSError as e:
            return None, f"cannot cache image: {e}"
    logger.debug("cached image %s from %s", out.name, hint)
    return out, ""


def _resolve_data_uri(src: str) -> tuple[Path | None, str]:
    m = _DATA_URI_RE.match(src)
    if not m:
        return None, "malformed data URI"
    b64 = src[m.end():]
    # Cheap guard before decoding: base64 inflates ~4/3.
    if len(b64) > MAX_IMAGE_BYTES * 4 // 3 + 16:
        return None, "data URI too large"
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        return None, f"invalid base64 in data URI: {e}"
    return _store_bytes(data, "data-uri")


async def _resolve_url(src: str, timeout_s: int = 20) -> tuple[Path | None, str]:
    import httpx

    from syne.security import is_url_safe_async

    ok, reason = await is_url_safe_async(src)
    if not ok:
        return None, f"blocked URL ({reason})"

    current = src
    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout_s,
            headers={"User-Agent": "SyneDocImage/1.0"},
        ) as client:
            for _ in range(4):
                ok, reason = await is_url_safe_async(current)
                if not ok:
                    return None, f"blocked redirect ({reason})"

                r = await client.get(current)
                if r.is_redirect:
                    nxt = r.headers.get("location")
                    if not nxt:
                        return None, "redirect without location header"
                    current = str(httpx.URL(current).join(nxt))
                    continue

                r.raise_for_status()
                data = r.content
                if len(data) > MAX_IMAGE_BYTES:
                    return None, f"image too large ({len(data)} bytes)"
                return _store_bytes(data, current)

            return None, "too many redirects"
    except Exception as e:  # network, TLS, HTTP status
        return None, f"download failed: {type(e).__name__}: {e}"


async def resolve_image_ref(src: str) -> tuple[str | None, str]:
    """Resolve one image reference to a local file path.

    Returns ``(path, "")`` on success or ``(None, reason)`` on failure.
    Never raises.
    """
    src = (src or "").strip()
    if not src:
        return None, "empty image source"

    try:
        if src.lower().startswith("data:"):
            path, err = _resolve_data_uri(src)
        elif src.lower().startswith(("http://", "https://")):
            path, err = await _resolve_url(src)
        elif "://" in src:
            return None, "unsupported URL scheme (only http/https/data)"
        else:
            path, err = _resolve_local(src)
    except Exception as e:  # defensive — resolution must never break a document
        logger.warning("image resolve error for %r: %s", src, e)
        return None, f"resolve error: {type(e).__name__}: {e}"

    if path is None:
        return None, err
    return str(path), ""


async def resolve_image_map(srcs: list[str]) -> dict[str, tuple[str | None, str]]:
    """Resolve many refs concurrently. Returns ``{src: (path|None, reason)}``."""
    import asyncio

    uniq = list(dict.fromkeys(s for s in srcs if s))
    if not uniq:
        return {}
    results = await asyncio.gather(
        *(resolve_image_ref(s) for s in uniq), return_exceptions=True
    )
    out: dict[str, tuple[str | None, str]] = {}
    for src, res in zip(uniq, results):
        if isinstance(res, BaseException):
            out[src] = (None, f"resolve error: {type(res).__name__}: {res}")
        else:
            out[src] = res
    return out


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def image_size_px(path: str) -> tuple[int, int]:
    """Return (width, height) in pixels. Falls back to (0, 0) when unreadable."""
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(path) as im:
            return int(im.width), int(im.height)
    except Exception as e:
        logger.debug("cannot read image size for %s: %s", path, e)
        return 0, 0


def fit_box(
    path: str,
    max_w: float,
    max_h: float,
    want: tuple[float, str] | None = None,
) -> tuple[float, float]:
    """Compute a draw size that preserves aspect ratio and fits (max_w, max_h).

    ``max_w`` / ``max_h`` are in caller units (points for ReportLab, inches for
    Office). ``want`` is the parsed ``{width=...}`` attribute; ``%`` is taken as
    a fraction of ``max_w``, other units are ignored in favour of the natural
    size (the caller's unit system is not known here).
    """
    px_w, px_h = image_size_px(path)
    if px_w <= 0 or px_h <= 0:
        return max_w, max_h

    aspect = px_h / px_w

    target_w = max_w
    if want:
        value, unit = want
        if unit == "%" and value > 0:
            target_w = max_w * min(value, 100.0) / 100.0

    target_w = min(target_w, max_w)
    target_h = target_w * aspect
    if target_h > max_h:
        target_h = max_h
        target_w = target_h / aspect
    return target_w, target_h
