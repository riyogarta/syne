"""Office documents ability — create Word, Excel, PowerPoint files.

Actions:
- create_docx: Microsoft Word (.docx) from markdown-ish text
- create_xlsx: Microsoft Excel (.xlsx) from structured sheets data
- create_pptx: Microsoft PowerPoint (.pptx) from slide outline

Dependencies (lazy-installed via ensure_dependencies):
- python-docx (Word)
- openpyxl (Excel)
- python-pptx (PowerPoint)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re
import sys
import time

from syne.abilities.base import Ability

logger = logging.getLogger("syne.ability.office")


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str, default: str = "document") -> str:
    name = (name or "").strip() or default
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120] or default


def _ensure_ext(name: str, ext: str) -> str:
    n = (name or "").strip() or "output"
    ext = ext if ext.startswith(".") else f".{ext}"
    while n.lower().endswith(ext.lower()):
        n = n[: -len(ext)]
    n = n.rstrip(". ")
    return (n or "output") + ext


class OfficeAbility(Ability):
    name = "office"
    description = "Create Microsoft Office documents (Word, Excel, PowerPoint)."
    version = "1.0"
    permission = 0o770
    priority = False  # tool-call style, not pre-processing

    _DEPS = {
        "python-docx": "docx",
        "openpyxl": "openpyxl",
        "python-pptx": "pptx",
    }

    async def ensure_dependencies(self) -> tuple[bool, str]:
        """Install Office deps (python-docx, openpyxl, python-pptx)."""
        missing_pkgs = []
        for pip_name, import_name in self._DEPS.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_pkgs.append(pip_name)

        if not missing_pkgs:
            return True, ""

        logger.info(f"Installing Office deps: {', '.join(missing_pkgs)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", *missing_pkgs,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            return False, f"Install timed out. Manual: pip install {' '.join(missing_pkgs)}"
        if proc.returncode != 0:
            err = stderr.decode().strip()
            return False, f"Failed to install Office deps: {err}"
        return True, f"Installed: {', '.join(missing_pkgs)}"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "office",
                "description": (
                    "Create Microsoft Office documents. Actions: "
                    "create_docx (Word), create_xlsx (Excel), create_pptx (PowerPoint)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create_docx", "create_xlsx", "create_pptx"],
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title (used as filename if out_name not given).",
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "For create_docx: body text. Markdown-style: "
                                "# H1, ## H2, ### H3, - bullets, **bold**, *italic*. "
                                "Paragraphs separated by blank line."
                            ),
                        },
                        "sheets": {
                            "type": "string",
                            "description": (
                                "For create_xlsx: JSON array of sheets. Each sheet: "
                                '{"name": "Sheet1", "headers": ["A","B"], "rows": [[1,2],[3,4]]}. '
                                "Headers optional. Pass as JSON string."
                            ),
                        },
                        "slides": {
                            "type": "string",
                            "description": (
                                "For create_pptx: JSON array of slides. Each slide: "
                                '{"title": "Slide title", "bullets": ["point 1","point 2"]} '
                                'OR {"title": "...", "content": "free text"}. '
                                "First slide auto-styled as title slide. Pass as JSON string."
                            ),
                        },
                        "out_name": {"type": "string", "description": "Optional output filename."},
                    },
                    "required": ["action"],
                },
            },
        }

    def get_guide(self, enabled: bool, config: dict) -> str:
        if not enabled:
            return (
                "- Status: **disabled**\n"
                "- Enable: `update_ability(action='enable', name='office')`"
            )

        missing = []
        for mod in self._DEPS.values():
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)

        if missing:
            return (
                f"- Status: **not ready** (missing deps: {', '.join(missing)})\n"
                "- Will auto-install on first use, or run `pip install python-docx openpyxl python-pptx`"
            )

        return (
            "- Status: **ready**\n"
            "- Word: `office(action='create_docx', title='...', content='# Heading\\nText')`\n"
            "- Excel: `office(action='create_xlsx', title='...', sheets='[{\"name\":\"S1\",\"headers\":[\"A\"],\"rows\":[[1]]}]')`\n"
            "- PowerPoint: `office(action='create_pptx', title='...', slides='[{\"title\":\"S1\",\"bullets\":[\"p1\"]}]')`"
        )

    async def execute(self, params: dict, context: dict) -> dict:
        action = (params.get("action") or "").strip()
        title = params.get("title") or ""
        out_name = params.get("out_name") or ""

        session_id = str(context.get("session_id") or "")
        out_dir = self.get_output_dir(session_id=session_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            if action == "create_docx":
                content = params.get("content") or ""
                name = _ensure_ext(_safe_filename(out_name or title or f"document_{_now_ts()}"), ".docx")
                out_path = os.path.join(out_dir, name)
                _make_docx(out_path, title, content)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            if action == "create_xlsx":
                sheets_str = params.get("sheets") or ""
                if not sheets_str:
                    raise ValueError("sheets is required (JSON array)")
                try:
                    sheets = _json.loads(sheets_str) if isinstance(sheets_str, str) else sheets_str
                except _json.JSONDecodeError as e:
                    raise ValueError(f"Invalid sheets JSON: {e}")
                if not isinstance(sheets, list) or not sheets:
                    raise ValueError("sheets must be a non-empty JSON array")
                name = _ensure_ext(_safe_filename(out_name or title or f"workbook_{_now_ts()}"), ".xlsx")
                out_path = os.path.join(out_dir, name)
                _make_xlsx(out_path, sheets)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            if action == "create_pptx":
                slides_str = params.get("slides") or ""
                if not slides_str:
                    raise ValueError("slides is required (JSON array)")
                try:
                    slides = _json.loads(slides_str) if isinstance(slides_str, str) else slides_str
                except _json.JSONDecodeError as e:
                    raise ValueError(f"Invalid slides JSON: {e}")
                if not isinstance(slides, list) or not slides:
                    raise ValueError("slides must be a non-empty JSON array")
                name = _ensure_ext(_safe_filename(out_name or title or f"slides_{_now_ts()}"), ".pptx")
                out_path = os.path.join(out_dir, name)
                _make_pptx(out_path, title, slides)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────


def _add_inline_runs(paragraph, text: str) -> None:
    """Add text to a docx paragraph honoring **bold** and *italic* markers."""
    # Split on bold first, then italic within each chunk
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
            continue
        # Italic within non-bold chunk
        sub = re.split(r"(\*[^*]+\*)", part)
        for s in sub:
            if not s:
                continue
            if s.startswith("*") and s.endswith("*") and len(s) > 2:
                run = paragraph.add_run(s[1:-1])
                run.italic = True
            else:
                paragraph.add_run(s)


def _make_docx(out_path: str, title: str, content: str) -> None:
    from docx import Document

    doc = Document()
    if title:
        doc.add_heading(title, level=0)

    for block in (content or "").split("\n\n"):
        block = block.strip()
        if not block:
            continue

        # Multi-line bullet list?
        lines = block.splitlines()
        if all(re.match(r"^[-*]\s+", ln.strip()) for ln in lines if ln.strip()):
            for ln in lines:
                ln = ln.strip().lstrip("-*").strip()
                if ln:
                    p = doc.add_paragraph(style="List Bullet")
                    _add_inline_runs(p, ln)
            continue

        # Heading?
        m = re.match(r"^(#{1,6})\s+(.*)$", block)
        if m:
            level = min(len(m.group(1)), 4)
            doc.add_heading(m.group(2).strip(), level=level)
            continue

        # Regular paragraph (preserve internal newlines as soft breaks)
        p = doc.add_paragraph()
        first = True
        for ln in block.split("\n"):
            if not first:
                p.add_run().add_break()
            _add_inline_runs(p, ln)
            first = False

    doc.save(out_path)


def _make_xlsx(out_path: str, sheets: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    # openpyxl creates one default sheet; remove it and re-add named sheets
    default = wb.active
    wb.remove(default)

    bold = Font(bold=True)

    for i, sheet in enumerate(sheets):
        name = (sheet.get("name") or f"Sheet{i + 1}")[:31]  # Excel limit
        ws = wb.create_sheet(title=name)

        headers = sheet.get("headers") or []
        rows = sheet.get("rows") or []

        if headers:
            ws.append([str(h) for h in headers])
            for col_idx in range(1, len(headers) + 1):
                ws.cell(row=1, column=col_idx).font = bold

        for row in rows:
            if not isinstance(row, list):
                row = [row]
            ws.append(row)

        # Auto-size columns (approximate)
        max_cols = max(
            (len(headers) if headers else 0),
            max((len(r) for r in rows if isinstance(r, list)), default=0),
        )
        for col_idx in range(1, max_cols + 1):
            col_letter = ws.cell(row=1, column=col_idx).column_letter
            max_len = 10
            for cell in ws[col_letter]:
                v = cell.value
                if v is not None:
                    max_len = max(max_len, min(len(str(v)) + 2, 50))
            ws.column_dimensions[col_letter].width = max_len

    # If no sheets were added (defensive), add an empty one
    if not wb.sheetnames:
        wb.create_sheet(title="Sheet1")

    wb.save(out_path)


def _make_pptx(out_path: str, title: str, slides: list[dict]) -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    prs = Presentation()
    # Use 16:9 default

    # First slide: title slide if title is given OR use first item
    title_layout = prs.slide_layouts[0]  # Title slide
    bullet_layout = prs.slide_layouts[1]  # Title + content

    for i, slide_data in enumerate(slides):
        s_title = (slide_data.get("title") or "").strip()
        bullets = slide_data.get("bullets") or []
        free_content = (slide_data.get("content") or "").strip()

        if i == 0 and title and not s_title:
            # Use document title as first slide
            slide = prs.slides.add_slide(title_layout)
            slide.shapes.title.text = title
            if free_content or bullets:
                # Subtitle from free content or first bullet
                subtitle = slide.placeholders[1]
                subtitle.text = free_content or " ".join(str(b) for b in bullets[:3])
            continue

        # Regular content slide
        slide = prs.slides.add_slide(bullet_layout)
        slide.shapes.title.text = s_title or f"Slide {i + 1}"

        body = slide.placeholders[1]
        tf = body.text_frame
        tf.word_wrap = True

        if bullets:
            for j, b in enumerate(bullets):
                if j == 0:
                    p = tf.paragraphs[0]
                else:
                    p = tf.add_paragraph()
                p.text = str(b)
                p.level = 0
                for run in p.runs:
                    run.font.size = Pt(18)
        elif free_content:
            tf.text = free_content
            for p in tf.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(18)

    prs.save(out_path)
