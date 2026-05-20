"""Office documents ability — create and read Word, Excel, PowerPoint files.

Create actions:
- create_docx: Microsoft Word (.docx) from markdown-ish text
- create_xlsx: Microsoft Excel (.xlsx) from structured sheets data
- create_pptx: Microsoft PowerPoint (.pptx) from slide outline

Read actions:
- read_docx: Extract text from .docx (base64 or path)
- read_xlsx: Extract sheets/rows from .xlsx (base64 or path)
- read_pptx: Extract slides text from .pptx (base64 or path)

Also auto-extracts uploaded docx/xlsx/pptx via pre_process — LLM sees
the document content as plain text without needing to call any tool.

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
    description = "Create and read Microsoft Office documents (Word, Excel, PowerPoint)."
    version = "2.0"
    permission = 0o770
    # Priority pre-processing: auto-extract content from uploaded .docx/.xlsx/.pptx
    priority = True

    _DEPS = {
        "python-docx": "docx",
        "openpyxl": "openpyxl",
        "python-pptx": "pptx",
    }

    # MIME mapping for pre_process
    _DOCX_MIMES = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }
    _XLSX_MIMES = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    }
    _PPTX_MIMES = {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    }

    # Read limits — truncate to keep LLM context manageable
    _MAX_EXTRACTED_CHARS = 50_000
    _MAX_XLSX_ROWS_PER_SHEET = 200
    _MAX_XLSX_COLS_PER_SHEET = 50

    def handles_input_type(self, input_type: str) -> bool:
        return input_type == "document"

    async def pre_process(
        self, input_type: str, input_data: dict, user_prompt: str,
        config: dict | None = None,
    ) -> str | None:
        """Auto-extract text from uploaded Office docs. Returns None for other types
        (falls through to next handler, e.g. pdf ability)."""
        mime = (input_data.get("mime_type") or "").lower()
        filename = (input_data.get("filename") or "").lower()

        kind = None
        if mime in self._DOCX_MIMES or filename.endswith(".docx") or filename.endswith(".doc"):
            kind = "docx"
        elif mime in self._XLSX_MIMES or filename.endswith(".xlsx") or filename.endswith(".xls"):
            kind = "xlsx"
        elif mime in self._PPTX_MIMES or filename.endswith(".pptx") or filename.endswith(".ppt"):
            kind = "pptx"
        else:
            return None  # not an Office doc — let other abilities handle

        b64 = input_data.get("base64", "")
        path = input_data.get("path", "")
        if not b64 and not path:
            return None

        try:
            import base64
            if b64:
                content = base64.b64decode(b64)
            else:
                with open(path, "rb") as f:
                    content = f.read()
        except Exception as e:
            logger.warning(f"Office pre_process: read failed: {e}")
            return None

        try:
            if kind == "docx":
                text, meta = _read_docx_bytes(content)
                header = f"Word: {filename or 'document.docx'} ({meta['paragraphs']} paragraphs, {len(text)} chars)"
            elif kind == "xlsx":
                text, meta = _read_xlsx_bytes(
                    content,
                    max_rows=self._MAX_XLSX_ROWS_PER_SHEET,
                    max_cols=self._MAX_XLSX_COLS_PER_SHEET,
                )
                header = f"Excel: {filename or 'workbook.xlsx'} ({meta['sheets']} sheet(s), {len(text)} chars)"
            else:  # pptx
                text, meta = _read_pptx_bytes(content)
                header = f"PowerPoint: {filename or 'slides.pptx'} ({meta['slides']} slide(s), {len(text)} chars)"
        except Exception as e:
            logger.warning(f"Office pre_process: extraction failed ({kind}): {e}")
            return None

        if len(text) > self._MAX_EXTRACTED_CHARS:
            text = text[: self._MAX_EXTRACTED_CHARS] + f"\n\n[... truncated at {self._MAX_EXTRACTED_CHARS} chars]"

        return f"{header}\n\n{text}" if text.strip() else f"{header}\n\n[empty document]"

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
                    "Create or read Microsoft Office documents. Create actions: "
                    "create_docx (Word), create_xlsx (Excel), create_pptx (PowerPoint). "
                    "Read actions: read_docx, read_xlsx, read_pptx (from base64). "
                    "Uploaded Office files are auto-extracted via pre_process — no tool call needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "create_docx", "create_xlsx", "create_pptx",
                                "read_docx", "read_xlsx", "read_pptx",
                            ],
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
                        "file_base64": {
                            "type": "string",
                            "description": "For read_* actions: base64-encoded file bytes.",
                        },
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
            "- Auto-extracts content from uploaded .docx/.xlsx/.pptx (priority pre-process)\n"
            "- Create Word: `office(action='create_docx', title='...', content='# Heading\\nText')`\n"
            "- Create Excel: `office(action='create_xlsx', title='...', sheets='[{\"name\":\"S1\",\"headers\":[\"A\"],\"rows\":[[1]]}]')`\n"
            "- Create PPT: `office(action='create_pptx', title='...', slides='[{\"title\":\"S1\",\"bullets\":[\"p1\"]}]')`\n"
            "- Read base64: `office(action='read_docx'|'read_xlsx'|'read_pptx', file_base64='...')`"
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

            if action in ("read_docx", "read_xlsx", "read_pptx"):
                b64 = params.get("file_base64") or ""
                if not b64:
                    raise ValueError("file_base64 is required")
                import base64 as _b64
                content = _b64.b64decode(b64)
                if action == "read_docx":
                    text, meta = _read_docx_bytes(content)
                elif action == "read_xlsx":
                    text, meta = _read_xlsx_bytes(
                        content,
                        max_rows=self._MAX_XLSX_ROWS_PER_SHEET,
                        max_cols=self._MAX_XLSX_COLS_PER_SHEET,
                    )
                else:  # read_pptx
                    text, meta = _read_pptx_bytes(content)
                if len(text) > self._MAX_EXTRACTED_CHARS:
                    text = text[: self._MAX_EXTRACTED_CHARS]
                    meta["truncated"] = True
                return {"success": True, "result": {**meta, "text": text}}

            raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────────


def _read_docx_bytes(content: bytes) -> tuple[str, dict]:
    """Extract text from .docx bytes. Returns (text, meta)."""
    import io
    from docx import Document

    doc = Document(io.BytesIO(content))
    parts = []
    para_count = 0
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        # Convert heading style to markdown-ish
        if "heading" in style:
            level_match = re.search(r"(\d+)", style)
            level = int(level_match.group(1)) if level_match else 1
            parts.append(("#" * min(level, 6)) + " " + t)
        else:
            parts.append(t)
        para_count += 1

    # Also extract tables
    table_count = 0
    for tbl in doc.tables:
        table_count += 1
        parts.append(f"\n[Table {table_count}]")
        for row in tbl.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            parts.append(" | ".join(cells))

    text = "\n\n".join(parts)
    return text, {"paragraphs": para_count, "tables": table_count}


def _read_xlsx_bytes(content: bytes, max_rows: int = 200, max_cols: int = 50) -> tuple[str, dict]:
    """Extract sheets/rows from .xlsx bytes. Returns (text, meta).

    Each sheet is rendered as Markdown-style table preview.
    """
    import io
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    parts = []
    sheet_count = 0
    total_rows = 0
    for ws in wb.worksheets:
        sheet_count += 1
        parts.append(f"--- Sheet: {ws.title} ---")
        rows_emitted = 0
        truncated_rows = False
        truncated_cols = False
        for row in ws.iter_rows(values_only=True):
            if rows_emitted >= max_rows:
                truncated_rows = True
                break
            cells = list(row)
            if len(cells) > max_cols:
                cells = cells[:max_cols]
                truncated_cols = True
            # Skip fully empty rows
            if all(c is None or (isinstance(c, str) and not c.strip()) for c in cells):
                continue
            parts.append(" | ".join("" if c is None else str(c) for c in cells))
            rows_emitted += 1
            total_rows += 1
        if truncated_rows:
            parts.append(f"[... row limit {max_rows} reached]")
        if truncated_cols:
            parts.append(f"[... col limit {max_cols} applied]")
        parts.append("")  # blank line between sheets

    wb.close()
    text = "\n".join(parts).rstrip()
    return text, {"sheets": sheet_count, "rows": total_rows}


def _read_pptx_bytes(content: bytes) -> tuple[str, dict]:
    """Extract slides text from .pptx bytes. Returns (text, meta)."""
    import io
    from pptx import Presentation

    prs = Presentation(io.BytesIO(content))
    parts = []
    slide_count = 0
    for slide in prs.slides:
        slide_count += 1
        slide_lines = [f"--- Slide {slide_count} ---"]
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                t = "".join(run.text for run in para.runs).strip()
                if t:
                    slide_lines.append(t)
        parts.append("\n".join(slide_lines))

    text = "\n\n".join(parts)
    return text, {"slides": slide_count}


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
    from pptx.util import Pt

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
            if slide.shapes.title:
                slide.shapes.title.text = title
            if free_content or bullets:
                # Subtitle from free content or first bullet
                try:
                    subtitle = slide.placeholders[1]
                    subtitle.text = free_content or " ".join(str(b) for b in bullets[:3])
                except (KeyError, IndexError):
                    pass  # layout has no subtitle placeholder
            continue

        # Regular content slide
        slide = prs.slides.add_slide(bullet_layout)
        if slide.shapes.title:
            slide.shapes.title.text = s_title or f"Slide {i + 1}"

        try:
            body = slide.placeholders[1]
        except (KeyError, IndexError):
            # Fallback: add a textbox if layout has no body placeholder
            from pptx.util import Inches as _In
            body = slide.shapes.add_textbox(_In(0.5), _In(1.5), _In(9), _In(5))
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
