"""PDF ability for Syne.

Provides:
- make_from_text: Generate a readable PDF from plain text/markdown-ish text.
- make_from_url: Download URL (PDF or HTML) and produce a PDF.
- read_from_url: Download a PDF and extract text.
- make_comparison: Generate a landscape PDF containing a real table (Platypus Table)
  for a Syne vs OpenClaw plus/minus comparison (option A).

Notes:
- Designed to be loadable by Syne dynamic ability loader: module must expose a
  public Ability subclass.
- Output files are written to self.get_output_dir().
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

from syne.abilities.base import Ability

logger = logging.getLogger("syne.ability.pdf")


# -----------------------------
# Helpers
# -----------------------------

def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str, default: str = "document") -> str:
    name = (name or "").strip() or default
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120] or default


def _ensure_pdf_ext(name: str) -> str:
    """Ensure a filename ends with exactly one .pdf extension."""
    n = (name or "").strip() or "output"
    # Remove one or more trailing .pdf (case-insensitive)
    while n.lower().endswith(".pdf"):
        n = n[:-4]
    n = n.rstrip(". ")
    return (n or "output") + ".pdf"


def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _download(
    url: str,
    timeout_s: int = 20,
    verify_ssl: bool = True,
    max_bytes: int = 25 * 1024 * 1024,
) -> tuple[bytes, str]:
    if not _is_http_url(url):
        raise ValueError("Only http(s) URLs are allowed")

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout_s,
        verify=verify_ssl,
        headers={"User-Agent": "SynePDF/1.0"},
    ) as client:
        r = client.get(url)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        content = r.content

    if len(content) > max_bytes:
        raise ValueError(f"Download too large: {len(content)} bytes (max {max_bytes})")

    return content, ctype


def _extract_readable_text_from_html(html_bytes: bytes) -> str:
    # Lazy imports (keep module import light)
    from readability import Document
    from bs4 import BeautifulSoup

    html = html_bytes.decode("utf-8", errors="replace")
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "lxml")
    text = soup.get_text("\n")
    # Clean up excessive whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _md_inline_to_rl(text: str) -> str:
    """Convert markdown inline syntax to ReportLab Paragraph mini-HTML.

    Supports: `code`, **bold**, *italic*, _italic_, ~~strike~~, [text](url).
    HTML-special chars are escaped so raw < > & in user text don't break
    ReportLab's mini-HTML parser.
    """
    # Stash code spans BEFORE escaping so their content stays verbatim
    # and they're not re-interpreted by the bold/italic regexes.
    code_spans: list[str] = []

    def _stash_code(m: re.Match) -> str:
        code_spans.append(m.group(1))
        return f"\x00C{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`\n]+)`", _stash_code, text)

    # Escape HTML special chars on the non-code text
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Links — [text](url). URL is already escaped by the step above.
    text = re.sub(
        r"\[([^\]\n]+)\]\(([^)\n\s]+)\)",
        lambda m: f'<link href="{m.group(2)}" color="#1a73e8"><u>{m.group(1)}</u></link>',
        text,
    )
    # Bold **text** — apply before single-asterisk italic
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", text)
    # Strikethrough ~~text~~
    text = re.sub(r"~~([^~\n]+)~~", r"<strike>\1</strike>", text)
    # Italic *text* and _text_ — guard against word-internal _ or *
    text = re.sub(r"(?<![\*\w])\*([^*\n]+?)\*(?![\*\w])", r"<i>\1</i>", text)
    text = re.sub(r"(?<![_\w])_([^_\n]+?)_(?![_\w])", r"<i>\1</i>", text)

    # Restore code spans (escape their content too, but no markdown parsing)
    def _restore_code(m: re.Match) -> str:
        idx = int(m.group(1))
        raw = code_spans[idx].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<font face="Courier" backColor="#f3f3f3">&nbsp;{raw}&nbsp;</font>'

    text = re.sub(r"\x00C(\d+)\x00", _restore_code, text)
    return text


def _md_is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _md_is_table_separator(line: str) -> bool:
    s = line.strip()
    if "|" not in s:
        return False
    core = s.strip("|")
    cells = [c.strip() for c in core.split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{1,}:?", c or "") for c in cells)


def _split_table_row(line: str) -> list:
    s = line.strip()
    # handle escaped pipes \| -> placeholder
    s = s.replace("\\|", "\x00")
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip().replace("\x00", "|") for c in s.split("|")]


def _parse_pdf_table_alignments(separator_line: str) -> list[str]:
    """Return per-column alignment ('LEFT'|'CENTER'|'RIGHT') for the separator row."""
    s = (separator_line or "").strip().strip("|")
    out: list[str] = []
    for c in s.split("|"):
        c = c.strip()
        if c.startswith(":") and c.endswith(":"):
            out.append("CENTER")
        elif c.endswith(":"):
            out.append("RIGHT")
        else:
            out.append("LEFT")
    return out


def _make_md_table(header: list, rows: list, styles: dict, separator_line: str = ""):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm

    base = styles["body"]
    ncols = max(1, len(header))
    aligns = _parse_pdf_table_alignments(separator_line)
    while len(aligns) < ncols:
        aligns.append("LEFT")

    # Per-column cell styles so the inner Paragraph aligns its own text — needed
    # because ReportLab's TableStyle ALIGN only positions atom cells, not the
    # wrapped flowables we use here.
    cell_styles = [
        ParagraphStyle(
            f"TCell{i}", parent=base, fontSize=9, leading=12, spaceAfter=0,
            alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[aligns[i]],
        )
        for i in range(ncols)
    ]
    head_styles = [
        ParagraphStyle(
            f"THead{i}", parent=base, fontName="Helvetica-Bold",
            fontSize=9.5, leading=12, spaceAfter=0,
            alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[aligns[i]],
        )
        for i in range(ncols)
    ]

    header_cells = (header + [""] * ncols)[:ncols]
    data = [[Paragraph(_md_inline_to_rl(c), head_styles[i]) for i, c in enumerate(header_cells)]]
    for r in rows:
        cells = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(_md_inline_to_rl(c), cell_styles[i]) for i, c in enumerate(cells)])

    usable = 17 * cm  # A4 portrait minus ~2cm margins each side
    col_w = usable / ncols
    tbl = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Column-level ALIGN as belt-and-suspenders for any non-Paragraph atoms
    for col_idx, align in enumerate(aligns):
        style_cmds.append(("ALIGN", (col_idx, 0), (col_idx, -1), align))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _md_is_block_start(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("#") and re.match(r"^#{1,6}\s+", s):
        return True
    if s.startswith("```"):
        return True
    if re.fullmatch(r"[-_*]{3,}", s):
        return True
    if re.match(r"^[-*]\s+", s):
        return True
    if re.match(r"^\d+\.\s+", s):
        return True
    if s.startswith(">"):
        return True
    if s.startswith("|") and s.count("|") >= 2:
        return True
    return False


def _md_to_rl_story(text: str, styles: dict) -> list:
    """Parse markdown text into ReportLab flowables."""
    from reportlab.platypus import (
        Paragraph,
        Spacer,
        Preformatted,
        ListFlowable,
        ListItem,
    )
    try:
        from reportlab.platypus.flowables import HRFlowable
    except ImportError:  # very old reportlab
        HRFlowable = None  # type: ignore

    story: list = []
    lines = (text or "").splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        s = line.strip()

        # Blank line — paragraph separator, just advance
        if not s:
            i += 1
            continue

        # Fenced code block
        if s.startswith("```"):
            buf: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            if i < n:
                i += 1  # skip closing fence
            story.append(Preformatted("\n".join(buf), styles["code"]))
            story.append(Spacer(1, 4))
            continue

        # Horizontal rule
        if re.fullmatch(r"[-_*]{3,}", s):
            if HRFlowable is not None:
                story.append(HRFlowable(
                    width="100%", thickness=0.5, color="#999999",
                    spaceBefore=4, spaceAfter=6,
                ))
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            level = min(len(m.group(1)), 4)
            style = styles.get(f"h{level}", styles["body"])
            story.append(Paragraph(_md_inline_to_rl(m.group(2).strip()), style))
            i += 1
            continue

        # Blockquote
        if s.startswith(">"):
            quote_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            story.append(Paragraph(
                _md_inline_to_rl(" ".join(ql.strip() for ql in quote_lines)),
                styles["quote"],
            ))
            story.append(Spacer(1, 4))
            continue

        # Table (pipe-style markdown)
        if _md_is_table_row(line) and (i + 1) < n and _md_is_table_separator(lines[i + 1]):
            header = _split_table_row(line)
            separator_line = lines[i + 1]
            i += 2  # skip header + separator
            rows: list = []
            while i < n and _md_is_table_row(lines[i]):
                rows.append(_split_table_row(lines[i]))
                i += 1
            story.append(_make_md_table(header, rows, styles, separator_line))
            story.append(Spacer(1, 6))
            continue

        # Bullet list
        if re.match(r"^[-*]\s+", s):
            items: list[str] = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            list_items = [
                ListItem(Paragraph(_md_inline_to_rl(it.strip()), styles["body"]),
                         leftIndent=14)
                for it in items
            ]
            story.append(ListFlowable(
                list_items, bulletType="bullet", start="•",
                leftIndent=18, bulletFontSize=10,
            ))
            story.append(Spacer(1, 4))
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", s):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            list_items = [
                ListItem(Paragraph(_md_inline_to_rl(it.strip()), styles["body"]),
                         leftIndent=14)
                for it in items
            ]
            story.append(ListFlowable(
                list_items, bulletType="1", leftIndent=18, bulletFontSize=10,
            ))
            story.append(Spacer(1, 4))
            continue

        # Regular paragraph — collect until blank line or next block
        para_lines = [line]
        i += 1
        while i < n and lines[i].strip() and not _md_is_block_start(lines[i]):
            para_lines.append(lines[i])
            i += 1
        para_text = " ".join(pl.strip() for pl in para_lines)
        story.append(Paragraph(_md_inline_to_rl(para_text), styles["body"]))
        story.append(Spacer(1, 4))

    return story


def _make_pdf_from_text_platypus(out_path: str, title: str | None, text: str, pagesize) -> None:
    # Lazy imports
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesize,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title or "Document",
    )

    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        spaceBefore=8, spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=17,
        spaceBefore=6, spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "H3", parent=base["Heading3"],
        fontName="Helvetica-Bold", fontSize=11.5, leading=15,
        spaceBefore=4, spaceAfter=4,
    )
    h4 = ParagraphStyle(
        "H4", parent=base["Heading4"],
        fontName="Helvetica-Bold", fontSize=10.5, leading=14,
        spaceBefore=4, spaceAfter=4,
    )
    code = ParagraphStyle(
        "Code", parent=base["Code"],
        fontName="Courier", fontSize=9, leading=12,
        leftIndent=8, rightIndent=8,
        backColor="#f3f3f3", borderPadding=4,
        spaceBefore=4, spaceAfter=4,
    )
    quote = ParagraphStyle(
        "Quote", parent=body,
        leftIndent=14, textColor="#555555",
        borderColor="#cccccc", borderWidth=0,
        spaceBefore=4, spaceAfter=4,
    )
    md_styles = {"body": body, "h1": h1, "h2": h2, "h3": h3, "h4": h4,
                 "code": code, "quote": quote}

    story = []
    if title:
        # Title is plain text — escape in case it contains < > &, no markdown.
        safe_title = (title.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"))
        story.append(Paragraph(safe_title, h1))
        story.append(Spacer(1, 10))

    raw = (text or "").strip()
    if not raw:
        raw = "(empty)"

    story.extend(_md_to_rl_story(raw, md_styles))

    doc.build(story)


@dataclass
class _CmpRow:
    aspect: str
    syne_plus: str
    syne_minus: str
    openclaw_plus: str
    openclaw_minus: str


def _comparison_rows() -> list[_CmpRow]:
    # Concise but meaningful; intended for landscape table readability.
    return [
        _CmpRow(
            "Fokus",
            "Asisten self-host + ops-minded",
            "Lebih opinionated",
            "Produk/community-first, multi-platform",
            "Bisa terasa besar",
        ),
        _CmpRow(
            "Channel",
            "Telegram native + kontrol grup/user",
            "Telegram only, no other channels",
            "Signal, WhatsApp, Discord, Telegram, iMessage, IRC, Slack, Google Chat (dll)",
            "Kompleksitas integrasi",
        ),
        _CmpRow(
            "Memory",
            "PostgreSQL-native, persistent across restarts",
            "Fitur baru belum mature (observability/confidence)",
            "memU built-in + ekosistem plugin",
            "File-based + plugin, bukan native DB",
        ),
        _CmpRow(
            "LLM Providers",
            "Codex, Google, Together, OpenAI",
            "Jumlah provider lebih sedikit",
            "Banyak provider OOTB (Anthropic, OpenAI, Google, Groq, Together, dll)",
            "Konfigurasi bisa lebih ramai",
        ),
        _CmpRow(
            "Cost",
            "Bisa zero cost (via OAuth; tergantung use-case)",
            "Bisa tetap ada cost kalau pakai API berbayar",
            "Subscription-based (seringnya pakai Anthropic API key)",
            "Cost cenderung fixed/berulang",
        ),
        _CmpRow(
            "Maturity",
            "Masih berkembang, fokus stabilisasi core",
            "Fitur baru bisa berubah",
            "Lebih mature, komunitas aktif, update rutin",
            "Perubahan cepat bisa bikin breaking changes",
        ),
        _CmpRow(
            "Tools",
            "Ability/tooling modular",
            "Perlu governance izin",
            "Ecosystem skill luas",
            "Konsistensi bervariasi",
        ),
        _CmpRow(
            "OAuth/Ops",
            "Refresh hardened + status",
            "Maintenance provider",
            "UX onboarding kuat",
            "Ops detail bisa tersembunyi",
        ),
        _CmpRow(
            "Automation",
            "Scheduler once/interval/cron",
            "Risk spam jika salah",
            "Automasi via skill/integrasi",
            "Tidak selalu native",
        ),
        _CmpRow(
            "Deployment",
            "Ringan untuk VPS personal",
            "Butuh disiplin backup/config",
            "Onboarding komunitas",
            "Footprint/moving parts",
        ),
    ]


def _make_comparison_pdf(out_path: str, title: str | None = None) -> None:
    # Lazy imports
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors

    pagesize = landscape(A4)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesize,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=title or "Syne vs OpenClaw",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )
    cell = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
    )
    head = ParagraphStyle(
        "Head",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
    )

    story: list[Any] = []
    story.append(Paragraph(title or "Perbandingan Syne vs OpenClaw", h1))
    story.append(Spacer(1, 6))

    # Build table data with Paragraph cells (wrap)
    data: list[list[Any]] = [
        [
            Paragraph("Aspek", head),
            Paragraph("Syne (+)", head),
            Paragraph("Syne (-)", head),
            Paragraph("OpenClaw (+)", head),
            Paragraph("OpenClaw (-)", head),
        ]
    ]

    for r in _comparison_rows():
        data.append(
            [
                Paragraph(r.aspect, cell),
                Paragraph(r.syne_plus, cell),
                Paragraph(r.syne_minus, cell),
                Paragraph(r.openclaw_plus, cell),
                Paragraph(r.openclaw_minus, cell),
            ]
        )

    # Column widths tuned for A4 landscape (in points). Total width ~ 770.
    col_widths = [80, 170, 170, 170, 170]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(tbl)
    doc.build(story)


# -----------------------------
# Ability implementation
# -----------------------------


class PdfAbility(Ability):
    name = "pdf"
    description = "Create and read PDFs from text, URL, or uploaded document."
    version = "1.7"
    permission = 0o770

    # Priority pre-processing: when user uploads a PDF, extract text automatically
    # so the LLM gets the content as text in the message context.
    priority = True

    # PDF input limits
    _MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MB
    _MAX_TEXT_CHARS = 50_000  # ~12k tokens — safe for any LLM context + fast response

    # Page classification thresholds
    _VISION_MIN_TEXT_CHARS = 30  # below this → page treated as image-only
    _DRAWING_MIN_VECTORS = 20  # above this → page has technical drawing (CAD-like)
    _DRAWING_MAX_TEXT_CHARS = 800  # if also < this much text → drawing dominates

    # Vision budget
    _VISION_MAX_PAGES = 10  # max pages to send to vision (cost/time guard)
    _VISION_DPI = 150  # render resolution — balance quality vs size
    _VISION_CONCURRENCY = 5  # parallel vision calls (avoid provider rate limit)
    _VISION_OVERALL_TIMEOUT = 90  # seconds — give up if vision takes too long

    # Prompts (specialized per page type)
    _PROMPT_OCR = (
        "Transcribe all visible text in this PDF page verbatim. "
        "If there are diagrams, tables, or images, briefly describe them. "
        "Preserve the reading order top-to-bottom."
    )
    _PROMPT_DRAWING = (
        "This page contains a technical/architectural drawing (CAD, blueprint, floor plan, schematic). "
        "Read and report ALL of the following that are visible: "
        "dimensions and measurements (with units), scale notation, "
        "labels for rooms / parts / sections, "
        "overall dimensions of the structure, layout description "
        "(how rooms/parts are arranged relative to each other), "
        "any annotations, notes, or revision marks. "
        "Be precise with numbers. List items, do not narrate."
    )

    def handles_input_type(self, input_type: str) -> bool:
        return input_type == "document"

    async def pre_process(
        self, input_type: str, input_data: dict, user_prompt: str,
        config: dict | None = None,
    ) -> str | None:
        """Extract text from an uploaded PDF document.

        Returns the extracted text (with page-count header) so the LLM can
        process the PDF content as plain text. Returns None on non-PDF or
        extraction failure → conversation falls back to native handling.
        """
        mime = (input_data.get("mime_type") or "").lower()
        filename = input_data.get("filename") or ""
        # Only handle PDFs — let other docs fall through
        if "pdf" not in mime and not filename.lower().endswith(".pdf"):
            return None

        b64 = input_data.get("base64", "")
        if not b64:
            return None

        try:
            import base64
            content = base64.b64decode(b64)
        except Exception as e:
            logger.warning(f"PDF pre_process: base64 decode failed: {e}")
            return None

        if len(content) > self._MAX_PDF_BYTES:
            return f"[PDF too large: {len(content) / 1024 / 1024:.1f} MB, max {self._MAX_PDF_BYTES // 1024 // 1024} MB]"

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PDF pre_process: PyMuPDF (fitz) not installed")
            return None

        try:
            doc = fitz.open(stream=content, filetype="pdf")
            pages = doc.page_count

            # Phase 1: extract text + classify each page
            # Page modes: "text" (text-only), "image" (no text, vision OCR),
            # "drawing" (has text + many vectors → text + vision drawing prompt)
            per_page_text: list[str] = []
            per_page_mode: list[str] = []
            per_page_vector_count: list[int] = []
            for i in range(pages):
                page = doc.load_page(i)
                t = page.get_text("text").strip()
                per_page_text.append(t)

                # Count vector drawing elements (cheap call)
                vec_count = 0
                try:
                    vec_count = len(page.get_drawings())
                except Exception:
                    pass
                per_page_vector_count.append(vec_count)

                # Classify
                if len(t) < self._VISION_MIN_TEXT_CHARS:
                    per_page_mode.append("image")  # image-only → OCR
                elif vec_count >= self._DRAWING_MIN_VECTORS and len(t) < self._DRAWING_MAX_TEXT_CHARS:
                    per_page_mode.append("drawing")  # technical drawing → text + vision
                else:
                    per_page_mode.append("text")  # regular text page

            # Phase 2: vision processing for image + drawing pages (within budget)
            # Prioritize: drawing pages by vector count desc, then image pages
            drawing_idx = [i for i, m in enumerate(per_page_mode) if m == "drawing"]
            image_idx = [i for i, m in enumerate(per_page_mode) if m == "image"]
            drawing_idx.sort(key=lambda i: per_page_vector_count[i], reverse=True)
            vision_queue: list[tuple[int, str]] = [(i, "drawing") for i in drawing_idx] + \
                                                  [(i, "image") for i in image_idx]
            vision_budget = vision_queue[: self._VISION_MAX_PAGES]
            vision_skipped = len(vision_queue) - len(vision_budget)

            per_page_vision: dict[int, str] = {}
            vision_used = 0
            if vision_budget:
                try:
                    per_page_vision = await self._describe_pages_via_vision(doc, vision_budget)
                    vision_used = sum(1 for v in per_page_vision.values() if v)
                except Exception as e:
                    logger.warning(f"PDF vision fallback failed: {e}")

            doc.close()

            # Build final text: text + [Drawing/Vision] block per page
            parts = []
            for i in range(pages):
                t = per_page_text[i]
                v = per_page_vision.get(i, "")
                mode = per_page_mode[i]
                section = []
                if t.strip():
                    section.append(t)
                if v.strip():
                    label = "Drawing" if mode == "drawing" else "Vision OCR"
                    section.append(f"[{label}]\n{v}")
                if section:
                    parts.append(f"--- Page {i + 1} ---\n" + "\n\n".join(section))
            full_text = "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"PDF pre_process: extraction failed: {e}")
            return None

        if not full_text.strip():
            return f"[PDF '{filename or 'document.pdf'}' ({pages} pages): no extractable text and vision fallback unavailable]"

        truncated = ""
        if len(full_text) > self._MAX_TEXT_CHARS:
            full_text = full_text[: self._MAX_TEXT_CHARS]
            truncated = f"\n\n[... truncated at {self._MAX_TEXT_CHARS} chars]"

        # Header with stats
        n_text = sum(1 for m in per_page_mode if m == "text")
        n_drawing = sum(1 for m in per_page_mode if m == "drawing")
        n_image = sum(1 for m in per_page_mode if m == "image")
        stats = [f"{pages} pages", f"{len(full_text)} chars"]
        if n_text:
            stats.append(f"{n_text} text")
        if n_drawing:
            stats.append(f"{n_drawing} drawing")
        if n_image:
            stats.append(f"{n_image} image")
        if vision_used:
            stats.append(f"{vision_used} via vision")
        if vision_skipped:
            stats.append(f"{vision_skipped} skipped (vision limit {self._VISION_MAX_PAGES})")
        header = f"PDF: {filename or 'document.pdf'} ({', '.join(stats)})"
        return f"{header}\n\n{full_text}{truncated}"

    async def _describe_pages_via_vision(self, doc, page_jobs: list[tuple[int, str]]) -> dict[int, str]:
        """Render PDF pages as images and send to image_analysis ability in parallel.

        Args:
            doc: open fitz Document
            page_jobs: list of (page_index, mode) where mode is "drawing" or "image"
                       (selects the vision prompt).

        Returns dict {page_index: description}. Pages that fail are omitted.
        Bounded by _VISION_CONCURRENCY (parallelism) + _VISION_OVERALL_TIMEOUT.
        """
        import base64 as _b64
        from .image_analysis import ImageAnalysisAbility

        ia = ImageAnalysisAbility()

        # Pre-render all images sequentially (CPU-bound, fast)
        import fitz
        rendered: list[tuple[int, str, str]] = []  # (page_index, mode, image_b64)
        for idx, mode in page_jobs:
            try:
                page = doc.load_page(idx)
                zoom = self._VISION_DPI / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                img_b64 = _b64.b64encode(img_bytes).decode("ascii")
                rendered.append((idx, mode, img_b64))
            except Exception as e:
                logger.warning(f"PDF page {idx + 1} render failed: {e}")

        # Parallel vision calls with concurrency cap
        sem = asyncio.Semaphore(self._VISION_CONCURRENCY)

        async def _one(idx: int, mode: str, img_b64: str) -> tuple[int, str | None]:
            prompt = self._PROMPT_DRAWING if mode == "drawing" else self._PROMPT_OCR
            async with sem:
                try:
                    result = await ia.execute(
                        params={
                            "image_base64": img_b64,
                            "mime_type": "image/png",
                            "prompt": prompt,
                        },
                        context={},
                    )
                    if result.get("success") and result.get("result"):
                        return idx, str(result["result"]).strip()
                    logger.warning(f"Vision page {idx + 1} ({mode}) failed: {result.get('error', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Vision page {idx + 1} ({mode}) exception: {e}")
                return idx, None

        results: dict[int, str] = {}
        try:
            tasks = [_one(idx, mode, b64) for idx, mode, b64 in rendered]
            outputs = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=self._VISION_OVERALL_TIMEOUT,
            )
            for idx, desc in outputs:
                if desc:
                    results[idx] = desc
        except asyncio.TimeoutError:
            logger.warning(
                f"PDF vision overall timeout ({self._VISION_OVERALL_TIMEOUT}s) — "
                f"partial results: {len(results)}/{len(rendered)} pages"
            )
        return results

    # pip package name → import name mapping
    _DEPS = {
        "reportlab": "reportlab",
        "readability-lxml": "readability",
        "beautifulsoup4": "bs4",
        "PyMuPDF": "fitz",
    }

    async def ensure_dependencies(self) -> tuple[bool, str]:
        """Install PDF dependencies (reportlab, readability-lxml, beautifulsoup4, PyMuPDF)."""
        missing_pkgs = []
        for pip_name, import_name in self._DEPS.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_pkgs.append(pip_name)

        if not missing_pkgs:
            return True, ""

        logger.info(f"Installing PDF deps: {', '.join(missing_pkgs)}")
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", *missing_pkgs,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        except asyncio.TimeoutError:
            return False, f"Installation timed out after 30 minutes. Try installing manually:\n  pip install {' '.join(missing_pkgs)}"
        if proc.returncode != 0:
            err = stderr.decode().strip()
            return False, f"Failed to install PDF deps: {err}"

        return True, f"Installed: {', '.join(missing_pkgs)}"

    def get_schema(self) -> dict:
        return {
            "name": "pdf",
            "description": "Create/read PDFs from text or URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "make_from_text",
                            "make_from_url",
                            "read_from_url",
                            "read_from_base64",
                            "make_comparison",
                        ],
                    },
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "url": {"type": "string"},
                    "pdf_base64": {"type": "string", "description": "Base64-encoded PDF (for read_from_base64)"},
                    "out_name": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 20},
                    "verify_ssl": {"type": "boolean", "default": True},
                },
                "required": ["action"],
            },
        }



    def get_guide(self, enabled: bool, config: dict) -> str:
        if not enabled:
            return (
                "- Status: **disabled**\n"
                "- Enable: `update_ability(action='enable', name='pdf')`"
            )

        # Dependency check kept lightweight (imports only, no heavy work)
        missing = []
        for mod in ("reportlab", "readability", "bs4", "fitz"):
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)

        if missing:
            return (
                f"- Status: **not ready** (missing deps: {', '.join(missing)})\n"
                "- Use after deps installed: `pdf(action='make_from_text', title='...', text='...')`\n"
                "- Note: `read_from_url` requires PyMuPDF (`fitz`)"
            )

        return (
            "- Status: **ready**\n"
            "- Auto-extracts text from PDFs uploaded by users (priority pre-process)\n"
            "- Make from text: `pdf(action='make_from_text', title='...', text='...')`\n"
            "- Make from URL: `pdf(action='make_from_url', url='https://...', out_name='file.pdf')`\n"
            "- Read PDF URL: `pdf(action='read_from_url', url='https://.../file.pdf')`\n"
            "- Read PDF base64: `pdf(action='read_from_base64', pdf_base64='...')`\n"
            "- Comparison PDF: `pdf(action='make_comparison', title='...')`"
        )


    async def execute(self, params: dict, context: dict) -> dict:
        action = (params.get("action") or "").strip()
        title = params.get("title")
        text = params.get("text")
        url = params.get("url")
        out_name = params.get("out_name")
        timeout_s = int(params.get("timeout_s") or 20)
        verify_ssl = bool(params.get("verify_ssl") if params.get("verify_ssl") is not None else True)

        session_id = str(context.get("session_id") or "")
        out_dir = self.get_output_dir(session_id=session_id)
        os.makedirs(out_dir, exist_ok=True)

        try:
            if action == "make_from_text":
                from reportlab.lib.pagesizes import A4

                name = _ensure_pdf_ext(_safe_filename(out_name or title or f"text_{_now_ts()}"))
                out_path = os.path.join(out_dir, name)
                _make_pdf_from_text_platypus(out_path, title, text or "", pagesize=A4)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            if action == "make_from_url":
                if not url:
                    raise ValueError("url is required")

                content, ctype = _download(url, timeout_s=timeout_s, verify_ssl=verify_ssl)

                # If it's already a PDF, just save it
                if ctype == "application/pdf" or url.lower().endswith(".pdf"):
                    name = _ensure_pdf_ext(_safe_filename(out_name or title or f"download_{_now_ts()}"))
                    out_path = os.path.join(out_dir, name)
                    with open(out_path, "wb") as f:
                        f.write(content)
                    return {"success": True, "result": {"file_path": out_path, "source": "pdf"}, "media": out_path}

                # Otherwise treat as HTML and create PDF from extracted readable text
                from reportlab.lib.pagesizes import A4

                extracted = _extract_readable_text_from_html(content)
                name = _ensure_pdf_ext(_safe_filename(out_name or title or f"url_{_now_ts()}"))
                out_path = os.path.join(out_dir, name)
                # Put URL on top for traceability
                combined = f"Source URL:\n{url}\n\n" + (extracted or "")
                _make_pdf_from_text_platypus(out_path, title or "From URL", combined, pagesize=A4)
                return {"success": True, "result": {"file_path": out_path, "source": "html"}, "media": out_path}

            if action == "read_from_url":
                if not url:
                    raise ValueError("url is required")

                content, ctype = _download(url, timeout_s=timeout_s, verify_ssl=verify_ssl)
                if not (ctype == "application/pdf" or url.lower().endswith(".pdf")):
                    raise ValueError(f"URL does not look like a PDF (content-type={ctype})")

                # Extract text
                import fitz  # PyMuPDF

                doc = fitz.open(stream=content, filetype="pdf")
                texts = []
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    texts.append(page.get_text("text").strip())
                full_text = "\n\n".join([t for t in texts if t])

                return {
                    "success": True,
                    "result": {
                        "page_count": doc.page_count,
                        "text": full_text,
                    },
                }

            if action == "read_from_base64":
                pdf_b64 = params.get("pdf_base64") or ""
                if not pdf_b64:
                    raise ValueError("pdf_base64 is required")
                import base64
                content = base64.b64decode(pdf_b64)
                if len(content) > self._MAX_PDF_BYTES:
                    raise ValueError(f"PDF too large: {len(content)} bytes")

                import fitz  # PyMuPDF
                doc = fitz.open(stream=content, filetype="pdf")
                texts = []
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    texts.append(page.get_text("text").strip())
                full_text = "\n\n".join([t for t in texts if t])
                pages = doc.page_count
                doc.close()

                return {
                    "success": True,
                    "result": {
                        "page_count": pages,
                        "text": full_text[: self._MAX_TEXT_CHARS],
                        "truncated": len(full_text) > self._MAX_TEXT_CHARS,
                    },
                }

            if action == "make_comparison":
                name = _ensure_pdf_ext(_safe_filename(out_name or title or f"syne_vs_openclaw_{_now_ts()}"))
                out_path = os.path.join(out_dir, name)
                _make_comparison_pdf(out_path, title=title)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            return {"success": False, "error": str(e)}
