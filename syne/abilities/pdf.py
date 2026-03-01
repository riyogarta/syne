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

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceAfter=10,
    )

    story = []
    if title:
        story.append(Paragraph(title, h1))
        story.append(Spacer(1, 10))

    raw = (text or "").strip()
    if not raw:
        raw = "(empty)"

    # Split paragraphs by blank lines; convert single newlines to <br/>
    for para in re.split(r"\n\s*\n", raw):
        para = para.strip()
        if not para:
            continue
        para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        para = para.replace("\n", "<br/>")
        story.append(Paragraph(para, body))

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
    description = "Create and read PDFs from text or URL."
    version = "1.5"

    # tool-call style, not pre-processing
    priority = False

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
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", *missing_pkgs,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
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
                            "make_comparison",
                        ],
                    },
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "url": {"type": "string"},
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
            "- Make from text: `pdf(action='make_from_text', title='...', text='...')`\n"
            "- Make from URL: `pdf(action='make_from_url', url='https://...', out_name='file.pdf')`\n"
            "- Read PDF URL: `pdf(action='read_from_url', url='https://.../file.pdf')`\n"
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

            if action == "make_comparison":
                name = _ensure_pdf_ext(_safe_filename(out_name or title or f"syne_vs_openclaw_{_now_ts()}"))
                out_path = os.path.join(out_dir, name)
                _make_comparison_pdf(out_path, title=title)
                return {"success": True, "result": {"file_path": out_path}, "media": out_path}

            raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            return {"success": False, "error": str(e)}
