"""Render tests: PDF / Word / PowerPoint share the same inline markup rules."""

import pytest

from syne.abilities import _inline_md

docx = pytest.importorskip("docx")
pptx = pytest.importorskip("pptx")
pytest.importorskip("reportlab")
fitz = pytest.importorskip("fitz")

from syne.abilities import office, pdf  # noqa: E402

PETITION = (
    "**2. Penghentian Standar GB/T sebagai *Stranded Asset***\n\n"
    "**tidak membatalkan garansi inti (*Lifetime Core Component Warranty*) yang berlaku**\n\n"
    "Kerugian sebagai *early adopters* dan **lihat [sumber](https://example.com) ini**."
)


# --- PDF -------------------------------------------------------------------

def test_pdf_inline_nested():
    html = pdf._md_inline_to_rl("**a *b***")
    assert html == "<b>a </b><b><i>b</i></b>"


def test_pdf_inline_escapes_html():
    assert pdf._md_inline_to_rl("a < b & **c**") == "a &lt; b &amp; <b>c</b>"


def test_pdf_inline_link_inside_bold():
    html = pdf._md_inline_to_rl("**x [y](https://e.com) z**")
    assert '<link href="https://e.com" color="#1a73e8"><u><b>y</b></u></link>' in html
    assert "**" not in html


def test_pdf_whatsapp_scope():
    with _inline_md.syntax_scope("whatsapp"):
        assert pdf._md_inline_to_rl("*tebal* _miring_") == "<b>tebal</b> <i>miring</i>"


def _pdf_text(path):
    doc = fitz.open(path)
    try:
        return "\n".join(p.get_text() for p in doc)
    finally:
        doc.close()


def _pdf_fonts(path):
    doc = fitz.open(path)
    try:
        names = set()
        for page in doc:
            for f in page.get_fonts():
                names.add(f[3])
        return names
    finally:
        doc.close()


def test_pdf_petition_has_no_stray_markers(tmp_path):
    from reportlab.lib.pagesizes import A4

    out = tmp_path / "p.pdf"
    pdf._make_pdf_from_text_platypus(str(out), "Judul", PETITION, pagesize=A4, images={})
    text = _pdf_text(out)
    assert "*" not in text
    assert "Stranded Asset" in text and "Lifetime Core Component Warranty" in text
    fonts = _pdf_fonts(out)
    assert "Helvetica-BoldOblique" in fonts
    assert "Helvetica-Oblique" in fonts


def test_pdf_whatsapp_build(tmp_path):
    from reportlab.lib.pagesizes import A4

    out = tmp_path / "wa.pdf"
    with _inline_md.syntax_scope("whatsapp"):
        pdf._make_pdf_from_text_platypus(
            str(out), None, "*Tebal* dan _miring_ dan ~coret~", pagesize=A4, images={}
        )
    text = _pdf_text(out)
    assert "*" not in text and "_" not in text and "~" not in text


# --- Word ------------------------------------------------------------------

def _docx_runs(path):
    d = docx.Document(str(path))
    runs = []
    for p in d.paragraphs:
        for r in p.runs:
            runs.append((r.text, bool(r.bold), bool(r.italic)))
    return d, runs


def test_docx_petition_nested_styles(tmp_path):
    out = tmp_path / "p.docx"
    office._make_docx(str(out), "Judul", PETITION, images={})
    d, runs = _docx_runs(out)
    full = "\n".join(p.text for p in d.paragraphs)
    assert "*" not in full
    assert ("Stranded Asset", True, True) in runs
    assert ("Lifetime Core Component Warranty", True, True) in runs
    assert ("early adopters", False, True) in runs


def test_docx_link_inside_bold_keeps_bold(tmp_path):
    from docx.oxml.ns import qn

    out = tmp_path / "l.docx"
    office._make_docx(str(out), "", "**lihat [sumber](https://e.com) ini**", images={})
    d = docx.Document(str(out))
    body = d.element.body
    links = body.findall(".//" + qn("w:hyperlink"))
    assert len(links) == 1
    assert links[0].find(".//" + qn("w:b")) is not None
    assert "**" not in "".join(t.text or "" for t in body.iter(qn("w:t")))


def test_docx_list_without_blank_line(tmp_path):
    out = tmp_path / "list.docx"
    office._make_docx(str(out), "", "Intro:\n- satu\n- dua", images={})
    d = docx.Document(str(out))
    texts = [p.text for p in d.paragraphs if p.text]
    assert texts == ["Intro:", "satu", "dua"]


def test_docx_heading_then_paragraph_without_blank_line(tmp_path):
    out = tmp_path / "h.docx"
    office._make_docx(str(out), "", "## Judul\nIsi paragraf", images={})
    d = docx.Document(str(out))
    paras = [p for p in d.paragraphs if p.text]
    assert [p.text for p in paras] == ["Judul", "Isi paragraf"]
    assert "Heading" in paras[0].style.name


def test_docx_whatsapp(tmp_path):
    out = tmp_path / "wa.docx"
    with _inline_md.syntax_scope("whatsapp"):
        office._make_docx(str(out), "", "*Tebal _miring_*", images={})
    _, runs = _docx_runs(out)
    assert ("Tebal ", True, False) in runs
    assert ("miring", True, True) in runs


def test_docx_snake_case_untouched(tmp_path):
    out = tmp_path / "s.docx"
    office._make_docx(str(out), "", "file nama_file_ku.py", images={})
    _, runs = _docx_runs(out)
    assert runs == [("file nama_file_ku.py", False, False)]


# --- PowerPoint ------------------------------------------------------------

def _pptx_runs(path):
    prs = pptx.Presentation(str(path))
    runs = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    runs.append((r.text, bool(r.font.bold), bool(r.font.italic)))
    return runs


def test_pptx_nested_styles(tmp_path):
    out = tmp_path / "s.pptx"
    slides = [
        {"title": "Petisi", "subtitle": "x"},
        {"title": "Alasan", "bullets": ["**sebagai *Stranded Asset***", "biasa ~~coret~~"]},
        {"title": "Terima Kasih"},
    ]
    office._make_pptx(str(out), "Petisi", slides, images={})
    runs = _pptx_runs(out)
    assert ("Stranded Asset", True, True) in runs
    assert not any("*" in t or "~~" in t for t, _, _ in runs)


def test_pptx_whatsapp(tmp_path):
    out = tmp_path / "wa.pptx"
    slides = [{"title": "A", "subtitle": "b"}, {"title": "B", "bullets": ["*tebal* _miring_"]}]
    with _inline_md.syntax_scope("whatsapp"):
        office._make_pptx(str(out), "A", slides, images={})
    runs = _pptx_runs(out)
    assert ("tebal", True, False) in runs
    assert ("miring", False, True) in runs


# --- Ability parameter -----------------------------------------------------

@pytest.mark.asyncio
async def test_pdf_ability_rejects_unknown_syntax(tmp_path, monkeypatch):
    ab = pdf.PdfAbility() if hasattr(pdf, "PdfAbility") else None
    if ab is None:
        pytest.skip("PdfAbility class name differs")
    monkeypatch.setattr(ab, "get_output_dir", lambda session_id=None: str(tmp_path))
    res = await ab.execute(
        {"action": "make_from_text", "text": "x", "syntax": "html"}, {"session_id": "t"}
    )
    assert res["success"] is False
    assert "syntax" in res["error"].lower()
