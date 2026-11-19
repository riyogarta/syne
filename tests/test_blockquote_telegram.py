"""Telegram blockquote (Riyo, 19 Nov 2026) - v1.23.09.

Lines starting with "> " reached the chat as a raw '>' (escaped to &gt;).
They are now sent as <blockquote>, which Telegram HTML renders as a quote.
"""
from syne.communication.formatting import markdown_to_telegram_html as md


def test_single_quote_line():
    assert md("> Siang Mas, terima kasih.") == "<blockquote>Siang Mas, terima kasih.</blockquote>"


def test_consecutive_lines_form_one_quote():
    out = md("Template:\n\n> baris satu\n> baris dua\n\nSelesai.")
    assert "<blockquote>baris satu\nbaris dua</blockquote>" in out
    assert out.count("<blockquote>") == 1
    assert out.startswith("Template:") and out.endswith("Selesai.")


def test_separate_quotes():
    out = md("> satu\n\nteks\n\n> dua")
    assert out.count("<blockquote>") == 2


def test_empty_quote_line_inside_quote():
    assert md("> a\n>\n> b") == "<blockquote>a\n\nb</blockquote>"


def test_inline_formatting_and_escaping_inside_quote():
    assert md("> **tegas** a < b") == "<blockquote><b>tegas</b> a &lt; b</blockquote>"


def test_not_a_quote():
    assert "<blockquote>" not in md("a > b")
    assert "<blockquote>" not in md(">= 5 juta")
    assert "<blockquote>" not in md("->panah")
    assert "&gt;" in md("a > b")


def test_quote_inside_code_block_stays_literal():
    out = md("```\n> bukan kutipan\n```")
    assert out == "<pre>&gt; bukan kutipan</pre>"


def test_empty_quote_alone_is_dropped():
    assert "<blockquote>" not in md(">")
