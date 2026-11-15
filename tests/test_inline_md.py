"""Tests for syne.abilities._inline_md (shared inline markup parser)."""

import pytest

from syne.abilities._inline_md import (
    Segment,
    has_markup,
    normalize_syntax,
    parse_inline,
    plain_text,
    syntax_scope,
)


def styled(text, syntax=None):
    """Compact view: list of (text, flags) where flags is a sorted str."""
    out = []
    for s in parse_inline(text, syntax):
        f = "".join(
            c for c, on in (("b", s.bold), ("i", s.italic), ("s", s.strike), ("c", s.code)) if on
        )
        if s.link:
            f += "@" + s.link
        out.append((s.text, f))
    return out


# --- Markdown basics -------------------------------------------------------

def test_plain_text_untouched():
    assert styled("halo dunia") == [("halo dunia", "")]


def test_bold_and_italic():
    assert styled("a **b** c *d* e") == [
        ("a ", ""), ("b", "b"), (" c ", ""), ("d", "i"), (" e", ""),
    ]


def test_double_underscore_bold_and_single_underscore_italic():
    assert styled("__b__ dan _i_") == [("b", "b"), (" dan ", ""), ("i", "i")]


def test_strike_and_code():
    assert styled("~~x~~ `a*b*c`") == [("x", "s"), (" ", ""), ("a*b*c", "c")]


def test_triple_star_is_bold_italic():
    assert styled("***x***") == [("x", "bi")]


# --- The two petition cases ------------------------------------------------

def test_italic_closing_together_with_bold():
    assert styled("**sebagai *Stranded Asset***") == [
        ("sebagai ", "b"), ("Stranded Asset", "bi"),
    ]


def test_italic_in_parentheses_inside_bold():
    assert styled("**inti (*Lifetime Warranty*) yang**") == [
        ("inti (", "b"), ("Lifetime Warranty", "bi"), (") yang", "b"),
    ]


def test_bold_inside_italic():
    assert styled("*miring **tebal** miring*") == [
        ("miring ", "i"), ("tebal", "bi"), (" miring", "i"),
    ]


# --- Links -----------------------------------------------------------------

def test_link():
    assert styled("lihat [docs](https://x.y/z)") == [
        ("lihat ", ""), ("docs", "@https://x.y/z"),
    ]


def test_link_inside_bold():
    assert styled("**lihat [docs](https://x) ya**") == [
        ("lihat ", "b"), ("docs", "b@https://x"), (" ya", "b"),
    ]


def test_bold_inside_link_label():
    assert styled("[**a**](https://x)") == [("a", "b@https://x")]


# --- Things that must stay literal -----------------------------------------

def test_snake_case_not_italic():
    assert plain_text("file nama_file_ku.py") == "file nama_file_ku.py"
    assert not has_markup("nama_file_ku")


def test_arithmetic_star_not_italic():
    assert styled("5 * 3 * 2") == [("5 * 3 * 2", "")]


def test_unclosed_markers_kept():
    assert plain_text("**belum ditutup") == "**belum ditutup"


def test_backslash_escape():
    assert styled(r"\*bukan miring\*") == [("*bukan miring*", "")]


def test_horizontal_rule_like_text_is_literal_inline():
    assert plain_text("***") == "***"


# --- WhatsApp syntax -------------------------------------------------------

def test_whatsapp_styles():
    assert styled("*tebal* _miring_ ~coret~", "whatsapp") == [
        ("tebal", "b"), (" ", ""), ("miring", "i"), (" ", ""), ("coret", "s"),
    ]


def test_whatsapp_nested():
    assert styled("*tebal _miring_ tebal*", "whatsapp") == [
        ("tebal ", "b"), ("miring", "bi"), (" tebal", "b"),
    ]


def test_whatsapp_no_intraword():
    assert plain_text("a*b*c snake_case_x", "whatsapp") == "a*b*c snake_case_x"


def test_whatsapp_code():
    assert styled("```kode *x*```", "whatsapp") == [("kode *x*", "c")]


def test_markdown_vs_whatsapp_single_star():
    assert styled("*x*") == [("x", "i")]
    assert styled("*x*", "whatsapp") == [("x", "b")]


# --- Syntax selection ------------------------------------------------------

def test_syntax_scope_sets_default():
    with syntax_scope("whatsapp"):
        assert styled("*x*") == [("x", "b")]
    assert styled("*x*") == [("x", "i")]


@pytest.mark.parametrize("val,exp", [
    (None, "markdown"), ("", "markdown"), ("MD", "markdown"),
    ("WhatsApp", "whatsapp"), ("wa", "whatsapp"),
])
def test_normalize_syntax(val, exp):
    assert normalize_syntax(val) == exp


def test_normalize_syntax_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_syntax("html")


def test_segment_plain():
    assert Segment("x").plain
    assert not Segment("x", bold=True).plain
