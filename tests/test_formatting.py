"""Tests for syne.communication.formatting module."""

import pytest

from syne.communication.formatting import markdown_to_telegram_html


class TestMarkdownToTelegramHtml:
    """Tests for markdown_to_telegram_html()."""

    # --- Bold ---

    def test_bold_double_asterisk(self):
        result = markdown_to_telegram_html("This is **bold** text")
        assert "<b>bold</b>" in result

    def test_bold_double_underscore(self):
        result = markdown_to_telegram_html("This is __bold__ text")
        assert "<b>bold</b>" in result

    # --- Italic ---

    def test_italic_single_asterisk(self):
        result = markdown_to_telegram_html("This is *italic* text")
        assert "<i>italic</i>" in result

    def test_italic_single_underscore(self):
        result = markdown_to_telegram_html("This is _italic_ text")
        assert "<i>italic</i>" in result

    def test_underscore_in_word_not_italic(self):
        result = markdown_to_telegram_html("file_name_here")
        # Underscores mid-word should NOT be converted to italic
        assert "<i>" not in result

    # --- Inline code ---

    def test_inline_code(self):
        result = markdown_to_telegram_html("Use `print()` to debug")
        assert "<code>print()</code>" in result

    def test_inline_code_html_escaped(self):
        result = markdown_to_telegram_html("Try `x < 5 && y > 3`")
        assert "<code>" in result
        assert "&lt;" in result
        assert "&amp;" in result

    # --- Code blocks ---

    def test_code_block(self):
        text = "```\nprint('hello')\n```"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "</pre>" in result
        assert "print" in result

    def test_code_block_with_language(self):
        text = "```python\nx = 42\n```"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "x = 42" in result

    def test_code_block_escapes_html(self):
        # Content with HTML special chars (but not actual HTML tags that
        # would trigger the _html_to_telegram pre-processing pass)
        text = "```\nx < 5 && y > 3\n```"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "&lt;" in result
        assert "&amp;" in result

    # --- Links ---

    def test_link(self):
        result = markdown_to_telegram_html("Visit [Google](https://google.com)")
        assert '<a href="https://google.com">Google</a>' in result

    def test_link_with_text_around(self):
        result = markdown_to_telegram_html("Go to [site](https://x.com) now")
        assert '<a href="https://x.com">site</a>' in result

    # --- Strikethrough ---

    def test_strikethrough(self):
        result = markdown_to_telegram_html("This is ~~deleted~~ text")
        assert "<s>deleted</s>" in result

    # --- Headers ---

    def test_header_h1(self):
        result = markdown_to_telegram_html("# Title")
        assert "<b>Title</b>" in result

    def test_header_h3(self):
        result = markdown_to_telegram_html("### Section")
        assert "<b>Section</b>" in result

    # --- Plain text ---

    def test_plain_text_unchanged(self):
        result = markdown_to_telegram_html("Hello, world!")
        assert "Hello, world!" in result

    def test_plain_text_html_escaped(self):
        result = markdown_to_telegram_html("Use x < 5 & y > 3")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_empty_string(self):
        result = markdown_to_telegram_html("")
        assert result == ""

    def test_none_passthrough(self):
        result = markdown_to_telegram_html(None)
        assert result is None

    # --- Mixed formatting ---

    def test_bold_and_italic_together(self):
        result = markdown_to_telegram_html("This is **bold** and *italic*")
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result

    def test_bold_with_inline_code(self):
        result = markdown_to_telegram_html("Run **this** with `code`")
        assert "<b>this</b>" in result
        assert "<code>code</code>" in result

    def test_multiple_code_spans(self):
        result = markdown_to_telegram_html("Use `foo` and `bar`")
        assert "<code>foo</code>" in result
        assert "<code>bar</code>" in result

    def test_link_and_bold(self):
        result = markdown_to_telegram_html("**Important**: see [docs](https://docs.com)")
        assert "<b>Important</b>" in result
        assert '<a href="https://docs.com">docs</a>' in result

    # --- Tables ---

    def test_simple_table(self):
        text = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "</pre>" in result
        assert "Alice" in result
        assert "Name" in result

    def test_table_alignment(self):
        text = "| Col1 | Col2 |\n|---|---|\n| a | b |"
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        # Table should be rendered with pipe alignment
        assert "|" in result

    def test_table_with_surrounding_text(self):
        text = "Results:\n| Key | Value |\n|---|---|\n| x | 1 |\nDone."
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "Results:" in result
        assert "Done." in result

    # --- Multiline ---

    def test_multiline_with_code_and_text(self):
        text = "Here is the code:\n```\nx = 1\n```\nAnd **done**."
        result = markdown_to_telegram_html(text)
        assert "<pre>" in result
        assert "<b>done</b>" in result
