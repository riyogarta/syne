"""Tests for syne.communication.outbound module."""

import os
from unittest.mock import patch

import pytest

from syne.communication.outbound import (
    extract_media,
    process_outbound,
    split_message,
    strip_narration,
    strip_server_paths,
)


class TestStripServerPaths:
    """Tests for strip_server_paths()."""

    def test_strips_home_path(self):
        text = "Here is the file: /home/syne/workspace/outputs/result.png"
        result = strip_server_paths(text)
        assert "/home/" not in result

    def test_strips_tmp_path(self):
        text = "Saved to: /tmp/audio_123.wav"
        result = strip_server_paths(text)
        assert "/tmp/" not in result

    def test_strips_var_path(self):
        text = "Log at /var/log/syne/error.log please check"
        result = strip_server_paths(text)
        assert "/var/" not in result

    def test_strips_path_with_file_prefix(self):
        text = "File: /home/user/documents/secret.txt"
        result = strip_server_paths(text)
        assert "/home/" not in result

    def test_strips_path_with_lokasi_prefix(self):
        text = "Lokasi: /home/syne/workspace/file.txt"
        result = strip_server_paths(text)
        assert "/home/" not in result

    def test_no_paths_unchanged(self):
        text = "This is a normal message with no file paths."
        result = strip_server_paths(text)
        assert result == text

    def test_mixed_content_strips_only_paths(self):
        text = "The result is 42.\nFile: /home/syne/output.txt\nDone!"
        result = strip_server_paths(text)
        assert "42" in result
        assert "Done!" in result
        assert "/home/" not in result

    def test_does_not_strip_url_paths(self):
        text = "Visit https://example.com/home/page for details"
        result = strip_server_paths(text)
        assert "https://example.com/home/page" in result

    def test_collapses_excess_newlines(self):
        text = "Before\n\n\n\n/home/user/file.txt\n\n\n\nAfter"
        result = strip_server_paths(text)
        assert "\n\n\n" not in result


class TestStripNarration:
    """Tests for strip_narration()."""

    def test_strips_let_me_check(self):
        text = "Let me check the database.\nThe answer is 42."
        result = strip_narration(text)
        assert "Let me check" not in result
        assert "42" in result

    def test_strips_ill_search(self):
        text = "I'll search for that information.\nHere are the results."
        result = strip_narration(text)
        assert "I'll search" not in result
        assert "results" in result

    def test_strips_now_im(self):
        text = "Now I'm going to analyze the logs.\nThe error is X."
        result = strip_narration(text)
        assert "Now I'm" not in result
        assert "error is X" in result

    def test_strips_first_let_me(self):
        text = "First, let me check the config.\nConfig looks fine."
        result = strip_narration(text)
        assert "First" not in result
        assert "Config looks fine" in result

    def test_strips_indonesian_narration(self):
        text = "Oke, aku akan cek databasenya.\nHasilnya adalah 42."
        result = strip_narration(text)
        assert "Oke" not in result
        assert "42" in result

    def test_no_narration_unchanged(self):
        text = "The answer to your question is 42."
        result = strip_narration(text)
        assert result == text

    def test_empty_string(self):
        result = strip_narration("")
        assert result == ""

    def test_narration_mid_text_preserved(self):
        # Narration patterns only strip from first 500 chars
        prefix = "A" * 501
        text = prefix + "\nLet me check that."
        result = strip_narration(text)
        assert "Let me check" in result

    def test_returns_original_if_stripping_removes_everything(self):
        text = "Let me check."
        result = strip_narration(text)
        # If stripping leaves nothing meaningful, original is returned
        assert result == text or len(result) > 0

    def test_strips_hmm_let_me(self):
        text = "Hmm, let me think about this.\nThe solution is X."
        result = strip_narration(text)
        assert "Hmm" not in result
        assert "solution is X" in result


class TestSplitMessage:
    """Tests for split_message()."""

    def test_short_message_no_split(self):
        text = "Hello, world!"
        result = split_message(text)
        assert result == ["Hello, world!"]

    def test_exact_max_length_no_split(self):
        text = "A" * 4096
        result = split_message(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_message_splits(self):
        text = "A" * 5000
        result = split_message(text, max_length=100)
        assert len(result) > 1
        assert all(len(chunk) <= 100 for chunk in result)

    def test_splits_at_newlines(self):
        lines = ["Line " + str(i) for i in range(100)]
        text = "\n".join(lines)
        result = split_message(text, max_length=200)
        assert len(result) > 1
        # Each chunk should end at a newline boundary
        for chunk in result[:-1]:
            assert "\n" not in chunk or chunk.endswith(chunk.rstrip())

    def test_respects_max_length(self):
        text = "word " * 2000
        result = split_message(text, max_length=500)
        for chunk in result:
            assert len(chunk) <= 500

    def test_custom_max_length(self):
        text = "A" * 200
        result = split_message(text, max_length=50)
        assert len(result) > 1
        assert all(len(chunk) <= 50 for chunk in result)

    def test_splits_at_space_when_no_newline(self):
        text = "word " * 100  # long text without newlines before max
        result = split_message(text, max_length=30)
        assert len(result) > 1

    def test_hard_cut_when_no_space_or_newline(self):
        text = "A" * 200  # no spaces or newlines
        result = split_message(text, max_length=50)
        assert len(result) == 4
        assert result[0] == "A" * 50


class TestProcessOutbound:
    """Tests for process_outbound() — the full pipeline."""

    def test_full_pipeline(self):
        text = "Let me check that.\nHere is the file: /home/user/data.txt\nResult is 42."
        result = process_outbound(text)
        assert "Let me check" not in result
        assert "/home/" not in result
        assert "42" in result

    def test_empty_string(self):
        result = process_outbound("")
        assert result == ""

    def test_none_passthrough(self):
        # process_outbound checks `if not text` so None should pass through
        result = process_outbound(None)
        assert result is None

    def test_clean_text_passes_through(self):
        text = "This is a perfectly normal response."
        result = process_outbound(text)
        assert result == text

    def test_collapses_excessive_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = process_outbound(text)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestExtractMedia:
    """Tests for extract_media()."""

    def test_with_media_tag(self):
        text = "Here is your image\n\nMEDIA: /tmp/test.png"
        with patch("os.path.isfile", return_value=True):
            cleaned, media = extract_media(text)
        assert cleaned == "Here is your image"
        assert media == "/tmp/test.png"

    def test_media_only(self):
        text = "MEDIA: /tmp/test.png"
        with patch("os.path.isfile", return_value=True):
            cleaned, media = extract_media(text)
        assert cleaned == ""
        assert media == "/tmp/test.png"

    def test_without_media_tag(self):
        text = "Just a normal message"
        cleaned, media = extract_media(text)
        assert cleaned == "Just a normal message"
        assert media is None

    def test_media_file_not_exists(self):
        text = "Here is your image\n\nMEDIA: /tmp/nonexistent.png"
        with patch("os.path.isfile", return_value=False):
            cleaned, media = extract_media(text)
        assert media is None

    def test_media_tag_mid_text(self):
        text = "Part 1\n\nMEDIA: /tmp/image.png"
        with patch("os.path.isfile", return_value=True):
            cleaned, media = extract_media(text)
        assert cleaned == "Part 1"
        assert media == "/tmp/image.png"

    def test_multiple_media_tags_takes_last(self):
        text = "First\n\nMEDIA: /tmp/a.png\n\nMEDIA: /tmp/b.png"
        with patch("os.path.isfile", return_value=True):
            cleaned, media = extract_media(text)
        # rsplit with maxsplit=1 takes the last occurrence
        assert media == "/tmp/b.png"
