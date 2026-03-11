"""Tests for syne.communication.tags module."""

import pytest

from syne.communication.tags import parse_react_tags, parse_reply_tag


class TestParseReplyTag:
    """Tests for parse_reply_tag()."""

    def test_no_tag(self):
        text = "Just a normal response."
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=100)
        assert cleaned == "Just a normal response."
        assert reply_id is None

    def test_reply_to_current_with_message_id(self):
        text = "[[reply_to_current]] Here is my reply."
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=42)
        assert "reply_to_current" not in cleaned
        assert "Here is my reply" in cleaned
        assert reply_id == 42

    def test_reply_to_current_with_spaces(self):
        text = "[[ reply_to_current ]] Replying now."
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=99)
        assert "reply_to_current" not in cleaned
        assert reply_id == 99

    def test_reply_to_current_no_incoming_message(self):
        text = "[[reply_to_current]] Some text."
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=None)
        assert "reply_to_current" not in cleaned
        assert reply_id is None

    def test_reply_to_specific_id(self):
        text = "[[reply_to:555]] Quoting that message."
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=100)
        assert "reply_to" not in cleaned
        assert "555" not in cleaned
        assert reply_id == 555

    def test_reply_to_specific_id_with_spaces(self):
        text = "[[ reply_to: 123 ]] Response."
        cleaned, reply_id = parse_reply_tag(text)
        assert reply_id == 123
        assert "reply_to" not in cleaned

    def test_reply_tag_at_end(self):
        text = "My response here [[reply_to_current]]"
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=50)
        assert cleaned == "My response here"
        assert reply_id == 50

    def test_reply_to_current_takes_precedence(self):
        # If both tags somehow exist, reply_to_current is checked first
        text = "[[reply_to_current]] [[reply_to:999]] text"
        cleaned, reply_id = parse_reply_tag(text, incoming_message_id=10)
        assert reply_id == 10

    def test_default_incoming_message_id(self):
        text = "No tags here."
        cleaned, reply_id = parse_reply_tag(text)
        assert reply_id is None


class TestParseReactTags:
    """Tests for parse_react_tags()."""

    def test_no_tags(self):
        text = "Just a normal response."
        cleaned, emojis = parse_react_tags(text)
        assert cleaned == "Just a normal response."
        assert emojis == []

    def test_single_react(self):
        text = "Got it! [[react:thumbs_up]]"
        cleaned, emojis = parse_react_tags(text)
        assert "react" not in cleaned
        assert len(emojis) == 1
        assert emojis[0] == "thumbs_up"

    def test_multiple_reacts(self):
        text = "Nice! [[react:heart]] [[react:fire]]"
        cleaned, emojis = parse_react_tags(text)
        assert len(emojis) == 2
        assert "heart" in emojis
        assert "fire" in emojis
        assert "react" not in cleaned

    def test_text_cleaning(self):
        text = "Before [[react:smile]] After"
        cleaned, emojis = parse_react_tags(text)
        assert "Before" in cleaned
        assert "After" in cleaned
        assert "[[" not in cleaned
        assert "]]" not in cleaned

    def test_react_with_spaces_in_tag(self):
        text = "OK [[ react: wave ]]"
        cleaned, emojis = parse_react_tags(text)
        assert emojis == ["wave"]
        assert "react" not in cleaned

    def test_react_with_emoji_character(self):
        text = "Sure [[react:ok_hand]]"
        cleaned, emojis = parse_react_tags(text)
        assert emojis == ["ok_hand"]

    def test_mixed_react_and_text(self):
        text = "I acknowledge your message. [[react:thumbs_up]] Will work on it."
        cleaned, emojis = parse_react_tags(text)
        assert "I acknowledge" in cleaned
        assert "Will work on it" in cleaned
        assert emojis == ["thumbs_up"]
