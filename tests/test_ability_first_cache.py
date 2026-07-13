"""Tests for _ability_first_preprocess populating self._cached_input_data.

The tool loop in _handle_tool_calls auto-fills image_base64 / audio_base64 /
document_base64 for ability tool calls that arrive without pixels — reading
from self._cached_input_data. Before the 2026-07-13 fix, that cache was
initialised to {} and never populated, so a well-succeeded pre_process left
later tool-call retries with empty bytes and the ability returned
'Either image_url or image_base64 is required'. From the user side this
surfaced as 'Syne says can't read the image, but on retry it can.'

These tests lock the snapshot behaviour so the mechanism cannot silently
regress into dead code again.
"""

from unittest.mock import MagicMock

import pytest

from syne.conversation import Conversation


# ---------------------------------------------------------------------------
# Minimal stubs for the collaborators _ability_first_preprocess touches
# ---------------------------------------------------------------------------


class _StubAbilityInstance:
    """Duck-types the ability instance surface the method uses."""

    def __init__(self, input_type: str, result):
        self._input_type = input_type
        self._result = result
        self.priority = True

    def handles_input_type(self, t: str) -> bool:
        return t == self._input_type

    async def pre_process(self, input_type, input_data, prompt, config=None):
        return self._result

    async def ensure_dependencies(self):
        return True, ""


class _StubRegistered:
    def __init__(self, name: str, instance):
        self.name = name
        self.instance = instance
        self.config = {}
        self.deps_ensured = True


class _StubAbilitiesRegistry:
    def __init__(self, entries):
        self._entries = entries

    def list_enabled(self, _access):
        return self._entries


class _StubConversation:
    """Minimal duck-typed Conversation for _ability_first_preprocess."""

    def __init__(self, abilities, provider_supports_vision: bool = True):
        self.abilities = abilities
        self._cached_input_data = {}
        self._message_metadata = None
        self.provider = MagicMock()
        self.provider.supports_vision = provider_supports_vision

    def _inline_text_document(self, _input_data):
        # Not exercised by the image tests below; return None so the
        # native-fallback branch behaves like the real one for our
        # purposes ("no native support — strip and show error").
        return None


# ---------------------------------------------------------------------------
# The regression the fix is guarding
# ---------------------------------------------------------------------------


class TestCachedInputData:

    async def test_cache_populated_when_ability_succeeds(self):
        """Ability succeeds → description injected → raw image stripped from
        metadata → but the cache MUST still hold the raw bytes so a
        subsequent LLM tool call can re-analyse without needing the user
        to re-attach the photo."""
        ability = _StubAbilityInstance(
            "image", result="a receipt showing total 150000"
        )
        conv = _StubConversation(
            abilities=_StubAbilitiesRegistry([
                _StubRegistered("image_analysis", ability)
            ]),
        )
        metadata = {"image": {"base64": "AAAA", "mime_type": "image/jpeg"}}

        new_msg, new_meta = await Conversation._ability_first_preprocess(
            conv, "berapa totalnya?", metadata
        )

        # Existing behaviour still holds:
        assert "receipt showing total 150000" in new_msg
        assert (new_meta or {}).get("image") is None

        # The fix: cache retains the raw bytes for a retry.
        assert conv._cached_input_data.get("image") is not None
        assert conv._cached_input_data["image"]["base64"] == "AAAA"
        assert conv._cached_input_data["image"]["mime_type"] == "image/jpeg"

    async def test_cache_populated_when_ability_fails(self):
        """Ability fails but the LLM has native vision → metadata is kept AND
        the cache still holds the bytes. Either delivery path (native
        pixels in cache metadata, or explicit tool call reading the vault)
        stays wired."""
        ability = _StubAbilityInstance("image", result=None)  # returns None ⇒ failure
        conv = _StubConversation(
            abilities=_StubAbilitiesRegistry([
                _StubRegistered("image_analysis", ability)
            ]),
            provider_supports_vision=True,
        )
        metadata = {"image": {"base64": "BBBB", "mime_type": "image/jpeg"}}

        _, new_meta = await Conversation._ability_first_preprocess(
            conv, "baca receipt", metadata
        )

        # Native-fallback path keeps the metadata for the driver to send.
        assert new_meta.get("image", {}).get("base64") == "BBBB"
        # Cache also captured it for the tool-call retry path.
        assert conv._cached_input_data["image"]["base64"] == "BBBB"

    async def test_no_media_no_cache_pollution(self):
        """A plain text turn (no image/audio/document in metadata) must not
        populate the cache at all — the tool loop's retry path stays
        inert when there's nothing to inject."""
        ability = _StubAbilityInstance("image", result=None)
        conv = _StubConversation(
            abilities=_StubAbilitiesRegistry([
                _StubRegistered("image_analysis", ability)
            ]),
        )
        metadata = {"original_text": "halo apa kabar"}

        await Conversation._ability_first_preprocess(
            conv, "halo apa kabar", metadata
        )

        assert conv._cached_input_data == {}

    async def test_cache_covers_all_media_kinds(self):
        """All three declared input types (image, audio, document) get
        snapshotted the same way — not just image."""
        # No matching ability for audio/document → they'll fall through.
        # The snapshot must still run BEFORE any ability logic per input.
        ability = _StubAbilityInstance("image", result="ok")
        conv = _StubConversation(
            abilities=_StubAbilitiesRegistry([
                _StubRegistered("image_analysis", ability)
            ]),
        )
        metadata = {
            "image": {"base64": "IMG", "mime_type": "image/jpeg"},
            "audio": {"base64": "AUD", "mime_type": "audio/ogg"},
            "document": {"base64": "DOC", "mime_type": "application/pdf"},
        }

        await Conversation._ability_first_preprocess(
            conv, "cek semuanya", metadata
        )

        assert conv._cached_input_data["image"]["base64"] == "IMG"
        assert conv._cached_input_data["audio"]["base64"] == "AUD"
        assert conv._cached_input_data["document"]["base64"] == "DOC"

    async def test_no_abilities_early_return_no_cache(self):
        """If the conversation has no abilities registry at all, the method
        early-returns without touching the cache. Verifies the guard at
        the top of the function still short-circuits correctly."""
        conv = _StubConversation(abilities=None)
        metadata = {"image": {"base64": "AAAA", "mime_type": "image/jpeg"}}

        new_msg, new_meta = await Conversation._ability_first_preprocess(
            conv, "baca", metadata
        )

        # Nothing changed, nothing cached — pass-through.
        assert new_msg == "baca"
        assert new_meta is metadata
        assert conv._cached_input_data == {}
