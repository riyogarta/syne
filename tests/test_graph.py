"""Tests for syne/memory/graph.py — knowledge graph extraction and recall."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from syne.memory.graph import _parse_extraction, extract_and_store


# ── _parse_extraction (pure function) ────────────────────────


class TestParseExtraction:
    def test_valid_json(self):
        raw = json.dumps({
            "entities": [{"name": "Alice", "type": "person", "description": "a friend"}],
            "relations": [{"subject": "Alice", "predicate": "lives_in", "object": "Jakarta"}],
        })
        result = _parse_extraction(raw)
        assert result is not None
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1
        assert result["entities"][0]["name"] == "Alice"

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"entities": [], "relations": []}\n```'
        result = _parse_extraction(raw)
        assert result is not None
        assert result["entities"] == []
        assert result["relations"] == []

    def test_empty_string(self):
        assert _parse_extraction("") is None

    def test_none_input(self):
        assert _parse_extraction(None) is None

    def test_no_json(self):
        assert _parse_extraction("I don't know what to extract") is None

    def test_invalid_json(self):
        assert _parse_extraction('{"entities": [broken') is None

    def test_entity_missing_name(self):
        raw = json.dumps({
            "entities": [{"type": "person", "description": "no name"}],
            "relations": [],
        })
        result = _parse_extraction(raw)
        assert result is not None
        assert len(result["entities"]) == 0

    def test_entity_missing_type(self):
        raw = json.dumps({
            "entities": [{"name": "Alice", "description": "no type"}],
            "relations": [],
        })
        result = _parse_extraction(raw)
        assert result is not None
        assert len(result["entities"]) == 0

    def test_relation_missing_predicate(self):
        raw = json.dumps({
            "entities": [],
            "relations": [{"subject": "A", "object": "B"}],
        })
        result = _parse_extraction(raw)
        assert result is not None
        assert len(result["relations"]) == 0

    def test_relation_missing_subject(self):
        raw = json.dumps({
            "entities": [],
            "relations": [{"predicate": "likes", "object": "B"}],
        })
        result = _parse_extraction(raw)
        assert len(result["relations"]) == 0

    def test_mixed_valid_invalid_entities(self):
        raw = json.dumps({
            "entities": [
                {"name": "Alice", "type": "person", "description": "valid"},
                {"type": "place"},  # missing name
                {"name": "Jakarta", "type": "place", "description": "city"},
            ],
            "relations": [],
        })
        result = _parse_extraction(raw)
        assert len(result["entities"]) == 2
        assert result["entities"][0]["name"] == "Alice"
        assert result["entities"][1]["name"] == "Jakarta"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the extraction:\n{"entities": [{"name": "Bob", "type": "person", "description": ""}], "relations": []}\nDone.'
        result = _parse_extraction(raw)
        assert result is not None
        assert len(result["entities"]) == 1


# ── extract_and_store (async, needs mocks) ───────────────────


class TestExtractAndStore:
    @pytest.fixture
    def mock_configs(self):
        configs = {
            "graph.enabled": True,
            "graph.extractor_driver": "provider",
        }

        async def _get(key, default=None):
            return configs.get(key, default)

        return _get, configs

    async def test_disabled_graph(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        configs["graph.enabled"] = False

        with patch("syne.memory.graph.get_config", side_effect=get_config):
            result = await extract_and_store(mock_provider, "test content", 1)
        assert result is False

    async def test_provider_extraction_success(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        extraction = {
            "entities": [{"name": "Alice", "type": "person", "description": ""}],
            "relations": [{"subject": "Alice", "predicate": "lives_in", "object": "Jakarta"}],
        }
        mock_provider.chat.return_value = MagicMock(content=json.dumps(extraction))

        with patch("syne.memory.graph.get_config", side_effect=get_config), \
             patch("syne.memory.graph._store_graph", new_callable=AsyncMock) as mock_store:
            result = await extract_and_store(mock_provider, "Alice lives in Jakarta", 1)

        assert result is True
        mock_store.assert_called_once()

    async def test_empty_extraction(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        mock_provider.chat.return_value = MagicMock(
            content=json.dumps({"entities": [], "relations": []})
        )

        with patch("syne.memory.graph.get_config", side_effect=get_config), \
             patch("syne.memory.graph._store_graph", new_callable=AsyncMock) as mock_store:
            result = await extract_and_store(mock_provider, "nothing here", 1)

        assert result is False
        mock_store.assert_not_called()

    async def test_provider_exception(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        mock_provider.chat.side_effect = Exception("API error")

        with patch("syne.memory.graph.get_config", side_effect=get_config):
            result = await extract_and_store(mock_provider, "test", 1)

        assert result is False

    async def test_speaker_name_passed(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        mock_provider.chat.return_value = MagicMock(
            content=json.dumps({"entities": [], "relations": []})
        )

        with patch("syne.memory.graph.get_config", side_effect=get_config):
            await extract_and_store(mock_provider, "test", 1, speaker_name="Riyo")

        call_args = mock_provider.chat.call_args
        user_msg = call_args[0][0][1].content  # second message (user)
        assert "The speaker is Riyo" in user_msg

    async def test_no_speaker_name(self, mock_provider, mock_configs):
        get_config, configs = mock_configs
        mock_provider.chat.return_value = MagicMock(
            content=json.dumps({"entities": [], "relations": []})
        )

        with patch("syne.memory.graph.get_config", side_effect=get_config):
            await extract_and_store(mock_provider, "test", 1, speaker_name="")

        call_args = mock_provider.chat.call_args
        user_msg = call_args[0][0][1].content
        assert "The speaker is" not in user_msg
