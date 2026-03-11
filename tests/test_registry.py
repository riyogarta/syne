"""Tests for syne.abilities.registry module."""

import pytest
from unittest.mock import MagicMock

from syne.abilities.registry import AbilityRegistry, RegisteredAbility


def _make_mock_ability(name="test_ability", description="A test ability", version="1.0.0"):
    """Create a mock Ability instance with required attributes."""
    mock = MagicMock()
    mock.name = name
    mock.description = description
    mock.version = version
    return mock


class TestAbilityRegistryInit:
    """Tests for AbilityRegistry.__init__()."""

    def test_empty_registry(self):
        registry = AbilityRegistry()
        assert registry.list_all() == []

    def test_internal_dict_exists(self):
        registry = AbilityRegistry()
        assert isinstance(registry._abilities, dict)
        assert len(registry._abilities) == 0


class TestRegister:
    """Tests for AbilityRegistry.register()."""

    def test_register_mock_ability(self):
        registry = AbilityRegistry()
        ability = _make_mock_ability()
        registry.register(ability)
        assert "test_ability" in registry._abilities

    def test_registered_ability_fields(self):
        registry = AbilityRegistry()
        ability = _make_mock_ability(name="calc", description="Calculator", version="2.0")
        registry.register(ability, source="installed", module_path="syne.abilities.calc")

        reg = registry._abilities["calc"]
        assert isinstance(reg, RegisteredAbility)
        assert reg.name == "calc"
        assert reg.description == "Calculator"
        assert reg.version == "2.0"
        assert reg.source == "installed"
        assert reg.module_path == "syne.abilities.calc"
        assert reg.instance is ability
        assert reg.enabled is True

    def test_register_with_config(self):
        registry = AbilityRegistry()
        ability = _make_mock_ability()
        registry.register(ability, config={"key": "value"})
        assert registry._abilities["test_ability"].config == {"key": "value"}

    def test_register_overwrites_existing(self):
        registry = AbilityRegistry()
        ability1 = _make_mock_ability(description="first")
        ability2 = _make_mock_ability(description="second")
        registry.register(ability1)
        registry.register(ability2)
        assert registry._abilities["test_ability"].description == "second"
        assert len(registry._abilities) == 1


class TestUnregister:
    """Tests for AbilityRegistry.unregister()."""

    def test_unregister_existing(self):
        registry = AbilityRegistry()
        ability = _make_mock_ability()
        registry.register(ability)
        registry.unregister("test_ability")
        assert "test_ability" not in registry._abilities
        assert registry.list_all() == []

    def test_unregister_nonexistent(self):
        """Unregistering a missing name should not raise."""
        registry = AbilityRegistry()
        registry.unregister("nonexistent")  # Should not raise


class TestGet:
    """Tests for AbilityRegistry.get()."""

    def test_get_found(self):
        registry = AbilityRegistry()
        ability = _make_mock_ability(name="search")
        registry.register(ability)
        result = registry.get("search")
        assert result is not None
        assert result.name == "search"

    def test_get_not_found(self):
        registry = AbilityRegistry()
        result = registry.get("nonexistent")
        assert result is None


class TestListAll:
    """Tests for AbilityRegistry.list_all()."""

    def test_empty(self):
        registry = AbilityRegistry()
        assert registry.list_all() == []

    def test_with_abilities(self):
        registry = AbilityRegistry()
        registry.register(_make_mock_ability(name="a"))
        registry.register(_make_mock_ability(name="b"))
        registry.register(_make_mock_ability(name="c"))
        result = registry.list_all()
        assert len(result) == 3
        names = {r.name for r in result}
        assert names == {"a", "b", "c"}

    def test_returns_list_of_registered_ability(self):
        registry = AbilityRegistry()
        registry.register(_make_mock_ability())
        result = registry.list_all()
        assert all(isinstance(r, RegisteredAbility) for r in result)
