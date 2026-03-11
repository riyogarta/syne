"""Tests for syne.abilities.loader — bundled loading, dynamic loading, registration."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from syne.abilities.base import Ability
from syne.abilities.registry import AbilityRegistry
from syne.abilities.loader import (
    load_bundled_abilities,
    load_dynamic_ability,
    load_dynamic_ability_safe,
    _resolve_module_to_filepath,
)


# ── Test Ability for dynamic loading ─────────────────────────────────


class _TestAbility(Ability):
    name = "test_dynamic"
    description = "A test ability"
    version = "1.0"

    async def execute(self, params, context):
        return {"success": True, "result": "test"}

    def get_guide(self, enabled, config):
        return "Test guide"

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "test_dynamic",
                "description": "Test",
                "parameters": {"type": "object", "properties": {}},
            },
        }


# ── load_bundled_abilities ───────────────────────────────────────────


class TestLoadBundledAbilities:
    def test_loads_into_registry(self):
        registry = AbilityRegistry()
        count = load_bundled_abilities(registry)
        assert count >= 0  # May vary based on available bundled abilities
        assert len(registry.list_all()) == count

    def test_all_have_names(self):
        registry = AbilityRegistry()
        load_bundled_abilities(registry)
        for ability in registry.list_all():
            assert ability.name
            assert ability.source == "bundled"

    def test_all_disabled_by_default(self):
        """Bundled abilities are registered as disabled until DB sync enables them."""
        registry = AbilityRegistry()
        load_bundled_abilities(registry)
        for ability in registry.list_all():
            assert ability.enabled is False


# ── load_dynamic_ability ─────────────────────────────────────────────


class TestLoadDynamicAbility:
    def test_load_nonexistent_returns_none(self):
        result = load_dynamic_ability("syne.abilities.nonexistent_xyz_test")
        assert result is None

    def test_load_module_without_ability_returns_none(self):
        # json module has no Ability subclass
        result = load_dynamic_ability("json")
        assert result is None


class TestLoadDynamicAbilitySafe:
    def test_returns_error_on_failure(self):
        ability, err = load_dynamic_ability_safe("syne.abilities.nonexistent_xyz")
        assert ability is None
        assert err is not None
        assert isinstance(err, str)

    def test_returns_error_for_no_ability_class(self):
        ability, err = load_dynamic_ability_safe("json")
        assert ability is None
        assert "No Ability subclass" in err


# ── _resolve_module_to_filepath ──────────────────────────────────────


class TestResolveModuleToFilepath:
    def test_file_path_passthrough(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"# test")
            path = f.name
        try:
            result = _resolve_module_to_filepath(path)
            assert result == path
        finally:
            os.unlink(path)

    def test_nonexistent_file_path(self):
        result = _resolve_module_to_filepath("/tmp/nonexistent_ability_xyz.py")
        assert result is None

    def test_dotted_path_to_real_module(self):
        # syne.abilities.base should resolve to the actual file
        result = _resolve_module_to_filepath("syne.abilities.base")
        assert result is not None
        assert result.endswith("base.py")

    def test_dotted_path_nonexistent(self):
        result = _resolve_module_to_filepath("syne.abilities.nonexistent_xyz")
        assert result is None


# ── AbilityRegistry execute (integration) ────────────────────────────


class TestAbilityRegistryExecute:
    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        registry = AbilityRegistry()
        result = await registry.execute("nonexistent", {}, {})
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_disabled(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "disabled_one"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        registry.register(mock_ability, enabled=False)

        result = await registry.execute("disabled_one", {}, {})
        assert result["success"] is False
        assert "disabled" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_permission_denied(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "restricted"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        registry.register(mock_ability, enabled=True, permission=0o700)

        result = await registry.execute("restricted", {}, {"access_level": "public"})
        assert result["success"] is False
        assert "permission" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "good"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.validate_config = AsyncMock(return_value=(True, ""))
        mock_ability.execute = AsyncMock(return_value={"success": True, "result": "ok"})
        registry.register(mock_ability, enabled=True, permission=0o700)

        result = await registry.execute("good", {}, {"access_level": "owner"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_config_invalid(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "badconfig"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.validate_config = AsyncMock(return_value=(False, "Missing API key"))
        registry.register(mock_ability, enabled=True, permission=0o700)

        result = await registry.execute("badconfig", {}, {"access_level": "owner"})
        assert result["success"] is False
        assert "API key" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "slow"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.validate_config = AsyncMock(return_value=(True, ""))

        async def slow_execute(params, context):
            import asyncio
            await asyncio.sleep(999)

        mock_ability.execute = slow_execute
        registry.register(mock_ability, enabled=True, permission=0o700)

        # Patch timeout to be very short
        with patch("syne.abilities.registry.EXECUTE_TIMEOUT", 0.01):
            result = await registry.execute("slow", {}, {"access_level": "owner"})
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracked(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "failing"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.validate_config = AsyncMock(return_value=(True, ""))
        mock_ability.execute = AsyncMock(side_effect=RuntimeError("boom"))
        registry.register(mock_ability, enabled=True, permission=0o700)

        result = await registry.execute("failing", {}, {"access_level": "owner"})
        assert result["success"] is False
        reg = registry.get("failing")
        assert reg.consecutive_failures == 1


# ── AbilityRegistry enable/disable ───────────────────────────────────


class TestAbilityRegistryEnableDisable:
    @pytest.mark.asyncio
    async def test_enable_not_found(self):
        registry = AbilityRegistry()
        ok, msg = await registry.enable("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_disable_not_found(self):
        registry = AbilityRegistry()
        result = await registry.disable("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_enable_calls_ensure_dependencies(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "dep_check"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.ensure_dependencies = AsyncMock(return_value=(True, "installed chromium"))
        registry.register(mock_ability, enabled=False, permission=0o700)

        ok, msg = await registry.enable("dep_check")
        assert ok is True
        assert "enabled" in msg.lower()
        mock_ability.ensure_dependencies.assert_called_once()

    @pytest.mark.asyncio
    async def test_enable_fails_on_dep_failure(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "bad_deps"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        mock_ability.ensure_dependencies = AsyncMock(return_value=(False, "chromium not found"))
        registry.register(mock_ability, enabled=False, permission=0o700)

        ok, msg = await registry.enable("bad_deps")
        assert ok is False
        assert "chromium" in msg

    @pytest.mark.asyncio
    async def test_disable_sets_false(self):
        registry = AbilityRegistry()
        mock_ability = MagicMock()
        mock_ability.name = "todisable"
        mock_ability.description = "test"
        mock_ability.version = "1.0"
        registry.register(mock_ability, enabled=True, permission=0o700)

        result = await registry.disable("todisable")
        assert result is True
        assert registry.get("todisable").enabled is False


# ── Ability base class ───────────────────────────────────────────────


class TestAbilityBase:
    def test_output_dir(self):
        ability = _TestAbility()
        outdir = ability.get_output_dir()
        assert "workspace/outputs" in outdir
        assert os.path.isdir(outdir)

    def test_output_dir_with_session(self):
        ability = _TestAbility()
        outdir = ability.get_output_dir(session_id="123")
        assert "session_123" in outdir

    def test_handles_input_type_default_false(self):
        ability = _TestAbility()
        assert ability.handles_input_type("image") is False

    @pytest.mark.asyncio
    async def test_pre_process_default_none(self):
        ability = _TestAbility()
        result = await ability.pre_process("image", {}, "test")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_config_no_required(self):
        ability = _TestAbility()
        ok, err = await ability.validate_config({})
        assert ok is True

    @pytest.mark.asyncio
    async def test_ensure_dependencies_default(self):
        ability = _TestAbility()
        ok, msg = await ability.ensure_dependencies()
        assert ok is True


# ── AbilityRegistry schema generation ────────────────────────────────


class TestAbilitySchema:
    def test_list_enabled_filters_by_access(self):
        registry = AbilityRegistry()
        mock = MagicMock()
        mock.name = "owner_only"
        mock.description = "test"
        mock.version = "1.0"
        registry.register(mock, enabled=True, permission=0o700)

        # Owner can see it
        assert len(registry.list_enabled("owner")) == 1
        # Public cannot
        assert len(registry.list_enabled("public")) == 0
        # Blocked sees nothing
        assert len(registry.list_enabled("blocked")) == 0

    def test_list_enabled_skips_disabled(self):
        registry = AbilityRegistry()
        mock = MagicMock()
        mock.name = "off"
        mock.description = "test"
        mock.version = "1.0"
        registry.register(mock, enabled=False, permission=0o777)

        assert len(registry.list_enabled("owner")) == 0
