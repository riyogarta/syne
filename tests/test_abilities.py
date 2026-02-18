"""Tests for the Syne Ability System."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from syne.abilities.base import Ability
from syne.abilities.registry import AbilityRegistry, RegisteredAbility


# ============================================================
# Test Fixtures
# ============================================================

class MockAbility(Ability):
    """A simple mock ability for testing."""
    
    name = "mock_ability"
    description = "A mock ability for testing"
    version = "1.0"
    
    async def execute(self, params: dict, context: dict) -> dict:
        if params.get("fail"):
            return {"success": False, "error": "Simulated failure"}
        return {
            "success": True,
            "result": f"Executed with params: {params}",
        }
    
    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "mock_ability",
                "description": "A mock ability for testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "fail": {"type": "boolean"},
                    },
                    "required": ["input"],
                },
            },
        }
    
    def get_required_config(self) -> list[str]:
        return ["API_KEY"]


class PublicMockAbility(MockAbility):
    """Mock ability with public access level."""
    name = "public_mock"
    description = "Public mock ability"
    
    def get_required_config(self) -> list[str]:
        return []
    
    def get_schema(self) -> dict:
        schema = super().get_schema()
        schema["function"]["name"] = "public_mock"
        schema["function"]["description"] = "Public mock ability"
        return schema


class AdminMockAbility(MockAbility):
    """Mock ability requiring admin access."""
    name = "admin_mock"
    description = "Admin mock ability"
    
    def get_schema(self) -> dict:
        schema = super().get_schema()
        schema["function"]["name"] = "admin_mock"
        schema["function"]["description"] = "Admin mock ability"
        return schema


@pytest.fixture
def registry():
    """Create a fresh AbilityRegistry for each test."""
    return AbilityRegistry()


@pytest.fixture
def mock_ability():
    """Create a MockAbility instance."""
    return MockAbility()


@pytest.fixture
def public_ability():
    """Create a PublicMockAbility instance."""
    return PublicMockAbility()


@pytest.fixture
def admin_ability():
    """Create an AdminMockAbility instance."""
    return AdminMockAbility()


# ============================================================
# Test Base Ability Interface
# ============================================================

class TestBaseAbility:
    """Tests for the base Ability class."""
    
    def test_ability_has_required_attributes(self, mock_ability):
        """Ability should have name, description, version."""
        assert mock_ability.name == "mock_ability"
        assert mock_ability.description == "A mock ability for testing"
        assert mock_ability.version == "1.0"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_ability):
        """Execute should return success dict."""
        result = await mock_ability.execute({"input": "test"}, {})
        assert result["success"] is True
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_ability):
        """Execute should return error on failure."""
        result = await mock_ability.execute({"fail": True}, {})
        assert result["success"] is False
        assert "error" in result
    
    def test_get_schema_returns_valid_structure(self, mock_ability):
        """Schema should be OpenAI-compatible."""
        schema = mock_ability.get_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "mock_ability"
        assert "parameters" in schema["function"]
    
    def test_get_required_config(self, mock_ability):
        """Should return list of required config keys."""
        required = mock_ability.get_required_config()
        assert isinstance(required, list)
        assert "API_KEY" in required
    
    @pytest.mark.asyncio
    async def test_validate_config_success(self, mock_ability):
        """Validate config should pass with required keys."""
        is_valid, error = await mock_ability.validate_config({"API_KEY": "test_key"})
        assert is_valid is True
        assert error == ""
    
    @pytest.mark.asyncio
    async def test_validate_config_missing_key(self, mock_ability):
        """Validate config should fail without required keys."""
        is_valid, error = await mock_ability.validate_config({})
        assert is_valid is False
        assert "API_KEY" in error


# ============================================================
# Test Ability Registry
# ============================================================

class TestAbilityRegistry:
    """Tests for the AbilityRegistry."""
    
    def test_register_ability(self, registry, mock_ability):
        """Should register an ability."""
        registry.register(mock_ability, source="bundled")
        assert registry.get("mock_ability") is not None
    
    def test_unregister_ability(self, registry, mock_ability):
        """Should unregister an ability."""
        registry.register(mock_ability)
        registry.unregister("mock_ability")
        assert registry.get("mock_ability") is None
    
    def test_get_nonexistent_ability(self, registry):
        """Get should return None for nonexistent ability."""
        assert registry.get("nonexistent") is None
    
    def test_list_all(self, registry, mock_ability, public_ability):
        """Should list all registered abilities."""
        registry.register(mock_ability)
        registry.register(public_ability)
        all_abilities = registry.list_all()
        assert len(all_abilities) == 2
    
    def test_list_enabled_filters_disabled(self, registry, mock_ability, public_ability):
        """Should filter out disabled abilities."""
        registry.register(mock_ability, enabled=True)
        registry.register(public_ability, enabled=False)
        enabled = registry.list_enabled("owner")
        assert len(enabled) == 1
        assert enabled[0].name == "mock_ability"
    
    def test_list_enabled_filters_by_access_level(self, registry, public_ability, admin_ability):
        """Should filter by access level."""
        registry.register(public_ability, requires_access_level="public")
        registry.register(admin_ability, requires_access_level="admin")
        
        # Public user should only see public ability
        public_abilities = registry.list_enabled("public")
        assert len(public_abilities) == 1
        assert public_abilities[0].name == "public_mock"
        
        # Admin should see both
        admin_abilities = registry.list_enabled("admin")
        assert len(admin_abilities) == 2


class TestAbilityRegistrySchema:
    """Tests for schema generation."""
    
    def test_to_openai_schema(self, registry, mock_ability):
        """Should convert to OpenAI schema format."""
        registry.register(mock_ability)
        schemas = registry.to_openai_schema("owner")
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_ability"
    
    def test_to_openai_schema_filters_by_access(self, registry, public_ability, admin_ability):
        """Should filter schemas by access level."""
        registry.register(public_ability, requires_access_level="public")
        registry.register(admin_ability, requires_access_level="admin")
        
        public_schemas = registry.to_openai_schema("public")
        assert len(public_schemas) == 1
        assert public_schemas[0]["function"]["name"] == "public_mock"


class TestAbilityRegistryExecution:
    """Tests for ability execution."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, registry, public_ability):
        """Should execute ability and return result."""
        registry.register(public_ability, requires_access_level="public")
        result = await registry.execute(
            "public_mock",
            {"input": "test"},
            {"access_level": "public"},
        )
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_not_found(self, registry):
        """Should return error for nonexistent ability."""
        result = await registry.execute(
            "nonexistent",
            {},
            {"access_level": "owner"},
        )
        assert result["success"] is False
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_disabled(self, registry, mock_ability):
        """Should return error for disabled ability."""
        registry.register(mock_ability, enabled=False)
        result = await registry.execute(
            "mock_ability",
            {},
            {"access_level": "owner"},
        )
        assert result["success"] is False
        assert "disabled" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_insufficient_access(self, registry, admin_ability):
        """Should return error for insufficient access."""
        registry.register(admin_ability, requires_access_level="admin")
        result = await registry.execute(
            "admin_mock",
            {},
            {"access_level": "public"},
        )
        assert result["success"] is False
        assert "Insufficient permissions" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_config_validation_failure(self, registry, mock_ability):
        """Should return error if config validation fails."""
        registry.register(mock_ability, config={})  # Missing API_KEY
        result = await registry.execute(
            "mock_ability",
            {},
            {"access_level": "owner"},
        )
        assert result["success"] is False
        assert "API_KEY" in result["error"]


class TestAbilityRegistryEnableDisable:
    """Tests for enable/disable functionality."""
    
    @pytest.mark.asyncio
    async def test_enable_ability(self, registry, mock_ability):
        """Should enable a disabled ability."""
        registry.register(mock_ability, enabled=False)
        
        # Mock DB update
        with patch.object(registry, '_update_db_enabled', new_callable=AsyncMock):
            success = await registry.enable("mock_ability")
        
        assert success is True
        assert registry.get("mock_ability").enabled is True
    
    @pytest.mark.asyncio
    async def test_disable_ability(self, registry, mock_ability):
        """Should disable an enabled ability."""
        registry.register(mock_ability, enabled=True)
        
        with patch.object(registry, '_update_db_enabled', new_callable=AsyncMock):
            success = await registry.disable("mock_ability")
        
        assert success is True
        assert registry.get("mock_ability").enabled is False
    
    @pytest.mark.asyncio
    async def test_enable_nonexistent(self, registry):
        """Should return False for nonexistent ability."""
        success = await registry.enable("nonexistent")
        assert success is False


# ============================================================
# Test Bundled Abilities (Import Only)
# ============================================================

class TestBundledAbilitiesExist:
    """Verify bundled abilities can be imported."""
    
    def test_image_gen_ability_exists(self):
        """ImageGenAbility should be importable."""
        from syne.abilities.image_gen import ImageGenAbility
        ability = ImageGenAbility()
        assert ability.name == "image_gen"
        assert "TOGETHER_API_KEY" not in ability.get_required_config()  # Falls back to env
    
    def test_image_analysis_ability_exists(self):
        """ImageAnalysisAbility should be importable."""
        from syne.abilities.image_analysis import ImageAnalysisAbility
        ability = ImageAnalysisAbility()
        assert ability.name == "image_analysis"
    
    def test_all_bundled_abilities_have_valid_schemas(self):
        """All bundled abilities should have valid OpenAI schemas."""
        from syne.abilities.image_gen import ImageGenAbility
        from syne.abilities.image_analysis import ImageAnalysisAbility
        
        for ability_cls in [ImageGenAbility, ImageAnalysisAbility]:
            ability = ability_cls()
            schema = ability.get_schema()
            
            assert schema["type"] == "function"
            assert "function" in schema
            assert schema["function"]["name"] == ability.name
            assert "parameters" in schema["function"]
            assert schema["function"]["parameters"]["type"] == "object"


# ============================================================
# Test Loader
# ============================================================

class TestLoader:
    """Tests for the ability loader."""
    
    def test_load_bundled_abilities(self):
        """Should load all bundled abilities."""
        from syne.abilities.loader import load_bundled_abilities
        
        registry = AbilityRegistry()
        count = load_bundled_abilities(registry)
        
        assert count >= 3  # image_gen, image_analysis, maps
        assert registry.get("image_gen") is not None
        assert registry.get("image_analysis") is not None
        assert registry.get("maps") is not None
    
    def test_get_bundled_ability_classes(self):
        """Should return list of ability classes."""
        from syne.abilities.loader import get_bundled_ability_classes
        
        classes = get_bundled_ability_classes()
        assert len(classes) >= 3  # image_gen, image_analysis, maps
        
        # All should be Ability subclasses
        for cls in classes:
            assert issubclass(cls, Ability)
