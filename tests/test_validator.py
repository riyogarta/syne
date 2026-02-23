"""Tests for ability validation."""

import pytest
from syne.abilities.validator import (
    validate_syntax,
    validate_structure,
    validate_tool_schema,
    validate_ability_file,
)


# === Syntax validation ===

class TestValidateSyntax:
    def test_valid_python(self):
        ok, err = validate_syntax("x = 1\nprint(x)")
        assert ok
        assert err == ""

    def test_syntax_error(self):
        ok, err = validate_syntax("def foo(\n  pass")
        assert not ok
        assert "Syntax error" in err

    def test_empty_code(self):
        ok, err = validate_syntax("")
        assert ok  # Empty file is valid Python

    def test_indentation_error(self):
        ok, err = validate_syntax("def foo():\npass")
        assert not ok
        assert "Syntax error" in err


# === Structure validation ===

VALID_ABILITY = '''
from syne.abilities.base import Ability

class TestAbility(Ability):
    name = "test"
    description = "A test ability"
    
    async def execute(self, params, context):
        return {"success": True}
    
    def get_schema(self):
        return {"type": "function", "function": {"name": "test", "parameters": {"type": "object", "properties": {}}}}
'''

MISSING_NAME = '''
from syne.abilities.base import Ability

class TestAbility(Ability):
    description = "A test ability"
    
    async def execute(self, params, context):
        return {"success": True}
    
    def get_schema(self):
        return {}
'''

MISSING_EXECUTE = '''
from syne.abilities.base import Ability

class TestAbility(Ability):
    name = "test"
    description = "A test ability"
    
    def get_schema(self):
        return {}
'''

NO_ABILITY_CLASS = '''
class NotAnAbility:
    name = "test"
    
    def execute(self):
        pass
'''

ANNOTATED_ATTRS = '''
from syne.abilities.base import Ability

class TestAbility(Ability):
    name: str = "test"
    description: str = "A test ability"
    
    async def execute(self, params, context):
        return {"success": True}
    
    def get_schema(self):
        return {}
'''


class TestValidateStructure:
    def test_valid_ability(self):
        ok, err = validate_structure(VALID_ABILITY)
        assert ok, err

    def test_missing_name(self):
        ok, err = validate_structure(MISSING_NAME)
        assert not ok
        assert "missing 'name'" in err

    def test_missing_execute(self):
        ok, err = validate_structure(MISSING_EXECUTE)
        assert not ok
        assert "missing 'execute()'" in err

    def test_no_ability_class(self):
        ok, err = validate_structure(NO_ABILITY_CLASS)
        assert not ok
        assert "No Ability subclass" in err

    def test_annotated_attributes(self):
        ok, err = validate_structure(ANNOTATED_ATTRS)
        assert ok, err


# === Tool schema validation ===

class TestValidateToolSchema:
    def test_valid_standard_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert ok, err

    def test_valid_flat_schema(self):
        schema = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        }
        ok, err = validate_tool_schema(schema, "test")
        assert ok, err

    def test_type_none(self):
        """This is the Codex 400 bug — type=None."""
        schema = {
            "type": None,
            "function": {
                "name": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "type=None" in err

    def test_empty_schema(self):
        ok, err = validate_tool_schema({}, "test")
        assert not ok
        assert "Empty schema" in err

    def test_none_schema(self):
        ok, err = validate_tool_schema(None, "test")
        assert not ok

    def test_missing_function_name(self):
        schema = {
            "type": "function",
            "function": {
                "description": "No name",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "name" in err

    def test_property_with_none_type(self):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "broken": {"type": None, "description": "oops"},
                    },
                },
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "type=None" in err

    def test_property_value_none(self):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "broken": None,
                    },
                },
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "'broken' is None" in err

    def test_wrong_parameters_type(self):
        schema = {
            "type": "function",
            "function": {
                "name": "test",
                "parameters": {
                    "type": "array",  # wrong
                    "properties": {},
                },
            },
        }
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "type='array'" in err

    def test_missing_function_dict(self):
        schema = {"type": "function"}
        ok, err = validate_tool_schema(schema, "test")
        assert not ok
        assert "missing 'function'" in err
