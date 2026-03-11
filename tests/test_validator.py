"""Tests for syne.abilities.validator module."""

import pytest

from syne.abilities.validator import (
    validate_syntax,
    validate_structure,
    validate_tool_schema,
    _validate_function_def,
)


class TestValidateSyntax:
    """Tests for validate_syntax()."""

    def test_valid_python(self):
        code = "x = 1\ndef foo():\n    return x\n"
        ok, err = validate_syntax(code)
        assert ok is True
        assert err == ""

    def test_invalid_python(self):
        code = "def foo(\n"
        ok, err = validate_syntax(code)
        assert ok is False
        assert "Syntax error" in err

    def test_empty_string(self):
        ok, err = validate_syntax("")
        assert ok is True
        assert err == ""

    def test_source_label_in_error(self):
        code = "def broken("
        ok, err = validate_syntax(code, source_label="my_ability.py")
        assert ok is False
        assert "my_ability.py" in err


class TestValidateStructure:
    """Tests for validate_structure()."""

    def test_proper_ability_subclass(self):
        code = (
            "class MyAbility(Ability):\n"
            "    name = 'test'\n"
            "    description = 'A test ability'\n"
            "    def execute(self): pass\n"
            "    def get_schema(self): pass\n"
            "    def get_guide(self): pass\n"
        )
        ok, err = validate_structure(code)
        assert ok is True
        assert err == ""

    def test_no_ability_class(self):
        code = "class NotAnAbility:\n    pass\n"
        ok, err = validate_structure(code)
        assert ok is False
        assert "No Ability subclass found" in err

    def test_missing_required_methods(self):
        code = (
            "class MyAbility(Ability):\n"
            "    name = 'test'\n"
            "    description = 'A test ability'\n"
        )
        ok, err = validate_structure(code)
        assert ok is False
        assert "missing 'execute()' method" in err
        assert "missing 'get_schema()' method" in err
        assert "missing 'get_guide()' method" in err

    def test_missing_name_attribute(self):
        code = (
            "class MyAbility(Ability):\n"
            "    description = 'A test ability'\n"
            "    def execute(self): pass\n"
            "    def get_schema(self): pass\n"
            "    def get_guide(self): pass\n"
        )
        ok, err = validate_structure(code)
        assert ok is False
        assert "missing 'name' attribute" in err

    def test_missing_description_attribute(self):
        code = (
            "class MyAbility(Ability):\n"
            "    name = 'test'\n"
            "    def execute(self): pass\n"
            "    def get_schema(self): pass\n"
            "    def get_guide(self): pass\n"
        )
        ok, err = validate_structure(code)
        assert ok is False
        assert "missing 'description' attribute" in err

    def test_annotated_assignments(self):
        code = (
            "class MyAbility(Ability):\n"
            "    name: str = 'test'\n"
            "    description: str = 'A test ability'\n"
            "    def execute(self): pass\n"
            "    def get_schema(self): pass\n"
            "    def get_guide(self): pass\n"
        )
        ok, err = validate_structure(code)
        assert ok is True
        assert err == ""

    def test_syntax_error_in_code(self):
        code = "class MyAbility(Ability:\n"
        ok, err = validate_structure(code)
        assert ok is False
        assert "Cannot parse" in err

    def test_base_class_case_insensitive(self):
        """Base class matching uses case-insensitive check on 'ability'."""
        code = (
            "class MyThing(BaseAbility):\n"
            "    name = 'test'\n"
            "    description = 'desc'\n"
            "    def execute(self): pass\n"
            "    def get_schema(self): pass\n"
            "    def get_guide(self): pass\n"
        )
        ok, err = validate_structure(code)
        assert ok is True


class TestValidateToolSchema:
    """Tests for validate_tool_schema()."""

    def test_valid_standard_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "do_thing",
                "description": "Does a thing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"},
                    },
                },
            },
        }
        ok, err = validate_tool_schema(schema)
        assert ok is True
        assert err == ""

    def test_invalid_schema_wrong_type(self):
        schema = {"type": "not_function", "function": {"name": "x"}}
        ok, err = validate_tool_schema(schema)
        assert ok is False
        assert "unsupported type" in err

    def test_empty_schema(self):
        ok, err = validate_tool_schema({})
        assert ok is False
        assert "Empty schema" in err

    def test_none_schema(self):
        ok, err = validate_tool_schema(None)
        assert ok is False
        assert "Empty schema" in err

    def test_flat_format(self):
        """Flat format: {"name": ..., "parameters": ...} without "type"."""
        schema = {
            "name": "do_thing",
            "description": "Does a thing",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
        ok, err = validate_tool_schema(schema)
        assert ok is True
        assert err == ""

    def test_type_none(self):
        schema = {"type": None}
        ok, err = validate_tool_schema(schema)
        assert ok is False
        assert "type=None" in err

    def test_missing_function_dict(self):
        schema = {"type": "function"}
        ok, err = validate_tool_schema(schema)
        assert ok is False
        assert "missing 'function' dict" in err

    def test_non_dict_schema(self):
        ok, err = validate_tool_schema("not a dict")
        assert ok is False
        assert "not a dict" in err


class TestValidateFunctionDef:
    """Tests for _validate_function_def()."""

    def test_valid_function_def(self):
        fn = {
            "name": "my_func",
            "description": "A function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
            },
        }
        ok, err = _validate_function_def(fn, "test")
        assert ok is True
        assert err == ""

    def test_missing_name(self):
        fn = {"parameters": {"type": "object"}}
        ok, err = _validate_function_def(fn, "test")
        assert ok is False
        assert "missing or empty 'name'" in err

    def test_empty_name(self):
        fn = {"name": "", "parameters": {"type": "object"}}
        ok, err = _validate_function_def(fn, "test")
        assert ok is False
        assert "missing or empty 'name'" in err

    def test_missing_parameters_is_ok(self):
        """Parameters are optional (parameters=None means no params)."""
        fn = {"name": "my_func", "description": "desc"}
        ok, err = _validate_function_def(fn, "test")
        assert ok is True

    def test_invalid_parameters_type(self):
        fn = {
            "name": "my_func",
            "parameters": {"type": "array"},
        }
        ok, err = _validate_function_def(fn, "test")
        assert ok is False
        assert "parameters.type='array'" in err

    def test_none_property_value(self):
        fn = {
            "name": "my_func",
            "parameters": {
                "type": "object",
                "properties": {"bad_prop": None},
            },
        }
        ok, err = _validate_function_def(fn, "test")
        assert ok is False
        assert "property 'bad_prop' is None" in err

    def test_property_with_none_type(self):
        fn = {
            "name": "my_func",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": None}},
            },
        }
        ok, err = _validate_function_def(fn, "test")
        assert ok is False
        assert "property 'x' has type=None" in err
