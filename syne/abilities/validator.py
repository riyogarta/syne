"""Ability Validator — syntax, structure, and schema validation before registration."""

import ast
import logging
from typing import Optional

from .base import Ability

logger = logging.getLogger("syne.abilities.validator")


class AbilityValidationError(Exception):
    """Raised when ability code or schema fails validation."""
    pass


def validate_syntax(code: str, source_label: str = "<ability>") -> tuple[bool, str]:
    """Parse Python source to catch syntax errors.
    
    Args:
        code: Python source code string
        source_label: Label for error messages (filename or ability name)
        
    Returns:
        (True, "") on success, (False, error_message) on failure
    """
    try:
        ast.parse(code, filename=source_label)
        return True, ""
    except SyntaxError as e:
        msg = f"Syntax error in {source_label} at line {e.lineno}: {e.msg}"
        logger.error(msg)
        return False, msg


def validate_structure(code: str, source_label: str = "<ability>") -> tuple[bool, str]:
    """Validate that code contains a proper Ability subclass.
    
    Checks:
    - At least one class inherits from Ability (or BaseAbility/Ability pattern)
    - Class has required attributes: name, description
    - Class has required methods: execute, get_schema
    
    Args:
        code: Python source code string
        source_label: Label for error messages
        
    Returns:
        (True, "") on success, (False, error_message) on failure
    """
    try:
        tree = ast.parse(code, filename=source_label)
    except SyntaxError as e:
        return False, f"Cannot parse {source_label}: {e.msg}"
    
    # Find classes that look like Ability subclasses
    ability_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if any base class looks like Ability
            for base in node.bases:
                base_name = _get_name(base)
                if base_name and "ability" in base_name.lower():
                    ability_classes.append(node)
                    break
    
    if not ability_classes:
        return False, (
            f"No Ability subclass found in {source_label}. "
            "The module must contain a class that inherits from Ability."
        )
    
    # Validate each ability class
    errors = []
    for cls_node in ability_classes:
        cls_name = cls_node.name
        
        # Check for required class attributes (name, description)
        has_name = False
        has_description = False
        has_execute = False
        has_get_schema = False
        
        for item in cls_node.body:
            # Class-level assignments: name = "...", description = "..."
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "name":
                            has_name = True
                        elif target.id == "description":
                            has_description = True
            
            # Annotated assignments: name: str = "..."
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == "name" and item.value is not None:
                    has_name = True
                elif item.target.id == "description" and item.value is not None:
                    has_description = True
            
            # Methods
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "execute":
                    has_execute = True
                elif item.name == "get_schema":
                    has_get_schema = True
        
        if not has_name:
            errors.append(f"Class '{cls_name}' missing 'name' attribute")
        if not has_description:
            errors.append(f"Class '{cls_name}' missing 'description' attribute")
        if not has_execute:
            errors.append(f"Class '{cls_name}' missing 'execute()' method")
        if not has_get_schema:
            errors.append(f"Class '{cls_name}' missing 'get_schema()' method")
    
    if errors:
        return False, f"Structure errors in {source_label}: " + "; ".join(errors)
    
    return True, ""


def validate_tool_schema(schema: dict, ability_name: str = "<ability>") -> tuple[bool, str]:
    """Validate that a tool schema is well-formed for LLM API consumption.
    
    Checks:
    - Schema has 'type' field (must be 'function')
    - Schema has 'function' dict with 'name' and 'parameters'
    - Parameters has 'type' = 'object'
    - No None values in required fields
    
    Also accepts flat format {"name": ..., "parameters": ...} which
    will be normalized by the registry.
    
    Args:
        schema: The tool schema dict from get_schema()
        ability_name: Name for error messages
        
    Returns:
        (True, "") on success, (False, error_message) on failure
    """
    if not schema:
        return False, f"Empty schema from ability '{ability_name}'"
    
    if not isinstance(schema, dict):
        return False, f"Schema from '{ability_name}' is not a dict: {type(schema).__name__}"
    
    # Accept flat format (will be wrapped by registry)
    if "type" not in schema and "name" in schema:
        # Flat format — validate as function def
        return _validate_function_def(schema, ability_name)
    
    # Standard format: {"type": "function", "function": {...}}
    schema_type = schema.get("type")
    if schema_type is None:
        return False, (
            f"Schema from '{ability_name}' has type=None. "
            "Must be 'function'. This will cause API errors."
        )
    
    if schema_type != "function":
        return False, (
            f"Schema from '{ability_name}' has unsupported type='{schema_type}'. "
            "Must be 'function'."
        )
    
    fn = schema.get("function")
    if not fn or not isinstance(fn, dict):
        return False, f"Schema from '{ability_name}' missing 'function' dict"
    
    return _validate_function_def(fn, ability_name)


def _validate_function_def(fn: dict, ability_name: str) -> tuple[bool, str]:
    """Validate the inner function definition of a tool schema."""
    errors = []
    
    # Name is required and must be a non-empty string
    fn_name = fn.get("name")
    if not fn_name or not isinstance(fn_name, str):
        errors.append("missing or empty 'name'")
    
    # Parameters should be a dict with type=object
    params = fn.get("parameters")
    if params is not None:
        if not isinstance(params, dict):
            errors.append(f"'parameters' is {type(params).__name__}, expected dict")
        else:
            param_type = params.get("type")
            if param_type != "object":
                errors.append(f"parameters.type='{param_type}', expected 'object'")
            
            # Check properties if present
            props = params.get("properties")
            if props is not None and not isinstance(props, dict):
                errors.append(f"parameters.properties is {type(props).__name__}, expected dict")
            
            # Check for None values in properties (common bug)
            if isinstance(props, dict):
                for prop_name, prop_def in props.items():
                    if prop_def is None:
                        errors.append(f"property '{prop_name}' is None")
                    elif isinstance(prop_def, dict):
                        prop_type = prop_def.get("type")
                        if prop_type is None:
                            errors.append(f"property '{prop_name}' has type=None")
    
    # Check for None values in top-level fields
    for key in ("name", "description"):
        if key in fn and fn[key] is None:
            errors.append(f"'{key}' is None")
    
    if errors:
        return False, f"Schema errors in '{ability_name}': " + "; ".join(errors)
    
    return True, ""


def validate_ability_instance(instance: Ability, ability_name: str = "") -> tuple[bool, str]:
    """Validate a loaded ability instance — check schema output.
    
    This is a sandboxed dry-run: instantiate, call get_schema(),
    and validate the output.
    
    Args:
        instance: An instantiated Ability object
        ability_name: Name for error messages (falls back to instance.name)
        
    Returns:
        (True, "") on success, (False, error_message) on failure
    """
    name = ability_name or getattr(instance, "name", "<unknown>")
    
    # Check required attributes exist
    if not getattr(instance, "name", None):
        return False, f"Ability instance has no 'name' attribute"
    
    if not getattr(instance, "description", None):
        return False, f"Ability '{name}' has no 'description' attribute"
    
    # Call get_schema() and validate output
    try:
        schema = instance.get_schema()
    except Exception as e:
        return False, f"Ability '{name}' get_schema() raised {type(e).__name__}: {e}"
    
    return validate_tool_schema(schema, name)


def validate_ability_file(filepath: str) -> tuple[bool, str]:
    """Full validation pipeline for an ability source file.
    
    Runs all static checks (syntax + structure) without importing.
    
    Args:
        filepath: Path to the Python source file
        
    Returns:
        (True, "") on success, (False, error_message) on first failure
    """
    try:
        with open(filepath, "r") as f:
            code = f.read()
    except (OSError, IOError) as e:
        return False, f"Cannot read ability file '{filepath}': {e}"
    
    # 1. Syntax check
    ok, err = validate_syntax(code, filepath)
    if not ok:
        return False, err
    
    # 2. Structure check
    ok, err = validate_structure(code, filepath)
    if not ok:
        return False, err
    
    return True, ""


def _get_name(node: ast.AST) -> Optional[str]:
    """Extract name string from an AST node (Name or Attribute)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None
