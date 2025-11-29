# apps/core/utils.py

"""
Core utility functions for the mediaprocessor application.

This module provides common utility functions used across
multiple apps including validation, formatting, and helpers.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from django.utils import timezone

logger = logging.getLogger(__name__)


# JSON UTILITIES

def validate_json_structure(
    data: Any,
    schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate a data structure against a simple schema.
    
    Schema format:
    {
        "field_name": {
            "type": "string" | "integer" | "float" | "boolean" | "list" | "dict",
            "required": True | False,
            "nullable": True | False,
            "min": <number>,  # for numeric types
            "max": <number>,  # for numeric types
            "min_length": <int>,  # for strings/lists
            "max_length": <int>,  # for strings/lists
            "choices": [<values>],  # for choice fields
            "pattern": "<regex>",  # for strings
        }
    }
    
    Args:
        data: Data to validate
        schema: Schema definition
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(data, dict):
        return False, [f"Expected a dictionary, got {type(data).__name__}"]
    
    for field_name, field_schema in schema.items():
        field_errors = _validate_field(data, field_name, field_schema)
        errors.extend(field_errors)
    
    return len(errors) == 0, errors


def _validate_field(
    data: Dict[str, Any],
    field_name: str,
    field_schema: Dict[str, Any]
) -> List[str]:
    """
    Validate a single field against its schema.
    
    Args:
        data: Full data dictionary
        field_name: Name of the field to validate
        field_schema: Schema for this field
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    is_required = field_schema.get("required", False)
    is_nullable = field_schema.get("nullable", False)
    field_type = field_schema.get("type", "string")
    
    # Check if field exists
    if field_name not in data:
        if is_required:
            errors.append(f"Required field '{field_name}' is missing")
        return errors
    
    value = data[field_name]
    
    # Check for null value
    if value is None:
        if not is_nullable and is_required:
            errors.append(f"Field '{field_name}' cannot be null")
        return errors
    
    # Type validation
    type_validators = {
        "string": (str, "a string"),
        "integer": ((int,), "an integer"),
        "float": ((int, float), "a number"),
        "boolean": (bool, "a boolean"),
        "list": (list, "a list"),
        "dict": (dict, "a dictionary"),
    }
    
    if field_type in type_validators:
        expected_types, type_name = type_validators[field_type]
        # Handle special case for booleans (don't allow int for bool)
        if field_type == "boolean" and isinstance(value, bool):
            pass  # Valid
        elif field_type == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field_name}' must be {type_name}, got {type(value).__name__}")
            return errors
        elif not isinstance(value, expected_types):
            errors.append(f"Field '{field_name}' must be {type_name}, got {type(value).__name__}")
            return errors
    
    # Numeric range validation
    if field_type in ("integer", "float"):
        min_val = field_schema.get("min")
        max_val = field_schema.get("max")
        
        if min_val is not None and value < min_val:
            errors.append(f"Field '{field_name}' must be at least {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            errors.append(f"Field '{field_name}' must be at most {max_val}, got {value}")
    
    # String/List length validation
    if field_type in ("string", "list"):
        min_length = field_schema.get("min_length")
        max_length = field_schema.get("max_length")
        
        if min_length is not None and len(value) < min_length:
            errors.append(
                f"Field '{field_name}' must have at least {min_length} "
                f"{'characters' if field_type == 'string' else 'items'}, got {len(value)}"
            )
        
        if max_length is not None and len(value) > max_length:
            errors.append(
                f"Field '{field_name}' must have at most {max_length} "
                f"{'characters' if field_type == 'string' else 'items'}, got {len(value)}"
            )
    
    # Choice validation
    choices = field_schema.get("choices")
    if choices is not None and value not in choices:
        errors.append(
            f"Field '{field_name}' must be one of {choices}, got '{value}'"
        )
    
    # Pattern validation for strings
    if field_type == "string":
        pattern = field_schema.get("pattern")
        if pattern is not None:
            if not re.match(pattern, value):
                errors.append(
                    f"Field '{field_name}' does not match required pattern"
                )
    
    return errors


def safe_json_loads(
    json_string: str,
    default: Any = None
) -> Tuple[Any, Optional[str]]:
    """
    Safely parse a JSON string.
    
    Args:
        json_string: String to parse
        default: Default value if parsing fails
        
    Returns:
        Tuple of (parsed_data, error_message)
        error_message is None if successful
    """
    if not json_string:
        return default, "Empty JSON string"
    
    try:
        return json.loads(json_string), None
    except json.JSONDecodeError as e:
        return default, f"Invalid JSON: {str(e)}"


def safe_json_dumps(
    data: Any,
    default: str = "{}"
) -> Tuple[str, Optional[str]]:
    """
    Safely serialize data to JSON string.
    
    Args:
        data: Data to serialize
        default: Default string if serialization fails
        
    Returns:
        Tuple of (json_string, error_message)
        error_message is None if successful
    """
    try:
        return json.dumps(data), None
    except (TypeError, ValueError) as e:
        return default, f"JSON serialization failed: {str(e)}"


# UUID UTILITIES

def is_valid_uuid(value: Any) -> bool:
    """
    Check if a value is a valid UUID.
    
    Args:
        value: Value to check
        
    Returns:
        True if valid UUID, False otherwise
    """
    if value is None:
        return False
    
    if isinstance(value, UUID):
        return True
    
    try:
        UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def parse_uuid(
    value: Any,
    field_name: str = "id"
) -> Tuple[Optional[UUID], Optional[str]]:
    """
    Parse a value as a UUID.
    
    Args:
        value: Value to parse
        field_name: Field name for error messages
        
    Returns:
        Tuple of (uuid_object, error_message)
        error_message is None if successful
    """
    if value is None:
        return None, f"{field_name} is required"
    
    if isinstance(value, UUID):
        return value, None
    
    try:
        return UUID(str(value)), None
    except (ValueError, AttributeError):
        return None, f"Invalid UUID format for {field_name}: '{value}'"


# STRING UTILITIES

def truncate_string(
    text: str,
    max_length: int,
    suffix: str = "..."
) -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if not text:
        return text
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def sanitize_string(
    text: str,
    allowed_chars: Optional[str] = None,
    replacement: str = "_"
) -> str:
    """
    Sanitize a string by removing/replacing invalid characters.
    
    Args:
        text: Text to sanitize
        allowed_chars: Regex pattern for allowed characters
        replacement: Character to replace invalid chars with
        
    Returns:
        Sanitized string
    """
    if not text:
        return text
    
    if allowed_chars is None:
        # Default: alphanumeric, underscore, hyphen
        allowed_chars = r'[^a-zA-Z0-9_\-]'
    
    return re.sub(allowed_chars, replacement, text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in a string.
    
    - Strips leading/trailing whitespace
    - Collapses multiple spaces into one
    - Replaces tabs and newlines with spaces
    - Removes control characters (except whitespace)
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized string
    """
    if not text:
        return text
    
    # Remove control characters (except tab, newline, carriage return which are whitespace)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Collapse all whitespace (spaces, tabs, newlines) into single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# DATE/TIME UTILITIES

def get_expiration_datetime(days: int = 7) -> datetime:
    """
    Get a datetime for expiration in the future.
    
    Args:
        days: Number of days until expiration
        
    Returns:
        Timezone-aware datetime
    """
    return timezone.now() + timedelta(days=days)


def is_expired(expiration_datetime: Optional[datetime]) -> bool:
    """
    Check if a datetime has expired.
    
    Args:
        expiration_datetime: Datetime to check
        
    Returns:
        True if expired or None, False otherwise
    """
    if expiration_datetime is None:
        return False
    
    return timezone.now() >= expiration_datetime


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 30m 15s"
    """
    if seconds < 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string like "1.5 MB"
    """
    if size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    
    return f"{size:.2f} {units[unit_index]}"


# LIST/DICT UTILITIES

def get_nested_value(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Get a nested value from a dictionary using dot notation.
    
    Args:
        data: Dictionary to search
        path: Path like "key1.key2.key3"
        default: Default value if not found
        separator: Path separator
        
    Returns:
        Found value or default
    """
    if not data or not path:
        return default
    
    keys = path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(
    data: Dict[str, Any],
    path: str,
    value: Any,
    separator: str = "."
) -> Dict[str, Any]:
    """
    Set a nested value in a dictionary using dot notation.
    
    Args:
        data: Dictionary to modify
        path: Path like "key1.key2.key3"
        value: Value to set
        separator: Path separator
        
    Returns:
        Modified dictionary
    """
    if not path:
        return data
    
    keys = path.split(separator)
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data


def merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any],
    deep: bool = True
) -> Dict[str, Any]:
    """
    Merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge on top
        deep: Whether to merge nested dictionaries
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result


# VALIDATION HELPERS

def validate_email_format(email: str) -> bool:
    """
    Validate email format using a simple regex.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid format
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url_format(url: str) -> bool:
    """
    Validate URL format using a simple regex.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid format
    """
    if not url:
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))


def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """
    Clamp a value to a range.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))