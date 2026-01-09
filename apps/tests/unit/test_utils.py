"""
Unit tests for core utility functions.
"""

from datetime import timedelta
from uuid import uuid4

from django.utils import timezone

from apps.core.utils import (
    # JSON utilities
    validate_json_structure,
    safe_json_loads,
    safe_json_dumps,
    # UUID utilities
    is_valid_uuid,
    parse_uuid,
    # String utilities
    truncate_string,
    sanitize_string,
    normalize_whitespace,
    # Date/Time utilities
    get_expiration_datetime,
    is_expired,
    format_duration,
    format_file_size,
    # List/Dict utilities
    get_nested_value,
    set_nested_value,
    merge_dicts,
    # Validation helpers
    validate_url_format,
    clamp,
)


# JSON UTILITIES TESTS

class TestValidateJsonStructure:
    """Tests for validate_json_structure function."""
    
    def test_valid_structure(self):
        """Test validating a valid structure."""
        data = {"name": "test", "age": 25}
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "integer", "required": True},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is True
        assert errors == []
    
    def test_missing_required_field(self):
        """Test validation with missing required field."""
        data = {"name": "test"}
        schema = {
            "name": {"type": "string", "required": True},
            "age": {"type": "integer", "required": True},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is False
        assert any("age" in error for error in errors)
    
    def test_wrong_type(self):
        """Test validation with wrong type."""
        data = {"name": 123}
        schema = {
            "name": {"type": "string", "required": True},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is False
        assert any("string" in error.lower() for error in errors)
    
    def test_integer_range(self):
        """Test integer range validation."""
        data = {"age": 150}
        schema = {
            "age": {"type": "integer", "required": True, "min": 0, "max": 120},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is False
        assert any("120" in error for error in errors)
    
    def test_string_length(self):
        """Test string length validation."""
        data = {"code": "AB"}
        schema = {
            "code": {"type": "string", "required": True, "min_length": 3},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is False
        assert any("3" in error for error in errors)
    
    def test_choices_validation(self):
        """Test choices validation."""
        data = {"status": "invalid"}
        schema = {
            "status": {"type": "string", "required": True, "choices": ["active", "inactive"]},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is False
        assert any("one of" in error.lower() for error in errors)
    
    def test_nullable_field(self):
        """Test nullable field validation."""
        data = {"name": None}
        schema = {
            "name": {"type": "string", "required": True, "nullable": True},
        }
        
        is_valid, errors = validate_json_structure(data, schema)
        
        assert is_valid is True
    
    def test_non_dict_input(self):
        """Test with non-dict input."""
        is_valid, errors = validate_json_structure("not a dict", {})
        
        assert is_valid is False
        assert any("dictionary" in error.lower() for error in errors)


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""
    
    def test_valid_json(self):
        """Test parsing valid JSON."""
        result, error = safe_json_loads('{"key": "value"}')
        
        assert result == {"key": "value"}
        assert error is None
    
    def test_invalid_json(self):
        """Test parsing invalid JSON."""
        result, error = safe_json_loads("{invalid}")
        
        assert result is None
        assert error is not None
        assert "Invalid JSON" in error
    
    def test_empty_string(self):
        """Test parsing empty string."""
        result, error = safe_json_loads("")
        
        assert result is None
        assert error is not None
    
    def test_with_default(self):
        """Test with default value."""
        result, error = safe_json_loads("", default=[])
        
        assert result == []


class TestSafeJsonDumps:
    """Tests for safe_json_dumps function."""
    
    def test_valid_data(self):
        """Test serializing valid data."""
        result, error = safe_json_dumps({"key": "value"})
        
        assert result == '{"key": "value"}'
        assert error is None
    
    def test_non_serializable(self):
        """Test serializing non-serializable data."""
        result, error = safe_json_dumps({"func": lambda x: x})
        
        assert result == "{}"
        assert error is not None


# UUID UTILITIES TESTS

class TestIsValidUuid:
    """Tests for is_valid_uuid function."""
    
    def test_valid_uuid_string(self):
        """Test with valid UUID string."""
        assert is_valid_uuid(str(uuid4())) is True
    
    def test_valid_uuid_object(self):
        """Test with UUID object."""
        assert is_valid_uuid(uuid4()) is True
    
    def test_invalid_uuid(self):
        """Test with invalid UUID."""
        assert is_valid_uuid("not-a-uuid") is False
    
    def test_none(self):
        """Test with None."""
        assert is_valid_uuid(None) is False


class TestParseUuid:
    """Tests for parse_uuid function."""
    
    def test_valid_string(self):
        """Test parsing valid UUID string."""
        uuid_str = str(uuid4())
        result, error = parse_uuid(uuid_str)
        
        assert result is not None
        assert error is None
        assert str(result) == uuid_str
    
    def test_uuid_object(self):
        """Test parsing UUID object."""
        uuid_obj = uuid4()
        result, error = parse_uuid(uuid_obj)
        
        assert result == uuid_obj
        assert error is None
    
    def test_invalid_string(self):
        """Test parsing invalid string."""
        result, error = parse_uuid("invalid")
        
        assert result is None
        assert error is not None
    
    def test_none(self):
        """Test parsing None."""
        result, error = parse_uuid(None, field_name="test_id")
        
        assert result is None
        assert "test_id" in error


# STRING UTILITIES TESTS

class TestTruncateString:
    """Tests for truncate_string function."""
    
    def test_short_string(self):
        """Test string shorter than max length."""
        result = truncate_string("hello", 10)
        assert result == "hello"
    
    def test_long_string(self):
        """Test string longer than max length."""
        result = truncate_string("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8
    
    def test_custom_suffix(self):
        """Test with custom suffix."""
        result = truncate_string("hello world", 9, suffix="…")
        assert result == "hello wo…"
    
    def test_empty_string(self):
        """Test with empty string."""
        result = truncate_string("", 10)
        assert result == ""


class TestSanitizeString:
    """Tests for sanitize_string function."""
    
    def test_basic_sanitization(self):
        """Test basic string sanitization."""
        result = sanitize_string("Hello World!")
        assert result == "Hello_World_"
    
    def test_custom_pattern(self):
        """Test with custom pattern."""
        result = sanitize_string("Hello123", allowed_chars=r'[^a-zA-Z]')
        assert result == "Hello___"
    
    def test_custom_replacement(self):
        """Test with custom replacement."""
        result = sanitize_string("Hello World!", replacement="-")
        assert result == "Hello-World-"
    
    def test_empty_string(self):
        """Test with empty string."""
        result = sanitize_string("")
        assert result == ""


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""
    
    def test_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        result = normalize_whitespace("hello    world")
        assert result == "hello world"
    
    def test_leading_trailing(self):
        """Test removing leading/trailing whitespace."""
        result = normalize_whitespace("  hello world  ")
        assert result == "hello world"
    
    def test_mixed_whitespace(self):
        """Test with tabs and newlines."""
        result = normalize_whitespace("hello\t\nworld")
        assert result == "hello world"
    
    def test_empty_string(self):
        """Test with empty string."""
        result = normalize_whitespace("")
        assert result == ""


# DATE/TIME UTILITIES TESTS

class TestGetExpirationDatetime:
    """Tests for get_expiration_datetime function."""
    
    def test_default_days(self):
        """Test with default 7 days."""
        result = get_expiration_datetime()
        now = timezone.now()
        
        # Should be approximately 7 days in the future
        diff = result - now
        assert 6 < diff.days < 8
    
    def test_custom_days(self):
        """Test with custom days."""
        result = get_expiration_datetime(days=30)
        now = timezone.now()
        
        diff = result - now
        assert 29 < diff.days < 31


class TestIsExpired:
    """Tests for is_expired function."""
    
    def test_past_date(self):
        """Test with past date."""
        past = timezone.now() - timedelta(days=1)
        assert is_expired(past) is True
    
    def test_future_date(self):
        """Test with future date."""
        future = timezone.now() + timedelta(days=1)
        assert is_expired(future) is False
    
    def test_none(self):
        """Test with None."""
        assert is_expired(None) is False


class TestFormatDuration:
    """Tests for format_duration function."""
    
    def test_seconds_only(self):
        """Test with seconds only."""
        assert format_duration(45) == "45s"
    
    def test_minutes_and_seconds(self):
        """Test with minutes and seconds."""
        assert format_duration(125) == "2m 5s"
    
    def test_hours_minutes_seconds(self):
        """Test with hours, minutes, and seconds."""
        assert format_duration(3725) == "1h 2m 5s"
    
    def test_negative(self):
        """Test with negative value."""
        assert format_duration(-10) == "0s"


class TestFormatFileSize:
    """Tests for format_file_size function."""
    
    def test_bytes(self):
        """Test with bytes."""
        assert format_file_size(500) == "500 B"
    
    def test_kilobytes(self):
        """Test with kilobytes."""
        assert format_file_size(2048) == "2.00 KB"
    
    def test_megabytes(self):
        """Test with megabytes."""
        assert format_file_size(1048576) == "1.00 MB"
    
    def test_gigabytes(self):
        """Test with gigabytes."""
        assert format_file_size(1073741824) == "1.00 GB"
    
    def test_negative(self):
        """Test with negative value."""
        assert format_file_size(-100) == "0 B"


# LIST/DICT UTILITIES TESTS

class TestGetNestedValue:
    """Tests for get_nested_value function."""
    
    def test_simple_path(self):
        """Test with simple path."""
        data = {"key": "value"}
        assert get_nested_value(data, "key") == "value"
    
    def test_nested_path(self):
        """Test with nested path."""
        data = {"level1": {"level2": {"level3": "deep"}}}
        assert get_nested_value(data, "level1.level2.level3") == "deep"
    
    def test_missing_key(self):
        """Test with missing key."""
        data = {"key": "value"}
        assert get_nested_value(data, "missing", default="default") == "default"
    
    def test_empty_data(self):
        """Test with empty data."""
        assert get_nested_value({}, "key", default="default") == "default"


class TestSetNestedValue:
    """Tests for set_nested_value function."""
    
    def test_simple_path(self):
        """Test with simple path."""
        data = {}
        result = set_nested_value(data, "key", "value")
        assert result == {"key": "value"}
    
    def test_nested_path(self):
        """Test with nested path."""
        data = {}
        result = set_nested_value(data, "level1.level2.key", "value")
        assert result == {"level1": {"level2": {"key": "value"}}}
    
    def test_existing_data(self):
        """Test with existing data."""
        data = {"existing": "data"}
        result = set_nested_value(data, "new", "value")
        assert result == {"existing": "data", "new": "value"}


class TestMergeDicts:
    """Tests for merge_dicts function."""
    
    def test_simple_merge(self):
        """Test simple merge."""
        base = {"a": 1}
        override = {"b": 2}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 2}
    
    def test_override_value(self):
        """Test overriding value."""
        base = {"a": 1}
        override = {"a": 2}
        result = merge_dicts(base, override)
        assert result == {"a": 2}
    
    def test_deep_merge(self):
        """Test deep merge."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = merge_dicts(base, override, deep=True)
        assert result == {"a": {"b": 1, "c": 2}}
    
    def test_shallow_merge(self):
        """Test shallow merge."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        result = merge_dicts(base, override, deep=False)
        assert result == {"a": {"c": 2}}


# VALIDATION HELPERS TESTS


class TestValidateUrlFormat:
    """Tests for validate_url_format function."""
    
    def test_valid_http_url(self):
        """Test with valid HTTP URL."""
        assert validate_url_format("http://example.com") is True
    
    def test_valid_https_url(self):
        """Test with valid HTTPS URL."""
        assert validate_url_format("https://example.com/path") is True
    
    def test_invalid_url_no_protocol(self):
        """Test without protocol."""
        assert validate_url_format("example.com") is False
    
    def test_invalid_url_invalid_protocol(self):
        """Test with invalid protocol."""
        assert validate_url_format("ftp://example.com") is False
    
    def test_empty_url(self):
        """Test with empty string."""
        assert validate_url_format("") is False


class TestClamp:
    """Tests for clamp function."""
    
    def test_value_in_range(self):
        """Test value within range."""
        assert clamp(5, 0, 10) == 5
    
    def test_value_below_min(self):
        """Test value below minimum."""
        assert clamp(-5, 0, 10) == 0
    
    def test_value_above_max(self):
        """Test value above maximum."""
        assert clamp(15, 0, 10) == 10
    
    def test_float_values(self):
        """Test with float values."""
        assert clamp(5.5, 0.0, 10.0) == 5.5