# tests/unit/test_registry.py
"""
Unit tests for the Operation Registry.
"""

import pytest
from apps.processors.registry import (
    OperationRegistry,
    OperationDefinition,
    ParameterSchema,
    ParameterType,
    MediaType,
    register_operation,
    get_registry,
)
from apps.processors.exceptions import (
    OperationNotFoundError,
    OperationRegistrationError,
    InvalidParametersError,
)


class TestParameterSchema:
    """Tests for ParameterSchema dataclass."""
    
    def test_parameter_schema_creation(self):
        """Test creating a basic parameter schema."""
        schema = ParameterSchema(
            name="quality",
            param_type=ParameterType.INTEGER,
            required=True,
            default=23,
            description="Quality value",
            min_value=18,
            max_value=28,
        )
        
        assert schema.name == "quality"
        assert schema.param_type == ParameterType.INTEGER
        assert schema.required is True
        assert schema.default == 23
        assert schema.description == "Quality value"
        assert schema.min_value == 18
        assert schema.max_value == 28
    
    def test_parameter_schema_to_dict(self):
        """Test converting parameter schema to dictionary."""
        schema = ParameterSchema(
            name="format",
            param_type=ParameterType.CHOICE,
            required=False,
            default="mp4",
            description="Output format",
            choices=["mp4", "webm", "mov"],
        )
        
        result = schema.to_dict()
        
        assert result["name"] == "format"
        assert result["type"] == "choice"
        assert result["required"] is False
        assert result["default"] == "mp4"
        assert result["choices"] == ["mp4", "webm", "mov"]
    
    def test_parameter_schema_to_dict_minimal(self):
        """Test converting minimal parameter schema to dictionary."""
        schema = ParameterSchema(
            name="test",
            param_type=ParameterType.STRING,
        )
        
        result = schema.to_dict()
        
        assert result["name"] == "test"
        assert result["type"] == "string"
        assert result["required"] is False
        assert "default" not in result
        assert "min" not in result
        assert "max" not in result
        assert "choices" not in result


class TestOperationDefinition:
    """Tests for OperationDefinition dataclass."""
    
    def test_operation_definition_creation(self):
        """Test creating an operation definition."""
        def dummy_handler(job, input_path, output_path, params):
            pass
        
        params = [
            ParameterSchema(
                name="quality",
                param_type=ParameterType.INTEGER,
                default=23,
            )
        ]
        
        op = OperationDefinition(
            name="test_operation",
            media_type=MediaType.VIDEO,
            handler=dummy_handler,
            parameters=params,
            description="Test operation",
            input_formats=["mp4", "avi"],
            output_formats=["mp4"],
        )
        
        assert op.name == "test_operation"
        assert op.media_type == MediaType.VIDEO
        assert op.handler == dummy_handler
        assert len(op.parameters) == 1
        assert op.description == "Test operation"
        assert op.input_formats == ["mp4", "avi"]
        assert op.output_formats == ["mp4"]
    
    def test_operation_definition_to_dict(self):
        """Test converting operation definition to dictionary."""
        def dummy_handler(job, input_path, output_path, params):
            pass
        
        params = [
            ParameterSchema(
                name="quality",
                param_type=ParameterType.INTEGER,
                default=23,
            )
        ]
        
        op = OperationDefinition(
            name="test_op",
            media_type=MediaType.IMAGE,
            handler=dummy_handler,
            parameters=params,
            description="Test",
        )
        
        result = op.to_dict()
        
        assert result["name"] == "test_op"
        assert result["media_type"] == "image"
        assert result["description"] == "Test"
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "quality"


class TestOperationRegistry:
    """Tests for OperationRegistry class."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Clear the registry before each test."""
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()
    
    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = OperationRegistry()
        registry2 = OperationRegistry()
        
        assert registry1 is registry2
    
    def test_get_registry_returns_singleton(self):
        """Test get_registry returns the singleton instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        assert registry1 is registry2
    
    def test_register_operation(self, setup_registry):
        """Test registering an operation."""
        registry = setup_registry
        
        def test_handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="test_operation",
            media_type=MediaType.VIDEO,
            handler=test_handler,
            description="A test operation",
        )
        
        assert registry.is_registered("test_operation")
    
    def test_register_operation_with_parameters(self, setup_registry):
        """Test registering an operation with parameters."""
        registry = setup_registry
        
        def test_handler(job, input_path, output_path, params):
            pass
        
        params = [
            ParameterSchema(
                name="quality",
                param_type=ParameterType.INTEGER,
                required=True,
                min_value=18,
                max_value=28,
            ),
            ParameterSchema(
                name="format",
                param_type=ParameterType.CHOICE,
                default="mp4",
                choices=["mp4", "webm"],
            ),
        ]
        
        registry.register_operation(
            name="video_compress",
            media_type=MediaType.VIDEO,
            handler=test_handler,
            parameters=params,
        )
        
        op = registry.get_operation("video_compress")
        assert len(op.parameters) == 2
    
    def test_register_duplicate_raises_error(self, setup_registry):
        """Test that registering a duplicate operation raises an error."""
        registry = setup_registry
        
        def handler1(job, input_path, output_path, params):
            pass
        
        def handler2(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="duplicate_op",
            media_type=MediaType.VIDEO,
            handler=handler1,
        )
        
        with pytest.raises(OperationRegistrationError) as exc_info:
            registry.register_operation(
                name="duplicate_op",
                media_type=MediaType.IMAGE,
                handler=handler2,
            )
        
        assert "already registered" in str(exc_info.value)
    
    def test_register_invalid_name_empty(self, setup_registry):
        """Test that registering with empty name raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        with pytest.raises(OperationRegistrationError):
            registry.register_operation(
                name="",
                media_type=MediaType.VIDEO,
                handler=handler,
            )
    
    def test_register_invalid_name_special_chars(self, setup_registry):
        """Test that registering with special characters raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        with pytest.raises(OperationRegistrationError):
            registry.register_operation(
                name="invalid-name!",
                media_type=MediaType.VIDEO,
                handler=handler,
            )
    
    def test_register_non_callable_handler(self, setup_registry):
        """Test that registering with non-callable handler raises error."""
        registry = setup_registry
        
        with pytest.raises(OperationRegistrationError):
            registry.register_operation(
                name="bad_handler",
                media_type=MediaType.VIDEO,
                handler="not a function",
            )
    
    def test_get_operation(self, setup_registry):
        """Test retrieving an operation."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            return "processed"
        
        registry.register_operation(
            name="get_test",
            media_type=MediaType.AUDIO,
            handler=handler,
        )
        
        op = registry.get_operation("get_test")
        
        assert op.name == "get_test"
        assert op.media_type == MediaType.AUDIO
        assert op.handler is handler
    
    def test_get_nonexistent_operation_raises_error(self, setup_registry):
        """Test that getting a nonexistent operation raises error."""
        registry = setup_registry
        
        with pytest.raises(OperationNotFoundError) as exc_info:
            registry.get_operation("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
    
    def test_is_registered_true(self, setup_registry):
        """Test is_registered returns True for registered operation."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="exists_test",
            media_type=MediaType.IMAGE,
            handler=handler,
        )
        
        assert registry.is_registered("exists_test") is True
    
    def test_is_registered_false(self, setup_registry):
        """Test is_registered returns False for unregistered operation."""
        registry = setup_registry
        
        assert registry.is_registered("nonexistent") is False
    
    def test_list_registered_operations(self, setup_registry):
        """Test listing all operations."""
        registry = setup_registry
        
        def handler1(job, input_path, output_path, params):
            pass
        
        def handler2(job, input_path, output_path, params):
            pass
        
        registry.register_operation(name="op1", media_type=MediaType.VIDEO, handler=handler1)
        registry.register_operation(name="op2", media_type=MediaType.IMAGE, handler=handler2)
        
        all_ops = registry.list_registered_operations()
        
        assert len(all_ops) == 2
        names = [op.name for op in all_ops]
        assert "op1" in names
        assert "op2" in names
    
    def test_list_operations_by_media_type(self, setup_registry):
        """Test filtering operations by media type."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(name="video1", media_type=MediaType.VIDEO, handler=handler)
        registry.register_operation(name="video2", media_type=MediaType.VIDEO, handler=handler)
        registry.register_operation(name="image1", media_type=MediaType.IMAGE, handler=handler)
        registry.register_operation(name="audio1", media_type=MediaType.AUDIO, handler=handler)
        
        video_ops = registry.list_operations_by_media_type(MediaType.VIDEO)
        image_ops = registry.list_operations_by_media_type(MediaType.IMAGE)
        audio_ops = registry.list_operations_by_media_type(MediaType.AUDIO)
        
        assert len(video_ops) == 2
        assert len(image_ops) == 1
        assert len(audio_ops) == 1
        
        assert all(op.media_type == MediaType.VIDEO for op in video_ops)
    
    def test_clear(self, setup_registry):
        """Test clearing the registry."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(name="op1", media_type=MediaType.VIDEO, handler=handler)
        registry.register_operation(name="op2", media_type=MediaType.IMAGE, handler=handler)
        
        assert len(registry.list_registered_operations()) == 2
        
        registry.clear()
        
        assert len(registry.list_registered_operations()) == 0


class TestParameterValidation:
    """Tests for parameter validation."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Clear and setup registry before each test."""
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()
    
    def test_validate_integer_parameter(self, setup_registry):
        """Test validating an integer parameter."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="int_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                    min_value=18,
                    max_value=28,
                )
            ],
        )
        
        result = registry.validate_parameters("int_test", {"quality": 23})
        
        assert result["quality"] == 23
    
    def test_validate_integer_from_string(self, setup_registry):
        """Test that string integers are converted."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="int_str_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                )
            ],
        )
        
        result = registry.validate_parameters("int_str_test", {"quality": "25"})
        
        assert result["quality"] == 25
        assert isinstance(result["quality"], int)
    
    def test_validate_integer_below_min_raises_error(self, setup_registry):
        """Test that integer below minimum raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="int_min_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                    min_value=18,
                )
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("int_min_test", {"quality": 10})
        
        assert "at least 18" in str(exc_info.value)
    
    def test_validate_integer_above_max_raises_error(self, setup_registry):
        """Test that integer above maximum raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="int_max_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                    max_value=28,
                )
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("int_max_test", {"quality": 50})
        
        assert "at most 28" in str(exc_info.value)
    
    def test_validate_float_parameter(self, setup_registry):
        """Test validating a float parameter."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="float_test",
            media_type=MediaType.AUDIO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="volume",
                    param_type=ParameterType.FLOAT,
                    required=True,
                    min_value=0.0,
                    max_value=2.0,
                )
            ],
        )
        
        result = registry.validate_parameters("float_test", {"volume": 1.5})
        
        assert result["volume"] == 1.5
        assert isinstance(result["volume"], float)
    
    def test_validate_boolean_true(self, setup_registry):
        """Test validating boolean True values."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="bool_test",
            media_type=MediaType.IMAGE,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="maintain_ratio",
                    param_type=ParameterType.BOOLEAN,
                    required=True,
                )
            ],
        )
        
        # Test various True representations
        for value in [True, "true", "True", "1", "yes", "on", 1]:
            result = registry.validate_parameters("bool_test", {"maintain_ratio": value})
            assert result["maintain_ratio"] is True
    
    def test_validate_boolean_false(self, setup_registry):
        """Test validating boolean False values."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="bool_false_test",
            media_type=MediaType.IMAGE,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="maintain_ratio",
                    param_type=ParameterType.BOOLEAN,
                    required=True,
                )
            ],
        )
        
        # Test various False representations
        for value in [False, "false", "False", "0", "no", "off", 0]:
            result = registry.validate_parameters("bool_false_test", {"maintain_ratio": value})
            assert result["maintain_ratio"] is False
    
    def test_validate_choice_parameter(self, setup_registry):
        """Test validating a choice parameter."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="choice_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="format",
                    param_type=ParameterType.CHOICE,
                    required=True,
                    choices=["mp4", "webm", "mov"],
                )
            ],
        )
        
        result = registry.validate_parameters("choice_test", {"format": "mp4"})
        
        assert result["format"] == "mp4"
    
    def test_validate_choice_invalid_value_raises_error(self, setup_registry):
        """Test that invalid choice raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="choice_invalid_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="format",
                    param_type=ParameterType.CHOICE,
                    required=True,
                    choices=["mp4", "webm"],
                )
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("choice_invalid_test", {"format": "avi"})
        
        assert "must be one of" in str(exc_info.value)
    
    def test_validate_string_parameter(self, setup_registry):
        """Test validating a string parameter."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="string_test",
            media_type=MediaType.IMAGE,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="watermark_text",
                    param_type=ParameterType.STRING,
                    required=True,
                )
            ],
        )
        
        result = registry.validate_parameters("string_test", {"watermark_text": "Copyright 2024"})
        
        assert result["watermark_text"] == "Copyright 2024"
    
    def test_validate_required_parameter_missing_raises_error(self, setup_registry):
        """Test that missing required parameter raises error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="required_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                )
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("required_test", {})
        
        assert "Required parameter" in str(exc_info.value)
        assert "quality" in str(exc_info.value)
    
    def test_validate_default_value_applied(self, setup_registry):
        """Test that default values are applied for missing optional parameters."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="default_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=False,
                    default=23,
                )
            ],
        )
        
        result = registry.validate_parameters("default_test", {})
        
        assert result["quality"] == 23
    
    def test_validate_unknown_parameters_raises_error(self, setup_registry):
        """Test that unknown parameters raise error."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="unknown_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                )
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("unknown_test", {"quality": 23, "unknown": "value"})
        
        assert "Unknown parameters" in str(exc_info.value)
    
    def test_validate_multiple_errors(self, setup_registry):
        """Test that multiple validation errors are collected."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="multi_error_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    required=True,
                ),
                ParameterSchema(
                    name="format",
                    param_type=ParameterType.CHOICE,
                    required=True,
                    choices=["mp4", "webm"],
                ),
            ],
        )
        
        with pytest.raises(InvalidParametersError) as exc_info:
            registry.validate_parameters("multi_error_test", {})
        
        # Both required parameters should be in the error
        error_msg = str(exc_info.value)
        assert "quality" in error_msg
        assert "format" in error_msg
    
    def test_validate_nonexistent_operation_raises_error(self, setup_registry):
        """Test that validating for nonexistent operation raises error."""
        registry = setup_registry
        
        with pytest.raises(OperationNotFoundError):
            registry.validate_parameters("nonexistent", {"param": "value"})


class TestRegisterOperationDecorator:
    """Tests for the register_operation decorator."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()
    
    def test_decorator_registers_operation(self, setup_registry):
        """Test that decorator properly registers an operation."""
        registry = setup_registry
        
        @register_operation(
            operation_name="decorated_op",
            media_type=MediaType.VIDEO,
            description="A decorated operation",
        )
        def my_operation(job, input_path, output_path, params):
            return "processed"
        
        assert registry.is_registered("decorated_op")
        op = registry.get_operation("decorated_op")
        assert op.description == "A decorated operation"
    
    def test_decorator_preserves_function(self, setup_registry):
        """Test that decorator preserves the original function."""
        @register_operation(
            operation_name="preserves_func",
            media_type=MediaType.IMAGE,
        )
        def original_function(job, input_path, output_path, params):
            return "original return value"
        
        # Function should still be callable
        result = original_function(None, None, None, None)
        assert result == "original return value"
    
    def test_decorator_with_full_parameters(self, setup_registry):
        """Test decorator with all parameters."""
        registry = setup_registry
        
        @register_operation(
            operation_name="full_params_op",
            media_type=MediaType.AUDIO,
            parameters=[
                ParameterSchema(
                    name="bitrate",
                    param_type=ParameterType.INTEGER,
                    default=128,
                    min_value=64,
                    max_value=320,
                )
            ],
            description="Full params test",
            input_formats=["mp3", "wav"],
            output_formats=["mp3"],
        )
        def full_operation(job, input_path, output_path, params):
            pass
        
        op = registry.get_operation("full_params_op")
        
        assert op.media_type == MediaType.AUDIO
        assert len(op.parameters) == 1
        assert op.parameters[0].name == "bitrate"
        assert op.input_formats == ["mp3", "wav"]
        assert op.output_formats == ["mp3"]


class TestGetOperationInfo:
    """Tests for getting operation info as dictionary."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield registry
        registry.clear()
    
    def test_get_operation_info(self, setup_registry):
        """Test getting operation info as dictionary."""
        registry = setup_registry
        
        def handler(job, input_path, output_path, params):
            pass
        
        registry.register_operation(
            name="info_test",
            media_type=MediaType.VIDEO,
            handler=handler,
            parameters=[
                ParameterSchema(
                    name="quality",
                    param_type=ParameterType.INTEGER,
                    default=23,
                    description="Quality value",
                )
            ],
            description="Test operation for info",
            input_formats=["mp4"],
            output_formats=["mp4"],
        )
        
        info = registry.get_operation_info("info_test")
        
        assert info["name"] == "info_test"
        assert info["media_type"] == "video"
        assert info["description"] == "Test operation for info"
        assert len(info["parameters"]) == 1
        assert info["parameters"][0]["name"] == "quality"
        assert info["input_formats"] == ["mp4"]
        assert info["output_formats"] == ["mp4"]