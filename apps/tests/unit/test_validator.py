# apps/tests/unit/test_validator.py

"""
Unit tests for the Validator Service.
"""

import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from apps.operations.services.validator import Validator, ValidationResult


class MockUploadedFile:
    """Mock Django UploadedFile for testing."""
    
    def __init__(
        self,
        name: str = "test_video.mp4",
        size: int = 1024,
        content_type: str = "video/mp4",
        content: bytes = None,
    ):
        self.name = name
        self.size = size
        self.content_type = content_type
        self._content = content or b"fake file content" * (size // 17 + 1)
        self._content = self._content[:size]
        self._position = 0
    
    def read(self, size: int = None) -> bytes:
        if size is None:
            data = self._content[self._position:]
            self._position = len(self._content)
        else:
            data = self._content[self._position:self._position + size]
            self._position += size
        return data
    
    def seek(self, position: int) -> None:
        self._position = position
    
    def chunks(self, chunk_size: int = 8192):
        self.seek(0)
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk


class MockOperationDefinition:
    """Mock OperationDefinition for testing."""
    
    def __init__(
        self,
        name: str = "video_compress",
        media_type_value: str = "video",
        description: str = "Test operation",
        input_formats: list = None,
        output_formats: list = None,
    ):
        self.operation_name = name
        self.media_type = MagicMock()
        self.media_type.value = media_type_value
        self.description = description
        self.input_formats = input_formats or []
        self.output_formats = output_formats or []


class MockRegistry:
    """Mock Operation Registry for testing."""
    
    def __init__(self, operations: dict = None):
        self._operations = operations or {}
    
    def is_registered(self, operation_name: str) -> bool:
        return operation_name in self._operations
    
    def get_operation(self, operation_name: str):
        if operation_name not in self._operations:
            from apps.processors.exceptions import OperationNotFoundError
            raise OperationNotFoundError(operation_name)
        return self._operations[operation_name]
    
    def list_registered_operations(self):
        return list(self._operations.values())
    
    def validate_parameters(self, operation_name: str, parameters: dict):
        if operation_name not in self._operations:
            from apps.processors.exceptions import OperationNotFoundError
            raise OperationNotFoundError(operation_name)
        # Simple mock validation - just return parameters
        return parameters


class MockRequest:
    """Mock Django HttpRequest for testing."""
    
    def __init__(self, session_key: str = None, has_session: bool = True):
        if has_session:
            self.session = MagicMock()
            self.session.session_key = session_key
            self.session.create = MagicMock()
            
            # Make session.create actually set a session key
            def create_session():
                self.session.session_key = f"test-session-{uuid4().hex[:8]}"
            
            self.session.create.side_effect = create_session
        else:
            # No session attribute
            pass


# VALIDATION RESULT TESTS

class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_valid_result_creation(self):
        """Test creating a valid result."""
        result = ValidationResult(
            is_valid=True,
            data={"key": "value"}
        )
        
        assert result.is_valid is True
        assert result.data == {"key": "value"}
        assert result.errors == []
    
    def test_invalid_result_creation(self):
        """Test creating an invalid result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"]
        )
        
        assert result.is_valid is False
        assert result.data == {}
        assert result.errors == ["Error 1", "Error 2"]
    
    def test_boolean_conversion_valid(self):
        """Test boolean conversion for valid result."""
        result = ValidationResult(is_valid=True)
        assert bool(result) is True
    
    def test_boolean_conversion_invalid(self):
        """Test boolean conversion for invalid result."""
        result = ValidationResult(is_valid=False)
        assert bool(result) is False
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(
            is_valid=True,
            data={"key": "value"},
            errors=[]
        )
        
        dict_result = result.to_dict()
        
        assert dict_result["is_valid"] is True
        assert dict_result["data"] == {"key": "value"}
        assert dict_result["errors"] == []


# OPERATION VALIDATION TESTS

class TestValidateOperationExists:
    """Tests for Validator.validate_operation_exists method."""
    
    def test_valid_operation(self):
        """Test validating an existing operation."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                name="video_compress",
                media_type_value="video",
                description="Compress video",
                input_formats=["mp4", "avi"],
                output_formats=["mp4"],
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_exists("video_compress")
            
            assert result.is_valid is True
            assert result.data["operation_name"] == "video_compress"
            assert result.data["media_type"] == "video"
    
    def test_nonexistent_operation(self):
        """Test validating a nonexistent operation."""
        mock_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_exists("nonexistent")
            
            assert result.is_valid is False
            assert "nonexistent" in result.errors[0]
    
    def test_empty_operation_name(self):
        """Test validating with empty operation name."""
        result = Validator.validate_operation_exists("")
        
        assert result.is_valid is False
        assert "required" in result.errors[0].lower()
    
    def test_none_operation_name(self):
        """Test validating with None operation name."""
        result = Validator.validate_operation_exists(None)
        
        assert result.is_valid is False
        assert "required" in result.errors[0].lower()
    
    def test_non_string_operation_name(self):
        """Test validating with non-string operation name."""
        result = Validator.validate_operation_exists(123)
        
        assert result.is_valid is False
        assert "string" in result.errors[0].lower()


class TestValidateOperationParameters:
    """Tests for Validator.validate_operation_parameters method."""
    
    def test_valid_parameters(self):
        """Test validating valid parameters."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition()
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_parameters(
                "video_compress",
                {"quality": 23}
            )
            
            assert result.is_valid is True
            assert result.data["validated_parameters"] == {"quality": 23}
    
    def test_invalid_parameters(self):
        """Test validating invalid parameters."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition()
        })
        
        # Mock validate_parameters to raise InvalidParametersError
        from apps.processors.exceptions import InvalidParametersError
        mock_registry.validate_parameters = MagicMock(
            side_effect=InvalidParametersError("video_compress", ["Invalid quality value"])
        )
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_parameters(
                "video_compress",
                {"quality": -1}
            )
            
            assert result.is_valid is False
            assert "Invalid quality value" in result.errors
    
    def test_none_parameters(self):
        """Test validating with None parameters."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition()
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_parameters("video_compress", None)
            
            assert result.is_valid is True
    
    def test_non_dict_parameters(self):
        """Test validating with non-dict parameters."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition()
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_parameters("video_compress", "invalid")
            
            assert result.is_valid is False
            assert "dictionary" in result.errors[0].lower()
    
    def test_nonexistent_operation(self):
        """Test validating parameters for nonexistent operation."""
        mock_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_parameters(
                "nonexistent",
                {"quality": 23}
            )
            
            assert result.is_valid is False


# FILE VALIDATION TESTS

class TestValidateFileForOperation:
    """Tests for Validator.validate_file_for_operation method."""
    
    def test_valid_file(self):
        """Test validating a valid file for an operation."""
        uploaded_file = MockUploadedFile(
            name="test_video.mp4",
            size=1024 * 1024,  # 1MB
            content_type="video/mp4"
        )
        
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                media_type_value="video",
                input_formats=["mp4", "avi", "mov"]
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch.object(
                Validator, '_get_max_file_size', return_value=500 * 1024 * 1024
            ):
                with patch.object(
                    Validator, '_get_supported_formats', return_value=["mp4", "avi", "mov"]
                ):
                    with patch(
                        'apps.operations.services.validator.FileManager.detect_mime_type',
                        return_value="video/mp4"
                    ):
                        with patch(
                            'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                            return_value="video"
                        ):
                            with patch(
                                'apps.operations.services.validator.FileManager.get_file_extension',
                                return_value="mp4"
                            ):
                                result = Validator.validate_file_for_operation(
                                    uploaded_file, "video_compress"
                                )
                                
                                assert result.is_valid is True
                                assert result.data["filename"] == "test_video.mp4"
                                assert result.data["media_type"] == "video"
    
    def test_no_file_uploaded(self):
        """Test validation when no file is uploaded."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition()
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_file_for_operation(None, "video_compress")
            
            assert result.is_valid is False
            assert "No file" in result.errors[0]
    
    def test_wrong_media_type(self):
        """Test validation when file media type doesn't match operation."""
        uploaded_file = MockUploadedFile(
            name="image.png",
            size=1024 * 1024,
            content_type="image/png"
        )
        
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                media_type_value="video",
                input_formats=[]  # Empty to test media type check
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch(
                'apps.operations.services.validator.FileManager.detect_mime_type',
                return_value="image/png"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                    return_value="image"
                ):
                    with patch(
                        'apps.operations.services.validator.FileManager.get_file_extension',
                        return_value="png"
                    ):
                        with patch.object(
                            Validator, '_get_max_file_size', return_value=50 * 1024 * 1024
                        ):
                            with patch.object(
                                Validator, '_get_supported_formats', return_value=["png", "jpg"]
                            ):
                                result = Validator.validate_file_for_operation(
                                    uploaded_file, "video_compress"
                                )
                                
                                assert result.is_valid is False
                                assert "video" in result.errors[0].lower()
    
    def test_file_too_large(self):
        """Test validation when file exceeds size limit."""
        uploaded_file = MockUploadedFile(
            name="large_video.mp4",
            size=600 * 1024 * 1024,  # 600MB
            content_type="video/mp4"
        )
        
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                media_type_value="video",
                input_formats=["mp4"]
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch(
                'apps.operations.services.validator.FileManager.detect_mime_type',
                return_value="video/mp4"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                    return_value="video"
                ):
                    with patch(
                        'apps.operations.services.validator.FileManager.get_file_extension',
                        return_value="mp4"
                    ):
                        with patch.object(
                            Validator, '_get_max_file_size', return_value=500 * 1024 * 1024
                        ):
                            with patch.object(
                                Validator, '_get_supported_formats', return_value=["mp4"]
                            ):
                                result = Validator.validate_file_for_operation(
                                    uploaded_file, "video_compress"
                                )
                                
                                assert result.is_valid is False
                                assert "exceeds" in result.errors[0].lower()
    
    def test_unsupported_format(self):
        """Test validation when file format is not supported."""
        uploaded_file = MockUploadedFile(
            name="video.wmv",
            size=1024 * 1024,
            content_type="video/x-ms-wmv"
        )
        
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                media_type_value="video",
                input_formats=["mp4", "avi"]  # wmv not in list
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch(
                'apps.operations.services.validator.FileManager.detect_mime_type',
                return_value="video/x-ms-wmv"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                    return_value="video"
                ):
                    with patch(
                        'apps.operations.services.validator.FileManager.get_file_extension',
                        return_value="wmv"
                    ):
                        result = Validator.validate_file_for_operation(
                            uploaded_file, "video_compress"
                        )
                        
                        assert result.is_valid is False
                        assert "not supported" in result.errors[0].lower()


class TestValidateFileBasic:
    """Tests for Validator.validate_file_basic method."""
    
    def test_valid_file(self):
        """Test validating a valid file."""
        uploaded_file = MockUploadedFile(
            name="test.mp4",
            size=1024,
            content_type="video/mp4"
        )
        
        with patch(
            'apps.operations.services.validator.FileManager.get_file_extension',
            return_value="mp4"
        ):
            with patch(
                'apps.operations.services.validator.FileManager.detect_mime_type',
                return_value="video/mp4"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                    return_value="video"
                ):
                    result = Validator.validate_file_basic(uploaded_file)
                    
                    assert result.is_valid is True
                    assert result.data["filename"] == "test.mp4"
                    assert result.data["media_type"] == "video"
    
    def test_no_file(self):
        """Test validation with no file."""
        result = Validator.validate_file_basic(None)
        
        assert result.is_valid is False
        assert "No file" in result.errors[0]
    
    def test_empty_file(self):
        """Test validation with empty file."""
        uploaded_file = MockUploadedFile(
            name="empty.mp4",
            size=0,
            content_type="video/mp4"
        )
        
        result = Validator.validate_file_basic(uploaded_file)
        
        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()
    
    def test_unknown_media_type(self):
        """Test validation with unknown media type."""
        uploaded_file = MockUploadedFile(
            name="document.xyz",
            size=1024,
            content_type="application/xyz"
        )
        
        with patch(
            'apps.operations.services.validator.FileManager.get_file_extension',
            return_value="xyz"
        ):
            with patch(
                'apps.operations.services.validator.FileManager.detect_mime_type',
                return_value="application/xyz"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                    return_value=None
                ):
                    result = Validator.validate_file_basic(uploaded_file)
                    
                    assert result.is_valid is False
                    assert "not recognized" in result.errors[0].lower()


# SESSION VALIDATION TESTS

class TestValidateSession:
    """Tests for Validator.validate_session method."""
    
    def test_existing_session(self):
        """Test validating an existing session."""
        request = MockRequest(session_key="existing-session-key")
        
        result = Validator.validate_session(request)
        
        assert result.is_valid is True
        assert result.data["session_key"] == "existing-session-key"
    
    def test_create_new_session(self):
        """Test creating a new session when none exists."""
        request = MockRequest(session_key=None)
        
        result = Validator.validate_session(request)
        
        assert result.is_valid is True
        assert result.data["session_key"] is not None
        request.session.create.assert_called_once()
    
    def test_no_request(self):
        """Test validation with no request object."""
        result = Validator.validate_session(None)
        
        assert result.is_valid is False
        assert "required" in result.errors[0].lower()
    
    def test_no_session_support(self):
        """Test validation when request has no session support."""
        request = MockRequest(has_session=False)
        
        result = Validator.validate_session(request)
        
        assert result.is_valid is False
        assert "session support" in result.errors[0].lower()


class TestGetOrCreateSession:
    """Tests for Validator.get_or_create_session method."""
    
    def test_get_existing_session(self):
        """Test getting an existing session key."""
        request = MockRequest(session_key="existing-key")
        
        session_key = Validator.get_or_create_session(request)
        
        assert session_key == "existing-key"
    
    def test_create_session(self):
        """Test creating a new session."""
        request = MockRequest(session_key=None)
        
        session_key = Validator.get_or_create_session(request)
        
        assert session_key is not None
    
    def test_raises_on_invalid(self):
        """Test that ValueError is raised for invalid request."""
        with pytest.raises(ValueError) as exc_info:
            Validator.get_or_create_session(None)
        
        assert "required" in str(exc_info.value).lower()


# JSON VALIDATION TESTS

class TestValidateJsonField:
    """Tests for Validator.validate_json_field method."""
    
    def test_valid_json_object(self):
        """Test validating a valid JSON object."""
        result = Validator.validate_json_field(
            data={"name": "test", "value": 123},
            required_keys=["name"],
            optional_keys=["value"]
        )
        
        assert result.is_valid is True
        assert result.data == {"name": "test", "value": 123}
    
    def test_missing_required_key(self):
        """Test validation with missing required key."""
        result = Validator.validate_json_field(
            data={"value": 123},
            required_keys=["name", "value"]
        )
        
        assert result.is_valid is False
        assert "name" in result.errors[0]
    
    def test_null_required_key(self):
        """Test validation with null required key."""
        result = Validator.validate_json_field(
            data={"name": None, "value": 123},
            required_keys=["name"]
        )
        
        assert result.is_valid is False
        assert "null" in result.errors[0].lower()
    
    def test_extra_keys_allowed(self):
        """Test that extra keys are allowed by default."""
        result = Validator.validate_json_field(
            data={"name": "test", "extra": "value"},
            required_keys=["name"],
            allow_extra_keys=True
        )
        
        assert result.is_valid is True
    
    def test_extra_keys_not_allowed(self):
        """Test that extra keys can be disallowed."""
        result = Validator.validate_json_field(
            data={"name": "test", "extra": "value"},
            required_keys=["name"],
            allow_extra_keys=False
        )
        
        assert result.is_valid is False
        assert "extra" in result.errors[0].lower()
    
    def test_non_dict_input(self):
        """Test validation with non-dict input."""
        result = Validator.validate_json_field(
            data="not a dict",
            required_keys=["name"]
        )
        
        assert result.is_valid is False
        assert "object" in result.errors[0].lower()


class TestValidateJsonString:
    """Tests for Validator.validate_json_string method."""
    
    def test_valid_json_string(self):
        """Test parsing a valid JSON string."""
        result = Validator.validate_json_string('{"key": "value", "number": 42}')
        
        assert result.is_valid is True
        assert result.data["parsed"] == {"key": "value", "number": 42}
    
    def test_empty_json_string(self):
        """Test parsing an empty string."""
        result = Validator.validate_json_string("")
        
        assert result.is_valid is False
        assert "empty" in result.errors[0].lower()
    
    def test_invalid_json_string(self):
        """Test parsing an invalid JSON string."""
        result = Validator.validate_json_string("{invalid json}")
        
        assert result.is_valid is False
        assert "Invalid JSON" in result.errors[0]
    
    def test_non_string_input(self):
        """Test with non-string input."""
        result = Validator.validate_json_string(123)
        
        assert result.is_valid is False
        assert "string" in result.errors[0].lower()


# UUID VALIDATION TESTS

class TestValidateUuid:
    """Tests for Validator.validate_uuid method."""
    
    def test_valid_uuid_string(self):
        """Test validating a valid UUID string."""
        uuid_str = str(uuid4())
        
        result = Validator.validate_uuid(uuid_str)
        
        assert result.is_valid is True
        assert result.data["uuid"] == uuid_str.lower()
    
    def test_valid_uuid_object(self):
        """Test validating a UUID object."""
        uuid_obj = uuid4()
        
        result = Validator.validate_uuid(uuid_obj)
        
        assert result.is_valid is True
        assert result.data["uuid"] == str(uuid_obj)
    
    def test_invalid_uuid_format(self):
        """Test validating an invalid UUID format."""
        result = Validator.validate_uuid("not-a-uuid")
        
        assert result.is_valid is False
        assert "Invalid UUID" in result.errors[0]
    
    def test_none_uuid(self):
        """Test validating None UUID."""
        result = Validator.validate_uuid(None)
        
        assert result.is_valid is False
        assert "required" in result.errors[0].lower()
    
    def test_custom_field_name(self):
        """Test using custom field name in error."""
        result = Validator.validate_uuid(None, field_name="operation_id")
        
        assert result.is_valid is False
        assert "operation_id" in result.errors[0]


# COMBINED VALIDATION TESTS

class TestValidateOperationRequest:
    """Tests for Validator.validate_operation_request method."""
    
    def test_valid_request(self):
        """Test validating a completely valid request."""
        uploaded_file = MockUploadedFile(
            name="test.mp4",
            size=1024 * 1024,
            content_type="video/mp4"
        )
        
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition(
                media_type_value="video",
                input_formats=["mp4"]
            )
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch(
                'apps.operations.services.validator.FileManager.get_file_extension',
                return_value="mp4"
            ):
                with patch(
                    'apps.operations.services.validator.FileManager.detect_mime_type',
                    return_value="video/mp4"
                ):
                    with patch(
                        'apps.operations.services.validator.FileManager.get_media_type_from_mime_type',
                        return_value="video"
                    ):
                        with patch.object(
                            Validator, '_get_max_file_size', return_value=500 * 1024 * 1024
                        ):
                            with patch.object(
                                Validator, '_get_supported_formats', return_value=["mp4"]
                            ):
                                result = Validator.validate_operation_request(
                                    operation_name="video_compress",
                                    parameters={"quality": 23},
                                    uploaded_file=uploaded_file
                                )
                                
                                assert result.is_valid is True
                                assert "operation" in result.data
                                assert "parameters" in result.data
                                assert "file" in result.data
    
    def test_invalid_operation(self):
        """Test validation with invalid operation."""
        uploaded_file = MockUploadedFile()
        mock_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_request(
                operation_name="nonexistent",
                parameters={},
                uploaded_file=uploaded_file
            )
            
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are collected."""
        mock_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            result = Validator.validate_operation_request(
                operation_name="nonexistent",
                parameters={"invalid": True},
                uploaded_file=None
            )
            
            assert result.is_valid is False
            # Should have at least operation error
            assert len(result.errors) >= 1


# UTILITY METHOD TESTS

class TestUtilityMethods:
    """Tests for Validator utility methods."""
    
    def test_get_max_file_size_from_settings(self):
        """Test getting max file size from settings."""
        with patch('apps.operations.services.validator.settings') as mock_settings:
            mock_settings.MAX_FILE_SIZE = {'video': 1000000000}
            
            result = Validator._get_max_file_size('video')
            
            assert result == 1000000000
    
    def test_get_max_file_size_default(self):
        """Test getting default max file size."""
        with patch('apps.operations.services.validator.settings') as mock_settings:
            mock_settings.MAX_FILE_SIZE = {}
            
            result = Validator._get_max_file_size('video')
            
            assert result == 524288000  # 500MB default
    
    def test_get_supported_formats_from_settings(self):
        """Test getting supported formats from settings."""
        with patch('apps.operations.services.validator.settings') as mock_settings:
            mock_settings.SUPPORTED_FORMATS = {'video': ['mp4', 'mkv']}
            
            result = Validator._get_supported_formats('video')
            
            assert result == ['mp4', 'mkv']
    
    def test_get_supported_formats_default(self):
        """Test getting default supported formats."""
        with patch('apps.operations.services.validator.settings') as mock_settings:
            mock_settings.SUPPORTED_FORMATS = {}
            
            result = Validator._get_supported_formats('image')
            
            assert 'jpg' in result
            assert 'png' in result