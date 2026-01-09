"""
Validator Service for handling input validation.

This module provides centralized validation including:
- Validating operation existence
- Validating operation parameters
- Validating files for operations
- Validating and managing sessions
- Validating JSON fields and structures
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest

from .file_manager import (
    FileManager
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """
    Result object for validation operations.
    
    Provides a consistent structure for validation results
    with success status, validated data, and error messages.
    """

    def __init__(
        self,
        is_valid: bool,
        data: Optional[Dict[str, Any]] = None,
        errors: Optional[List[str]] = None,
    ):
        """
        Initialize a validation result.
        
        Args:
            is_valid: Whether validation passed
            data: Validated and normalized data (if valid)
            errors: List of error messages (if invalid)
        """
        self.is_valid = is_valid
        self.data = data or {}
        self.errors = errors or []

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean contexts."""
        return self.is_valid
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "data": self.data,
            "errors": self.errors,
        }


class Validator:
    """
    Service class for handling validation logic.
    
    All methods are static as this is a stateless service.
    Provides centralized validation for operations, parameters,
    files, and sessions.
    """

    # OPERATION VALIDATION METHODS

    @staticmethod
    def validate_operation_exists(operation_name: str) -> ValidationResult:
        """
        Validate that an operation exists in the registry.
        
        Args:
            operation_name: Name of the operation to validate
            
        Returns:
            ValidationResult with operation info if valid
        """
        from apps.processors.registry import get_registry
        from apps.processors.exceptions import OperationNotFoundError

        if not operation_name:
            return ValidationResult(
                is_valid=False,
                errors=["Operation name is required."]
            )

        if not isinstance(operation_name, str):
            return ValidationResult(
                is_valid=False,
                errors=["Operation name must be a string."]
            )
        
        registry = get_registry()

        if not registry.is_registered(operation_name):
            available_operations = [op for op in registry.list_registered_operations()]
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"Operation '{operation_name}' is not registered. "
                    f"Available operations: {', '.join(available_operations) if available_operations else 'none'}"
                ]
            )
        
        # Get operation info for the result
        operation_def = registry.get_operation(operation_name)

        return ValidationResult(
            is_valid=True,
            data={
                "operation_name": operation_name,
                "media_type": operation_def.media_type.value,
                "description": operation_def.description,
                "input_formats": operation_def.input_formats,
                "output_formats": operation_def.output_formats,
            }
        )
    

    @staticmethod
    def validate_operation_parameters(
        operation_name: str,
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate parameters for an operation against its schema.
        
        This method wraps the registry's validate_parameters method
        and returns a ValidationResult instead of raising exceptions.
        
        Args:
            operation_name: Name of the operation
            parameters: User-provided parameters to validate
            
        Returns:
            ValidationResult with validated parameters if valid
        """
        from apps.processors.registry import get_registry
        from apps.processors.exceptions import (
            OperationNotFoundError,
            InvalidParametersError,
        )

        # First validate that the operation exists
        operation_result = Validator.validate_operation_exists(operation_name)
        if not operation_result.is_valid:
            return operation_result
        
        # Ensure parameters is a dict
        if parameters is None:
            parameters = {}
        
        if not isinstance(parameters, dict):
            return ValidationResult(
                is_valid=False,
                errors=["Parameters must be a dictionary/object."]
            )
        
        registry = get_registry()

        try:
            validated_params = registry.validate_parameters(
                operation_name,
                parameters
            )
            return ValidationResult(
                is_valid=True,
                data={"validated_parameters": validated_params}
            )
        except InvalidParametersError as e:
            return ValidationResult(
                is_valid=False,
                errors=e.errors
            )
        except Exception as e:
            logger.error(f"Unexpected error validating parameters: {e}")
            return ValidationResult(
                is_valid=False,
                errors=["Parameter validation failed due to an unexpected error."]
            )
    
    # FILE VALIDATION METHODS

    @staticmethod
    def validate_file_for_operation(
        uploaded_file: UploadedFile,
        operation_name: str,
    ) -> ValidationResult:
        """
        Validate that a file is suitable for a specific operation.
        
        Checks:
        - File is provided and valid
        - File type matches operation's expected media type
        - File size is within limits
        - File format is supported
        
        Args:
            uploaded_file: The Django UploadedFile object
            operation_name: Name of the operation
            
        Returns:
            ValidationResult with file info if valid
        """
        from apps.processors.registry import get_registry

        errors = []

        # Validae file is provided
        if uploaded_file is None:
            return ValidationResult(
                is_valid=False,
                errors=["No file was uploaded"]
            )
        
        # Validate operation exists
        operation_result = Validator.validate_operation_exists(operation_name)
        if not operation_result.is_valid:
            return operation_result
        
        # Get operation's expected media type
        registry = get_registry()
        operation_def = registry.get_operation(operation_name)
        expected_media_type = operation_def.media_type.value

        # Get file info
        filename = uploaded_file.name or "unnamed"
        file_size = uploaded_file.size
        extension = FileManager.get_file_extension(filename)

        # Detect MIME type and media type
        mime_type = FileManager.detect_mime_type(uploaded_file, extension)
        detected_media_type = FileManager.get_media_type_from_mime_type(mime_type)

        # Validate media type matches (if operation specifies input formats)
        if operation_def.input_formats:
            if extension.lower() not in [fmt.lower() for fmt in operation_def.input_formats]:
                errors.append(
                    f"File format '.{extension}' is not supported for operation '{operation_name}'. "
                    f"Supported formats: {', '.join(operation_def.input_formats)}"
                )
        elif detected_media_type is None:
            errors.append(
                f"Unable to determine file type for '{filename}'. "
                f"Please upload a valid {expected_media_type} file."
            )
        elif detected_media_type != expected_media_type:
            errors.append(
                f"Operation '{operation_name}' requires a {expected_media_type} file, "
                f"but received a {detected_media_type} file."
            )
        
        # Validate file size
        if detected_media_type:
            max_size = Validator._get_max_file_size(detected_media_type)
            if file_size > max_size:
                max_size_mb = max_size / (1024 * 1024)
                file_size_mb = file_size / (1024 * 1024)
                errors.append(
                    f"File size ({file_size_mb:.2f} MB) exceeds the maximum "
                    f"allowed size ({max_size_mb:.0f} MB) for {detected_media_type} files."
                )
        
        # Validate file format is supported
        if detected_media_type:
            supported_formats = Validator._get_supported_formats(detected_media_type)
            if extension.lower() not in supported_formats:
                errors.append(
                    f"File format '.{extension}' is not supported. "
                    f"Supported {detected_media_type} formats: {', '.join(supported_formats)}"
                )
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        return ValidationResult(
            is_valid=True,
            data={
                "filename": filename,
                "file_size": file_size,
                "mime_type": mime_type,
                "media_type": detected_media_type,
                "extension": extension,
            }
        )
    

    @staticmethod
    def validate_file_basic(uploaded_file: UploadedFile) -> ValidationResult:
        """
        Perform basic validation on an uploaded file without operation context.
        
        Validates:
        - File is provided
        - File has a name
        - File has content
        - File type is recognized
        
        Args:
            uploaded_file: The Django UploadedFile object
            
        Returns:
            ValidationResult with file info if valid
        """
        errors = []

        if uploaded_file is None:
            return ValidationResult(
                is_valid=False,
                errors=["No file was uploaded."]
            )
        
        # Check filename
        filename = getattr(uploaded_file, 'name', None)
        if not filename:
            errors.append("Uploaded file must have a name.")
        
        # Check file size
        file_size = getattr(uploaded_file, 'size', 0)
        if file_size == 0:
            errors.append("Uploaded file is empty")
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        # Get file details
        extension = FileManager.get_file_extension(filename)
        mime_type = FileManager.detect_mime_type(uploaded_file, extension)
        media_type = FileManager.get_media_type_from_mime_type(mime_type)
        
        if media_type is None:
            return ValidationResult(
                is_valid=False,
                errors=[
                    f"File type '{extension}' is not recognized as a supported media type. "
                    "Please upload a video, image, or audio file."
                ]
            )
        
        return ValidationResult(
            is_valid=True,
            data={
                "filename": filename,
                "file_size": file_size,
                "extension": extension,
                "mime_type": mime_type,
                "media_type": media_type,
            }
        )
    
    # SESSION VALIDATION METHODS

    @staticmethod
    def validate_session(request: HttpRequest) -> ValidationResult:
        """
        Validate and ensure a session exists for a request.
        
        Creates a session if one doesn't exist.
        
        Args:
            request: Django HttpRequest object
            
        Returns:
            ValidationResult with session_key
        """
        if request is None:
            return ValidationResult(
                is_valid=False,
                errors=["Request object is required for session validation."]
            )
        
        # Check iif session exists, create if not
        if not hasattr(request, 'session'):
            return ValidationResult(
                is_valid=False,
                errors=["Request object does not have session support"]
            )
        
        if not request.session.session_key:
            request.session.create()
        
        session_key = request.session.session_key

        if not session_key:
            return ValidationResult(
                is_valid=False,
                errors=["Failed to create or retrieve session."]
            )
        
        return ValidationResult(
            is_valid=True,
            data={"session_key": session_key}
        )
    

    @staticmethod
    def get_or_create_session(request: HttpRequest) -> str:
        """
        Get existing session key or create a new session.
        
        This is a convenience method that returns just the session key.
        
        Args:
            request: Django HttpRequest object
            
        Returns:
            Session key string
            
        Raises:
            ValueError: If session cannot be created
        """
        result = Validator.validate_session(request)

        if not result.is_valid:
            raise ValueError("; ".join(result.errors))
        
        return result.data["session_key"]
    
    # JSON VALIDATION METHODS

    @staticmethod
    def validate_json_field(
        data: Any,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        allow_extra_keys: bool = False,
    ) -> ValidationResult:
        """
        Validate a JSON field/object structure.
        
        Args:
            data: Data to validate (should be dict after JSON parsing)
            required_keys: List of required keys
            optional_keys: List of optional keys
            allow_extra_keys: Whether to allow keys not in required or optional
            
        Returns:
            ValidationResult with validated data
        """
        errors = []
        required_keys = required_keys or []
        optional_keys = optional_keys or []

        # Check if data is a dict
        if not isinstance(data, dict):
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected a JSON object, got {type(data).__name__}"]
            )
        
        # Check for required keys
        for key in required_keys:
            if key not in data:
                errors.append(f"Required field '{key}' is missing")
            elif data[key] is None:
                errors.append(f"Required field '{key}' cannot be null")
        
        # Check for unknown keys
        if not allow_extra_keys:
            known_keys = set(required_keys) | set(optional_keys)
            unknown_keys = set(data.keys()) - known_keys
            if unknown_keys:
                errors.append(f"Unknown fields: {', '.join(sorted(unknown_keys))}")
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        return ValidationResult(is_valid=True, data=data)
    

    @staticmethod
    def validate_json_string(json_string: str) -> ValidationResult:
        """
        Validate and parse a JSON string.
        
        Args:
            json_string: String to parse as JSON
            
        Returns:
            ValidationResult with parsed data
        """
        if not json_string:
            return ValidationResult(
                is_valid=False,
                errors=["JSON string is empty"]
            )
        
        if not isinstance(json_string, str):
            return ValidationResult(
                is_valid=False,
                errors=[f"Expected a string, got {type(json_string).__name__}"]
            )
        
        try:
            parsed_data = json.loads(json_string)
            return ValidationResult(
                is_valid=True,
                data={"parsed": parsed_data}
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {str(e)}"]
            )
    
    # UUID VALIDATION

    @staticmethod
    def validate_uuid(value: Any, field_name: str = "id") -> ValidationResult:
        """
        Validate that a value is a valid UUID.
        
        Args:
            value: Value to validate
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult with UUID string
        """
        import uuid

        if value is None:
            return ValidationResult(
                is_valid=False,
                errors=[f"{field_name} is required."]
            )
        
        # If already a UUID object, convert to string
        if isinstance(value, uuid.UUID):
            return ValidationResult(
                is_valid=True,
                data={"uuid": str(value)}
            )
        
        # Convert to string and validate
        str_value = str(value).strip()
        
        # UUID regex pattern
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        if not uuid_pattern.match(str_value):
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid UUID format for {field_name}: '{str_value}'"]
            )
        
        try:
            parsed_uuid = uuid.UUID(str_value)
            return ValidationResult(
                is_valid=True,
                data={"uuid": str(parsed_uuid)}
            )
        except ValueError:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid UUID for {field_name}: '{str_value}'"]
            )
    
    # COMBINED VALIDATION METHODS

    @staticmethod
    def validate_operation_request(
        operation_name: str,
        parameters: Dict[str, Any],
        uploaded_file: UploadedFile,
    ) -> ValidationResult:
        """
        Perform complete validation for an operation request.
        
        Validates:
        - Operation exists
        - Parameters are valid
        - File is suitable for the operation
        
        Args:
            operation_name: Name of the operation
            parameters: Operation parameters
            uploaded_file: The uploaded file
            
        Returns:
            ValidationResult with all validated data
        """
        all_errors = []
        validated_data = {}

        # Validate operation exists
        operation_result = Validator.validate_operation_exists(operation_name)
        if not operation_result.is_valid:
            all_errors.extend(operation_result.errors)
        else:
            validated_data["operation"] = operation_result.data
        
        # Validate parameters (only if operation exists)
        if operation_result.is_valid:
            params_result = Validator.validate_operation_parameters(
                operation_name,
                parameters
            )
            if not params_result.is_valid:
                all_errors.extend(params_result.errors)
            else:
                validated_data["parameters"] = params_result.data.get("validated_parameters", {})
        
        # Validate file for operation (only if operation exists)
        if operation_result.is_valid:
            file_result = Validator.validate_file_for_operation(
                uploaded_file,
                operation_name
            )
            if not file_result.is_valid:
                all_errors.extend(file_result.errors)
            else:
                validated_data["file"] = file_result.data
        
        if all_errors:
            return ValidationResult(is_valid=False, errors=all_errors)
        
        return ValidationResult(is_valid=True, data=validated_data)
    
    # UTILITY METHODS

    @staticmethod
    def _get_max_file_size(media_type: str) -> int:
        """
        Get the maximum file size for a media type.
        
        Args:
            media_type: Media type category
            
        Returns:
            Maximum file size in bytes
        """
        max_sizes = getattr(settings, 'MAX_FILE_SIZE', {})
        
        defaults = {
            'video': 524288000,  # 500MB
            'image': 52428800,   # 50MB
            'audio': 104857600,  # 100MB
        }
        
        return max_sizes.get(media_type, defaults.get(media_type, 52428800))
    

    @staticmethod
    def _get_supported_formats(media_type: str) -> List[str]:
        """
        Get the supported formats for a media type.
        
        Args:
            media_type: Media type category
            
        Returns:
            List of supported format extensions
        """
        supported = getattr(settings, 'SUPPORTED_FORMATS', {})
        
        defaults = {
            'video': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
            'image': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
            'audio': ['mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'],
        }
        
        return supported.get(media_type, defaults.get(media_type, []))