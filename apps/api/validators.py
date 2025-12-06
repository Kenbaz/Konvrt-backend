# apps/api/validators.py
"""
Custom validators for the API layer.

This module provides reusable validators for file uploads,
operation parameters, and other API inputs.
"""

import logging
import mimetypes
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from rest_framework import serializers

logger = logging.getLogger(__name__)


class FileValidator:
    """
    Validator for uploaded files.
    
    Validates file size, type, and format based on media type.
    """
    
    def __init__(
        self,
        media_type: Optional[str] = None,
        max_size: Optional[int] = None,
        allowed_extensions: Optional[List[str]] = None,
        allowed_mime_types: Optional[List[str]] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            media_type: The expected media type ('video', 'image', 'audio')
            max_size: Maximum file size in bytes
            allowed_extensions: List of allowed file extensions
            allowed_mime_types: List of allowed MIME types
        """
        self.media_type = media_type
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions
        self.allowed_mime_types = allowed_mime_types
    
    def __call__(self, value: UploadedFile) -> UploadedFile:
        """Validate the uploaded file."""
        self.validate_not_empty(value)
        self.validate_size(value)
        self.validate_extension(value)
        self.validate_mime_type(value)
        return value
    
    def validate_not_empty(self, file: UploadedFile) -> None:
        """Ensure file is not empty."""
        if not file:
            raise serializers.ValidationError("No file was uploaded.")
        
        if file.size <= 0:
            raise serializers.ValidationError("The uploaded file is empty.")
    
    def validate_size(self, file: UploadedFile) -> None:
        """Validate file size."""
        max_size = self.max_size
        
        # If no explicit max size, use settings based on media type
        if max_size is None and self.media_type:
            max_size = settings.MAX_FILE_SIZE.get(self.media_type)
        
        if max_size and file.size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            file_size_mb = file.size / (1024 * 1024)
            raise serializers.ValidationError(
                f"File size ({file_size_mb:.1f} MB) exceeds the maximum "
                f"allowed size ({max_size_mb:.0f} MB)."
            )
    
    def validate_extension(self, file: UploadedFile) -> None:
        """Validate file extension."""
        filename = file.name or ""
        extension = self._get_extension(filename)
        
        if not extension:
            raise serializers.ValidationError(
                "Could not determine file type. "
                "Please ensure the file has a valid extension."
            )
        
        allowed = self.allowed_extensions
        
        # If no explicit allowed extensions, use settings based on media type
        if allowed is None and self.media_type:
            allowed = settings.SUPPORTED_FORMATS.get(self.media_type, [])
        
        if allowed and extension not in allowed:
            raise serializers.ValidationError(
                f"File type '.{extension}' is not supported. "
                f"Allowed types: {', '.join(allowed)}"
            )
    
    def validate_mime_type(self, file: UploadedFile) -> None:
        """Validate MIME type."""
        if not self.allowed_mime_types:
            return
        
        # Try to get MIME type from file
        mime_type = file.content_type
        if not mime_type:
            # Try to guess from filename
            mime_type, _ = mimetypes.guess_type(file.name or "")
        
        if mime_type and mime_type not in self.allowed_mime_types:
            raise serializers.ValidationError(
                f"MIME type '{mime_type}' is not supported."
            )
    
    def _get_extension(self, filename: str) -> str:
        """Extract and normalize file extension."""
        if '.' in filename:
            return filename.rsplit('.', 1)[-1].lower()
        return ""


class VideoFileValidator(FileValidator):
    """Validator specifically for video files."""
    
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(
            media_type='video',
            max_size=max_size,
        )


class ImageFileValidator(FileValidator):
    """Validator specifically for image files."""
    
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(
            media_type='image',
            max_size=max_size,
        )


class AudioFileValidator(FileValidator):
    """Validator specifically for audio files."""
    
    def __init__(self, max_size: Optional[int] = None):
        super().__init__(
            media_type='audio',
            max_size=max_size,
        )


def validate_file_size(file: UploadedFile, max_size: int) -> None:
    """
    Validate that a file doesn't exceed the maximum size.
    
    Args:
        file: The uploaded file
        max_size: Maximum size in bytes
        
    Raises:
        serializers.ValidationError: If file exceeds max size
    """
    if file.size > max_size:
        max_size_mb = max_size / (1024 * 1024)
        file_size_mb = file.size / (1024 * 1024)
        raise serializers.ValidationError(
            f"File size ({file_size_mb:.1f} MB) exceeds maximum "
            f"allowed size ({max_size_mb:.0f} MB)."
        )


def validate_file_type(
    file: UploadedFile,
    allowed_extensions: List[str],
) -> None:
    """
    Validate that a file has an allowed extension.
    
    Args:
        file: The uploaded file
        allowed_extensions: List of allowed extensions (without dots)
        
    Raises:
        serializers.ValidationError: If extension not allowed
    """
    filename = file.name or ""
    if '.' in filename:
        extension = filename.rsplit('.', 1)[-1].lower()
    else:
        extension = ""
    
    if not extension:
        raise serializers.ValidationError(
            "Could not determine file type. "
            "Please ensure the file has a valid extension."
        )
    
    if extension not in allowed_extensions:
        raise serializers.ValidationError(
            f"File type '.{extension}' is not supported. "
            f"Allowed types: {', '.join(allowed_extensions)}"
        )


def validate_operation_exists(operation_name: str) -> None:
    """
    Validate that an operation is registered.
    
    Args:
        operation_name: Name of the operation
        
    Raises:
        serializers.ValidationError: If operation not found
    """
    from apps.processors.registry import get_registry
    
    registry = get_registry()
    if not registry.is_registered(operation_name):
        available = registry.list_operations()
        available_names = [op['operation_name'] for op in available]
        raise serializers.ValidationError(
            f"Operation '{operation_name}' is not available. "
            f"Available operations: {', '.join(available_names)}"
        )


def validate_operation_parameters(
    operation_name: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate parameters for an operation.
    
    Args:
        operation_name: Name of the operation
        parameters: Parameters to validate
        
    Returns:
        Validated and normalized parameters
        
    Raises:
        serializers.ValidationError: If parameters are invalid
    """
    from apps.processors.registry import get_registry
    from apps.processors.exceptions import InvalidParametersError
    
    registry = get_registry()
    
    try:
        return registry.validate_parameters(operation_name, parameters)
    except InvalidParametersError as e:
        raise serializers.ValidationError({
            'parameters': e.errors if hasattr(e, 'errors') else [str(e)]
        })


def validate_uuid(value: str, field_name: str = "ID") -> str:
    """
    Validate that a string is a valid UUID.
    
    Args:
        value: String to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated UUID string
        
    Raises:
        serializers.ValidationError: If not a valid UUID
    """
    import uuid
    
    try:
        uuid.UUID(str(value))
        return str(value)
    except (ValueError, AttributeError):
        raise serializers.ValidationError(
            f"'{value}' is not a valid {field_name}."
        )


def validate_json_string(value: str) -> Dict[str, Any]:
    """
    Validate and parse a JSON string.
    
    Args:
        value: JSON string to validate
        
    Returns:
        Parsed JSON object
        
    Raises:
        serializers.ValidationError: If not valid JSON
    """
    import json
    
    if not isinstance(value, str):
        if isinstance(value, dict):
            return value
        raise serializers.ValidationError(
            "Value must be a JSON string or object."
        )
    
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise serializers.ValidationError(
                "JSON must be an object, not an array or primitive."
            )
        return parsed
    except json.JSONDecodeError as e:
        raise serializers.ValidationError(
            f"Invalid JSON: {str(e)}"
        )


def validate_positive_integer(
    value: Any,
    field_name: str = "Value",
    max_value: Optional[int] = None,
) -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        field_name: Name for error messages
        max_value: Optional maximum value
        
    Returns:
        Validated integer
        
    Raises:
        serializers.ValidationError: If validation fails
    """
    try:
        int_value = int(value)
    except (ValueError, TypeError):
        raise serializers.ValidationError(
            f"{field_name} must be an integer."
        )
    
    if int_value < 0:
        raise serializers.ValidationError(
            f"{field_name} must be a positive integer."
        )
    
    if max_value is not None and int_value > max_value:
        raise serializers.ValidationError(
            f"{field_name} must not exceed {max_value}."
        )
    
    return int_value