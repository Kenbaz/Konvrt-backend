"""
Exceptions for the processors app.

This module defines custom exceptions for media processing operations,
including errors from FFmpeg, file validation, and general processing.
"""

class ProcessorException(Exception):
    """Base exception for all processor-related errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "PROCESSOR_ERROR"
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary representation."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }
    

class OperationNotFoundError(ProcessorException):
    """Raised when an operation is not found in the registry."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        super().__init__(
            message=f"Operation '{operation_name}' is not registered.",
            code="OPERATION_NOT_FOUND",
            details={"operation_name": operation_name}
        )


class OperationRegistrationError(ProcessorException):
    """Raised when there is an error registering an operation."""

    def __init__(self, operation_name: str, reason: str):
        self.operation_name = operation_name
        self.reason = reason
        super().__init__(
            message=f"Failed to register operation '{operation_name}': {reason}",
            code="OPERATION_REGISTRATION_ERROR",
            details={"operation_name": operation_name, "reason": reason}
        )


class InvalidParametersError(ProcessorException):
    """Raised when operation parameters are invalid."""

    def __init__(self, operation_name: str, errors: list):
        self.operation_name = operation_name
        self.errors = errors
        error_message = "; ".join(errors)
        super().__init__(
            message=f"Invalid parameters for operation '{operation_name}': {error_message}",
            code="INVALID_PARAMETERS",
            details={"operation_name": operation_name, "errors": errors}
        )


class ProcessingError(ProcessorException):
    """Raised when processing fails."""

    def __init__(
        self,
        operation_name: str,
        reason: str,
        is_retryable: bool = False,
        original_error: Exception = None,
    ):
        self.operation_name = operation_name
        self.reason = reason
        self.is_retryable = is_retryable
        self.original_error = original_error
        
        super().__init__(
            message=f"Processing failed for operation '{operation_name}': {reason}",
            code="PROCESSING_ERROR",
            details={
                "operation_name": operation_name,
                "reason": reason,
                "is_retryable": is_retryable,
            }
        )


class FFmpegError(ProcessorException):
    """Raised when FFmpeg processing fails."""

    def __init__(
        self,
        command: str,
        stderr: str,
        return_code: int,
        is_retryable: bool = False,
    ):
        self.command = command
        self.stderr = stderr
        self.return_code = return_code
        self.is_retryable = is_retryable
        
        # Extract a user-friendly message from stderr
        user_message = self._extract_user_message(stderr)
        
        super().__init__(
            message=f"FFmpeg error (code {return_code}): {user_message}",
            code="FFMPEG_ERROR",
            details={
                "command": command,
                "return_code": return_code,
                "stderr": stderr[:2000],  # Limit stderr length
                "is_retryable": is_retryable,
            }
        )
    
    def _extract_user_message(self, stderr: str) -> str:
        """
        Extract a user-friendly message from FFmpeg stderr.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            User-friendly error message
        """
        stderr_lower = stderr.lower()
        
        # Common FFmpeg errors and their user-friendly messages
        error_patterns = [
            ("no such file or directory", "Input file not found"),
            ("invalid data found", "Input file appears to be corrupted"),
            ("codec not found", "Required codec is not available"),
            ("encoder not found", "Required encoder is not available"),
            ("decoder not found", "Required decoder is not available"),
            ("permission denied", "Permission denied accessing file"),
            ("no space left", "Insufficient disk space"),
            ("out of memory", "Insufficient memory for processing"),
            ("unrecognized option", "Invalid FFmpeg option specified"),
            ("does not contain any stream", "Input file has no media streams"),
            ("could not find codec", "Could not find required codec"),
            ("unsupported codec", "Input uses an unsupported codec"),
        ]
        
        for pattern, message in error_patterns:
            if pattern in stderr_lower:
                return message
        
        # Return first line of stderr if no pattern matches
        first_line = stderr.split('\n')[0].strip()
        if first_line:
            return first_line[:200]
        
        return "FFmpeg processing failed"


class FFprobeError(ProcessorException):
    """Raised when FFprobe analysis fails."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        
        super().__init__(
            message=f"FFprobe failed for '{file_path}': {reason}",
            code="FFPROBE_ERROR",
            details={
                "file_path": file_path,
                "reason": reason,
            }
        )


class FileValidationError(ProcessorException):
    """Raised when file validation fails."""
    
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(
            message=f"File validation failed for '{filename}': {reason}",
            code="FILE_VALIDATION_ERROR",
            details={"filename": filename, "reason": reason}
        )


class UnsupportedFormatError(ProcessorException):
    """Raised when a file format is not supported."""
    
    def __init__(
        self,
        filename: str,
        format_type: str,
        supported_formats: list = None,
    ):
        self.filename = filename
        self.format_type = format_type
        self.supported_formats = supported_formats or []
        
        if supported_formats:
            supported_str = ", ".join(supported_formats)
            message = f"Format '{format_type}' is not supported. Supported formats: {supported_str}"
        else:
            message = f"Format '{format_type}' is not supported"
        
        super().__init__(
            message=message,
            code="UNSUPPORTED_FORMAT",
            details={
                "filename": filename,
                "format": format_type,
                "supported_formats": supported_formats,
            }
        )


class UnsupportedCodecError(ProcessorException):
    """Raised when a codec is not supported."""
    
    def __init__(
        self,
        filename: str,
        codec: str,
        media_type: str,
        supported_codecs: list = None,
    ):
        self.filename = filename
        self.codec = codec
        self.media_type = media_type
        self.supported_codecs = supported_codecs or []
        
        super().__init__(
            message=f"{media_type.capitalize()} codec '{codec}' is not supported for processing",
            code="UNSUPPORTED_CODEC",
            details={
                "filename": filename,
                "codec": codec,
                "media_type": media_type,
                "supported_codecs": supported_codecs,
            }
        )


class MediaDurationError(ProcessorException):
    """Raised when media duration exceeds limits or cannot be determined."""
    
    def __init__(
        self,
        filename: str,
        duration: float = None,
        max_duration: float = None,
    ):
        self.filename = filename
        self.duration = duration
        self.max_duration = max_duration
        
        if duration is None:
            message = f"Could not determine duration for '{filename}'"
        elif max_duration and duration > max_duration:
            message = f"Media duration ({duration:.1f}s) exceeds maximum allowed ({max_duration}s)"
        else:
            message = f"Invalid media duration: {duration}"
        
        super().__init__(
            message=message,
            code="MEDIA_DURATION_ERROR",
            details={
                "filename": filename,
                "duration": duration,
                "max_duration": max_duration,
            }
        )


class ImageProcessingError(ProcessorException):
    """Raised when image processing with Pillow fails."""

    def __init__(self, filename: str, reason: str, operation: str = None):
        self.filename = filename
        self.reason = reason
        self.operation = operation
        
        if operation:
            message = f"Image {operation} failed for '{filename}': {reason}"
        else:
            message = f"Image processing failed for '{filename}': {reason}"
        
        super().__init__(
            message=message,
            code="IMAGE_PROCESSING_ERROR",
            details={
                "filename": filename,
                "reason": reason,
                "operation": operation,
            }
        )


class OutputCreationError(ProcessorException):
    """Raised when output file cannot be created or is invalid."""
    
    def __init__(self, output_path: str, reason: str):
        self.output_path = output_path
        self.reason = reason
        
        super().__init__(
            message=f"Failed to create output file '{output_path}': {reason}",
            code="OUTPUT_CREATION_ERROR",
            details={
                "output_path": output_path,
                "reason": reason,
            }
        )


class TempFileError(ProcessorException):
    """Raised when temporary file operations fail."""
    
    def __init__(self, operation: str, path: str, reason: str):
        self.operation = operation
        self.path = path
        self.reason = reason
        
        super().__init__(
            message=f"Temp file {operation} failed for '{path}': {reason}",
            code="TEMP_FILE_ERROR",
            details={
                "operation": operation,
                "path": path,
                "reason": reason,
            }
        )


class ResourceLimitError(ProcessorException):
    """Raised when a resource limit is exceeded."""
    
    def __init__(self, resource: str, current: float, limit: float, unit: str = ""):
        self.resource = resource
        self.current = current
        self.limit = limit
        self.unit = unit
        
        unit_str = f" {unit}" if unit else ""
        message = f"{resource} limit exceeded: {current}{unit_str} (limit: {limit}{unit_str})"
        
        super().__init__(
            message=message,
            code="RESOURCE_LIMIT_ERROR",
            details={
                "resource": resource,
                "current": current,
                "limit": limit,
                "unit": unit,
            }
        )


class ProcessingTimeoutError(ProcessorException):
    """Raised when processing times out."""

    def __init__(self, operation_name: str, timeout_seconds: int):
        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        
        super().__init__(
            message=f"Operation '{operation_name}' timed out after {timeout_seconds} seconds",
            code="PROCESSING_TIMEOUT",
            details={
                "operation_name": operation_name,
                "timeout_seconds": timeout_seconds,
            }
        )


class ProcessingCancelledError(ProcessorException):
    """Raised when processing is cancelled by user or system."""
    
    def __init__(self, operation_id: str, reason: str = "User cancelled"):
        self.operation_id = operation_id
        self.reason = reason
        
        super().__init__(
            message=f"Processing cancelled: {reason}",
            code="PROCESSING_CANCELLED",
            details={
                "operation_id": operation_id,
                "reason": reason,
            }
        )


class AudioProcessingError(ProcessorException):
    """Raised when audio processing fails."""

    def __init__(self, filename: str, reason: str, operation: str = None):
        self.filename = filename
        self.reason = reason
        self.operation = operation
        
        if operation:
            message = f"Audio {operation} failed for '{filename}': {reason}"
        else:
            message = f"Audio processing failed for '{filename}': {reason}"
        
        super().__init__(
            message=message,
            code="AUDIO_PROCESSING_ERROR",
            details={
                "filename": filename,
                "reason": reason,
                "operation": operation,
            }
        )