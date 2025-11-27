# apps/processors/exceptions.py

class ProcessorException(Exception):
    """Base exception for all processor-related errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "PROCESSOR_ERROR"
        self.details = details or {}
        super().__init__(message)
    

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

    def __init__(self, operation_name: str, reason: str, is_retryable: bool = False):
        self.operation_name = operation_name
        self.reason = reason
        self.is_retryable = is_retryable
        super().__init__(
            message=f"Processing failed for operation '{operation_name}': {reason}",
            code="PROCESSING_ERROR",
            details={
                "operation_name": operation_name,
                "reason": reason,
                "is_retryable": is_retryable
            }
        )


class FFmpegError(ProcessorException):
    """Raised when FFmpeg processing fails."""

    def __init__(self, command: str, stderr: str, return_code: int, is_retryable: bool = False):
        self.command = command
        self.stderr = stderr
        self.return_code = return_code
        super().__init__(
            operation_name="ffmpeg",
            reason=f"FFmpeg exited with code {return_code}: {stderr[:500]}",
            is_retryable=is_retryable
        )
        self.code = "FFMPEG_ERROR"
        self.details.update({
            "command": command,
            "return_code": return_code,
            "stderr": stderr[:1000], # Limit stderr length
        })


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