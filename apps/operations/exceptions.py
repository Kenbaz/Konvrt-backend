# apps/operations/exceptions.py

class OperationsException(Exception):
    """Base exception for all operations-related errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "OPERATIONS_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class FileTooLargeError(OperationsException):
    """Raised when an uploaded file exceeds the size limit"""

    def __init__(self, filename: str, file_size: int, max_size: int, media_type: str):
        self.filename = filename
        self.file_size = file_size
        self.max_size = max_size
        self.media_type = media_type

        # Convert to human-readable sizes
        file_size_mb = file_size / (1024 * 1024)
        max_size_mb = max_size / (1024 * 1024)

        super().__init__(
            message=f"File '{filename}' ({file_size_mb:.2f} MB) exceeds the maximum size limit of {max_size_mb:.0f} MB for {media_type} files.",
            code="FILE_TOO_LARGE",
            details={
                "filename": filename,
                "file_size": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "max_size": max_size,
                "max_size_mb": round(max_size_mb, 0),
                "media_type": media_type,
            }
        )


class UnsupportedFileFormatError(OperationsException):
    """Raised when a file format is not supported."""
    
    def __init__(self, filename: str, extension: str, media_type: str, supported_formats: list):
        self.filename = filename
        self.extension = extension
        self.media_type = media_type
        self.supported_formats = supported_formats
        
        supported_str = ", ".join(supported_formats)
        super().__init__(
            message=f"File format '.{extension}' is not supported for {media_type}. Supported formats: {supported_str}",
            code="UNSUPPORTED_FORMAT",
            details={
                "filename": filename,
                "extension": extension,
                "media_type": media_type,
                "supported_formats": supported_formats,
            }
        )


class FileCorruptedError(OperationsException):
    """Raised when a file appears to be corrupted or unreadable."""
    
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(
            message=f"File '{filename}' appears to be corrupted or invalid: {reason}",
            code="FILE_CORRUPTED",
            details={
                "filename": filename,
                "reason": reason,
            }
        )


class FileNotFoundError(OperationsException):
    """Raised when an expected file is not found."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__(
            message=f"File not found: {file_path}",
            code="FILE_NOT_FOUND",
            details={
                "file_path": file_path,
            }
        )


class OperationNotFoundError(OperationsException):
    """Raised when an operation record is not found in the database."""
    
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        super().__init__(
            message=f"Operation with ID '{operation_id}' was not found.",
            code="OPERATION_NOT_FOUND",
            details={
                "operation_id": operation_id,
            }
        )


class OperationAccessDeniedError(OperationsException):
    """Raised when a user tries to access an operation they don't own."""
    
    def __init__(self, operation_id: str, session_key: str = None):
        self.operation_id = operation_id
        self.session_key = session_key
        super().__init__(
            message=f"Access denied to operation '{operation_id}'.",
            code="OPERATION_ACCESS_DENIED",
            details={
                "operation_id": operation_id,
            }
        )


class OperationInProgressError(OperationsException):
    """Raised when trying to delete or modify an operation that is in progress."""
    
    def __init__(self, operation_id: str, status: str):
        self.operation_id = operation_id
        self.status = status
        super().__init__(
            message=f"Cannot modify operation '{operation_id}' while it is in '{status}' status.",
            code="OPERATION_IN_PROGRESS",
            details={
                "operation_id": operation_id,
                "status": status,
            }
        )


class StorageError(OperationsException):
    """Raised when there's an error with file storage operations."""
    
    def __init__(self, operation: str, path: str, reason: str):
        self.operation = operation
        self.path = path
        self.reason = reason
        super().__init__(
            message=f"Storage error during {operation}: {reason}",
            code="STORAGE_ERROR",
            details={
                "operation": operation,
                "path": path,
                "reason": reason,
            }
        )


class MimeTypeDetectionError(OperationsException):
    """Raised when unable to detect file MIME type."""
    
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(
            message=f"Unable to detect MIME type for file '{filename}'.",
            code="MIME_TYPE_DETECTION_ERROR",
            details={
                "filename": filename,
            }
        )