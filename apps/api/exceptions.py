"""
Custom exception handling for the API layer.

This module provides a unified exception handler that converts all exceptions
to a standardized API response format. It handles both DRF exceptions and
custom application exceptions.
"""
import logging
from typing import Any, Dict, Optional

from django.core.exceptions import PermissionDenied, ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler

logger = logging.getLogger(__name__)


# HTTP status code mapping for custom exceptions
EXCEPTION_STATUS_CODES = {
    # File-related errors
    "FILE_TOO_LARGE": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "UNSUPPORTED_FORMAT": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "FILE_CORRUPTED": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "FILE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "STORAGE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "MIME_TYPE_DETECTION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
    
    # Operation-related errors
    "OPERATION_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "OPERATION_ACCESS_DENIED": status.HTTP_403_FORBIDDEN,
    "OPERATION_IN_PROGRESS": status.HTTP_409_CONFLICT,
    "JOB_ALREADY_PROCESSING": status.HTTP_409_CONFLICT,
    "OPERATION_NOT_RETRYABLE": status.HTTP_400_BAD_REQUEST,
    "OPERATION_NOT_DELETABLE": status.HTTP_400_BAD_REQUEST,
    "OPERATION_QUEUING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "INVALID_OPERATION_STATE": status.HTTP_400_BAD_REQUEST,
    "OPERATION_VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
    
    # Processor-related errors
    "INVALID_PARAMETERS": status.HTTP_400_BAD_REQUEST,
    "PROCESSING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "FFMPEG_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "IMAGE_PROCESSING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "AUDIO_PROCESSING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "OUTPUT_CREATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "TEMP_FILE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "RESOURCE_LIMIT_ERROR": status.HTTP_507_INSUFFICIENT_STORAGE,
    "PROCESSING_TIMEOUT": status.HTTP_504_GATEWAY_TIMEOUT,
    "PROCESSING_CANCELLED": status.HTTP_499_CLIENT_CLOSED_REQUEST if hasattr(status, 'HTTP_499_CLIENT_CLOSED_REQUEST') else 499,
    
    # General errors
    "OPERATIONS_ERROR": status.HTTP_400_BAD_REQUEST,
    "PROCESSOR_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
}


class APIError(APIException):
    """
    Base API exception with standardized error format.
    
    All API errors should use this format for consistency.
    """
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "API_ERROR"
    default_detail = "An error occurred."
    
    def __init__(
        self,
        detail: str = None,
        code: str = None,
        status_code: int = None,
        errors: list = None,
        metadata: dict = None,
    ):
        self.detail = detail or self.default_detail
        self.code = code or self.default_code
        if status_code is not None:
            self.status_code = status_code
        self.errors = errors or []
        self.metadata = metadata or {}
        super().__init__(detail=self.detail, code=self.code)


class ValidationError(APIError):
    """Raised when request validation fails."""
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "VALIDATION_ERROR"
    default_detail = "Request validation failed."


class NotFoundError(APIError):
    """Raised when a requested resource is not found."""
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "NOT_FOUND"
    default_detail = "The requested resource was not found."


class PermissionDeniedError(APIError):
    """Raised when access to a resource is denied."""
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "PERMISSION_DENIED"
    default_detail = "You do not have permission to perform this action."


class ConflictError(APIError):
    """Raised when there is a conflict with the current state."""
    status_code = status.HTTP_409_CONFLICT
    default_code = "CONFLICT"
    default_detail = "The request conflicts with the current state of the resource."


class RateLimitExceededError(APIError):
    """Raised when rate limit is exceeded."""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_code = "RATE_LIMIT_EXCEEDED"
    default_detail = "Too many requests. Please try again later."


class ServiceUnavailableError(APIError):
    """Raised when a required service is unavailable."""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "SERVICE_UNAVAILABLE"
    default_detail = "The service is temporarily unavailable."


def build_error_response(
    message: str,
    code: str,
    status_code: int,
    details: Optional[Dict[str, Any]] = None,
    errors: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized error response.
    
    Args:
        message: Human-readable error message
        code: Machine-readable error code
        status_code: HTTP status code
        details: Additional error details
        errors: List of specific errors (for validation)
        metadata: Additional metadata
        
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        }
    }

    if details:
        response["error"]["details"] = details
    
    if errors:
        response["error"]["errors"] = errors
    
    if metadata:
        response["error"]["metadata"] = metadata

    return response


def custom_exception_handler(exc: Exception, context: dict) -> Optional[Response]:
    """
    Custom exception handler for Django REST Framework.
    
    This handler converts all exceptions to a standardized format and ensures
    consistent error responses across the API.
    
    Args:
        exc: The exception that was raised
        context: Dictionary containing request, view, args, kwargs
        
    Returns:
        Response object with standardized error format
    """
    # Get the request for logging
    request = context.get("request")
    view = context.get("view")

    # Let DRF handle standard exceptions first
    response = drf_exception_handler(exc, context)

    # Handle DRF exceptions
    if response is not None:
        return _handle_drf_exception(exc, response, request, view)
    
    # Handle custom application exceptions
    return _handle_custom_exception(exc, request, view)


def _handle_drf_exception(
    exc: Exception,
    response: Response,
    request,
    view,
) -> Response:
    """Handle DRF-specific exceptions."""

    # Extract error information
    if hasattr(exc, "detail"):
        if isinstance(exc.detail, dict):
            # Validation errors with field-specific messages
            errors = []
            for field, messages in exc.detail.items():
                if isinstance(messages, list):
                    for msg in messages:
                        errors.append({"field": field, "message": str(msg)})
                else:
                    errors.append({"field": field, "message": str(messages)})
            message = "Validation failed."
            code = "VALIDATION_ERROR"
        elif isinstance(exc.detail, list):
            errors = [{"message": str(msg)} for msg in exc.detail]
            message = str(exc.detail[0]) if exc.detail else "An error occurred."
            code = getattr(exc, "default_code", "API_ERROR").upper()
        else:
            message = str(exc.detail)
            code = getattr(exc, "default_code", "API_ERROR").upper()
            errors = []
    else:
        message = str(exc)
        code = "API_ERROR"
        errors = []
    
    error_response = build_error_response(
        message=message,
        code=code,
        status_code=response.status_code,
        errors=errors if errors else None,
    )

    # Log the error
    _log_error(exc, request, view, response.status_code)

    response.data = error_response
    return response


def _handle_custom_exception(
    exc: Exception,
    request,
    view,
) -> Optional[Response]:
    """Handle custom application exceptions."""
    from apps.operations.exceptions import OperationsException
    from apps.processors.exceptions import ProcessorException

    # Handle Operations exceptions
    if isinstance(exc, OperationsException):
        status_code = EXCEPTION_STATUS_CODES.get(exc.code, status.HTTP_400_BAD_REQUEST)
        error_response = build_error_response(
            message=exc.message,
            code=exc.code,
            status_code=status_code,
            details=exc.details if exc.details else None,
        )
        _log_error(exc, request, view, status_code)
        return Response(error_response, status=status_code)
    
    # Handle Processor exceptions
    if isinstance(exc, ProcessorException):
        status_code = EXCEPTION_STATUS_CODES.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        error_response = build_error_response(
            message=exc.message,
            code=exc.code,
            status_code=status_code,
            details=exc.details if exc.details else None,
        )
        _log_error(exc, request, view, status_code)
        return Response(error_response, status=status_code)
    
    # Handle Django Http404
    if isinstance(exc, Http404):
        error_response = build_error_response(
            message=str(exc) if str(exc) else "The requested resource was not found.",
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )
        _log_error(exc, request, view, status.HTTP_404_NOT_FOUND)
        return Response(error_response, status=status.HTTP_404_NOT_FOUND)
    
    # Handle Django's PermissionDenied
    if isinstance(exc, PermissionDenied):
        error_response = build_error_response(
            message=str(exc) if str(exc) else "Permission denied.",
            code="PERMISSION_DENIED",
            status_code=status.HTTP_403_FORBIDDEN,
        )
        _log_error(exc, request, view, status.HTTP_403_FORBIDDEN)
        return Response(error_response, status=status.HTTP_403_FORBIDDEN)
    
    # Handle Django's ValidationError
    if isinstance(exc, DjangoValidationError):
        if hasattr(exc, "message_dict"):
            errors = []
            for field, messages in exc.message_dict.items():
                for msg in messages:
                    errors.append({"field": field, "message": str(msg)})
            message = "Validation failed."
        else:
            errors = [{"message": str(msg)} for msg in exc.messages]
            message = str(exc.messages[0]) if exc.messages else "Validation failed."
        
        error_response = build_error_response(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            errors=errors,
        )
        _log_error(exc, request, view, status.HTTP_400_BAD_REQUEST)
        return Response(error_response, status=status.HTTP_400_BAD_REQUEST)
    
    # Handle unexpected exceptions (500 errors)
    logger.exception(
        f"Unhandled exception in {view.__class__.__name__ if view else 'unknown'}: {exc}",
        exc_info=exc,
    )
    
    error_response = build_error_response(
        message="An unexpected error occurred. Please try again later.",
        code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    return Response(error_response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _log_error(exc: Exception, request, view, status_code: int) -> None:
    """Log error details for debugging."""
    view_name = view.__class__.__name__ if view else "unknown"
    method = request.method if request else "unknown"
    path = request.path if request else "unknown"
    
    log_message = f"API Error [{status_code}] {method} {path} - {view_name}: {exc}"
    
    if status_code >= 500:
        logger.error(log_message, exc_info=exc)
    elif status_code >= 400:
        logger.warning(log_message)
    else:
        logger.info(log_message)