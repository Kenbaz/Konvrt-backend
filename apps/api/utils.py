# apps/api/utils.py

"""
Utility functions for the API layer.

This module provides helper functions for creating standardized API responses,
handling sessions, and other common API operations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from django.http import HttpRequest
from rest_framework import status
from urllib.parse import quote
from rest_framework.response import Response

logger = logging.getLogger(__name__)


def success_response(
    data: Optional[Union[Dict, List, Any]] = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
    metadata: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Create a standardized success response.
    
    Args:
        data: The response data (can be dict, list, or any serializable value)
        message: Optional success message
        status_code: HTTP status code (default 200)
        metadata: Optional metadata (pagination info, etc.)
        
    Returns:
        Response object with standardized success format
    """
    response_data = {
        "success": True
    }

    if data is not None:
        response_data["data"] = data
    
    if message:
        response_data["message"] = message
    
    if metadata:
        response_data["metadata"] = metadata
    
    return Response(response_data, status=status_code)


def error_response(
    message: str,
    code: str = "ERROR",
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Create a standardized error response.
    
    Args:
        message: Human-readable error message
        code: Machine-readable error code
        status_code: HTTP status code (default 400)
        details: Additional error details
        errors: List of specific errors (for validation)
        metadata: Optional metadata
        
    Returns:
        Response object with standardized error format
    """
    response_data = {
        "success": False,
        "error": {
            "message": message,
            "code": code,
        }
    }

    if details:
        response_data["error"]["details"] = details

    if errors:
        response_data["error"]["errors"] = errors

    if metadata:
        response_data["metadata"] = metadata

    return Response(response_data, status=status_code)


def paginated_response(
    data: List[Any],
    page: int,
    page_size: int,
    total_count: int,
    message: Optional[str] = None,
) -> Response:
    """
    Create a standardized paginated response.
    
    Args:
        data: List of items for current page
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total_count: Total number of items across all pages
        message: Optional message
        
    Returns:
        Response with pagination metadata
    """
    total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0
    has_next = page < total_pages
    has_previous = page > 1

    metadata = {
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_previous": has_previous,
        }
    }

    return success_response(
        data=data,
        message=message,
        metadata=metadata
    )


def get_session_key(request: HttpRequest) -> str:
    """
    Get or create a session key for the request.
    
    This ensures every request has a valid session key for tracking
    anonymous users and their operations.
    
    Args:
        request: The HTTP request object
        
    Returns:
        The session key as a string
    """
    # Ensure session exists
    if not request.session.session_key:
        request.session.create()
    
    return request.session.session_key


def get_client_ip(request: HttpRequest) -> str:
    """
    Get the client's IP address from the request.
    
    Handles proxied requests by checking X-Forwarded-For header.
    
    Args:
        request: The HTTP request object
        
    Returns:
        Client IP address as string
    """
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        # Take the first IP in the list
        ip = x_forwarded_for.split(",")[0].strip()
    else:
        ip = request.META.get("REMOTE_ADDR", "unknown")
    
    return ip


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Human-readable size string (e.g., "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string (e.g., "2m 30s")
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def sanitize_filename_for_response(filename: str) -> str:
    """
    Sanitize a filename for use in Content-Disposition header.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for HTTP headers
    """
    # Remove any path components
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove any characters that might cause issues in headers
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Ensure it's not empty and doesn't start with a dot
    if not sanitized or sanitized.startswith("."):
        sanitized = "file" + sanitized
    
    return sanitized


def build_download_headers(
    filename: str,
    file_size: Optional[int] = None,
    content_type: Optional[str] = None,
    inline: bool = False,
) -> Dict[str, str]:
    """
    Build HTTP headers for file download responses.
    
    Creates proper Content-Disposition header with both ASCII and UTF-8
    encoded filename for maximum browser compatibility.
    
    Args:
        filename: The filename to suggest for download
        file_size: Optional file size in bytes
        content_type: Optional MIME type
        inline: If True, suggest inline display instead of download
        
    Returns:
        Dictionary of HTTP headers
    """
    headers = {}
    
    # Sanitize filename for header
    safe_filename = sanitize_filename_for_header(filename)
    
    # Build Content-Disposition header
    disposition_type = "inline" if inline else "attachment"
    
    ascii_filename = safe_filename.encode('ascii', 'replace').decode('ascii')
    utf8_filename = quote(filename, safe='')
    
    content_disposition = f'{disposition_type}; filename="{ascii_filename}"; filename*=UTF-8\'\'{utf8_filename}'
    headers['Content-Disposition'] = content_disposition
    
    if file_size is not None:
        headers['Content-Length'] = str(file_size)
    
    if content_type:
        headers['Content-Type'] = content_type
    
    headers['Cache-Control'] = 'private, no-cache, no-store, must-revalidate'
    headers['Pragma'] = 'no-cache'
    headers['Expires'] = '0'
    
    headers['Access-Control-Expose-Headers'] = 'Content-Disposition, Content-Length, Content-Type'
    
    return headers


def sanitize_filename_for_header(filename: str) -> str:
    """
    Sanitize a filename for use in HTTP headers.
    
    Removes or replaces characters that could cause issues in
    Content-Disposition headers.
    
    Args:
        filename: The original filename
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "download"
    
    sanitized = filename
    
    # Replace backslashes and forward slashes
    sanitized = sanitized.replace('\\', '_').replace('/', '_')
    
    # Replace quotes
    sanitized = sanitized.replace('"', "'").replace('\n', '').replace('\r', '')
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    if not sanitized or sanitized.strip() == '':
        return "download"
    
    return sanitized.strip()


def parse_boolean_param(value: Any, default: bool = False) -> bool:
    """
    Parse a boolean parameter from query string or request data.
    
    Args:
        value: The value to parse
        default: Default value if parsing fails
        
    Returns:
        Boolean value
    """
    if value is None:
        return default
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    return default


def parse_int_param(
    value: Any,
    default: int = 0,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """
    Parse an integer parameter with optional bounds.
    
    Args:
        value: The value to parse
        default: Default value if parsing fails
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Integer value (bounded if limits provided)
    """
    if value is None:
        return default
    
    try:
        result = int(value)
    except (ValueError, TypeError):
        return default
    
    if min_value is not None:
        result = max(result, min_value)
    
    if max_value is not None:
        result = min(result, max_value)
    
    return result