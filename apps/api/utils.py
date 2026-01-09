"""
API utility functions for the media processing platform.

This module provides common utility functions for API responses,
session management, and other API-related operations.
"""

import logging
from typing import Any, Dict, List, Optional

from django.http import HttpRequest
from rest_framework import status
from rest_framework.response import Response

logger = logging.getLogger(__name__)


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
    metadata: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
        metadata: Optional additional metadata
        
    Returns:
        DRF Response object
    """
    response_data: Dict[str, Any] = {
        "success": True,
    }
    
    if data is not None:
        response_data["data"] = data
    
    if message:
        response_data["message"] = message
    
    if metadata:
        response_data.update(metadata)
    
    return Response(response_data, status=status_code)


def error_response(
    message: str,
    code: str = "ERROR",
    status_code: int = status.HTTP_400_BAD_REQUEST,
    errors: Optional[List[Dict[str, Any]]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Response:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        code: Error code for client handling
        status_code: HTTP status code
        errors: Optional list of field-specific errors
        details: Optional additional error details
        
    Returns:
        DRF Response object
    """
    response_data: Dict[str, Any] = {
        "success": False,
        "error": {
            "message": message,
            "code": code,
        },
    }
    
    if errors:
        response_data["error"]["errors"] = errors
    
    if details:
        response_data["error"]["details"] = details
    
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
    Get the session key for the request.
    
    The SessionMiddleware in apps.core.middleware handles:
    1. Loading session from X-Session-ID header
    2. Falling back to cookie-based session
    3. Creating a new session if needed
    
    This function simply returns the session key that the middleware
    has already set up.
    
    Args:
        request: The HTTP request object
        
    Returns:
        The session key as a string
    """
    if not request.session.session_key:
        request.session.create()
        logger.warning("Session not initialized by middleware, creating new one")
    
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
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def build_download_headers(
    filename: str,
    content_type: str,
    file_size: Optional[int] = None,
    inline: bool = False,
) -> Dict[str, str]:
    """
    Build HTTP headers for file download response.
    
    Args:
        filename: Name of the file for Content-Disposition
        content_type: MIME type of the file
        file_size: Optional file size for Content-Length
        inline: If True, display in browser; if False, force download
        
    Returns:
        Dictionary of HTTP headers
    """
    disposition = "inline" if inline else "attachment"
    
    headers = {
        "Content-Type": content_type,
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "Cache-Control": "private, no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    
    if file_size is not None:
        headers["Content-Length"] = str(file_size)
    
    return headers