# apps/core/middleware.py

"""
Custom middleware for the media processor application.

This module provides middleware classes for request processing,
including session management and request logging.
"""

import logging
import time
from typing import Callable

from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)


class SessionMiddleware:
    """
    Middleware to ensure every request has a valid session.
    
    This is particularly important for anonymous users who need
    sessions to track their operations.
    """
    
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Ensure session exists for all API requests
        if request.path.startswith('/api/'):
            if not request.session.session_key:
                request.session.create()
                logger.debug(f"Created new session: {request.session.session_key[:8]}...")
        
        response = self.get_response(request)
        return response


class RequestLoggingMiddleware:
    """
    Middleware for logging API requests and responses.
    
    Logs request method, path, status code, and duration.
    """
    
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Only log API requests
        if not request.path.startswith('/api/'):
            return self.get_response(request)
        
        # Record start time
        start_time = time.time()
        
        # Process request
        response = self.get_response(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log the request
        self._log_request(request, response, duration)
        
        return response
    
    def _log_request(
        self,
        request: HttpRequest,
        response: HttpResponse,
        duration: float
    ) -> None:
        """Log request details."""
        method = request.method
        path = request.path
        status_code = response.status_code
        duration_ms = duration * 1000
        
        # Get client IP
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            client_ip = x_forwarded_for.split(',')[0].strip()
        else:
            client_ip = request.META.get('REMOTE_ADDR', 'unknown')
        
        # Get session key (truncated for privacy)
        session_key = request.session.session_key
        session_display = f"{session_key[:8]}..." if session_key else "no-session"
        
        log_message = (
            f"{method} {path} - {status_code} - "
            f"{duration_ms:.2f}ms - {client_ip} - {session_display}"
        )
        
        if status_code >= 500:
            logger.error(log_message)
        elif status_code >= 400:
            logger.warning(log_message)
        else:
            logger.info(log_message)


class CORSDebugMiddleware:
    """
    Middleware for debugging CORS issues in development.
    
    To be used only in development environments.
    """
    
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        
        # Log CORS-related headers for debugging
        if request.method == 'OPTIONS':
            logger.debug(
                f"CORS Preflight: {request.path} - "
                f"Origin: {request.META.get('HTTP_ORIGIN', 'none')} - "
                f"Method: {request.META.get('HTTP_ACCESS_CONTROL_REQUEST_METHOD', 'none')}"
            )
        
        return response