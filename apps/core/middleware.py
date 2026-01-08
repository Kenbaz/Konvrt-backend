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
from django.contrib.sessions.backends.db import SessionStore
from django.utils import timezone

logger = logging.getLogger(__name__)

# Custom header name for session ID (used for Safari/iOS compatibility)
SESSION_HEADER_NAME = 'X-Session-ID'


class SessionMiddleware:
    """
    Middleware to ensure every request has a valid session.
    
    This middleware:
    1. Checks for a valid session ID in the X-Session-ID header first
    2. If found and valid, loads that session into the request
    3. If not found, ensures a new session exists
    4. Adds the session ID to response headers
    """
    
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response
    
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Only process API requests
        if request.path.startswith('/api/'):
            self._ensure_session(request)
        
        response = self.get_response(request)
        
        if request.path.startswith('/api/') and request.session.session_key:
            response[SESSION_HEADER_NAME] = request.session.session_key
            existing_expose = response.get('Access-Control-Expose-Headers', '')
            if existing_expose:
                if SESSION_HEADER_NAME not in existing_expose:
                    response['Access-Control-Expose-Headers'] = f"{existing_expose}, {SESSION_HEADER_NAME}"
            else:
                response['Access-Control-Expose-Headers'] = SESSION_HEADER_NAME
        
        return response
    
    def _ensure_session(self, request: HttpRequest) -> None:
        """
        Ensure the request has a valid session.
        
        Priority:
        1. Check X-Session-ID header for a valid session
        2. Use existing cookie session if valid
        3. Create new session if neither exists
        """
        header_session_id = request.META.get(
            f'HTTP_{SESSION_HEADER_NAME.upper().replace("-", "_")}'
        )
        
        if header_session_id:
            session_store = SessionStore(session_key=header_session_id)
            
            if session_store.exists(header_session_id):
                try:
                    session_data = session_store.load()
                    
                    expiry_age = session_store.get_expiry_age()
                    if expiry_age > 0:
                        request.session = session_store
                        logger.debug(f"Loaded session from header: {header_session_id[:8]}...")
                        return
                    else:
                        logger.debug(f"Header session expired: {header_session_id[:8]}...")
                except Exception as e:
                    logger.debug(f"Error loading header session {header_session_id[:8]}...: {e}")
            else:
                logger.debug(f"Header session does not exist: {header_session_id[:8]}...")
        
        # Fall back to cookie session or create new one
        if not request.session.session_key:
            request.session.create()
            logger.debug(f"Created new session: {request.session.session_key[:8]}...")


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
        
        # Check if session came from header
        header_session = request.META.get('HTTP_X_SESSION_ID')
        if header_session and session_key and header_session == session_key:
            session_display += " (header)"
        
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