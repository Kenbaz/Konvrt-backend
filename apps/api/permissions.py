# apps/api/permissions.py

"""
Custom permissions for the API layer.

This module provides permission classes for controlling access
to API resources based on session ownership and other criteria.
"""

import logging

from rest_framework.permissions import BasePermission
from rest_framework.request import Request

from .utils import get_session_key

logger = logging.getLogger(__name__)


class IsSessionOwner(BasePermission):
    """
    Permission class that checks if the request session owns the resource.
    
    This is used for anonymous users who are identified by session key.
    The resource must have a `session_key` attribute that matches the
    request's session key.
    """
    
    message = "You do not have permission to access this resource."
    
    def has_object_permission(
        self,
        request: Request,
        view,
        obj,
    ) -> bool:
        """
        Check if the session owns the object.
        
        Args:
            request: The request object
            view: The view being accessed
            obj: The object being accessed
            
        Returns:
            True if session owns object, False otherwise
        """
        session_key = get_session_key(request)
        
        # Check if object has session_key attribute
        if hasattr(obj, 'session_key'):
            return obj.session_key == session_key
        
        # If object doesn't have session_key, deny access
        return False


class IsSessionOwnerOrAdmin(BasePermission):
    """
    Permission class that allows session owners and admin users.
    
    Admin users can access any resource.
    Regular users can only access resources they own.
    """
    
    message = "You do not have permission to access this resource."
    
    def has_permission(self, request: Request, view) -> bool:
        """
        Check if user has general permission.
        
        Admin users always have permission.
        Others need a valid session.
        """
        # Admin users always have permission
        if request.user and request.user.is_staff:
            return True
        
        # For anonymous users, ensure session exists
        return bool(get_session_key(request))
    
    def has_object_permission(
        self,
        request: Request,
        view,
        obj,
    ) -> bool:
        """
        Check if the session owns the object or user is admin.
        """
        # Admin users can access anything
        if request.user and request.user.is_staff:
            return True
        
        session_key = get_session_key(request)
        
        # Check session ownership
        if hasattr(obj, 'session_key'):
            return obj.session_key == session_key
        
        return False


class HasValidSession(BasePermission):
    """
    Permission class that requires a valid session.
    
    This ensures that the client has cookies enabled and
    a session has been established.
    """
    
    message = "A valid session is required. Please ensure cookies are enabled."
    
    def has_permission(self, request: Request, view) -> bool:
        """
        Check if request has a valid session.
        """
        session_key = get_session_key(request)
        return bool(session_key)


class IsReadOnly(BasePermission):
    """
    Permission class that only allows read operations.
    
    Useful for public endpoints that shouldn't allow modifications.
    """
    
    SAFE_METHODS = ('GET', 'HEAD', 'OPTIONS')
    
    message = "This resource is read-only."
    
    def has_permission(self, request: Request, view) -> bool:
        """
        Check if request method is safe (read-only).
        """
        return request.method in self.SAFE_METHODS


class CanAccessOperation(BasePermission):
    """
    Permission class specifically for operation access.
    
    Checks that the operation belongs to the session or
    the user is an admin.
    """
    
    message = "You do not have permission to access this operation."
    
    def has_object_permission(
        self,
        request: Request,
        view,
        obj,
    ) -> bool:
        """
        Check if user can access the operation.
        """
        # Admin users can access anything
        if request.user and request.user.is_staff:
            return True
        
        session_key = get_session_key(request)
        
        # Check session ownership
        if hasattr(obj, 'session_key'):
            if obj.session_key == session_key:
                return True
        
        # Check user ownership (for authenticated users)
        if request.user and request.user.is_authenticated:
            if hasattr(obj, 'user') and obj.user == request.user:
                return True
        
        return False


class CanDownloadFile(BasePermission):
    """
    Permission class for file downloads.
    
    Checks that:
    1. User owns the operation
    2. Operation is completed
    3. File exists
    """
    
    message = "You cannot download this file."
    
    def has_object_permission(
        self,
        request: Request,
        view,
        obj,
    ) -> bool:
        """
        Check if user can download the file.
        
        Args:
            obj: The Operation instance
        """
        # First check ownership
        session_key = get_session_key(request)
        
        is_owner = False
        if hasattr(obj, 'session_key') and obj.session_key == session_key:
            is_owner = True
        if request.user and request.user.is_staff:
            is_owner = True
        if request.user and hasattr(obj, 'user') and obj.user == request.user:
            is_owner = True
        
        if not is_owner:
            self.message = "You do not have permission to access this operation."
            return False
        
        # Check operation is completed
        if hasattr(obj, 'status') and obj.status != 'completed':
            self.message = f"Operation is not complete. Current status: {obj.status}"
            return False
        
        # Check output file exists
        if hasattr(obj, 'files'):
            has_output = obj.files.filter(file_type='output').exists()
            if not has_output:
                self.message = "No output file available for this operation."
                return False
        
        return True