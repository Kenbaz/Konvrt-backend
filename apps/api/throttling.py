"""
Custom throttling classes for API rate limiting.

This module provides various throttle classes to prevent abuse and ensure
fair usage of the API resources.
"""

from rest_framework.throttling import AnonRateThrottle, SimpleRateThrottle


class AnonBurstRateThrottle(AnonRateThrottle):
    """
    Throttle for burst requests from anonymous users.
    
    This limits short-term request bursts to prevent abuse.
    Default: 60 requests per minute.
    """
    scope = 'anon_burst'


class AnonSustainedRateThrottle(AnonRateThrottle):
    """
    Throttle for sustained requests from anonymous users.
    
    This limits long-term request volume to prevent abuse.
    Default: 1000 requests per day.
    """
    scope = 'anon_sustained'


class UploadRateThrottle(SimpleRateThrottle):
    """
    Throttle specifically for file upload operations.
    
    This is more restrictive since uploads are resource-intensive.
    Default: 10 uploads per hour.
    
    Uses session key for tracking to allow anonymous users.
    """
    scope = 'uploads'
    
    def get_cache_key(self, request, view):
        """
        Get the cache key for rate limiting.
        
        Uses session key for anonymous users, allowing per-session limiting.
        """
        if request.user and request.user.is_authenticated:
            ident = request.user.pk
        else:
            # Use session key for anonymous users
            if not request.session.session_key:
                request.session.create()
            ident = request.session.session_key
        
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }


class StatusCheckRateThrottle(SimpleRateThrottle):
    """
    Throttle for status check/polling operations.
    
    This is more permissive since clients may poll frequently for updates.
    Default: 120 requests per minute.
    
    Uses session key for tracking to allow anonymous users.
    """
    scope = 'status_checks'
    
    def get_cache_key(self, request, view):
        """
        Get the cache key for rate limiting.
        
        Uses session key for anonymous users, allowing per-session limiting.
        """
        if request.user and request.user.is_authenticated:
            ident = request.user.pk
        else:
            # Use session key for anonymous users
            if not request.session.session_key:
                request.session.create()
            ident = request.session.session_key
        
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }


class DownloadRateThrottle(SimpleRateThrottle):
    """
    Throttle for file download operations.
    
    Moderate rate limit for downloads.
    Default: 30 downloads per hour.
    """
    scope = 'downloads'
    rate = '30/hour'
    
    def get_cache_key(self, request, view):
        """
        Get the cache key for rate limiting.
        
        Uses session key for anonymous users.
        """
        if request.user and request.user.is_authenticated:
            ident = request.user.pk
        else:
            if not request.session.session_key:
                request.session.create()
            ident = request.session.session_key
        
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }