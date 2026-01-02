# config/urls.py

"""
Root URL configuration for the Media Processor.

This module defines the root URL patterns that route to different parts
of the application:

    /admin/          - Django admin interface
    /api/v1/         - REST API endpoints (versioned)
    /django-rq/      - Redis Queue dashboard (development only)
    /media/          - Media files (development only)

API Versioning:
    The API is versioned using URL path prefixing (/api/v1/).
    This allows for future API versions without breaking existing clients.

Security Notes:
    - The RQ dashboard is only enabled in DEBUG mode
    - Media file serving is only enabled in DEBUG mode (use nginx/CDN in production)
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # API v1 Endpoints
    path('api/v1/', include('apps.api.urls', namespace='api')),
]

if settings.DEBUG:# Provides debugging information panel in the browser
    urlpatterns += [
        path('__debug__/', include('debug_toolbar.urls')),
    ]

    # Django RQ Dashboard
    urlpatterns += [
        path('django-rq/', include('django_rq.urls')),
    ]

    # Serve media files during development
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    urlpatterns += static(
        settings.STATIC_URL,
        document_root=settings.STATIC_ROOT
    )