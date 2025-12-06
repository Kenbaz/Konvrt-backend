"""
URL configuration for the API layer.

This module defines all API endpoints using Django REST Framework's
DefaultRouter for ViewSets and standard path() for APIViews.

API Versioning: All endpoints are mounted under /api/v1/

Endpoints:
    Operations:
        - GET    /api/v1/operations/           - List user's operations
        - POST   /api/v1/operations/           - Create new operation
        - GET    /api/v1/operations/{id}/      - Get operation details
        - DELETE /api/v1/operations/{id}/      - Delete operation
        - GET    /api/v1/operations/{id}/status/   - Get lightweight status (polling)
        - GET    /api/v1/operations/{id}/download/ - Download output file
        - POST   /api/v1/operations/{id}/retry/    - Retry failed operation
        - POST   /api/v1/operations/{id}/cancel/   - Cancel queued operation
    
    Operation Definitions:
        - GET    /api/v1/operation-definitions/        - List available operations
        - GET    /api/v1/operation-definitions/{name}/ - Get operation details
    
    Health & Monitoring:
        - GET    /api/v1/health/       - Health check endpoint
        - GET    /api/v1/queues/       - Queue statistics
        - GET    /api/v1/session/      - Current session info
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    HealthCheckView,
    OperationDefinitionViewSet,
    OperationViewSet,
    QueueStatsView,
    SessionInfoView,
)

# Create router and register viewsets
router = DefaultRouter()

router.register(r'operations', OperationViewSet, basename='operation')
router.register(r'operation-definitions', OperationDefinitionViewSet, basename='operation-definition')


# URL patterns for API v1
# The router handles ViewSet URLs, while path() handles APIView URLs

urlpatterns = [
    path('', include(router.urls)), # Include all ViewSet routes

    path('health/', HealthCheckView.as_view(), name='health-check'),
    path('queues/', QueueStatsView.as_view(), name='queue-stats'),
    path('session/', SessionInfoView.as_view(), name='session-info'),
]

# App name for URL namespacing
# Allows using 'api:operation-list', 'api:health-check', etc.
app_name = 'api'