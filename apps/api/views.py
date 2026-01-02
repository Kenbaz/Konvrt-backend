# apps/api/views.py

"""
API views for the media processing platform.

This module provides ViewSets and views for:
- Managing operations (create, list, retrieve, delete)
- Checking operation status
- Downloading processed files
- Listing available operations
- Health checks
"""

import logging
import os
import shutil
from typing import Any, Dict

from django.conf import settings
from django.db import connection
from django.http import FileResponse, Http404
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .exceptions import NotFoundError, PermissionDeniedError
from .pagination import OperationPagination
from .serializers import (
    FileSerializer,
    HealthCheckSerializer,
    OperationCreateSerializer,
    OperationDefinitionListSerializer,
    OperationDefinitionSerializer,
    OperationListSerializer,
    OperationSerializer,
    OperationStatusSerializer,
    QueueStatsSerializer,
)
from .throttling import (
    DownloadRateThrottle,
    StatusCheckRateThrottle,
    UploadRateThrottle,
)
from .utils import (
    build_download_headers,
    error_response,
    get_session_key,
    success_response,
)

logger = logging.getLogger(__name__)


class OperationViewSet(viewsets.ViewSet):
    """
    ViewSet for managing media processing operations.
    
    Provides endpoints for:
    - POST /operations/ - Create a new operation
    - GET /operations/ - List user's operations
    - GET /operations/{id}/ - Get operation details
    - DELETE /operations/{id}/ - Delete an operation
    - GET /operations/{id}/status/ - Get lightweight status for polling
    - GET /operations/{id}/download/ - Download the output file
    - POST /operations/{id}/retry/ - Retry a failed operation
    - POST /operations/{id}/cancel/ - Cancel a queued operation
    """
    
    pagination_class = OperationPagination
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    
    def get_throttles(self):
        """Apply different throttles based on action."""
        if self.action == 'create':
            return [UploadRateThrottle()]
        elif self.action in ['status', 'retrieve']:
            return [StatusCheckRateThrottle()]
        elif self.action == 'download':
            return [DownloadRateThrottle()]
        return super().get_throttles()
    

    def list(self, request: Request) -> Response:
        """
        List all operations for the current session.
        
        Query Parameters:
            - status: Filter by status (pending, queued, processing, completed, failed)
            - operation: Filter by operation name
            - limit: Number of results per page (default 50)
            - offset: Offset for pagination
        
        Returns:
            Paginated list of operations
        """
        from apps.operations.models import Operation
        
        session_key = get_session_key(request)
        
        # Base queryset - only non-deleted operations for this session
        queryset = Operation.objects.filter(
            session_key=session_key,
            is_deleted=False,
        ).prefetch_related('files')
        
        # Apply filters
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        operation_filter = request.query_params.get('operation')
        if operation_filter:
            queryset = queryset.filter(operation=operation_filter)
        
        # Order by creation date (newest first)
        queryset = queryset.order_by('-created_at')
        
        # Paginate
        paginator = self.pagination_class()
        paginated_queryset = paginator.paginate_queryset(queryset, request)
        
        # Serialize
        serializer = OperationListSerializer(
            paginated_queryset,
            many=True,
            context={'request': request}
        )
        
        return paginator.get_paginated_response(serializer.data)
    

    def create(self, request: Request) -> Response:
        """
        Create a new operation and queue it for processing.
        
        Request Body (multipart/form-data):
            - operation: Name of the operation (required)
            - parameters: JSON object with operation parameters (optional)
            - file: The file to process (required)
        
        Returns:
            Created operation details
        """
        session_key = get_session_key(request)
        
        serializer = OperationCreateSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if not serializer.is_valid():
            return error_response(
                message="Validation failed",
                code="VALIDATION_ERROR",
                status_code=status.HTTP_400_BAD_REQUEST,
                errors=[
                    {"field": field, "message": str(errors[0])}
                    for field, errors in serializer.errors.items()
                ],
            )
        
        # Create and queue the operation
        operation = serializer.save()
        
        # Return full operation details
        response_serializer = OperationSerializer(
            operation,
            context={'request': request}
        )
        
        logger.info(
            f"Created operation {operation.id} for session {session_key[:8]}..."
        )
        
        return success_response(
            data=response_serializer.data,
            message="Operation created and queued for processing",
            status_code=status.HTTP_201_CREATED,
        )
    

    def retrieve(self, request: Request, pk: str = None) -> Response:
        """
        Get detailed information about a specific operation.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Returns:
            Operation details including files
        """
        operation = self._get_operation(request, pk)
        
        serializer = OperationSerializer(
            operation,
            context={'request': request}
        )
        
        return success_response(data=serializer.data)
    

    def destroy(self, request: Request, pk: str = None) -> Response:
        """
        Delete an operation and its associated files.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Returns:
            Success message
        """
        from apps.operations.services.operations_manager import OperationsManager
        
        session_key = get_session_key(request)
        
        try:
            OperationsManager.delete_operation(pk, session_key)
        except Exception as e:
            logger.error(f"Failed to delete operation {pk}: {e}")
            raise
        
        logger.info(f"Deleted operation {pk}")
        
        return success_response(
            message="Operation deleted successfully",
            status_code=status.HTTP_200_OK,
        )
    

    @action(detail=True, methods=['get'])
    def status(self, request: Request, pk: str = None) -> Response:
        """
        Get lightweight status information for polling.
        
        This endpoint is optimized for frequent polling during processing.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Returns:
            Minimal status info: id, status, progress, eta_seconds
        """
        operation = self._get_operation(request, pk)
        
        serializer = OperationStatusSerializer(
            operation,
            context={'request': request}
        )
        
        return success_response(data=serializer.data)
    

    @action(detail=True, methods=['get'])
    def download(self, request: Request, pk: str = None) -> Response:
        """
        Download the output file for a completed operation.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Query Parameters:
            - inline: If 'true', suggest inline display instead of download
        
        Returns:
            File response with appropriate headers
        """
        operation = self._get_operation(request, pk)
        
        # Check operation is completed
        if operation.status != 'completed':
            return error_response(
                message=f"Operation is not complete. Current status: {operation.status}",
                code="OPERATION_NOT_COMPLETE",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        
        # Get output file
        output_file = operation.files.filter(file_type='output').first()
        if not output_file:
            return error_response(
                message="No output file available for this operation",
                code="OUTPUT_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Build full file path
        file_path = os.path.join(settings.MEDIA_ROOT, output_file.file_path)
        
        if not os.path.exists(file_path):
            logger.error(f"Output file not found on disk: {file_path}")
            return error_response(
                message="Output file not found on server",
                code="FILE_NOT_FOUND",
                status_code=status.HTTP_404_NOT_FOUND,
            )
        
        # Determine if inline display is requested
        inline = request.query_params.get('inline', '').lower() == 'true'
        
        # Build response headers
        headers = build_download_headers(
            filename=output_file.file_name,
            file_size=output_file.file_size,
            content_type=output_file.mime_type,
            inline=inline,
        )
        
        # Return file response
        response = FileResponse(
            open(file_path, 'rb'),
            content_type=output_file.mime_type,
        )
        
        for key, value in headers.items():
            response[key] = value
        
        logger.info(f"Serving download for operation {pk}: {output_file.file_name}")
        
        return response
    

    @action(detail=True, methods=['post'])
    def retry(self, request: Request, pk: str = None) -> Response:
        """
        Retry a failed operation.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Returns:
            Updated operation details
        """
        from apps.operations.services.operations_manager import OperationsManager
        
        session_key = get_session_key(request)
        
        # Retry the operation
        OperationsManager.retry_operation(pk, session_key)
        
        # Get updated operation
        operation = self._get_operation(request, pk)
        
        serializer = OperationSerializer(
            operation,
            context={'request': request}
        )
        
        logger.info(f"Retried operation {pk}")
        
        return success_response(
            data=serializer.data,
            message="Operation queued for retry",
        )
    

    @action(detail=True, methods=['post'])
    def cancel(self, request: Request, pk: str = None) -> Response:
        """
        Cancel a pending or queued operation.
        
        Path Parameters:
            - pk: Operation ID (UUID)
        
        Returns:
            Updated operation details
        """
        from apps.operations.services.operations_manager import OperationsManager
        
        session_key = get_session_key(request)
        
        # Cancel the operation
        OperationsManager.cancel_operation(pk, session_key)
        
        # Get updated operation
        operation = self._get_operation(request, pk)
        
        serializer = OperationSerializer(
            operation,
            context={'request': request}
        )
        
        logger.info(f"Cancelled operation {pk}")
        
        return success_response(
            data=serializer.data,
            message="Operation cancelled",
        )
    

    def _get_operation(self, request: Request, pk: str):
        """
        Get an operation by ID with ownership verification.
        
        Args:
            request: The request object
            pk: Operation ID
            
        Returns:
            Operation instance
            
        Raises:
            NotFoundError: If operation not found
            PermissionDeniedError: If operation doesn't belong to session
        """
        
        from apps.operations.exceptions import (
            OperationNotFoundError,
            OperationAccessDeniedError,
        )
        from apps.operations.services.operations_manager import OperationsManager
        
        session_key = get_session_key(request)
        
        try:
            operation = OperationsManager.get_operation(pk, session_key)
            return operation
        except OperationNotFoundError:
            raise NotFoundError(
                detail=f"Operation with ID '{pk}' was not found.",
                code="OPERATION_NOT_FOUND",
            )
        except OperationAccessDeniedError:
            raise PermissionDeniedError(
                detail="You do not have permission to access this operation.",
                code="OPERATION_ACCESS_DENIED",
            )


class OperationDefinitionViewSet(viewsets.ViewSet):
    """
    ViewSet for listing available operations.
    
    Provides endpoints for:
    - GET /operation-types/ - List all available operations
    - GET /operation-types/{name}/ - Get details for a specific operation
    """
    

    def list(self, request: Request) -> Response:
        """
        List all available operations.
        
        Query Parameters:
            - media_type: Filter by media type (video, image, audio)
        
        Returns:
            List of available operations with their schemas
        """
        from apps.processors.registry import get_registry, MediaType
        
        registry = get_registry()
        
        # Apply media_type filter if provided
        media_type_filter = request.query_params.get('media_type')
        if media_type_filter:
            # Convert string to MediaType enum
            try:
                media_type_enum = MediaType(media_type_filter)
                operation_defs = registry.list_operations_by_media_type(media_type_enum)
            except ValueError:
                # Invalid media type, return empty list
                operation_defs = []
        else:
            operation_defs = registry.list_registered_operations()
        
        serializer = OperationDefinitionSerializer(
            operation_defs,
            many=True,
            context={'request': request}
        )
        
        return success_response(
            data=serializer.data,
            metadata={
                "total_count": len(operation_defs),
                "media_types": ["video", "image", "audio"],
            }
        )
    

    def retrieve(self, request: Request, pk: str = None) -> Response:
        """
        Get details for a specific operation.
        
        Path Parameters:
            - pk: Operation name
        
        Returns:
            Operation definition with parameter schema
        """
        from apps.processors.registry import get_registry
        from apps.processors.exceptions import OperationNotFoundError
        
        registry = get_registry()
        
        try:
            operation_def = registry.get_operation(pk)
        except OperationNotFoundError:
            raise NotFoundError(
                detail=f"Operation '{pk}' is not available.",
                code="OPERATION_NOT_FOUND",
            )
        
        serializer = OperationDefinitionSerializer(
            operation_def,
            context={'request': request}
        )
        
        return success_response(data=serializer.data)


class HealthCheckView(APIView):
    """
    Health check endpoint for monitoring.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - Queue status
    - Storage availability
    """
    
    # No authentication or throttling for health checks
    authentication_classes = []
    permission_classes = []
    throttle_classes = []
    
    def get(self, request: Request) -> Response:
        """
        Perform health check and return status.
        
        Query Parameters:
            - detailed: If 'true', include detailed component status
        
        Returns:
            Health status with component details
        """
        detailed = request.query_params.get('detailed', '').lower() == 'true'
        
        health_data = {
            "status": "healthy",
            "timestamp": timezone.now().isoformat(),
            "version": getattr(settings, 'API_VERSION', 'v1'),
        }
        
        # Check components
        components = {}
        all_healthy = True
        
        # Database check
        db_status = self._check_database()
        components["database"] = db_status
        if db_status["status"] != "healthy":
            all_healthy = False
        
        # Redis check
        redis_status = self._check_redis()
        components["redis"] = redis_status
        if redis_status["status"] != "healthy":
            all_healthy = False
        
        # Storage check
        storage_status = self._check_storage()
        components["storage"] = storage_status
        if storage_status["status"] != "healthy":
            all_healthy = False
        
        # Queue stats (if detailed)
        if detailed:
            queue_stats = self._get_queue_stats()
            components["queues"] = queue_stats
        
        # Update overall status
        if not all_healthy:
            health_data["status"] = "degraded"
        
        if detailed:
            health_data["components"] = components
        
        # Return appropriate status code
        status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return Response(health_data, status=status_code)
    
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return {"status": "healthy", "message": "Database connection OK"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "message": str(e)}
    

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            import django_rq
            
            # Try to get a queue and ping Redis
            queue = django_rq.get_queue('video_queue')
            queue.connection.ping()
            return {"status": "healthy", "message": "Redis connection OK"}
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "message": str(e)}
    

    def _check_storage(self) -> Dict[str, Any]:
        """Check storage availability."""
        try:
            # Check that storage directories exist and are writable
            for dir_name in ['uploads', 'outputs', 'temp']:
                dir_path = os.path.join(settings.MEDIA_ROOT, dir_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                
                # Check if writable by creating a temp file
                test_file = os.path.join(dir_path, '.health_check')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except (IOError, OSError) as e:
                    return {
                        "status": "unhealthy",
                        "message": f"Directory {dir_name} is not writable: {e}"
                    }
            
            # Get disk usage
            total, used, free = shutil.disk_usage(settings.MEDIA_ROOT)
            free_gb = free / (1024 ** 3)
            
            if free_gb < 1:
                return {
                    "status": "warning",
                    "message": f"Low disk space: {free_gb:.2f} GB free",
                    "free_space_gb": round(free_gb, 2),
                }
            
            return {
                "status": "healthy",
                "message": "Storage OK",
                "free_space_gb": round(free_gb, 2),
            }
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {"status": "unhealthy", "message": str(e)}
    
    
    def _get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            from apps.operations.services.operations_manager import OperationsManager
            
            stats = OperationsManager.get_queue_stats()
            return {"status": "healthy", "queues": stats}
        except Exception as e:
            logger.error(f"Queue stats check failed: {e}")
            return {"status": "unknown", "message": str(e)}


class QueueStatsView(APIView):
    """
    View for getting queue statistics.
    
    Provides detailed information about job queues.
    """
    
    def get(self, request: Request) -> Response:
        """
        Get statistics for all queues.
        
        Returns:
            Queue statistics including counts and worker status
        """
        try:
            from apps.operations.services.operations_manager import OperationsManager
            
            stats = OperationsManager.get_queue_stats()
            
            # Format for response
            queue_data = []
            for queue_name, queue_stats in stats.items():
                queue_data.append({
                    "queue_name": queue_name,
                    "queued": queue_stats.get("queued", 0),
                    "started": queue_stats.get("started", 0),
                    "failed": queue_stats.get("failed", 0),
                })
            
            return success_response(
                data=queue_data,
                metadata={
                    "timestamp": timezone.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return error_response(
                message="Failed to retrieve queue statistics",
                code="QUEUE_STATS_ERROR",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SessionInfoView(APIView):
    """
    View for getting current session information.
    
    Useful for debugging and client session management.
    """
    
    def get(self, request: Request) -> Response:
        """
        Get information about the current session.
        
        Returns:
            Session ID and statistics
        """
        from apps.operations.models import Operation
        
        session_key = get_session_key(request)
        
        # Get operation counts for this session
        operation_counts = {}
        for status_choice in ['pending', 'queued', 'processing', 'completed', 'failed']:
            count = Operation.objects.filter(
                session_key=session_key,
                is_deleted=False,
                status=status_choice,
            ).count()
            operation_counts[status_choice] = count
        
        total_operations = sum(operation_counts.values())
        
        return success_response(
            data={
                "session_id": session_key,
                "operations": {
                    "total": total_operations,
                    "by_status": operation_counts,
                },
            }
        )