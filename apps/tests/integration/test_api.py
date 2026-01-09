"""
Integration tests for the API endpoints.

This module provides comprehensive tests for all API endpoints including:
- Operations CRUD (create, list, retrieve, delete)
- Operation status polling
- File downloads
- Operation definitions listing
- Health checks
- Input validation
- Error handling
"""

import io
import json
import os
from typing import Dict
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from django.conf import settings
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient, APITestCase


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def create_test_file(
    filename: str = "test_video.mp4",
    content: bytes = b"fake video content",
    content_type: str = "video/mp4",
) -> io.BytesIO:
    """Create a test file for upload."""
    file = io.BytesIO(content)
    file.name = filename
    file.content_type = content_type
    return file


# =============================================================================
# Base Test Class
# =============================================================================

class APITestBase(APITestCase):
    """Base class for API tests with common setup."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        super().setUpClass()
        # Ensure storage directories exist
        for dir_name in ['uploads', 'outputs', 'temp']:
            dir_path = os.path.join(settings.MEDIA_ROOT, dir_name)
            os.makedirs(dir_path, exist_ok=True)
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = APIClient()
        # Ensure session is created for each request
        self.client.session.create()
    
    def get_session_key(self) -> str:
        """Get the current session key."""
        if not self.client.session.session_key:
            self.client.session.create()
        return self.client.session.session_key
    
    def create_test_operation(
        self,
        operation_name: str = "video_compress",
        status: str = "pending",
        session_key: str = None,
        parameters: Dict = None,
        progress: int = 0,
    ):
        """Create a test operation in the database."""
        from apps.operations.models import Operation
        
        return Operation.objects.create(
            session_key=session_key or self.get_session_key(),
            operation=operation_name,
            status=status,
            parameters=parameters or {},
            progress=progress,
        )
    
    def create_test_file_record(
        self,
        operation,
        file_type: str = "input",
        file_name: str = "test_video.mp4",
        file_size: int = 1024,
        mime_type: str = "video/mp4",
    ):
        """Create a test file record in the database."""
        from apps.operations.models import File
        
        return File.objects.create(
            operation=operation,
            file_type=file_type,
            file_path=f"{file_type}s/{operation.session_key}/{operation.id}/{file_name}",
            file_name=file_name,
            file_size=file_size,
            mime_type=mime_type,
        )


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheckEndpoint(APITestBase):
    """Tests for the health check endpoint."""
    
    def test_health_check_returns_status(self):
        """Test health check returns proper status structure."""
        response = self.client.get('/api/v1/health/')
        
        # Should return either healthy or degraded (depending on Redis availability)
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn(data['status'], ['healthy', 'degraded'])
    
    def test_health_check_detailed_returns_components(self):
        """Test detailed health check returns component status."""
        response = self.client.get('/api/v1/health/?detailed=true')
        
        # Should return either healthy or degraded
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE])
        data = response.json()
        self.assertIn('components', data)
        self.assertIn('database', data['components'])
        self.assertIn('storage', data['components'])


# =============================================================================
# Operation Definition Tests
# =============================================================================

class TestOperationDefinitionsEndpoint(APITestBase):
    """Tests for the operation definitions endpoint."""
    
    def test_list_operation_definitions_returns_200(self):
        """Test listing operation definitions returns 200."""
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.list_operations.return_value = [
                {'operation_name': 'video_compress', 'media_type': 'video'},
                {'operation_name': 'image_resize', 'media_type': 'image'},
            ]
            mock_registry.get_operation.return_value = MagicMock(
                name='video_compress',
                media_type=MagicMock(value='video'),
                description='Compress video',
                parameters=[],
                input_formats=['mp4'],
                output_formats=['mp4'],
            )
            mock_get_registry.return_value = mock_registry
            
            response = self.client.get('/api/v1/operation-definitions/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_list_operation_definitions_filter_by_media_type(self):
        """Test filtering operation definitions by media type."""
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.list_operations.return_value = [
                {'operation_name': 'video_compress', 'media_type': 'video'},
            ]
            mock_registry.get_operation.return_value = MagicMock(
                name='video_compress',
                media_type=MagicMock(value='video'),
                description='Compress video',
                parameters=[],
                input_formats=['mp4'],
                output_formats=['mp4'],
            )
            mock_get_registry.return_value = mock_registry
            
            response = self.client.get('/api/v1/operation-definitions/?media_type=video')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_retrieve_operation_definition_returns_200(self):
        """Test retrieving a specific operation definition."""
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get_operation.return_value = MagicMock(
                name='video_compress',
                media_type=MagicMock(value='video'),
                description='Compress video files',
                parameters=[],
                input_formats=['mp4', 'avi'],
                output_formats=['mp4'],
            )
            mock_get_registry.return_value = mock_registry
            
            response = self.client.get('/api/v1/operation-definitions/video_compress/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])
    
    def test_retrieve_nonexistent_operation_returns_404(self):
        """Test retrieving a non-existent operation returns 404."""
        from apps.processors.exceptions import OperationNotFoundError
        
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.get_operation.side_effect = OperationNotFoundError("nonexistent")
            mock_get_registry.return_value = mock_registry
            
            response = self.client.get('/api/v1/operation-definitions/nonexistent/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


# =============================================================================
# Operation CRUD Tests
# =============================================================================

class TestOperationListEndpoint(APITestBase):
    """Tests for the operation list endpoint."""
    
    def test_list_operations_returns_200(self):
        """Test listing operations returns 200."""
        response = self.client.get('/api/v1/operations/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        # Response uses 'data' not 'results'
        self.assertIn('data', data)
        self.assertTrue(data['success'])
    
    def test_list_operations_returns_own_operations(self):
        """Test that only own session operations are returned."""
        # Get the session key that the client will use
        session_key = self.get_session_key()
        
        # Create operations for this session
        op1 = self.create_test_operation(session_key=session_key)
        op2 = self.create_test_operation(session_key=session_key)
        
        # Create operation for different session
        other_op = self.create_test_operation(session_key="other_session_key")
        
        response = self.client.get('/api/v1/operations/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        
        operation_ids = [str(op['id']) for op in data['data']]
        # Should contain our operations
        self.assertIn(str(op1.id), operation_ids)
        self.assertIn(str(op2.id), operation_ids)
        # Should NOT contain other session's operation
        self.assertNotIn(str(other_op.id), operation_ids)
    
    def test_list_operations_filters_by_status(self):
        """Test filtering operations by status."""
        session_key = self.get_session_key()
        op_pending = self.create_test_operation(status="pending", session_key=session_key)
        op_completed = self.create_test_operation(status="completed", session_key=session_key)
        
        response = self.client.get('/api/v1/operations/?status=pending')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        for op in data['data']:
            self.assertEqual(op['status'], 'pending')
    
    def test_list_operations_pagination(self):
        """Test operations list pagination."""
        # Create multiple operations
        session_key = self.get_session_key()
        for i in range(15):
            self.create_test_operation(session_key=session_key)
        
        with patch('apps.api.utils.get_session_key', return_value=session_key):
            response = self.client.get('/api/v1/operations/?limit=10')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        # Pagination is in metadata
        self.assertIn('metadata', data)
        self.assertIn('pagination', data['metadata'])
        self.assertEqual(data['metadata']['pagination']['total_count'], 15)
        self.assertLessEqual(len(data['data']), 10)


class TestOperationCreateEndpoint(APITestBase):
    """Tests for the operation create endpoint."""
    
    def test_create_operation_without_file_returns_400(self):
        """Test creating an operation without a file returns 400."""
        response = self.client.post(
            '/api/v1/operations/',
            {
                'operation': 'video_compress',
                'parameters': json.dumps({}),
            },
            format='multipart',
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_create_operation_with_invalid_operation_returns_400(self):
        """Test creating an operation with invalid operation name returns 400."""
        test_file = create_test_file()
        
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            registry = MagicMock()
            registry.is_registered.return_value = False
            mock_get_registry.return_value = registry
            
            response = self.client.post(
                '/api/v1/operations/',
                {
                    'operation': 'nonexistent_operation',
                    'parameters': json.dumps({}),
                    'file': test_file,
                },
                format='multipart',
            )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_create_operation_validates_file_type(self):
        """Test that file type is validated against operation requirements."""
        # Create a text file for a video operation
        test_file = create_test_file(
            filename="test.txt",
            content=b"text content",
            content_type="text/plain",
        )
        
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            registry = MagicMock()
            registry.is_registered.return_value = True
            registry.validate_parameters.return_value = {}
            registry.get_operation.return_value = MagicMock(
                media_type=MagicMock(value='video')
            )
            mock_get_registry.return_value = registry
            
            response = self.client.post(
                '/api/v1/operations/',
                {
                    'operation': 'video_compress',
                    'parameters': json.dumps({}),
                    'file': test_file,
                },
                format='multipart',
            )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class TestOperationRetrieveEndpoint(APITestBase):
    """Tests for the operation retrieve endpoint."""
    
    def test_retrieve_operation_returns_200(self):
        """Test retrieving an operation returns 200."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(session_key=session_key)
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.return_value = operation
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(str(data['data']['id']), str(operation.id))
    
    def test_retrieve_nonexistent_operation_returns_404(self):
        """Test retrieving a non-existent operation returns 404."""
        from apps.operations.exceptions import OperationNotFoundError
        
        fake_id = uuid4()
        session_key = self.get_session_key()
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.side_effect = OperationNotFoundError(str(fake_id))
            
            response = self.client.get(f'/api/v1/operations/{fake_id}/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_retrieve_other_session_operation_returns_403(self):
        """Test retrieving another session's operation returns 403."""
        from apps.operations.exceptions import OperationAccessDeniedError
        
        operation = self.create_test_operation(session_key="other_session")
        session_key = self.get_session_key()
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.side_effect = OperationAccessDeniedError(str(operation.id))
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


class TestOperationDeleteEndpoint(APITestBase):
    """Tests for the operation delete endpoint."""
    
    def test_delete_operation_returns_200(self):
        """Test deleting an operation returns 200."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="completed",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.delete_operation') as mock_delete:
            mock_delete.return_value = None
            
            response = self.client.delete(f'/api/v1/operations/{operation.id}/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_delete_nonexistent_operation_returns_404(self):
        """Test deleting a non-existent operation returns 404."""
        from apps.operations.exceptions import OperationNotFoundError
        
        fake_id = uuid4()
        session_key = self.get_session_key()
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.delete_operation') as mock_delete:
            mock_delete.side_effect = OperationNotFoundError(str(fake_id))
            
            response = self.client.delete(f'/api/v1/operations/{fake_id}/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


# =============================================================================
# Operation Status Tests
# =============================================================================

class TestOperationStatusEndpoint(APITestBase):
    """Tests for the operation status endpoint (polling)."""
    
    def test_status_endpoint_returns_200(self):
        """Test status endpoint returns 200."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="processing",
            progress=50,
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.return_value = operation
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/status/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['status'], 'processing')
        self.assertEqual(data['data']['progress'], 50)
    
    def test_status_endpoint_returns_lightweight_response(self):
        """Test status endpoint returns minimal fields for efficiency."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(session_key=session_key)
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.return_value = operation
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/status/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        # Should contain essential fields only
        self.assertIn('id', data['data'])
        self.assertIn('status', data['data'])
        self.assertIn('progress', data['data'])


# =============================================================================
# Operation Download Tests
# =============================================================================

class TestOperationDownloadEndpoint(APITestBase):
    """Tests for the operation download endpoint."""
    
    def test_download_incomplete_operation_returns_400(self):
        """Test downloading an incomplete operation returns 400."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="processing",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.return_value = operation
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/download/')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_download_operation_without_output_returns_404(self):
        """Test downloading an operation without output file returns 404."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="completed",
        )
        # No output file created
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.return_value = operation
            
            response = self.client.get(f'/api/v1/operations/{operation.id}/download/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


# =============================================================================
# Operation Retry Tests
# =============================================================================

class TestOperationRetryEndpoint(APITestBase):
    """Tests for the operation retry endpoint."""
    
    def test_retry_failed_operation_returns_200(self):
        """Test retrying a failed operation returns 200."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="failed",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.retry_operation') as mock_retry, \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_retry.return_value = None
            # After retry, operation should be queued
            operation.status = "queued"
            mock_get.return_value = operation
            
            response = self.client.post(f'/api/v1/operations/{operation.id}/retry/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_retry_non_failed_operation_returns_400(self):
        """Test retrying a non-failed operation returns 400."""
        from apps.operations.exceptions import OperationNotRetryableError
        
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="completed",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.retry_operation') as mock_retry:
            mock_retry.side_effect = OperationNotRetryableError(str(operation.id), "completed")
            
            response = self.client.post(f'/api/v1/operations/{operation.id}/retry/')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Operation Cancel Tests
# =============================================================================

class TestOperationCancelEndpoint(APITestBase):
    """Tests for the operation cancel endpoint."""
    
    def test_cancel_queued_operation_returns_200(self):
        """Test cancelling a queued operation returns 200."""
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="queued",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.cancel_operation') as mock_cancel, \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_cancel.return_value = None
            operation.status = "cancelled"
            mock_get.return_value = operation
            
            response = self.client.post(f'/api/v1/operations/{operation.id}/cancel/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_cancel_completed_operation_returns_400(self):
        """Test cancelling a completed operation returns 400."""
        from apps.operations.exceptions import InvalidOperationStateError
        
        session_key = self.get_session_key()
        operation = self.create_test_operation(
            session_key=session_key,
            status="completed",
        )
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.cancel_operation') as mock_cancel:
            mock_cancel.side_effect = InvalidOperationStateError(
                str(operation.id), "completed", ["pending", "queued"]
            )
            
            response = self.client.post(f'/api/v1/operations/{operation.id}/cancel/')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Session Info Tests
# =============================================================================

class TestSessionInfoEndpoint(APITestBase):
    """Tests for the session info endpoint."""
    
    def test_session_info_returns_200(self):
        """Test session info endpoint returns 200."""
        response = self.client.get('/api/v1/session/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('session_id', data['data'])
        self.assertIn('operations', data['data'])
    
    def test_session_info_returns_operation_counts(self):
        """Test session info returns correct operation counts."""
        session_key = self.get_session_key()
        
        # Create operations with different statuses
        self.create_test_operation(session_key=session_key, status="pending")
        self.create_test_operation(session_key=session_key, status="completed")
        self.create_test_operation(session_key=session_key, status="completed")
        
        with patch('apps.api.utils.get_session_key', return_value=session_key):
            response = self.client.get('/api/v1/session/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['data']['operations']['total'], 3)
        self.assertEqual(data['data']['operations']['by_status']['pending'], 1)
        self.assertEqual(data['data']['operations']['by_status']['completed'], 2)


# =============================================================================
# Queue Stats Tests
# =============================================================================

class TestQueueStatsEndpoint(APITestBase):
    """Tests for the queue stats endpoint."""
    
    def test_queue_stats_returns_200(self):
        """Test queue stats endpoint returns 200."""
        with patch('apps.operations.services.operations_manager.OperationsManager.get_queue_stats') as mock_stats:
            mock_stats.return_value = {
                'video_queue': {'queued': 5, 'started': 2, 'failed': 0},
                'image_queue': {'queued': 10, 'started': 1, 'failed': 1},
                'audio_queue': {'queued': 3, 'started': 0, 'failed': 0},
            }
            
            response = self.client.get('/api/v1/queues/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertTrue(data['success'])


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation(APITestBase):
    """Tests for input validation."""
    
    def test_invalid_uuid_returns_error(self):
        """Test invalid UUID format returns 400 or 404."""
        response = self.client.get('/api/v1/operations/not-a-uuid/')
        
        # Should return either 400 (bad request) or 404 (not found)
        self.assertIn(response.status_code, [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        ])
    
    def test_empty_file_upload_returns_400(self):
        """Test uploading an empty file returns 400."""
        empty_file = create_test_file(content=b"")
        
        response = self.client.post(
            '/api/v1/operations/',
            {
                'operation': 'video_compress',
                'parameters': json.dumps({}),
                'file': empty_file,
            },
            format='multipart',
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_invalid_json_parameters_returns_400(self):
        """Test invalid JSON in parameters returns 400."""
        test_file = create_test_file()
        
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            registry = MagicMock()
            registry.is_registered.return_value = True
            mock_get_registry.return_value = registry
            
            response = self.client.post(
                '/api/v1/operations/',
                {
                    'operation': 'video_compress',
                    'parameters': 'not valid json {{{',
                    'file': test_file,
                },
                format='multipart',
            )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# =============================================================================
# Error Response Format Tests
# =============================================================================

class TestErrorResponseFormat(APITestBase):
    """Tests for consistent error response format."""
    
    def test_404_error_has_consistent_format(self):
        """Test 404 errors have consistent format."""
        from apps.operations.exceptions import OperationNotFoundError
        
        fake_id = uuid4()
        session_key = self.get_session_key()
        
        with patch('apps.api.utils.get_session_key', return_value=session_key), \
             patch('apps.operations.services.operations_manager.OperationsManager.get_operation') as mock_get:
            mock_get.side_effect = OperationNotFoundError(str(fake_id))
            
            response = self.client.get(f'/api/v1/operations/{fake_id}/')
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        data = response.json()
        self.assertFalse(data.get('success', True))
        self.assertIn('error', data)
    
    def test_validation_error_has_consistent_format(self):
        """Test validation errors have consistent format."""
        response = self.client.post(
            '/api/v1/operations/',
            {
                'operation': 'video_compress',
                # Missing required 'file' field
            },
            format='multipart',
        )
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertFalse(data.get('success', True))


# =============================================================================
# CORS Tests
# =============================================================================

class TestCORSHeaders(APITestBase):
    """Tests for CORS headers."""
    
    def test_cors_headers_on_preflight(self):
        """Test CORS headers are present on preflight requests."""
        response = self.client.options(
            '/api/v1/operations/',
            HTTP_ORIGIN='http://localhost:3000',
            HTTP_ACCESS_CONTROL_REQUEST_METHOD='POST',
        )
        
        # Should not return an error
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT])


# =============================================================================
# Content Type Tests
# =============================================================================

class TestContentTypes(APITestBase):
    """Tests for content type handling."""
    
    def test_json_response_content_type(self):
        """Test API returns JSON content type."""
        response = self.client.get('/api/v1/operations/')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('application/json', response['Content-Type'])
    
    def test_accepts_multipart_form_data(self):
        """Test API accepts multipart form data for uploads."""
        test_file = create_test_file()
        
        with patch('apps.processors.registry.get_registry') as mock_get_registry:
            registry = MagicMock()
            registry.is_registered.return_value = False
            mock_get_registry.return_value = registry
            
            response = self.client.post(
                '/api/v1/operations/',
                {
                    'operation': 'video_compress',
                    'parameters': json.dumps({}),
                    'file': test_file,
                },
                format='multipart',
            )
        
        # Should process the request (even if it fails validation)
        self.assertIn(response.status_code, [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_201_CREATED,
            status.HTTP_200_OK,
        ])