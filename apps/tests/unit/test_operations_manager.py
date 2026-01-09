import pytest
from datetime import timedelta
from unittest.mock import patch, MagicMock, PropertyMock
from uuid import uuid4

from django.utils import timezone


class MockUploadedFile:
    """Mock Django UploadedFile for testing."""
    
    def __init__(
        self,
        name: str = "test_video.mp4",
        size: int = 1024,
        content_type: str = "video/mp4",
    ):
        self.name = name
        self.size = size
        self.content_type = content_type
        self._content = b"fake content" * (size // 12 + 1)
        self._content = self._content[:size]
        self._position = 0
    
    def read(self, size=None):
        if size is None:
            data = self._content[self._position:]
            self._position = len(self._content)
        else:
            data = self._content[self._position:self._position + size]
            self._position += size
        return data
    
    def seek(self, position):
        self._position = position
    
    def chunks(self, chunk_size=8192):
        self.seek(0)
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk


class MockOperation:
    """Mock Operation model for testing."""
    
    def __init__(
        self,
        id=None,
        session_key="test-session",
        operation="video_compress",
        status="pending",
        parameters=None,
        progress=0,
        error_message=None,
        created_at=None,
        started_at=None,
        completed_at=None,
        expires_at=None,
        is_deleted=False,
        user=None,
    ):
        self.id = id or uuid4()
        self.session_key = session_key
        self.operation = operation
        self.status = status
        self.parameters = parameters or {}
        self.progress = progress
        self.error_message = error_message
        self.created_at = created_at or timezone.now()
        self.started_at = started_at
        self.completed_at = completed_at
        self.expires_at = expires_at
        self.is_deleted = is_deleted
        self.user = user
        self._saved_fields = []
    
    def save(self, update_fields=None):
        self._saved_fields = update_fields or []
    
    def can_be_deleted(self):
        return self.status in ["completed", "failed"] or self.is_expired
    
    def can_be_retried(self):
        return self.status == "failed"
    
    @property
    def is_expired(self):
        if not self.expires_at:
            return False
        return timezone.now() >= self.expires_at
    
    @property
    def processing_time(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class MockFile:
    """Mock File model for testing."""
    
    def __init__(
        self,
        id=None,
        operation=None,
        file_type="input",
        file_path="uploads/session/op/file.mp4",
        file_name="file.mp4",
        file_size=1024,
        mime_type="video/mp4",
        metadata=None,
    ):
        self.id = id or uuid4()
        self.operation = operation
        self.file_type = file_type
        self.file_path = file_path
        self.file_name = file_name
        self.file_size = file_size
        self.mime_type = mime_type
        self.metadata = metadata or {}


class MockRegistry:
    """Mock Operation Registry for testing."""
    
    def __init__(self, operations=None):
        self._operations = operations or {}
    
    def is_registered(self, operation_name):
        return operation_name in self._operations
    
    def get_operation(self, operation_name):
        if operation_name not in self._operations:
            raise Exception(f"Operation not found: {operation_name}")
        return self._operations[operation_name]
    
    def validate_parameters(self, operation_name, parameters):
        return parameters


class MockOperationDefinition:
    """Mock OperationDefinition for testing."""
    
    def __init__(self, name, media_type_value="video"):
        self.operation_name = name
        self.media_type = MagicMock()
        self.media_type.value = media_type_value
    
    def get_media_type(self):
        return self.media_type.value


@pytest.mark.django_db
class TestOperationsManagerCreateOperation:
    """Tests for OperationsManager.create_operation method."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry with a test operation."""
        operation_def = MockOperationDefinition("video_compress", "video")
        registry = MockRegistry({"video_compress": operation_def})
        return registry
    
    @pytest.fixture
    def mock_file_manager(self):
        """Create a mock FileManager."""
        return MagicMock()
    
    def test_create_operation_success(self, mock_registry, mock_file_manager):
        """Test successful operation creation."""
        uploaded_file = MockUploadedFile()
        operation_id = uuid4()
        
        # Mock the Operation model
        mock_operation = MockOperation(id=operation_id)
        
        mock_file_info = {
            "file_path": "uploads/test-session/op-id/test_video.mp4",
            "file_name": "test_video.mp4",
            "file_size": 1024,
            "mime_type": "video/mp4",
            "media_type": "video",
        }
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch('apps.operations.models.Operation.objects') as mock_operation_manager:
                with patch('apps.operations.models.File.objects') as mock_file_model:
                    with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                        mock_operation_manager.create.return_value = mock_operation
                        mock_fm.save_uploaded_file.return_value = mock_file_info
                        
                        from apps.operations.services.operations_manager import OperationsManager
                        
                        result = OperationsManager.create_operation(
                            session_key="test-session",
                            operation_name="video_compress",
                            parameters={"quality": 23},
                            uploaded_file=uploaded_file,
                        )
                        
                        assert result.id == operation_id
                        mock_operation_manager.create.assert_called_once()
                        mock_fm.save_uploaded_file.assert_called_once()
                        mock_file_model.create.assert_called_once()
    
    def test_create_operation_not_found(self, mock_file_manager):
        """Test operation creation fails for unknown operation."""
        uploaded_file = MockUploadedFile()
        
        # Registry without the operation
        empty_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=empty_registry):
            from apps.operations.services.operations_manager import OperationsManager
            from apps.processors.exceptions import OperationNotFoundError
            
            with pytest.raises(OperationNotFoundError):
                OperationsManager.create_operation(
                    session_key="test-session",
                    operation_name="nonexistent_operation",
                    parameters={},
                    uploaded_file=uploaded_file,
                )
    
    def test_create_operation_with_user(self, mock_registry, mock_file_manager):
        """Test operation creation with authenticated user."""
        uploaded_file = MockUploadedFile()
        operation_id = uuid4()
        mock_user = MagicMock()
        mock_user.id = 123
        
        mock_operation = MockOperation(id=operation_id, user=mock_user)
        
        mock_file_info = {
            "file_path": "uploads/test-session/op-id/test_video.mp4",
            "file_name": "test_video.mp4",
            "file_size": 1024,
            "mime_type": "video/mp4",
            "media_type": "video",
        }
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            with patch('apps.operations.models.Operation.objects') as mock_operation_manager:
                with patch('apps.operations.models.File.objects') as mock_file_model:
                    with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                        mock_operation_manager.create.return_value = mock_operation
                        mock_fm.save_uploaded_file.return_value = mock_file_info
                        
                        from apps.operations.services.operations_manager import OperationsManager
                        
                        result = OperationsManager.create_operation(
                            session_key="test-session",
                            operation_name="video_compress",
                            parameters={"quality": 23},
                            uploaded_file=uploaded_file,
                            user=mock_user,
                        )
                        
                        # Verify user was passed to create
                        create_call = mock_operation_manager.create.call_args
                        assert create_call.kwargs.get('user') == mock_user


class TestOperationsManagerQueueOperation:
    """Tests for OperationsManager.queue_operation method."""
    
    def test_queue_operation_success(self):
        """Test successful operation queuing."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="pending")
        rq_job_id = "rq-job-123"
        
        mock_rq_job = MagicMock()
        mock_rq_job.id = rq_job_id
        
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_rq_job
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('django_rq.get_queue', return_value=mock_queue):
                with patch('apps.processors.registry.get_registry') as mock_get_registry:
                    mock_op_manager.get.return_value = mock_operation
                    
                    mock_registry = MockRegistry({
                        "video_compress": MockOperationDefinition("video_compress", "video")
                    })
                    mock_get_registry.return_value = mock_registry
                    
                    from apps.operations.services.operations_manager import OperationsManager
                    
                    result = OperationsManager.queue_operation(operation_id)
                    
                    assert result == rq_job_id
                    assert mock_operation.status == "queued"
                    mock_queue.enqueue.assert_called_once()
    
    def test_queue_operation_already_processing(self):
        """Test queuing fails for already processing operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import OperationAlreadyProcessingError
            
            with pytest.raises(OperationAlreadyProcessingError) as exc_info:
                OperationsManager.queue_operation(operation_id)
            
            assert "processing" in str(exc_info.value)
    
    def test_queue_failed_operation_for_retry(self):
        """Test queuing a failed operation (for retry)."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="failed")
        rq_job_id = "rq-job-456"
        
        mock_rq_job = MagicMock()
        mock_rq_job.id = rq_job_id
        
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_rq_job
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('django_rq.get_queue', return_value=mock_queue):
                with patch('apps.processors.registry.get_registry') as mock_get_registry:
                    mock_op_manager.get.return_value = mock_operation
                    
                    mock_registry = MockRegistry({
                        "video_compress": MockOperationDefinition("video_compress", "video")
                    })
                    mock_get_registry.return_value = mock_registry
                    
                    from apps.operations.services.operations_manager import OperationsManager
                    
                    result = OperationsManager.queue_operation(operation_id)
                    
                    assert result == rq_job_id


class TestOperationsManagerStatusUpdates:
    """Tests for OperationsManager status update methods."""
    
    def test_start_operation_success(self):
        """Test successfully starting an operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="queued")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.start_operation(operation_id)
            
            assert result.status == "processing"
            assert result.started_at is not None
    
    def test_start_operation_wrong_status(self):
        """Test starting an operation in wrong status raises error."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="pending")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import InvalidOperationStateError
            
            with pytest.raises(InvalidOperationStateError):
                OperationsManager.start_operation(operation_id)
    
    def test_update_operation_progress(self):
        """Test updating operation progress."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing", progress=0)
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.update_operation_progress(operation_id, 50)
            
            assert result.progress == 50
    
    def test_update_operation_progress_clamps_values(self):
        """Test that progress is clamped to 0-100."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing", progress=0)
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            # Test clamping above 100
            result = OperationsManager.update_operation_progress(operation_id, 150)
            assert result.progress == 100
            
            # Test clamping below 0
            mock_operation.progress = 50
            result = OperationsManager.update_operation_progress(operation_id, -10)
            assert result.progress == 0
    
    def test_update_operation_progress_with_status(self):
        """Test updating progress with status change."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing", progress=0)
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.update_operation_progress(operation_id, 100, status="completed")
            
            assert result.progress == 100
            assert result.status == "completed"
    
    @pytest.mark.django_db
    def test_complete_operation_success(self):
        """Test successfully completing an operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing")
        
        mock_output_info = {
            "file_path": "outputs/session/op/output.mp4",
            "file_name": "output.mp4",
            "file_size": 2048,
            "mime_type": "video/mp4",
        }
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.models.File.objects') as mock_file_model:
                with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                    mock_op_manager.get.return_value = mock_operation
                    mock_fm.move_to_output.return_value = mock_output_info
                    mock_fm.cleanup_temp_directory.return_value = True
                    
                    from apps.operations.services.operations_manager import OperationsManager
                    
                    result = OperationsManager.complete_operation(
                        operation_id=operation_id,
                        output_file_path="/tmp/output.mp4",
                        output_filename="output.mp4",
                        session_key="test-session",
                    )
                    
                    assert result.status == "completed"
                    assert result.progress == 100
                    assert result.completed_at is not None
                    assert result.expires_at is not None
                    mock_file_model.create.assert_called_once()
                    mock_fm.cleanup_temp_directory.assert_called_once()
    
    def test_fail_operation_success(self):
        """Test failing an operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.get.return_value = mock_operation
                mock_fm.cleanup_temp_directory.return_value = True
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.fail_operation(operation_id, "Processing error occurred")
                
                assert result.status == "failed"
                assert result.error_message == "Processing error occurred"
                assert result.completed_at is not None
                mock_fm.cleanup_temp_directory.assert_called_once()
    
    def test_fail_operation_truncates_long_error(self):
        """Test that long error messages are truncated."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, status="processing")
        
        long_error = "x" * 3000
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.get.return_value = mock_operation
                mock_fm.cleanup_temp_directory.return_value = True
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.fail_operation(operation_id, long_error)
                
                assert len(result.error_message) <= 2000


class TestOperationsManagerRetrieval:
    """Tests for OperationsManager retrieval methods."""
    
    def test_get_operation_success(self):
        """Test retrieving an operation with correct session."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, session_key="test-session")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_operation(operation_id, "test-session")
            
            assert result.id == operation_id
    
    def test_get_operation_access_denied(self):
        """Test retrieving an operation with wrong session raises error."""
        operation_id = uuid4()
        mock_operation = MockOperation(id=operation_id, session_key="other-session")
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import OperationAccessDeniedError
            
            with pytest.raises(OperationAccessDeniedError):
                OperationsManager.get_operation(operation_id, "wrong-session")
    
    def test_get_operation_not_found(self):
        """Test retrieving a nonexistent operation raises error."""
        operation_id = uuid4()
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            from apps.operations.models import Operation
            mock_op_manager.get.side_effect = Operation.DoesNotExist()
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import OperationNotFoundError
            
            with pytest.raises(OperationNotFoundError):
                OperationsManager.get_operation(operation_id, "test-session")
    
    def test_get_operation_status(self):
        """Test getting operation status for polling."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="processing",
            progress=75,
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_operation_status(operation_id, "test-session")
            
            assert result["id"] == str(operation_id)
            assert result["status"] == "processing"
            assert result["progress"] == 75
    
    def test_get_operation_status_with_error(self):
        """Test getting status for failed operation includes error."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="failed",
            error_message="Processing failed",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_operation_status(operation_id, "test-session")
            
            assert result["error_message"] == "Processing failed"
    
    def test_list_user_operations(self):
        """Test listing user's operations."""
        mock_operations = [
            MockOperation(id=uuid4(), session_key="test-session"),
            MockOperation(id=uuid4(), session_key="test-session"),
        ]
        
        mock_queryset = MagicMock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.__getitem__ = lambda self, key: mock_operations[:key.stop] if isinstance(key, slice) else mock_operations[key]
        mock_queryset.__iter__ = lambda self: iter(mock_operations)
        
        with patch('apps.operations.models.Operation.objects', mock_queryset):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.list_user_operations("test-session")
            
            assert len(result) == 2
    
    def test_list_user_operations_with_status_filter(self):
        """Test listing user's operations with status filter."""
        mock_queryset = MagicMock()
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        mock_queryset.__getitem__ = lambda self, key: []
        
        with patch('apps.operations.models.Operation.objects', mock_queryset):
            from apps.operations.services.operations_manager import OperationsManager
            
            OperationsManager.list_user_operations("test-session", status_filter="completed")
            
            # Verify filter was called with status
            filter_calls = mock_queryset.filter.call_args_list
            assert any("status" in str(call) for call in filter_calls)


@pytest.mark.django_db
class TestOperationsManagerDeletion:
    """Tests for OperationsManager deletion methods."""
    
    def test_delete_operation_success(self):
        """Test successfully deleting a completed operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="completed",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.get.return_value = mock_operation
                mock_fm.delete_operation_files.return_value = 2
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.delete_operation(operation_id, "test-session")
                
                assert result is True
                assert mock_operation.is_deleted is True
                mock_fm.delete_operation_files.assert_called_once()
    
    def test_delete_operation_not_deletable(self):
        """Test deleting a processing operation raises error."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="processing",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import OperationNotDeletableError
            
            with pytest.raises(OperationNotDeletableError):
                OperationsManager.delete_operation(operation_id, "test-session")
    
    def test_delete_operation_force(self):
        """Test force deleting a processing operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="processing",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.get.return_value = mock_operation
                mock_fm.delete_operation_files.return_value = 2
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.delete_operation(operation_id, "test-session", force=True)
                
                assert result is True
                assert mock_operation.is_deleted is True


class TestOperationsManagerRetry:
    """Tests for OperationsManager retry methods."""
    
    def test_retry_operation_success(self):
        """Test successfully retrying a failed operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="failed",
            error_message="Previous error",
            progress=50,
        )
        
        mock_rq_job = MagicMock()
        mock_rq_job.id = "rq-retry-123"
        
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_rq_job
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('django_rq.get_queue', return_value=mock_queue):
                with patch('apps.processors.registry.get_registry') as mock_get_registry:
                    mock_op_manager.get.return_value = mock_operation
                    
                    mock_registry = MockRegistry({
                        "video_compress": MockOperationDefinition("video_compress", "video")
                    })
                    mock_get_registry.return_value = mock_registry
                    
                    from apps.operations.services.operations_manager import OperationsManager
                    
                    result = OperationsManager.retry_operation(operation_id, "test-session")
                    
                    assert result == "rq-retry-123"
                    assert mock_operation.status == "queued"
                    assert mock_operation.progress == 0
                    assert mock_operation.error_message is None
    
    def test_retry_operation_not_retryable(self):
        """Test retrying a non-failed operation raises error."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="completed",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import OperationNotRetryableError
            
            with pytest.raises(OperationNotRetryableError):
                OperationsManager.retry_operation(operation_id, "test-session")


class TestOperationsManagerCancel:
    """Tests for OperationsManager cancel method."""
    
    def test_cancel_pending_operation(self):
        """Test cancelling a pending operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="pending",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.cancel_operation(operation_id, "test-session")
            
            assert result is True
            assert mock_operation.status == "failed"
            assert "cancelled" in mock_operation.error_message.lower()
    
    def test_cancel_queued_operation(self):
        """Test cancelling a queued operation."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="queued",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.cancel_operation(operation_id, "test-session")
            
            assert result is True
    
    def test_cancel_processing_operation_fails(self):
        """Test cancelling a processing operation raises error."""
        operation_id = uuid4()
        mock_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="processing",
        )
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            mock_op_manager.get.return_value = mock_operation
            
            from apps.operations.services.operations_manager import OperationsManager
            from apps.operations.exceptions import InvalidOperationStateError
            
            with pytest.raises(InvalidOperationStateError):
                OperationsManager.cancel_operation(operation_id, "test-session")


class TestOperationsManagerQueueHelpers:
    """Tests for OperationsManager queue helper methods."""
    
    def test_get_queue_for_video_operation(self):
        """Test getting queue for video operation."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition("video_compress", "video")
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_for_operation("video_compress")
            
            assert result == "video_queue"
    
    def test_get_queue_for_image_operation(self):
        """Test getting queue for image operation."""
        mock_registry = MockRegistry({
            "image_resize": MockOperationDefinition("image_resize", "image")
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_for_operation("image_resize")
            
            assert result == "image_queue"
    
    def test_get_queue_for_audio_operation(self):
        """Test getting queue for audio operation."""
        mock_registry = MockRegistry({
            "audio_convert": MockOperationDefinition("audio_convert", "audio")
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_for_operation("audio_convert")
            
            assert result == "audio_queue"
    
    def test_get_queue_for_unknown_operation(self):
        """Test getting queue for unknown operation defaults to image."""
        mock_registry = MockRegistry({})
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_for_operation("unknown_op")
            
            assert result == "image_queue"
    
    def test_get_timeout_for_video(self):
        """Test getting timeout for video operation."""
        mock_registry = MockRegistry({
            "video_compress": MockOperationDefinition("video_compress", "video")
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_timeout_for_operation("video_compress")
            
            assert result == 1800  # 30 minutes
    
    def test_get_timeout_for_image(self):
        """Test getting timeout for image operation."""
        mock_registry = MockRegistry({
            "image_resize": MockOperationDefinition("image_resize", "image")
        })
        
        with patch('apps.processors.registry.get_registry', return_value=mock_registry):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_timeout_for_operation("image_resize")
            
            assert result == 60  # 1 minute


class TestOperationsManagerFileHelpers:
    """Tests for OperationsManager file helper methods."""
    
    def test_get_input_file(self):
        """Test getting input file for an operation."""
        operation_id = uuid4()
        mock_file = MockFile(file_type="input")
        
        with patch('apps.operations.models.File.objects') as mock_file_manager:
            mock_file_manager.get.return_value = mock_file
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_input_file(operation_id)
            
            assert result.file_type == "input"
    
    def test_get_input_file_not_found(self):
        """Test getting input file when none exists."""
        operation_id = uuid4()
        
        with patch('apps.operations.models.File.objects') as mock_file_manager:
            from apps.operations.models import File
            mock_file_manager.get.side_effect = File.DoesNotExist()
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_input_file(operation_id)
            
            assert result is None
    
    def test_get_output_file(self):
        """Test getting output file for an operation."""
        operation_id = uuid4()
        mock_file = MockFile(file_type="output")
        
        with patch('apps.operations.models.File.objects') as mock_file_manager:
            mock_file_manager.get.return_value = mock_file
            
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_output_file(operation_id)
            
            assert result.file_type == "output"


class TestOperationsManagerCleanup:
    """Tests for OperationsManager cleanup methods."""
    
    def test_cleanup_expired_operations(self):
        """Test cleaning up expired operations."""
        operation_id = uuid4()
        expired_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="completed",
            expires_at=timezone.now() - timedelta(days=1),
        )
        
        mock_queryset = MagicMock()
        mock_queryset.__iter__ = lambda self: iter([expired_operation])
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.filter.return_value = mock_queryset
                mock_fm.delete_operation_files.return_value = 2
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.cleanup_expired_operations()
                
                assert result["operations_processed"] == 1
                assert result["operations_deleted"] == 1
                assert expired_operation.is_deleted is True
    
    def test_cleanup_expired_operations_dry_run(self):
        """Test cleanup dry run doesn't delete anything."""
        operation_id = uuid4()
        expired_operation = MockOperation(
            id=operation_id,
            session_key="test-session",
            status="completed",
            expires_at=timezone.now() - timedelta(days=1),
        )
        
        mock_queryset = MagicMock()
        mock_queryset.__iter__ = lambda self: iter([expired_operation])
        
        with patch('apps.operations.models.Operation.objects') as mock_op_manager:
            with patch('apps.operations.services.operations_manager.FileManager') as mock_fm:
                mock_op_manager.filter.return_value = mock_queryset
                
                from apps.operations.services.operations_manager import OperationsManager
                
                result = OperationsManager.cleanup_expired_operations(dry_run=True)
                
                assert result["operations_processed"] == 1
                assert result["operations_deleted"] == 1
                assert expired_operation.is_deleted is False  # Not actually deleted
                mock_fm.delete_operation_files.assert_not_called()


class TestOperationsManagerQueueStats:
    """Tests for OperationsManager queue stats method."""
    
    def test_get_queue_stats_success(self):
        """Test getting queue statistics."""
        mock_queue = MagicMock()
        mock_queue.count = 5
        mock_queue.started_operation_registry.count = 2
        mock_queue.failed_operation_registry.count = 1
        
        with patch('django_rq.get_queue', return_value=mock_queue):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_stats()
            
            assert "video_queue" in result
            assert "image_queue" in result
            assert "audio_queue" in result
            assert result["video_queue"]["queued"] == 5
            assert result["video_queue"]["started"] == 2
            assert result["video_queue"]["failed"] == 1
    
    def test_get_queue_stats_redis_unavailable(self):
        """Test queue stats returns empty when Redis unavailable."""
        with patch('django_rq.get_queue', side_effect=Exception("Redis unavailable")):
            from apps.operations.services.operations_manager import OperationsManager
            
            result = OperationsManager.get_queue_stats()
            
            # Should return empty stats, not raise
            assert "video_queue" in result
            assert result["video_queue"]["queued"] == 0