# apps/tests/unit/test_workers.py

"""
Unit tests for the RQ workers module.

These tests verify:
- Operation processing workflow
- Error classification
- Retry logic
- Status updates
- File handling
"""

import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4
from datetime import datetime, timedelta

from apps.processors.rq_workers import (
    process_operation,
    complete_operation,
    fail_operation,
    classify_error,
    handle_job_error,
    get_worker_stats,
    cleanup_stale_operations,
    WorkerError,
    InputFileNotFoundError,
    ProcessingFailedError,
    MAX_RETRIES,
    RETRY_DELAYS,
)
from apps.processors.exceptions import OperationNotFoundError


class TestClassifyError:
    """Tests for error classification."""
    
    def test_classify_timeout_error(self):
        """Test classifying timeout errors."""
        error = TimeoutError("Operation timed out")
        result = classify_error(error)
        
        assert result['is_retryable'] is True
        assert result['error_code'] == 'TIMEOUT'
        assert 'timed out' in result['user_message'].lower()
    
    def test_classify_connection_error(self):
        """Test classifying connection errors."""
        error = ConnectionError("Connection refused")
        result = classify_error(error)
        
        assert result['is_retryable'] is True
        assert result['error_code'] == 'CONNECTION_ERROR'
    
    def test_classify_memory_error(self):
        """Test classifying memory errors."""
        error = MemoryError("Out of memory")
        result = classify_error(error)
        
        assert result['is_retryable'] is True
        assert result['error_code'] == 'MEMORY_ERROR'
    
    def test_classify_disk_full_error(self):
        """Test classifying disk full errors."""
        error = OSError("No space left on device")
        result = classify_error(error)
        
        assert result['is_retryable'] is True
        assert result['error_code'] == 'DISK_FULL'
    
    def test_classify_corrupt_file_error(self):
        """Test classifying corrupt file errors."""
        error = ValueError("File is corrupt and cannot be processed")
        result = classify_error(error)
        
        assert result['is_retryable'] is False
        assert result['error_code'] == 'CORRUPT_FILE'
    
    def test_classify_unsupported_format_error(self):
        """Test classifying unsupported format errors."""
        error = ValueError("Unsupported video format")
        result = classify_error(error)
        
        assert result['is_retryable'] is False
        assert result['error_code'] == 'UNSUPPORTED_FORMAT'
    
    def test_classify_codec_not_found_error(self):
        """Test classifying codec not found errors."""
        error = RuntimeError("Codec not found: libx264")
        result = classify_error(error)
        
        assert result['is_retryable'] is False
        assert result['error_code'] == 'CODEC_NOT_FOUND'
    
    def test_classify_permission_denied_error(self):
        """Test classifying permission denied errors."""
        error = PermissionError("Permission denied: /path/to/file")
        result = classify_error(error)
        
        assert result['is_retryable'] is False
        assert result['error_code'] == 'PERMISSION_DENIED'
    
    def test_classify_unknown_error(self):
        """Test classifying unknown errors."""
        error = RuntimeError("Some unknown error occurred")
        result = classify_error(error)
        
        assert result['is_retryable'] is False
        assert result['error_code'] == 'UNKNOWN_ERROR'
        assert 'unknown error' in result['user_message'].lower()
    
    def test_classify_redis_error(self):
        """Test classifying Redis errors."""
        error = ConnectionError("Redis connection refused")
        result = classify_error(error)
        
        assert result['is_retryable'] is True


class TestHandleJobError:
    """Tests for job error handling."""
    
    def test_handle_retryable_error_first_attempt(self):
        """Test handling retryable error on first attempt."""
        error = TimeoutError("Operation timed out")
        should_retry, delay = handle_job_error("test-op-id", error, retry_count=0)
        
        assert should_retry is True
        assert delay == RETRY_DELAYS[0]  # First retry delay
    
    def test_handle_retryable_error_second_attempt(self):
        """Test handling retryable error on second attempt."""
        error = TimeoutError("Operation timed out")
        should_retry, delay = handle_job_error("test-op-id", error, retry_count=1)
        
        assert should_retry is True
        assert delay == RETRY_DELAYS[1]  # Second retry delay
    
    def test_handle_retryable_error_max_retries_exceeded(self):
        """Test handling retryable error when max retries exceeded."""
        error = TimeoutError("Operation timed out")
        should_retry, delay = handle_job_error("test-op-id", error, retry_count=MAX_RETRIES)
        
        assert should_retry is False
        assert delay is None
    
    def test_handle_non_retryable_error(self):
        """Test handling non-retryable error."""
        error = ValueError("File is corrupt")
        should_retry, delay = handle_job_error("test-op-id", error, retry_count=0)
        
        assert should_retry is False
        assert delay is None


class TestWorkerExceptions:
    """Tests for worker exception classes."""
    
    def test_worker_error_defaults(self):
        """Test WorkerError default values."""
        error = WorkerError("Test error")
        
        assert error.message == "Test error"
        assert error.is_retryable is False
        assert error.error_code == "WORKER_ERROR"
    
    def test_worker_error_with_params(self):
        """Test WorkerError with custom parameters."""
        error = WorkerError(
            message="Custom error",
            is_retryable=True,
            error_code="CUSTOM_CODE"
        )
        
        assert error.message == "Custom error"
        assert error.is_retryable is True
        assert error.error_code == "CUSTOM_CODE"
    
    # def test_operation_not_found_error(self):
    #     """Test OperationNotFoundError."""
    #     error = OperationNotFoundError("123-456")
        
    #     assert "123-456" in error.message
    #     assert error.is_retryable is False
    #     assert error.error_code == "OPERATION_NOT_FOUND"
    
    def test_input_file_not_found_error(self):
        """Test InputFileNotFoundError."""
        error = InputFileNotFoundError("123-456", "/path/to/file")
        
        assert "123-456" in error.message
        assert "/path/to/file" in error.message
        assert error.is_retryable is False
        assert error.error_code == "INPUT_FILE_NOT_FOUND"
    
    def test_processing_failed_error(self):
        """Test ProcessingFailedError."""
        error = ProcessingFailedError("Processing failed", is_retryable=True)
        
        assert error.message == "Processing failed"
        assert error.is_retryable is True
        assert error.error_code == "PROCESSING_FAILED"


class TestCompleteOperation:
    """Tests for complete_operation function."""
    
    @pytest.fixture
    def mock_operation(self):
        """Create a mock operation."""
        operation = MagicMock()
        operation.id = uuid4()
        operation.parameters = {}
        return operation
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings."""
        with patch('apps.processors.rq_workers.settings') as mock:
            mock.OPERATION_EXPIRATION_DAYS = 7
            yield mock
    
    def test_complete_operation_sets_status(self, mock_operation, mock_settings):
        """Test that complete_operation sets correct status."""
        # Mock the import inside the function
        mock_status = MagicMock()
        mock_status.COMPLETED = 'completed'
        
        with patch('apps.processors.rq_workers.timezone') as mock_tz, \
             patch.dict('sys.modules', {'apps.operations.enums': MagicMock(OperationStatus=mock_status)}):
            mock_tz.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            
            complete_operation(mock_operation, "/output/path.mp4")
            
            assert mock_operation.status == 'completed'
            assert mock_operation.progress == 100
            assert mock_operation.error_message is None
    
    def test_complete_operation_sets_expiry(self, mock_operation, mock_settings):
        """Test that complete_operation sets expiry date."""
        mock_status = MagicMock()
        mock_status.COMPLETED = 'completed'
        
        with patch('apps.processors.rq_workers.timezone') as mock_tz, \
             patch.dict('sys.modules', {'apps.operations.enums': MagicMock(OperationStatus=mock_status)}):
            now = datetime(2024, 1, 1, 12, 0, 0)
            mock_tz.now.return_value = now
            
            complete_operation(mock_operation, "/output/path.mp4")
            
            expected_expiry = now + timedelta(days=7)
            assert mock_operation.expires_at == expected_expiry
    
    def test_complete_operation_stores_metadata(self, mock_operation, mock_settings):
        """Test that complete_operation stores processing metadata."""
        mock_status = MagicMock()
        mock_status.COMPLETED = 'completed'
        
        with patch('apps.processors.rq_workers.timezone') as mock_tz, \
             patch.dict('sys.modules', {'apps.operations.enums': MagicMock(OperationStatus=mock_status)}):
            mock_tz.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            
            metadata = {'codec': 'h264', 'duration': 120}
            complete_operation(mock_operation, "/output/path.mp4", metadata=metadata)
            
            assert '_processing_metadata' in mock_operation.parameters
            assert mock_operation.parameters['_processing_metadata'] == metadata


class TestFailOperation:
    """Tests for fail_operation function."""
    
    @pytest.fixture
    def mock_operation(self):
        """Create a mock operation."""
        operation = MagicMock()
        operation.id = uuid4()
        operation.parameters = {}
        return operation
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings."""
        with patch('apps.processors.rq_workers.settings') as mock:
            mock.OPERATION_EXPIRATION_DAYS = 7
            yield mock
    
    def test_fail_operation_sets_status(self, mock_operation, mock_settings):
        """Test that fail_operation sets correct status."""
        mock_status = MagicMock()
        mock_status.FAILED = 'failed'
        
        with patch('apps.processors.rq_workers.timezone') as mock_tz, \
             patch.dict('sys.modules', {'apps.operations.enums': MagicMock(OperationStatus=mock_status)}):
            mock_tz.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            
            fail_operation(mock_operation, "Test error message")
            
            assert mock_operation.status == 'failed'
            assert mock_operation.error_message == "Test error message"
    
    def test_fail_operation_stores_error_info(self, mock_operation, mock_settings):
        """Test that fail_operation stores error information."""
        mock_status = MagicMock()
        mock_status.FAILED = 'failed'
        
        with patch('apps.processors.rq_workers.timezone') as mock_tz, \
             patch.dict('sys.modules', {'apps.operations.enums': MagicMock(OperationStatus=mock_status)}):
            mock_tz.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            
            fail_operation(
                mock_operation,
                "Test error",
                is_retryable=True,
                error_code="TEST_ERROR"
            )
            
            assert '_error_info' in mock_operation.parameters
            error_info = mock_operation.parameters['_error_info']
            assert error_info['error_code'] == "TEST_ERROR"
            assert error_info['is_retryable'] is True


class TestProcessOperation:
    """Integration tests for process_operation function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp(prefix='test_worker_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_process_operation_not_found(self):
        """Test processing non-existent operation."""
        pass  # Would test with full Django setup
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_process_operation_updates_status_to_processing(self):
        """Test that processing updates status to PROCESSING."""
        pass  # Would test with full Django setup


class TestRetryConfiguration:
    """Tests for retry configuration constants."""
    
    def test_max_retries_value(self):
        """Test that MAX_RETRIES has expected value."""
        assert MAX_RETRIES == 2
    
    def test_retry_delays_length(self):
        """Test that RETRY_DELAYS has correct length."""
        assert len(RETRY_DELAYS) >= MAX_RETRIES
    
    def test_retry_delays_increasing(self):
        """Test that retry delays are increasing (exponential backoff)."""
        for i in range(len(RETRY_DELAYS) - 1):
            assert RETRY_DELAYS[i] < RETRY_DELAYS[i + 1]


class TestWorkerStats:
    """Tests for worker statistics functions."""
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_get_worker_stats_structure(self):
        """Test that get_worker_stats returns expected structure."""
        pass  # Would test with full Django setup
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_get_worker_stats_success_rate_calculation(self):
        """Test success rate calculation."""
        pass  # Would test with full Django setup


class TestCleanupStaleOperations:
    """Tests for cleanup_stale_operations function."""
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_cleanup_stale_operations_marks_as_failed(self):
        """Test that stale operations are marked as failed."""
        pass  # Would test with full Django setup
    
    @pytest.mark.skip(reason="Requires full Django model setup")
    def test_cleanup_stale_operations_with_no_stale(self):
        """Test cleanup when no stale operations exist."""
        pass  # Would test with full Django setup