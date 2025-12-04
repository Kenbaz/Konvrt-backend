# apps/tests/unit/test_queue_manager.py

"""
Unit tests for the Queue Manager module.

These tests verify:
- Queue selection logic
- Timeout configuration
- Queue statistics retrieval
- Job cancellation
- Queue health monitoring
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4
from datetime import datetime

from apps.operations.services.queue_manager import (
    QueueManager,
    QUEUE_MAPPING,
    TIMEOUT_MAPPING,
    WORKER_FUNCTION,
    enqueue_operation,
    get_queue_stats,
    get_queue_health,
    cancel_rq_job,
    get_failed_jobs,
    get_worker_info,
)


class TestQueueMapping:
    """Tests for queue mapping configuration."""
    
    def test_queue_mapping_has_video(self):
        """Test that video queue mapping exists."""
        assert "video" in QUEUE_MAPPING
        assert QUEUE_MAPPING["video"] == "video_queue"
    
    def test_queue_mapping_has_image(self):
        """Test that image queue mapping exists."""
        assert "image" in QUEUE_MAPPING
        assert QUEUE_MAPPING["image"] == "image_queue"
    
    def test_queue_mapping_has_audio(self):
        """Test that audio queue mapping exists."""
        assert "audio" in QUEUE_MAPPING
        assert QUEUE_MAPPING["audio"] == "audio_queue"


class TestTimeoutMapping:
    """Tests for timeout configuration."""
    
    def test_video_timeout_is_30_minutes(self):
        """Test that video timeout is 30 minutes (1800 seconds)."""
        assert TIMEOUT_MAPPING["video"] == 1800
    
    def test_image_timeout_is_1_minute(self):
        """Test that image timeout is 1 minute (60 seconds)."""
        assert TIMEOUT_MAPPING["image"] == 60
    
    def test_audio_timeout_is_5_minutes(self):
        """Test that audio timeout is 5 minutes (300 seconds)."""
        assert TIMEOUT_MAPPING["audio"] == 300


class TestWorkerFunction:
    """Tests for worker function configuration."""
    
    def test_worker_function_path(self):
        """Test that worker function path is correct."""
        assert WORKER_FUNCTION == "apps.processors.rq_workers.process_operation"


class TestQueueManagerGetQueueForMediaType:
    """Tests for get_queue_for_media_type method."""
    
    def test_get_queue_for_video(self):
        """Test getting queue for video media type."""
        result = QueueManager.get_queue_for_media_type("video")
        assert result == "video_queue"
    
    def test_get_queue_for_image(self):
        """Test getting queue for image media type."""
        result = QueueManager.get_queue_for_media_type("image")
        assert result == "image_queue"
    
    def test_get_queue_for_audio(self):
        """Test getting queue for audio media type."""
        result = QueueManager.get_queue_for_media_type("audio")
        assert result == "audio_queue"
    
    def test_get_queue_for_unknown_defaults_to_image(self):
        """Test that unknown media type defaults to image queue."""
        result = QueueManager.get_queue_for_media_type("unknown")
        assert result == "image_queue"


class TestQueueManagerGetTimeoutForMediaType:
    """Tests for get_timeout_for_media_type method."""
    
    def test_get_timeout_for_video(self):
        """Test getting timeout for video media type."""
        result = QueueManager.get_timeout_for_media_type("video")
        assert result == 1800
    
    def test_get_timeout_for_image(self):
        """Test getting timeout for image media type."""
        result = QueueManager.get_timeout_for_media_type("image")
        assert result == 60
    
    def test_get_timeout_for_audio(self):
        """Test getting timeout for audio media type."""
        result = QueueManager.get_timeout_for_media_type("audio")
        assert result == 300
    
    def test_get_timeout_for_unknown_defaults_to_60(self):
        """Test that unknown media type defaults to 60 seconds."""
        result = QueueManager.get_timeout_for_media_type("unknown")
        assert result == 60


class TestQueueManagerEnqueueOperation:
    """Tests for enqueue_operation method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_enqueue_operation_calls_django_rq(self):
        """Test that enqueue_operation calls django_rq correctly."""
        pass
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_enqueue_operation_uses_correct_timeout(self):
        """Test that enqueue_operation uses correct timeout."""
        pass
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_enqueue_operation_with_custom_timeout(self):
        """Test enqueue_operation with custom timeout."""
        pass


class TestQueueManagerGetRqJob:
    """Tests for get_rq_job method."""
    
    @pytest.mark.skip(reason="Requires rq to be installed")
    def test_get_rq_job_returns_job(self):
        """Test that get_rq_job returns the job."""
        pass
    
    @pytest.mark.skip(reason="Requires rq to be installed")
    def test_get_rq_job_returns_none_on_error(self):
        """Test that get_rq_job returns None on error."""
        pass


class TestQueueManagerCancelRqJob:
    """Tests for cancel_rq_job method."""
    
    def test_cancel_rq_job_success(self):
        """Test successful job cancellation."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_job = MagicMock()
            mock_job.is_finished = False
            mock_job.is_failed = False
            mock_get.return_value = mock_job
            
            result = QueueManager.cancel_rq_job("test-job-id")
            
            assert result is True
            mock_job.cancel.assert_called_once()
    
    def test_cancel_rq_job_not_found(self):
        """Test cancellation when job not found."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_get.return_value = None
            
            result = QueueManager.cancel_rq_job("non-existent-job")
            
            assert result is False
    
    def test_cancel_rq_job_already_finished(self):
        """Test cancellation when job already finished."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_job = MagicMock()
            mock_job.is_finished = True
            mock_job.is_failed = False
            mock_get.return_value = mock_job
            
            result = QueueManager.cancel_rq_job("finished-job")
            
            assert result is False
            mock_job.cancel.assert_not_called()
    
    def test_cancel_rq_job_already_failed(self):
        """Test cancellation when job already failed."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_job = MagicMock()
            mock_job.is_finished = False
            mock_job.is_failed = True
            mock_get.return_value = mock_job
            
            result = QueueManager.cancel_rq_job("failed-job")
            
            assert result is False
            mock_job.cancel.assert_not_called()


class TestQueueManagerGetJobStatus:
    """Tests for get_job_status method."""
    
    def test_get_job_status_returns_info(self):
        """Test that get_job_status returns job information."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_job = MagicMock()
            mock_job.id = "test-job-id"
            mock_job.get_status.return_value = "queued"
            mock_job.created_at = datetime(2024, 1, 1, 12, 0, 0)
            mock_job.started_at = None
            mock_job.ended_at = None
            mock_job.is_finished = False
            mock_job.is_failed = False
            mock_job.meta = {"key": "value"}
            mock_get.return_value = mock_job
            
            result = QueueManager.get_job_status("test-job-id")
            
            assert result['id'] == "test-job-id"
            assert result['status'] == "queued"
            assert result['created_at'] is not None
            assert result['meta'] == {"key": "value"}
    
    def test_get_job_status_returns_none_when_not_found(self):
        """Test that get_job_status returns None when job not found."""
        with patch.object(QueueManager, 'get_rq_job') as mock_get:
            mock_get.return_value = None
            
            result = QueueManager.get_job_status("non-existent-job")
            
            assert result is None


class TestQueueManagerGetQueueStats:
    """Tests for get_queue_stats method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_get_queue_stats_returns_all_queues(self):
        """Test that get_queue_stats returns stats for all queues."""
        pass
    
    def test_get_queue_stats_handles_import_error(self):
        """Test that get_queue_stats handles ImportError gracefully."""
        # When django_rq is not importable, should return empty stats
        result = QueueManager.get_queue_stats()
        
        # Should have all queue names
        for queue_name in QUEUE_MAPPING.values():
            assert queue_name in result


class TestQueueManagerGetQueueHealth:
    """Tests for get_queue_health method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_get_queue_health_healthy(self):
        """Test get_queue_health when system is healthy."""
        pass
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_get_queue_health_unhealthy_no_workers(self):
        """Test get_queue_health when no workers are running."""
        pass
    
    def test_get_queue_health_returns_structure(self):
        """Test that get_queue_health returns expected structure."""
        result = QueueManager.get_queue_health()
        
        assert 'healthy' in result
        assert 'redis_connected' in result
        assert 'workers_active' in result
        assert 'warnings' in result


class TestQueueManagerGetFailedJobs:
    """Tests for get_failed_jobs method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_get_failed_jobs_returns_list(self):
        """Test that get_failed_jobs returns a list of failed jobs."""
        pass
    
    def test_get_failed_jobs_returns_empty_on_error(self):
        """Test that get_failed_jobs returns empty list on error."""
        result = QueueManager.get_failed_jobs()
        assert isinstance(result, list)


class TestQueueManagerRetryFailedJob:
    """Tests for retry_failed_job method."""
    
    @pytest.mark.skip(reason="Requires rq to be installed")
    def test_retry_failed_job_success(self):
        """Test successful retry of failed job."""
        pass
    
    @pytest.mark.skip(reason="Requires rq to be installed")
    def test_retry_failed_job_not_failed(self):
        """Test retry when job is not failed."""
        pass


class TestQueueManagerClearFailedJobs:
    """Tests for clear_failed_jobs method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_clear_failed_jobs_clears_all(self):
        """Test clearing all failed jobs."""
        pass
    
    def test_clear_failed_jobs_returns_zero_on_error(self):
        """Test that clear_failed_jobs returns 0 on error."""
        result = QueueManager.clear_failed_jobs()
        assert result == 0


class TestQueueManagerGetWorkerInfo:
    """Tests for get_worker_info method."""
    
    @pytest.mark.skip(reason="Requires rq to be installed")
    def test_get_worker_info_returns_list(self):
        """Test that get_worker_info returns worker information."""
        pass
    
    def test_get_worker_info_returns_empty_on_error(self):
        """Test that get_worker_info returns empty list on error."""
        result = QueueManager.get_worker_info()
        assert isinstance(result, list)


class TestQueueManagerEstimateWaitTime:
    """Tests for estimate_wait_time method."""
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_estimate_wait_time_with_workers(self):
        """Test wait time estimation with active workers."""
        pass
    
    @pytest.mark.skip(reason="Requires django_rq to be installed")
    def test_estimate_wait_time_no_workers(self):
        """Test wait time estimation with no workers."""
        pass
    
    def test_estimate_wait_time_returns_none_on_error(self):
        """Test that estimate_wait_time returns None on error."""
        result = QueueManager.estimate_wait_time('video_queue')
        assert result is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_enqueue_operation_function(self):
        """Test enqueue_operation convenience function."""
        with patch.object(QueueManager, 'enqueue_operation') as mock_method:
            mock_method.return_value = "job-id"
            
            result = enqueue_operation(uuid4(), "video_queue")
            
            assert result == "job-id"
            mock_method.assert_called_once()
    
    def test_get_queue_stats_function(self):
        """Test get_queue_stats convenience function."""
        with patch.object(QueueManager, 'get_queue_stats') as mock_method:
            mock_method.return_value = {"video_queue": {}}
            
            result = get_queue_stats()
            
            assert result == {"video_queue": {}}
            mock_method.assert_called_once()
    
    def test_get_queue_health_function(self):
        """Test get_queue_health convenience function."""
        with patch.object(QueueManager, 'get_queue_health') as mock_method:
            mock_method.return_value = {"healthy": True}
            
            result = get_queue_health()
            
            assert result == {"healthy": True}
            mock_method.assert_called_once()
    
    def test_cancel_rq_job_function(self):
        """Test cancel_rq_job convenience function."""
        with patch.object(QueueManager, 'cancel_rq_job') as mock_method:
            mock_method.return_value = True
            
            result = cancel_rq_job("job-id")
            
            assert result is True
            mock_method.assert_called_once_with("job-id")
    
    def test_get_failed_jobs_function(self):
        """Test get_failed_jobs convenience function."""
        with patch.object(QueueManager, 'get_failed_jobs') as mock_method:
            mock_method.return_value = [{"id": "job-1"}]
            
            result = get_failed_jobs()
            
            assert result == [{"id": "job-1"}]
            mock_method.assert_called_once()
    
    def test_get_worker_info_function(self):
        """Test get_worker_info convenience function."""
        with patch.object(QueueManager, 'get_worker_info') as mock_method:
            mock_method.return_value = [{"name": "worker-1"}]
            
            result = get_worker_info()
            
            assert result == [{"name": "worker-1"}]
            mock_method.assert_called_once()