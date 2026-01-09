"""
Queue Manager Service for RQ job queue integration.

This module provides comprehensive queue management including:
- Enqueueing operations to appropriate queues
- Retrieving queue statistics
- Cancelling queued jobs
- Monitoring queue health
- Managing job lifecycle in Redis
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


# Queue name mapping based on media type
QUEUE_MAPPING = {
    "video": "video_queue",
    "image": "image_queue",
    "audio": "audio_queue",
}

# Default timeout in seconds for each media type
TIMEOUT_MAPPING = {
    "video": 1800,   # 30 minutes
    "image": 60,     # 1 minute
    "audio": 300,    # 5 minutes
}

WORKER_FUNCTION = "apps.processors.rq_workers.process_operation"


class QueueManager:
    """
    Service class for managing RQ job queues.
    
    Provides methods for:
    - Enqueueing operations
    - Getting queue statistics
    - Cancelling jobs
    - Monitoring queue health
    """

    @staticmethod
    def get_queue(queue_name: str):
        """
        Get an RQ queue by name.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            django_rq Queue instance
            
        Raises:
            ImportError: If django_rq is not installed
        """
        import django_rq
        return django_rq.get_queue(queue_name)
    

    @staticmethod
    def get_queue_for_media_type(media_type: str) -> str:
        """
        Get the queue name for a given media type.
        
        Args:
            media_type: 'video', 'image', or 'audio'
            
        Returns:
            Queue name string
        """
        return QUEUE_MAPPING.get(media_type, "image_queue")
    

    @staticmethod
    def get_timeout_for_media_type(media_type: str) -> int:
        """
        Get the timeout for a given media type.
        
        Args:
            media_type: 'video', 'image', or 'audio'
            
        Returns:
            Timeout in seconds
        """
        return TIMEOUT_MAPPING.get(media_type, 60)
    

    def enqueue_operation(
        operation_id: UUID,
        queue_name: str,
        timeout: Optional[int] = None,
        retry: Optional[int] = None,
        retry_delay: Optional[int] = None,
    ) -> str:
        """
        Enqueue an operation for processing.
        
        Args:
            operation_id: UUID of the operation to process
            queue_name: Name of the queue to enqueue to
            timeout: Optional timeout override in seconds
            retry: Optional number of retries
            retry_delay: Optional delay between retries in seconds
            
        Returns:
            The RQ job ID as a string
            
        Raises:
            Exception: If enqueueing fails
        """
        import django_rq
        from rq import Retry
        
        queue = django_rq.get_queue(queue_name)
        
        # Build job options
        job_kwargs = {
            'job_timeout': timeout or TIMEOUT_MAPPING.get(queue_name.replace('_queue', ''), 60),
        }
        
        # Add retry configuration if specified
        if retry is not None and retry > 0:
            delays = [retry_delay or 60] * retry  # Same delay for each retry
            job_kwargs['retry'] = Retry(max=retry, interval=delays)
        
        # Enqueue the job
        rq_job = queue.enqueue(
            WORKER_FUNCTION,
            str(operation_id),
            **job_kwargs
        )
        
        logger.info(
            f"Enqueued operation {operation_id} to {queue_name} "
            f"(rq_job_id={rq_job.id}, timeout={job_kwargs['job_timeout']}s)"
        )
        
        return rq_job.id
    

    @staticmethod
    def get_rq_job(job_id: str):
        """
        Get an RQ job by ID.
        
        Args:
            job_id: The RQ job ID
            
        Returns:
            RQ Job instance or None if not found
        """
        from rq.job import Job
        from django_rq import get_connection
        
        try:
            connection = get_connection()
            return Job.fetch(job_id, connection=connection)
        except Exception as e:
            logger.warning(f"Could not fetch RQ job {job_id}: {e}")
            return None
    

    @staticmethod
    def cancel_rq_job(job_id: str) -> bool:
        """
        Cancel an RQ job.
        
        Args:
            job_id: The RQ job ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        job = QueueManager.get_rq_job(job_id)

        if job is None:
            logger.warning(f"RQ job {job_id} not found for cancellation")
            return False
        
        try:
            # Check if job can be cancelled
            if job.is_finished:
                logger.info(f"RQ job {job_id} already finished")
                return False
            
            if job.is_failed:
                logger.info(f"RQ job {job_id} already failed")
                return False
            
            # Cancel the job
            job.cancel()
            logger.info(f"Cancelled RQ job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel RQ job {job_id}: {e}")
            return False
    

    @staticmethod
    def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an RQ job.
        
        Args:
            job_id: The RQ job ID
            
        Returns:
            Dictionary with job status info or None if not found
        """
        job = QueueManager.get_rq_job(job_id)
        
        if job is None:
            return None
        
        return {
            'id': job.id,
            'status': job.get_status(),
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'ended_at': job.ended_at.isoformat() if job.ended_at else None,
            'result': job.result if job.is_finished else None,
            'exc_info': str(job.exc_info) if job.is_failed else None,
            'meta': job.meta,
        }
    

    @staticmethod
    def get_queue_stats() -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics for all queues.
        
        Returns:
            Dictionary with queue stats:
            {
                "video_queue": {
                    "queued": 5,
                    "started": 2,
                    "failed": 0,
                    "deferred": 0,
                    "scheduled": 0,
                    "workers": 1,
                },
                ...
            }
        """
        stats = {}

        try:
            import django_rq
            from rq import Worker
            from django_rq import get_connection

            connection = get_connection()

            for queue_name in QUEUE_MAPPING.values():
                try:
                    queue = django_rq.get_queue(queue_name)

                    # Get worker count for this queue
                    workers = Worker.all(connection=connection)
                    queue_workers = [w for w in workers if queue_name in [q.name for q in w.queues]]

                    stats[queue_name] = {
                        "queued": queue.count,
                        "started": len(queue.started_job_registry),
                        "failed": len(queue.failed_job_registry),
                        "deferred": len(queue.deferred_job_registry),
                        "scheduled": len(queue.scheduled_job_registry),
                        "finished": len(queue.finished_job_registry),
                        "workers": len(queue_workers),
                    }
                except Exception as e:
                    logger.warning(f"Error getting stats for queue {queue_name}: {e}")
                    stats[queue_name] = {
                        "queued": 0,
                        "started": 0,
                        "failed": 0,
                        "deferred": 0,
                        "scheduled": 0,
                        "finished": 0,
                        "workers": 0,
                        "error": str(e),
                    }
        
        except ImportError:
            logger.error("django_rq not installed")
            for queue_name in QUEUE_MAPPING.values():
                stats[queue_name] = {
                    "queued": 0,
                    "started": 0,
                    "failed": 0,
                    "deferred": 0,
                    "scheduled": 0,
                    "finished": 0,
                    "workers": 0,
                    "error": "django_rq not installed",
                }
        
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            for queue_name in QUEUE_MAPPING.values():
                stats[queue_name] = {
                    "queued": 0,
                    "started": 0,
                    "failed": 0,
                    "deferred": 0,
                    "scheduled": 0,
                    "finished": 0,
                    "workers": 0,
                    "error": str(e),
                }
        
        return stats
    

    @staticmethod
    def get_queue_health() -> Dict[str, Any]:
        """
        Get overall queue health status.
        
        Returns:
            Dictionary with health status:
            {
                "healthy": True,
                "redis_connected": True,
                "workers_active": 4,
                "total_queued": 10,
                "total_failed": 2,
                "queues": {...},
                "warnings": [],
            }
        """
        health = {
            "healthy": True,
            "redis_connected": False,
            "workers_active": 0,
            "total_queued": 0,
            "total_failed": 0,
            "queues": {},
            "warnings": [],
        }

        try:
            import django_rq
            from django_rq import get_connection
            from rq import Worker

            # Check Redis connection
            connection = get_connection()
            connection.ping()
            health["redis_connected"] = True

            # Get all workers
            workers = Worker.all(connection=connection)
            health["workers_active"] = len([w for w in workers if w.state == 'busy' or w.state == 'idle'])

            # Get queue stats
            queue_stats = QueueManager.get_queue_stats()
            health["queues"] = queue_stats

            # Calculate totals
            for queue_name, stats in queue_stats.items():
                health["total_queued"] += stats.get("queued", 0)
                health["total_failed"] += stats.get("failed", 0)

                # Check for warnings
                if stats.get("workers", 0) == 0:
                    health["warnings"].append(f"No workers for {queue_name}")
                    health["healthy"] = False
                
                if stats.get("failed", 0) > 10:
                    health["warnings"].append(f"High failure count in {queue_name}")
            
            # Check if any workers are running
            if health["workers_active"] == 0:
                health["warnings"].append("No active workers detected")
                health["healthy"] = False
        
        except ImportError:
            health["healthy"] = False
            health["warnings"].append("django_rq not installed")
        
        except Exception as e:
            health["healthy"] = False
            health["warnings"].append(f"Redis connection error: {str(e)}")
        
        return health
    

    @staticmethod
    def get_failed_jobs(queue_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get failed jobs from queues.
        
        Args:
            queue_name: Optional specific queue name, or all queues if None
            limit: Maximum number of jobs to return
            
        Returns:
            List of failed job information dictionaries
        """
        failed_jobs = []

        try:
            import django_rq
            from rq.job import Job
            from django_rq import get_connection

            connection = get_connection()
            queues_to_check = [queue_name] if queue_name else list(QUEUE_MAPPING.values())
            
            for qname in queues_to_check:
                try:
                    queue = django_rq.get_queue(qname)
                    failed_registry = queue.failed_job_registry

                    job_ids = failed_registry.get_job_ids(0, limit)

                    for job_id in job_ids:
                        try:
                            job = Job.fetch(job_id, connection=connection)
                            failed_jobs.append({
                                'id': job.id,
                                'queue': qname,
                                'func_name': job.func_name,
                                'args': job.args,
                                'created_at': job.created_at.isoformat() if job.created_at else None,
                                'ended_at': job.ended_at.isoformat() if job.ended_at else None,
                                'exc_info': str(job.exc_info)[:500] if job.exc_info else None,
                            })
                        except Exception as e:
                            logger.warning(f"Error fetching failed job {job_id}: {e}")
                
                except Exception as e:
                    logger.warning(f"Error getting failed jobs from {qname}: {e}")
                
                if len(failed_jobs) >= limit:
                    break
        
        except Exception as e:
            logger.error(f"Error getting failed jobs: {e}")
        
        return failed_jobs[:limit]
    

    @staticmethod
    def retry_failed_job(job_id: str) -> Optional[str]:
        """
        Retry a failed job.
        
        Args:
            job_id: The RQ job ID to retry
            
        Returns:
            New job ID if requeued successfully, None otherwise
        """
        try:
            from rq.job import Job
            from django_rq import get_connection
            
            connection = get_connection()
            job = Job.fetch(job_id, connection=connection)
            
            if not job.is_failed:
                logger.warning(f"Job {job_id} is not failed, cannot retry")
                return None
            
            # Requeue the job
            job.requeue()
            logger.info(f"Requeued failed job {job_id}")
            
            return job.id
        
        except Exception as e:
            logger.error(f"Error retrying job {job_id}: {e}")
            return None
    

    @staticmethod
    def clear_failed_jobs(queue_name: Optional[str] = None) -> int:
        """
        Clear all failed jobs from queues.
        
        Args:
            queue_name: Optional specific queue name, or all queues if None
            
        Returns:
            Number of jobs cleared
        """
        cleared_count = 0

        try:
            import django_rq

            queues_to_clear = [queue_name] if queue_name else list(QUEUE_MAPPING.values())

            for qname in queues_to_clear:
                try:
                    queue = django_rq.get_queue(qname)
                    failed_registry = queue.failed_job_registry

                    # Get all failed job IDs
                    job_ids = failed_registry.get_job_ids()

                    # Remove each failed job
                    for job_id in job_ids:
                        try:
                            failed_registry.remove(job_id, delete_job=True)
                            cleared_count += 1
                        except Exception as e:
                            logger.warning(f"Error removing failed job {job_id}: {e}")
                
                except Exception as e:
                    logger.warning(f"Error clearing failed jobs from {qname}: {e}")
            
            logger.info(f"Cleared {cleared_count} failed jobs")
        
        except Exception as e:
            logger.error(f"Error clearing failed jobs: {e}")
        
        return cleared_count
    

    @staticmethod
    def get_worker_info() -> List[Dict[str, Any]]:
        """
        Get information about all workers.
        
        Returns:
            List of worker information dictionaries
        """
        workers_info = []
        
        try:
            from rq import Worker
            from django_rq import get_connection
            
            connection = get_connection()
            workers = Worker.all(connection=connection)
            
            for worker in workers:
                workers_info.append({
                    'name': worker.name,
                    'state': worker.state,
                    'queues': [q.name for q in worker.queues],
                    'current_job': worker.get_current_job_id(),
                    'successful_job_count': worker.successful_job_count,
                    'failed_job_count': worker.failed_job_count,
                    'total_working_time': worker.total_working_time,
                    'birth_date': worker.birth_date.isoformat() if worker.birth_date else None,
                    'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
                })
        
        except Exception as e:
            logger.error(f"Error getting worker info: {e}")
        
        return workers_info
    

    @staticmethod
    def estimate_wait_time(queue_name: str) -> Optional[int]:
        """
        Estimate wait time for a new job in seconds.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Estimated wait time in seconds, or None if cannot estimate
        """
        try:
            import django_rq
            from rq import Worker
            from django_rq import get_connection

            connection = get_connection()
            queue = django_rq.get_queue(queue_name)

            # Get queue length
            queue_length = queue.count

            # Get workers count
            workers = Worker.all(connection=connection)
            queue_workers = [w for w in workers if queue_name in [q.name for q in w.queues]]
            worker_count = len(queue_workers)

            if worker_count == 0:
                return None  # Cannot estimate without workers
            
            # Get average job duration from timeout
            media_type = queue_name.replace('_queue', '')
            avg_job_time = TIMEOUT_MAPPING.get(media_type, 60)

            # Estimate wait time
            estimate_wait = int((queue_length / worker_count) * avg_job_time)

            return estimate_wait
        
        except Exception as e:
            logger.warning(f"Error estimating wait time: {e}")
            return None
    

# Convenience functions for direct import
def enqueue_operation(
    operation_id: UUID,
    queue_name: str,
    **kwargs
) -> str:
    """Function to enqueue an operation."""
    return QueueManager.enqueue_operation(
        operation_id,
        queue_name,
        **kwargs
    )


def get_queue_stats() -> Dict[str, Dict[str, Any]]:
    """Function to get queue statistics."""
    return QueueManager.get_queue_stats()


def get_queue_health() -> Dict[str, Any]:
    """Function to get queue health status."""
    return QueueManager.get_queue_health()


def cancel_rq_job(job_id: str) -> bool:
    """Function to cancel an RQ job."""
    return QueueManager.cancel_rq_job(job_id)


def get_failed_jobs(queue_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Function to get failed jobs."""
    return QueueManager.get_failed_jobs(queue_name, limit)


def get_worker_info() -> List[Dict[str, Any]]:
    """Function to get worker information."""
    return QueueManager.get_worker_info()