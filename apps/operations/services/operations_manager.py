# apps/operations/services/operations_manager.py

"""
Operation Manager Service for handling operation lifecycle.

This module provides centralized job management including:
- Creating new jobs with file uploads
- Queuing jobs for processing
- Updating job status and progress
- Retrieving job information
- Deleting jobs and associated files
"""
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.utils import timezone

from ..models import Operation, File
from ..enums import OperationStatus, FileType
from ..exceptions import (
    OperationNotFoundError,
    OperationAccessDeniedError,
    OperationAlreadyProcessingError,
    OperationNotRetryableError,
    OperationNotDeletableError,
    OperationQueuingError,
    InvalidOperationStateError,
    OperationValidationError
)
from .file_manager import FileManager

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

# Default expiration period for completed operations
DEFAULT_EXPIRATION_DAYS = 7


class OperationsManager:
    """
    Service class for managing media operation lifecycle.
    
    All methods are static as this is a stateless service.
    """
    # OPERATION CREATION

    @staticmethod
    @transaction.atomic
    def create_operation(
        session_key: str,
        operation_name: str,
        parameters: Dict[str, Any],
        uploaded_file: UploadedFile,
        user=None
    ) -> Operation:
        """
        Create a new operation with an uploaded file.
        
        This method:
        1. Validates the operation exists in the registry
        2. Validates the parameters against the operation schema
        3. Creates the operation record
        4. Saves the uploaded file
        5. Creates the file record associated with the operation
        
        Args:
            session_key: User's session key for anonymous tracking
            operation_name: Name of the operation to perform
            parameters: Operation parameters from the user
            uploaded_file: The Django UploadedFile object
            user: Optional Django User instance (for authenticated users)
            
        Returns:
            The created Operation instance
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            InvalidParametersError: If parameters are invalid
            FileTooLargeError: If file exceeds size limit
            UnsupportedFileFormatError: If format not supported
        """
        from apps.processors.registry import get_registry

        registry = get_registry()

        # Validate operation exists
        if not registry.is_registered(operation_name):
            from apps.processors.exceptions import OperationNotFoundError
            raise OperationNotFoundError(operation_name)
        
        # Validate parameters
        validated_params = registry.validate_parameters(operation_name, parameters)

        # Get operation definition for media type
        operation_def = registry.get_operation(operation_name)

        # Create the operation record first
        operation = Operation.objects.create(
            session_key=session_key,
            user=user,
            operation=operation_name,
            parameters=validated_params,
            status=OperationStatus.PENDING,
            progress=0,
        )

        logger.info(
            f"Created job {operation.id} for '{operation_name}' operation"
            f"(session={session_key[:8]}...)"
        )

        try:
            # Save the uploaded file
            file_info = FileManager.save_uploaded_file(
                uploaded_file=uploaded_file,
                operation_id=str(operation.id),
                session_key=session_key,
            )

            # Create the input file record
            File.objects.create(
                operation=operation,
                file_type=FileType.INPUT,
                file_path=file_info['file_path'],
                file_name=file_info['file_name'],
                file_size=file_info['file_size'],
                mime_type=file_info['mime_type'],
                metadata={
                    "original_filename": uploaded_file.name,
                    "media_type": file_info['media_type'],
                },
            )

            logger.info(
                f"Saved input file for job {operation.id}: {file_info['file_name']} "
                f"({file_info['file_size']} bytes)"
            )
        
        except Exception as e:
            # If file save fails, rollback operation creation
            logger.error(f"Failed to save file for operation {operation.id}: {e}")
            raise

        return operation
    

    # Operation/Job Queueing

    @staticmethod
    def queue_operation(operation_id: UUID) -> str:
        """
        Queue a operation/job for processing.
        
        This method:
        1. Retrieves the operation/job
        2. Validates it can be queued
        3. Determines the appropriate queue
        4. Enqueues to Redis via django-rq
        5. Updates operation/job status to QUEUED
        
        Args:
            job_id: UUID of the operation/job to queue
            
        Returns:
            The RQ operation/job ID as a string
            
        Raises:
            OperationNotFoundError: If operation/job doesn't exist
            OperationAlreadyProcessingError: If operation/job is already processing
        """

        operation = OperationsManager._get_operation_by_id(operation_id)

        # Check if operation can be queued
        if operation.status not in [OperationStatus.PENDING, OperationStatus.FAILED]:
            raise OperationAlreadyProcessingError(
                operation_id=str(operation_id),
                current_status=operation.status
            )
        
        # Get the appropriate queue
        queue_name = OperationsManager.get_queue_for_operation(operation.operation)
        timeout = OperationsManager.get_timeout_for_operation(operation.operation)

        try:
            import django_rq

            queue = django_rq.get_queue(queue_name)

            # Enqueue the operation for processing
            # The worker will call process_operation with the operation ID
            rq_job = queue.enqueue(
                "apps.processors.rq_workers.process_operation",
                str(operation_id),
                job_timeout=timeout
            )

            # Update operation status to QUEUED
            operation.status = OperationStatus.QUEUED
            operation.save(update_fields=['status'])

            logger.info(
                f"Queued operation {operation_id} to {queue_name}"
                f"(rq_job_id={rq_job.id}, timeout={timeout}s)"
            )

            return rq_job.id
        
        except Exception as e:
            logger.error(f"Failed to queue operation {operation_id}: {e}")
            raise OperationQueuingError(
                operation_id=str(operation_id),
                reason=str(e)
            )
    

    @staticmethod
    def get_queue_for_operation(operation_name: str) -> str:
        """
        Get the appropriate queue name for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Queue name string
        """
        from apps.processors.registry import get_registry

        registry = get_registry()

        try:
            operation_def = registry.get_operation(operation_name)
            media_type = operation_def.get_media_type()
            return QUEUE_MAPPING.get(media_type, "image_queue")
        except Exception:
             # Default to image queue if operation not found
            logger.warning(
                f"Could not determine queue for operation '{operation_name}', "
                "defaulting to image_queue"
            )
            return "image_queue"
    

    @staticmethod
    def get_timeout_for_operation(operation_name: str) -> int:
        """
        Get the timeout in seconds for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Timeout in seconds
        """
        from apps.processors.registry import get_registry

        registry = get_registry()

        try:
            operation_def = registry.get_operation(operation_name)
            media_type = operation_def.media_type.value
            return TIMEOUT_MAPPING.get(media_type, 60)
        except Exception:
            # Default to 60 seconds if operation not found
            return 60
    

    # Operation status updates

    @staticmethod
    def start_operation(operation_id: UUID) -> Operation:
        """
        Mark a job as started (processing).
        
        Args:
            operation_id: UUID of the operation/job
            
        Returns:
            Updated Operation/job instance
            
        Raises:
            OperationNotFoundError: If operation/job doesn't exist
            InvalidOperationStateError: If operation/job is not in QUEUED status
        """
        operation = OperationsManager._get_operation_by_id(operation_id)

        if operation.status != OperationStatus.QUEUED:
            raise InvalidOperationStateError(
                operation_id=str(operation_id),
                current_status=operation.status,
                expected_statuses=[OperationStatus.QUEUED],
            )
        
        operation.status = OperationStatus.PROCESSING
        operation.started_at = timezone.now()
        operation.save(update_fields=['status', 'started_at'])

        logger.info(f"Started processing operation {operation_id}")

        return operation
    

    @staticmethod
    def update_operation_progress(
        operation_id: UUID,
        progress: int,
        status: Optional[str] = None
    ) -> Operation:
        """
        Update the progress of a operation/job.
        
        Args:
            job_id: UUID of the operation/job
            progress: Progress percentage (0-100)
            status: Optional new status
            
        Returns:
            Updated Operation instance
            
        Raises:
            OperationNotFoundError: If operation/job doesn't exist
        """
        operation = OperationsManager._get_operation_by_id(operation_id)

        # Clamp progress between 0 and 100
        operation.progress = max(0, min(100, progress))

        update_fields = ['progress']

        if status is not None:
            operation.status = status
            update_fields.append('status')
        
        operation.save(update_fields=update_fields)

        logger.debug(f"Updated operation {operation_id} progress to {progress}%")

        return operation
    

    @staticmethod
    @transaction.atomic
    def complete_operation(
        operation_id: UUID,
        output_file_path: str,
        output_filename: str,
        session_key: str,
    ) -> Operation:
        """
        Mark a operation/job as completed and register the output file.
        
        Args:
            operation_id: UUID of the operation/job
            output_file_path: Path to the processed output file (temp location)
            output_filename: Desired filename for the output
            session_key: User's session key
            
        Returns:
            Updated Operation instance
            
        Raises:
            OperationNotFoundError: If operation/job doesn't exist
        """
        operation = OperationsManager._get_operation_by_id(operation_id)

        now = timezone.now()

        # Move output file to permanent storage
        output_info = FileManager.move_to_output(
            temp_path=output_file_path,
            operation_id=str(operation_id),
            session_key=session_key,
            output_filename=output_filename,
        )

        # Create output file record
        File.objects.create(
            operation=operation,
            file_type=FileType.OUTPUT,
            file_path=output_info['file_path'],
            file_name=output_info['file_name'],
            file_size=output_info['file_size'],
            mime_type=output_info['mime_type'],
            metadata={},
        )

        # Update operation status to COMPLETED
        operation.status = OperationStatus.COMPLETED
        operation.progress = 100
        operation.completed_at = now
        operation.expires_at = now + timedelta(days=DEFAULT_EXPIRATION_DAYS)
        operation.save(update_fields=["status", "progress", "completed_at", "expires_at"])

        # Clean up temp directory
        FileManager.cleanup_temp_directory(str(operation_id))

        logger.info(
            f"Comepleted operation {operation_id} - output: {output_info['file_name']}"
            f"({output_info['file_size']} bytes)"
        )

        return operation
    

    @staticmethod
    def fail_operation(operation_id: UUID, error_message: str) -> Operation:
        """
        Mark a operation/job as failed with an error message.
        
        Args:
            operation_id: UUID of the operation/job
            error_message: Description of what went wrong
            
        Returns:
            Updated Operation instance
            
        Raises:
            OperationNotFoundError: If operation/job doesn't exist
        """
        operation = OperationsManager._get_operation_by_id(operation_id)

        now = timezone.now()

        operation.status = OperationStatus.FAILED
        operation.error_message = error_message[:2000]
        operation.completed_at = now
        operation.expires_at = now + timedelta(days=DEFAULT_EXPIRATION_DAYS)
        operation.save(update_fields=["status", "error_message", "completed_at", "expires_at"])

        # Clean up temp directory
        FileManager.cleanup_temp_directory(str(operation_id))

        logger.error(f"Operation {operation.id} failed: {error_message[:200]}")

        return operation
    

    # OPERATION RETRIEVAL

    @staticmethod
    def get_operation(operation_id: UUID, session_key: str) -> Operation:
        """
        Retrieve a operation with ownership verification.
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            Operation instance
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
        """
        operation = OperationsManager._get_operation_by_id(operation_id)

        # Verify session ownership
        if operation.session_key != session_key:
            raise OperationAccessDeniedError(
                operation_id=str(operation_id),
                session_key=session_key
            )
        
        return operation
    

    @staticmethod
    def get_operation_with_files(operation_id: UUID, session_key: str) -> Operation:
        """
        Retrieve an operation with its files, with ownership verification.
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            Operation instance with prefetched files
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
        """
        try:
            operation = Operation.objects.prefetch_related('files').get(id=operation_id, is_deleted=False)
        except Operation.DoesNotExist:
            raise OperationNotFoundError(str(operation_id))
        
        # Verify session ownership
        if operation.session_key != session_key:
            raise OperationAccessDeniedError(
                operation_id=str(operation_id),
                session_key=session_key
            )
        
        return operation
    

    @staticmethod
    def list_user_operations(
        session_key: str,
        limit: int = 50,
        status_filter: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[Operation]:
        """
        List all operations for a session.
        
        Args:
            session_key: User's session key
            limit: Maximum number of operations to return
            status_filter: Optional status to filter by
            include_deleted: Whether to include soft-deleted operations
            
        Returns:
            List of Operation instances
        """
        queryset = Operation.objects.filter(session_key=session_key)

        if not include_deleted:
            queryset = queryset.filter(is_deleted=False)
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Ordery by most recent first
        queryset = queryset.order_by('-created_at')

        if limit:
            queryset = queryset[:limit]
        
        return list(queryset)
    

    @staticmethod
    def get_operation_status(operation_id: UUID, session_key: str) -> Dict[str, Any]:
        """
        Get lightweight status information for an operation (for polling).
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            Dictionary with status information
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
        """
        operation = OperationsManager.get_operation(operation_id, session_key)

        result = {
            "id": str(operation.id),
            "status": operation.status,
            "progress": operation.progress,
        }

        # Add error message if failed
        if operation.status == OperationStatus.FAILED and operation.error_message:
            result["error_message"] = operation.error_message

        # Add processing time if completed
        if operation.processing_time is not None:
            result["processing_time_seconds"] = operation.processing_time
        
        return result
    

    # OPERATION DELETION
    @staticmethod
    @transaction.atomic
    def delete_operation(operation_id: UUID, session_key: str, force: bool = False) -> bool:
        """
        Delete a operation and its associated files (soft delete).
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            force: If True, delete regardless of status
            
        Returns:
            True if deleted successfully
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
            OperationNotDeletableError: If operation cannot be deleted
        """
        operation = OperationsManager.get_operation(operation_id, session_key)

        # Check if operation can be deleted
        if not force and not operation.can_be_deleted():
            raise OperationNotDeletableError(
                operation_id=str(operation_id),
                current_status=operation.status
            )
        
        # Delete physical files
        deleted_count = FileManager.delete_operation_files(
            operation_id=str(operation_id),
            session_key=session_key
        )

        # Soft delete the operation record
        operation.is_deleted = True
        operation.save(update_fields=['is_deleted'])

        logger.info(
            f"Deleted operation {operation_id} (soft delete, {deleted_count} files removed)"
        )

        return True
    

    @staticmethod
    def hard_delete_operation(operation_id: UUID) -> bool:
        """
        Permanently delete an operation and its files from the database.
        
        This should only be used by cleanup tasks.
        
        Args:
            job_id: UUID of the operation
            
        Returns:
            True if deleted successfully
        """
        try:
            operation = Operation.objects.get(id=operation_id)
            session_key = operation.session_key or "anonymous"

            # Delete physical files
            FileManager.delete_operation_files(
                operation_id=str(operation_id),
                session_key=session_key
            )

            # Permanently delete the operation record
            operation.delete()

            logger.info(f"Permanently deleted operation {operation_id}")
            return True
        
        except Operation.DoesNotExist:
            logger.warning(f"Operation {operation_id} not found for hard deletion")
            return False
    

    # OPERATION RETRY
    @staticmethod
    def retry_operation(operation_id: UUID, session_key: str) -> Operation:
        """
        Retry a failed operation.
        
        Args:
            job_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            New RQ operation ID
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
            OperationNotRetryableError: If operation cannot be retried
        """
        operation = OperationsManager.get_operation(operation_id, session_key)

        if not operation.can_be_retried():
            raise OperationNotRetryableError(
                operation_id=str(operation_id),
                current_status=operation.status
            )
        
        # Reset operation state
        operation.status = OperationStatus.PENDING
        operation.progress = 0
        operation.error_message = None
        operation.started_at = None
        operation.completed_at = None
        operation.expires_at = None
        operation.save(update_fields=[
            'status', 'progress', 'error_message',
            'started_at', 'completed_at', 'expires_at'
        ])

        logger.info(f"Reset operation {operation_id} for retry")

        # Queue the operation again
        return OperationsManager.queue_operation(operation_id)

    
    # UTILITY METHODS

    @staticmethod
    def _get_operation_by_id(operation_id: UUID) -> Operation:
        """
        Retrieve an operation by ID without ownership check.
        
        Args:
            operation_id: UUID of the operation
            
        Returns:
            Operation instance
            
        Raises:
            JobNotFoundError: If job doesn't exist
        """
        try:
            return Operation.objects.get(id=operation_id, is_deleted=False)
        except Operation.DoesNotExist:
            raise OperationNotFoundError(str(operation_id))
    

    @staticmethod
    def get_input_file(operation_id: UUID) -> Optional[File]:
        """
        Get the input file for an operation.
        
        Args:
            job_id: UUID of the operation
            
        Returns:
            File instance or None if not found
        """
        try:
            return File.objects.get(
                operation_id=operation_id,
                file_type=FileType.INPUT
            )
        except File.DoesNotExist:
            return None
    

    @staticmethod
    def get_output_file(operation_id: UUID) -> Optional[File]:
        """
        Get the output file for an operation.
        
        Args:
            job_id: UUID of the operation
            
        Returns:
            File instance or None if not found
        """
        try:
            return File.objects.get(
                operation_id=operation_id,
                file_type=FileType.OUTPUT
            )
        except File.DoesNotExist:
            return None
    

    @staticmethod
    def get_queue_stats() -> Dict[str, Dict[str, int]]:
        """
        Get statistics for all queues.
        
        Returns:
            Dictionary with queue stats:
            {
                "video_queue": {"queued": 5, "started": 2, "failed": 0},
                ...
            }
        """
        stats = {}

        try:
            import django_rq

            for queue_name in QUEUE_MAPPING.values():
                queue = django_rq.get_queue(queue_name)
                stats[queue_name] = {
                    "queued": queue.count,
                    "started": queue.started_operation_registry.count,
                    "failed": queue.failed_operation_registry.count,
                }
        
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            # Return empty stats if redis is unavailable
            for queue_name in QUEUE_MAPPING.values():
                stats[queue_name] = {
                    "queued": 0,
                    "started": 0,
                    "failed": 0,
                }
        
        return stats
    

    @staticmethod
    def cancel_operation(operation_id: UUID, session_key: str) -> bool:
        """
        Cancel a queued operation.
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            True if cancelled successfully
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
            InvalidOperationStateError: If operation is not in a cancellable state
        """
        operation = OperationsManager.get_operation(operation_id,
        session_key)

        if operation.status not in [OperationStatus.PENDING, OperationStatus.QUEUED]:
            raise InvalidOperationStateError(
                operation_id=str(operation_id),
                current_status=operation.status,
                expected_statuses=[OperationStatus.PENDING, OperationStatus.QUEUED],
            )
        
        # Mark as failed with cancellation message
        operation.status = OperationStatus.FAILED
        operation.error_message = "Operation was cancelled by the user."
        operation.completed_at = timezone.now()
        operation.expires_at = operation.completed_at + timedelta(days=DEFAULT_EXPIRATION_DAYS)
        operation.save(update_fields=["status", "error_message", "completed_at", "expires_at"])

        logger.info(f"Cancelled operation {operation_id}")

        return True
    

    @staticmethod
    def cleanup_expired_operations(dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up expired operations and their files.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            "operations_processed": 0,
            "operations_deleted": 0,
            "errors": [],
        }

        # Find expired operations that haven't been deleted
        expired_operations = Operation.objects.filter(
            is_deleted=False,
            expires_at__lte=timezone.now()
        )

        for operation in expired_operations:
            stats["operations_processed"] += 1
            try:
                if not dry_run:
                    # Delete files
                    FileManager.delete_operation_files(
                        operation_id=str(operation.id),
                        session_key=operation.session_key or "anonymous"
                    )

                    # Soft delete operation
                    operation.is_deleted = True
                    operation.save(update_fields=['is_deleted'])
                
                stats["operations_deleted"] += 1
            
            except Exception as e:
                error_msg = f"Failed to delete operation {operation.id}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        logger.info(
            f"Cleanup complete: {stats['operations_deleted']} operations deleted "
            f"({stats['operations_processed']} processed)"
        )

        return stats



