# apps/operations/services/operations_manager.py

"""
Operation Manager Service for handling operation lifecycle.

This module provides centralized job management including:
- Creating new jobs with file uploads (to Cloudinary or local storage)
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
        4. Saves the uploaded file (to Cloudinary or local storage)
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
            # Save the uploaded file (to Cloudinary or local storage)
            file_info = FileManager.save_uploaded_file(
                uploaded_file=uploaded_file,
                operation_id=str(operation.id),
                session_key=session_key,
            )

            # Create the input file record with all fields including Cloudinary
            File.objects.create(
                operation=operation,
                file_type=FileType.INPUT,
                file_path=file_info.get('file_path', ''),
                file_name=file_info['file_name'],
                file_size=file_info['file_size'],
                mime_type=file_info['mime_type'],
                # Cloudinary fields (empty strings if using local storage)
                cloudinary_public_id=file_info.get('cloudinary_public_id', ''),
                cloudinary_url=file_info.get('cloudinary_url', ''),
                cloudinary_resource_type=file_info.get('cloudinary_resource_type', ''),
                metadata={
                    "original_filename": uploaded_file.name,
                    "media_type": file_info['media_type'],
                    "storage_location": "cloudinary" if file_info.get('cloudinary_public_id') else "local",
                },
            )

            storage_type = "Cloudinary" if file_info.get('cloudinary_public_id') else "local"
            logger.info(
                f"Saved input file for job {operation.id}: {file_info['file_name']} "
                f"({file_info['file_size']} bytes, storage={storage_type})"
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
            media_type = operation_def.media_type.value
            return QUEUE_MAPPING.get(media_type, "video_queue")
        except Exception:
            # Default to video queue if operation not found
            return "video_queue"
    

    @staticmethod
    def get_timeout_for_operation(operation_name: str) -> int:
        """
        Get the appropriate timeout for an operation.
        
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
            return TIMEOUT_MAPPING.get(media_type, 1800)
        except Exception:
            # Default to video timeout if operation not found
            return 1800


    # OPERATION RETRIEVAL
    
    @staticmethod
    def get_operation(operation_id: UUID, session_key: str) -> Operation:
        """
        Retrieve an operation by ID with ownership validation.
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            Operation instance
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            OperationAccessDeniedError: If operation doesn't belong to session
        """
        try:
            operation = Operation.objects.prefetch_related('files').get(
                id=operation_id,
                is_deleted=False
            )
        except Operation.DoesNotExist:
            raise OperationNotFoundError(str(operation_id))
        
        # Verify ownership
        if operation.session_key != session_key:
            raise OperationAccessDeniedError(str(operation_id))
        
        return operation
    

    @staticmethod
    def list_operations(
        session_key: str,
        status_filter: Optional[str] = None,
        operation_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Operation]:
        """
        List operations for a session with optional filtering.
        
        Args:
            session_key: User's session key
            status_filter: Optional status to filter by
            operation_filter: Optional operation name to filter by
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of Operation instances
        """
        queryset = Operation.objects.filter(
            session_key=session_key,
            is_deleted=False
        ).prefetch_related('files')

        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        if operation_filter:
            queryset = queryset.filter(operation=operation_filter)
        
        queryset = queryset.order_by('-created_at')

        return list(queryset[offset:offset + limit])
    

    @staticmethod
    def count_operations(
        session_key: str,
        status_filter: Optional[str] = None
    ) -> int:
        """
        Count operations for a session.
        
        Args:
            session_key: User's session key
            status_filter: Optional status to filter by
            
        Returns:
            Count of matching operations
        """
        queryset = Operation.objects.filter(
            session_key=session_key,
            is_deleted=False
        )

        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset.count()
    

    # STATUS UPDATES

    @staticmethod
    def update_progress(operation_id: UUID, progress: int) -> None:
        """
        Update an operation's progress.
        
        Args:
            operation_id: UUID of the operation
            progress: Progress percentage (0-100)
        """
        # Clamp progress to valid range
        progress = max(0, min(100, progress))

        Operation.objects.filter(id=operation_id).update(progress=progress)
        logger.debug(f"Updated progress for operation {operation_id}: {progress}%")
    

    @staticmethod
    def mark_operation_started(operation_id: UUID) -> None:
        """
        Mark an operation as started (processing).
        
        Args:
            operation_id: UUID of the operation
        """
        Operation.objects.filter(id=operation_id).update(
            status=OperationStatus.PROCESSING,
            started_at=timezone.now(),
            progress=0
        )
        logger.info(f"Operation {operation_id} marked as PROCESSING")
    

    @staticmethod
    def mark_operation_completed(
        operation_id: UUID,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark an operation as successfully completed.
        
        Args:
            operation_id: UUID of the operation
            output_path: Path to the output file (local or Cloudinary public_id)
            metadata: Optional processing metadata
        """
        expiration_days = getattr(settings, 'OPERATION_EXPIRATION_DAYS', DEFAULT_EXPIRATION_DAYS)
        now = timezone.now()

        update_fields = {
            'status': OperationStatus.COMPLETED,
            'progress': 100,
            'completed_at': now,
            'expires_at': now + timedelta(days=expiration_days),
            'error_message': None
        }

        # Update with metadata if provided
        if metadata:
            operation = OperationsManager._get_operation_by_id(operation_id)
            params = operation.parameters or {}
            params['_processing_metadata'] = metadata
            update_fields['parameters'] = params
            operation.status = OperationStatus.COMPLETED
            operation.progress = 100
            operation.completed_at = now
            operation.expires_at = now + timedelta(days=expiration_days)
            operation.error_message = None
            operation.parameters = params
            operation.save()
        else:
            Operation.objects.filter(id=operation_id).update(**update_fields)

        logger.info(f"Operation {operation_id} marked as COMPLETED")
    

    @staticmethod
    def mark_operation_failed(
        operation_id: UUID,
        error_message: str,
        is_retryable: bool = False,
        error_code: str = "PROCESSING_ERROR"
    ) -> None:
        """
        Mark an operation as failed.
        
        Args:
            operation_id: UUID of the operation
            error_message: User-friendly error message
            is_retryable: Whether the operation can be retried
            error_code: Error code for classification
        """
        expiration_days = getattr(settings, 'OPERATION_EXPIRATION_DAYS', DEFAULT_EXPIRATION_DAYS)
        now = timezone.now()

        try:
            operation = OperationsManager._get_operation_by_id(operation_id)
            
            params = operation.parameters or {}
            params['_error_info'] = {
                'error_code': error_code,
                'is_retryable': is_retryable,
                'failed_at': now.isoformat()
            }

            operation.status = OperationStatus.FAILED
            operation.error_message = error_message
            operation.completed_at = now
            operation.expires_at = now + timedelta(days=expiration_days)
            operation.parameters = params
            operation.save(update_fields=[
                'status', 'error_message', 'completed_at', 'expires_at', 'parameters'
            ])

            logger.info(
                f"Operation {operation_id} marked as FAILED "
                f"(retryable={is_retryable}, code={error_code})"
            )
        except Exception as e:
            # Fallback to simple update if full update fails
            logger.error(f"Failed to fully update operation {operation_id}: {e}")
            Operation.objects.filter(id=operation_id).update(
                status=OperationStatus.FAILED,
                error_message=error_message,
                completed_at=timezone.now()
            )


    @staticmethod
    def get_operation_status(operation_id: UUID, session_key: str) -> Dict[str, Any]:
        """
        Get lightweight status information for polling.
        
        Args:
            operation_id: UUID of the operation
            session_key: User's session key for ownership check
            
        Returns:
            Dictionary with status information
        """
        operation = OperationsManager.get_operation(operation_id, session_key)

        result = {
            "id": str(operation.id),
            "status": operation.status,
            "progress": operation.progress,
            "is_complete": operation.status in [
                OperationStatus.COMPLETED, 
                OperationStatus.FAILED
            ],
            "has_output": False,
        }

        # Check for output file
        if operation.status == OperationStatus.COMPLETED:
            output_file = operation.files.filter(file_type=FileType.OUTPUT).first()
            result["has_output"] = output_file is not None
            if output_file:
                result["storage_location"] = (
                    "cloudinary" if output_file.cloudinary_public_id else "local"
                )

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
        
        # Collect Cloudinary files for deletion
        cloudinary_files = []
        for file_obj in operation.files.all():
            if file_obj.cloudinary_public_id:
                cloudinary_files.append({
                    'public_id': file_obj.cloudinary_public_id,
                    'resource_type': file_obj.cloudinary_resource_type or 'auto',
                })

        # Delete physical files (local and/or Cloudinary)
        deleted_count = FileManager.delete_operation_files(
            operation_id=str(operation_id),
            session_key=session_key,
            cloudinary_files=cloudinary_files if cloudinary_files else None,
        )

        # Soft delete the operation record
        operation.is_deleted = True
        operation.save(update_fields=['is_deleted'])

        storage_type = "cloudinary" if cloudinary_files else "local"
        logger.info(
            f"Deleted operation {operation_id} (soft delete, {deleted_count} files removed, "
            f"storage={storage_type})"
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
            operation = Operation.objects.prefetch_related('files').get(id=operation_id)
            session_key = operation.session_key or "anonymous"

            # Collect Cloudinary files for deletion
            cloudinary_files = []
            for file_obj in operation.files.all():
                if file_obj.cloudinary_public_id:
                    cloudinary_files.append({
                        'public_id': file_obj.cloudinary_public_id,
                        'resource_type': file_obj.cloudinary_resource_type or 'auto',
                    })

            # Delete physical files
            FileManager.delete_operation_files(
                operation_id=str(operation_id),
                session_key=session_key,
                cloudinary_files=cloudinary_files if cloudinary_files else None,
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
                    "started": queue.started_job_registry.count,
                    "failed": queue.failed_job_registry.count,
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
        operation = OperationsManager.get_operation(operation_id, session_key)

        cancellable_statuses = [
            OperationStatus.PENDING,
            OperationStatus.QUEUED,
            OperationStatus.PROCESSING,
        ]

        if operation.status not in cancellable_statuses:
            raise InvalidOperationStateError(
                operation_id=str(operation_id),
                current_status=operation.status,
                expected_statuses=cancellable_statuses,
            )
        
        # Try to cancel the RQ job if it's in the queue or processing
        if operation.status in [OperationStatus.QUEUED, OperationStatus.PROCESSING]:
            try:
                import django_rq
                
                # Get the appropriate queue
                queue_name = OperationsManager.get_queue_for_operation(operation.operation)
                queue = django_rq.get_queue(queue_name)
                
                # Try to find and cancel the RQ job
                for rq_job in queue.jobs:
                    if str(operation_id) in str(rq_job.args):
                        try:
                            rq_job.cancel()
                            logger.info(f"Cancelled RQ job for operation {operation_id}")
                        except Exception as e:
                            logger.warning(f"Could not cancel RQ job for operation {operation_id}: {e}")
                        break
            except Exception as e:
                logger.warning(f"Error attempting to cancel RQ job for operation {operation_id}: {e}")

        
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
            "cloudinary_files_deleted": 0,
            "local_files_deleted": 0,
            "errors": [],
        }

        # Find expired operations that haven't been deleted
        expired_operations = Operation.objects.filter(
            is_deleted=False,
            expires_at__lte=timezone.now()
        ).prefetch_related('files')

        for operation in expired_operations:
            stats["operations_processed"] += 1
            try:
                if not dry_run:
                    # Collect Cloudinary files
                    cloudinary_files = []
                    for file_obj in operation.files.all():
                        if file_obj.cloudinary_public_id:
                            cloudinary_files.append({
                                'public_id': file_obj.cloudinary_public_id,
                                'resource_type': file_obj.cloudinary_resource_type or 'auto',
                            })
                    
                    # Track storage type
                    if cloudinary_files:
                        stats["cloudinary_files_deleted"] += len(cloudinary_files)
                    else:
                        stats["local_files_deleted"] += operation.files.count()

                    # Delete files
                    FileManager.delete_operation_files(
                        operation_id=str(operation.id),
                        session_key=operation.session_key or "anonymous",
                        cloudinary_files=cloudinary_files if cloudinary_files else None,
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
            f"({stats['operations_processed']} processed, "
            f"{stats['cloudinary_files_deleted']} cloudinary files, "
            f"{stats['local_files_deleted']} local files)"
        )

        return stats