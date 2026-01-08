# apps/processors/rq_workers.py

"""
RQ Worker functions for processing media operations.

This module provides the entry point for Redis Queue (RQ) workers
to process media operations. It handles:
- Fetching operation details from the database
- Downloading input files from Cloudinary (when enabled)
- Executing the appropriate processor
- Uploading output files to Cloudinary (when enabled)
- Updating operation status
- Error handling and classification
- Cleanup of temporary files
- Real-time progress tracking with throttling
"""
import logging
import os
import shutil
import traceback
from datetime import timedelta
import subprocess
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from django.conf import settings
from django.utils import timezone


logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAYS = [60, 300]  # Seconds: 1 minute, 5 minutes (exponential backoff)

# Progress update configuration
PROGRESS_UPDATE_INTERVAL = 5.0  # Minimum seconds between progress updates
MIN_PERCENT_CHANGE = 5  # Minimum percent change to trigger update


class WorkerError(Exception):
    """Custom exception for worker errors."""
    
    def __init__(
        self,
        message: str,
        is_retryable: bool = False,
        error_code: str = "WORKER_ERROR",
    ):
        self.message = message
        self.is_retryable = is_retryable
        self.error_code = error_code
        super().__init__(message)


class InputFileNotFoundError(WorkerError):
    """Raised when the input file for an operation is not found."""
    
    def __init__(self, operation_id: str, file_path: str):
        super().__init__(
            message=f"Input file not found for operation {operation_id}: {file_path}",
            is_retryable=False,
            error_code="INPUT_FILE_NOT_FOUND",
        )


class CloudinaryDownloadError(WorkerError):
    """Raised when downloading from Cloudinary fails."""
    
    def __init__(self, operation_id: str, public_id: str, reason: str):
        super().__init__(
            message=f"Failed to download input file from Cloudinary for operation {operation_id}: {reason}",
            is_retryable=True,  # Cloudinary errors are usually temporary
            error_code="CLOUDINARY_DOWNLOAD_ERROR",
        )


class CloudinaryUploadError(WorkerError):
    """Raised when uploading to Cloudinary fails."""
    
    def __init__(self, operation_id: str, filename: str, reason: str):
        super().__init__(
            message=f"Failed to upload output file to Cloudinary for operation {operation_id}: {reason}",
            is_retryable=True,  # Cloudinary errors are usually temporary
            error_code="CLOUDINARY_UPLOAD_ERROR",
        )


class ProcessingFailedError(WorkerError):
    """Raised when processing fails."""
    
    def __init__(self, message: str, is_retryable: bool = False):
        super().__init__(
            message=message,
            is_retryable=is_retryable,
            error_code="PROCESSING_FAILED",
        )


def _use_cloudinary() -> bool:
    """Check if Cloudinary storage is enabled."""
    return getattr(settings, 'USE_CLOUDINARY', False)


def check_ffmpeg_on_startup():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        logger.info(f"FFmpeg version info:\n{result.stdout[:500]}")
        
        result2 = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10)
        has_libx264 = 'libx264' in result2.stdout
        logger.info(f"libx264 encoder available: {has_libx264}")
    except Exception as e:
        logger.error(f"FFmpeg check failed: {e}")


def create_worker_progress_callback(operation_id: str):
    """
    Create a progress callback function for worker use.
    
    This creates a callback that updates the operation's progress in the database
    with throttling to prevent excessive database writes.
    
    Args:
        operation_id: UUID string of the operation
        
    Returns:
        Callable that accepts (percent: int, eta_seconds: Optional[float])
    """
    from .utils.track_progress import ThrottledProgressCallback
    
    def update_progress_in_db(percent: int, eta_seconds: Optional[float] = None) -> None:
        """Update operation progress directly in database."""
        from apps.operations.models import Operation
        
        try:
            # Cap progress at 99% until completion (100% is set by complete_operation)
            clamped_percent = max(0, min(99, percent))
            
            # Use update() for efficiency - avoids loading the full model
            Operation.objects.filter(id=operation_id).update(progress=clamped_percent)
            
            logger.debug(
                f"Updated operation {operation_id} progress to {clamped_percent}%"
                + (f" (ETA: {eta_seconds:.1f}s)" if eta_seconds else "")
            )
        except Exception as e:
            # Log but don't fail - progress updates are non-critical
            logger.warning(f"Failed to update progress for operation {operation_id}: {e}")
    
    # Create throttled callback
    throttled_callback = ThrottledProgressCallback(
        callback=update_progress_in_db,
        min_interval=PROGRESS_UPDATE_INTERVAL,
        min_percent_change=MIN_PERCENT_CHANGE,
    )
    
    # Return a simple callable that uses the throttled callback
    def progress_callback(percent: int, eta_seconds: Optional[float] = None) -> None:
        throttled_callback(percent, eta_seconds)
    
    return progress_callback


def _get_input_file_path(operation_id: str, input_file, temp_dir: str) -> str:
    """
    Get the local path to the input file, downloading from Cloudinary if needed.
    
    Args:
        operation_id: UUID string of the operation
        input_file: File model instance
        temp_dir: Temporary directory for downloads
        
    Returns:
        Local path to the input file
        
    Raises:
        InputFileNotFoundError: If file cannot be found
        CloudinaryDownloadError: If Cloudinary download fails
    """
    from apps.operations.services.file_manager import FileManager
    
    # Check if file is stored in Cloudinary
    if input_file.cloudinary_public_id:
        logger.info(
            f"Downloading input file from Cloudinary for operation {operation_id}: "
            f"{input_file.cloudinary_public_id}"
        )
        
        try:
            # Download from Cloudinary to temp directory
            local_path = FileManager.download_from_cloudinary(
                public_id=input_file.cloudinary_public_id,
                resource_type=input_file.cloudinary_resource_type or 'auto',
                destination_dir=temp_dir,
            )
            
            logger.info(
                f"Downloaded input file from Cloudinary: {input_file.cloudinary_public_id} "
                f"-> {local_path}"
            )
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download from Cloudinary: {e}")
            raise CloudinaryDownloadError(
                operation_id=operation_id,
                public_id=input_file.cloudinary_public_id,
                reason=str(e),
            )
    
    # Local storage: construct path from MEDIA_ROOT
    if input_file.file_path:
        input_path = os.path.join(settings.MEDIA_ROOT, input_file.file_path)
        
        if os.path.exists(input_path):
            return input_path
        
        raise InputFileNotFoundError(operation_id, input_path)
    
    # No valid file path
    raise InputFileNotFoundError(operation_id, "No file path or Cloudinary ID found")


def _get_media_type_from_file(input_file) -> str:
    """
    Determine the media type from the input file.
    
    Args:
        input_file: File model instance
        
    Returns:
        Media type string: 'video', 'image', or 'audio'
    """
    mime_type = input_file.mime_type or ''
    
    if mime_type.startswith('video/'):
        return 'video'
    elif mime_type.startswith('image/'):
        return 'image'
    elif mime_type.startswith('audio/'):
        return 'audio'
    
    # Try to get from metadata
    metadata = input_file.metadata or {}
    if 'media_type' in metadata:
        return metadata['media_type']
    
    # Default to video for unknown types
    return 'video'


def process_operation(operation_id: str) -> Dict[str, Any]:
    """
    Main entry point for processing an operation.
    
    This function is called by RQ workers when a job is dequeued.
    It handles the complete processing lifecycle:
    1. Fetch operation from database
    2. Update status to PROCESSING
    3. Download input file from Cloudinary (if using cloud storage)
    4. Get the appropriate processor from registry
    5. Execute the processor with progress tracking
    6. Upload output to Cloudinary (if using cloud storage)
    7. Update operation status to COMPLETED or FAILED
    8. Cleanup temporary files
    
    Args:
        operation_id: UUID string of the operation to process
        
    Returns:
        Dictionary with processing result:
        - success: bool
        - operation_id: str
        - output_path: str (if successful)
        - error_message: str (if failed)
        
    Raises:
        WorkerError: If processing fails irrecoverably
    """
    logger.info(f"Starting processing for operation {operation_id}")

    from apps.operations.models import Operation, File
    from apps.operations.enums import OperationStatus, FileType
    from apps.operations.services.file_manager import FileManager
    from apps.processors.registry import get_registry
    from apps.processors.exceptions import OperationNotFoundError

    operation = None
    temp_output_path = None
    temp_dir = None
    downloaded_input_path = None

    try:
        # Fetch operation from database
        try:
            operation = Operation.objects.get(
                id=operation_id,
                is_deleted=False,
            )
        except Operation.DoesNotExist:
            raise OperationNotFoundError(operation_id)
    
        # Update operation status to PROCESSING
        operation.status = OperationStatus.PROCESSING
        operation.started_at = timezone.now()
        operation.progress = 0
        operation.save(update_fields=["status", "started_at", "progress"])

        logger.info(
            f"Operation {operation_id} started processing "
            f"(operation={operation.operation})"
        )

        # Get input file record
        try:
            input_file = File.objects.get(
                operation=operation,
                file_type=FileType.INPUT,
            )
        except File.DoesNotExist:
            raise InputFileNotFoundError(operation_id, "No input file found")
        
        # Ensure temp directory exists for this operation
        temp_dir = FileManager.ensure_temp_directory(operation_id)
        
        # Get input file path (downloads from Cloudinary if needed)
        input_path = _get_input_file_path(
            operation_id=operation_id,
            input_file=input_file,
            temp_dir=temp_dir,
        )
        
        # Track if we downloaded the input (for cleanup)
        if input_file.cloudinary_public_id:
            downloaded_input_path = input_path
        
        # Verify file exists
        if not os.path.exists(input_path):
            raise InputFileNotFoundError(operation_id, input_path)
        
        # Get processor from registry
        registry = get_registry()

        if not registry.is_registered(operation.operation):
            raise ProcessingFailedError(
                f"Operation '{operation.operation}' is not registered",
                is_retryable=False
            )
        
        operation_def = registry.get_operation(operation.operation)
        handler = operation_def.handler

        # Create progress callback with throttling
        progress_callback = create_worker_progress_callback(operation_id)
        
        # Execute the processor
        logger.info(f"Executing handler for operation {operation.operation}")

        result = handler(
            operation_id=UUID(operation_id),
            session_key=operation.session_key or "anonymous",
            input_path=input_path,
            parameters=operation.parameters or {},
            progress_callback=progress_callback,
        )

        # Handle result
        if result.success:
            temp_output_path = result.output_path

            # Move/upload output to permanent storage
            if temp_output_path and os.path.exists(temp_output_path):
                # Determine media type for Cloudinary resource type
                media_type = _get_media_type_from_file(input_file)
                
                try:
                    output_info = FileManager.move_to_output(
                        temp_path=temp_output_path,
                        operation_id=operation_id,
                        session_key=operation.session_key or "anonymous",
                        output_filename=result.output_filename,
                        media_type=media_type,
                    )
                except Exception as e:
                    if _use_cloudinary():
                        raise CloudinaryUploadError(
                            operation_id=operation_id,
                            filename=result.output_filename,
                            reason=str(e),
                        )
                    raise

                # Create output file record with Cloudinary fields
                File.objects.create(
                    operation=operation,
                    file_type=FileType.OUTPUT,
                    file_path=output_info.get('file_path', ''),
                    file_name=output_info['file_name'],
                    file_size=output_info['file_size'],
                    mime_type=output_info['mime_type'],
                    cloudinary_public_id=output_info.get('cloudinary_public_id', ''),
                    cloudinary_url=output_info.get('cloudinary_url', ''),
                    cloudinary_resource_type=output_info.get('cloudinary_resource_type', ''),
                    metadata=result.metadata,
                )

                # Mark as complete
                complete_operation(
                    operation=operation,
                    output_path=output_info.get('file_path') or output_info.get('cloudinary_public_id', ''),
                    metadata=result.metadata,
                )

                logger.info(
                    f"Operation {operation_id} completed successfully "
                    f"(output={output_info['file_name']}, "
                    f"size={output_info['file_size']} bytes, "
                    f"storage={'cloudinary' if output_info.get('cloudinary_public_id') else 'local'})"
                )

                # Clean up temp directory after successful processing
                _cleanup_temp_files(temp_dir, downloaded_input_path, temp_output_path)

                return {
                    "success": True,
                    "operation_id": operation_id,
                    "output_path": output_info.get('file_path') or output_info.get('cloudinary_url', ''),
                    "processing_time": result.processing_time_seconds,
                    "storage": "cloudinary" if output_info.get('cloudinary_public_id') else "local",
                }
            else:
                raise ProcessingFailedError(
                    "Processing succeeded but no output file was created",
                    is_retryable=True
                )
        else:
            # Processing failed
            raise ProcessingFailedError(
                result.error_message or "Processing failed with unknown error",
                is_retryable=result.is_retryable
            )
    
    except WorkerError as e:
        logger.error(f"Worker error for operation {operation_id}: {e.message}")
        if operation:
            fail_operation(
                operation=operation,
                error_message=e.message,
                is_retryable=e.is_retryable,
                error_code=e.error_code,
            )
        # Clean up on failure
        _cleanup_temp_files(temp_dir, downloaded_input_path, temp_output_path)
        raise
    
    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        logger.error(
            f"Unexpected error processing operation {operation_id}: "
            f"{error_message}\n{error_traceback}"
        )

        # Classify the error
        error_info = classify_error(e)
        
        if operation:
            fail_operation(
                operation=operation,
                error_message=error_info['user_message'],
                is_retryable=error_info['is_retryable'],
                error_code=error_info['error_code'],
            )
        
        # Clean up on failure
        _cleanup_temp_files(temp_dir, downloaded_input_path, temp_output_path)
        
        raise ProcessingFailedError(
            error_info['user_message'],
            is_retryable=error_info['is_retryable']
        )


def _cleanup_temp_files(
    temp_dir: Optional[str],
    downloaded_input_path: Optional[str],
    temp_output_path: Optional[str],
) -> None:
    """
    Clean up temporary files after processing.
    """
    # Clean up downloaded input file
    if downloaded_input_path and os.path.exists(downloaded_input_path):
        try:
            os.remove(downloaded_input_path)
            logger.debug(f"Cleaned up downloaded input: {downloaded_input_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up downloaded input: {e}")
    
    # Clean up temp output file (if not already moved)
    if temp_output_path and os.path.exists(temp_output_path):
        try:
            os.remove(temp_output_path)
            logger.debug(f"Cleaned up temp output: {temp_output_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp output: {e}")
    
    # Clean up temp directory
    if temp_dir and os.path.exists(temp_dir) and 'temp' in temp_dir:
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


def complete_operation(
    operation,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Mark an operation as successfully completed.
    
    Args:
        operation: Operation model instance
        output_path: Path to the output file (local path or Cloudinary public_id)
        metadata: Optional processing metadata
    """
    from apps.operations.enums import OperationStatus

    # Calculate expiration (7 days from completion)
    expiration_days = getattr(settings, 'OPERATION_EXPIRATION_DAYS', 7)

    operation.status = OperationStatus.COMPLETED
    operation.progress = 100
    operation.completed_at = timezone.now()
    operation.expires_at = operation.completed_at + timedelta(days=expiration_days)
    operation.error_message = None

    # Store processing metadata if provided
    if metadata:
        existing_params = operation.parameters or {}
        existing_params['_processing_metadata'] = metadata
        operation.parameters = existing_params
    
    operation.save(update_fields=[
        "status",
        "progress",
        "completed_at",
        "expires_at",
        "error_message",
        "parameters",
    ])

    logger.info(f"Operation {operation.id} marked as COMPLETED")


def fail_operation(
    operation,
    error_message: str,
    is_retryable: bool = False,
    error_code: str = "PROCESSING_ERROR",
) -> None:
    """
    Mark an operation as failed.
    
    Args:
        operation: Operation model instance
        error_message: User-friendly error message
        is_retryable: Whether the operation can be retried
        error_code: Error code for classification
    """
    from apps.operations.enums import OperationStatus

    # Calculate expiration (7 days from failure)
    expiration_days = getattr(settings, 'OPERATION_EXPIRATION_DAYS', 7)
    
    operation.status = OperationStatus.FAILED
    operation.completed_at = timezone.now()
    operation.expires_at = operation.completed_at + timedelta(days=expiration_days)
    operation.error_message = error_message
    
    # Store retry info in parameters
    params = operation.parameters or {}
    params['_error_info'] = {
        'error_code': error_code,
        'is_retryable': is_retryable,
        'failed_at': operation.completed_at.isoformat(),
    }
    operation.parameters = params
    
    operation.save(update_fields=[
        'status', 'completed_at', 'expires_at', 'error_message', 'parameters'
    ])
    
    logger.info(
        f"Operation {operation.id} marked as FAILED "
        f"(retryable={is_retryable}, code={error_code})"
    )


def classify_error(error: Exception) -> Dict[str, Any]:
    """
    Classify an error and determine if it's retryable.
    
    Categorizes errors as:
    - Retryable: Redis errors, disk full, timeouts, temporary network issues, Cloudinary errors
    - Non-retryable: Corrupted files, unsupported codecs, invalid parameters
    
    Args:
        error: The exception to classify
        
    Returns:
        Dictionary with:
        - is_retryable: bool
        - error_code: str
        - user_message: str (user-friendly message)
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check for specific error types first
    retryable_errors = {
        'TimeoutError': ('TIMEOUT', 'Operation timed out. Please try again.'),
        'ConnectionError': ('CONNECTION_ERROR', 'Connection error. Please try again.'),
        'MemoryError': ('MEMORY_ERROR', 'Not enough memory. Please try with a smaller file.'),
        'CloudinaryDownloadError': ('CLOUDINARY_DOWNLOAD_ERROR', 'Failed to download file. Please try again.'),
        'CloudinaryUploadError': ('CLOUDINARY_UPLOAD_ERROR', 'Failed to upload file. Please try again.'),
        'OSError': None,  # Needs further inspection
        'IOError': None,  # Needs further inspection
    }
    
    if error_type in retryable_errors:
        info = retryable_errors[error_type]
        if info:
            return {
                'is_retryable': True,
                'error_code': info[0],
                'user_message': info[1],
            }
    
    # Check error message patterns
    retryable_patterns = [
        ('no space left', 'DISK_FULL', 'Not enough disk space. Please try again later.'),
        ('disk quota', 'DISK_QUOTA', 'Disk quota exceeded. Please try again later.'),
        ('connection refused', 'CONNECTION_REFUSED', 'Service temporarily unavailable.'),
        ('redis', 'REDIS_ERROR', 'Queue service error. Please try again.'),
        ('timeout', 'TIMEOUT', 'Operation timed out. Please try again.'),
        ('temporary', 'TEMPORARY_ERROR', 'Temporary error. Please try again.'),
        ('cloudinary', 'CLOUDINARY_ERROR', 'Cloud storage error. Please try again.'),
        ('network', 'NETWORK_ERROR', 'Network error. Please try again.'),
    ]
    
    for pattern, code, message in retryable_patterns:
        if pattern in error_str:
            return {
                'is_retryable': True,
                'error_code': code,
                'user_message': message,
            }
    
    # Non-retryable patterns
    non_retryable_patterns = [
        ('corrupt', 'CORRUPT_FILE', 'The file appears to be corrupted.'),
        ('invalid data', 'INVALID_DATA', 'The file contains invalid data.'),
        ('unsupported', 'UNSUPPORTED_FORMAT', 'The file format is not supported.'),
        ('codec not found', 'CODEC_NOT_FOUND', 'Required codec is not available.'),
        ('permission denied', 'PERMISSION_DENIED', 'Permission denied accessing files.'),
        ('not found', 'FILE_NOT_FOUND', 'Required file was not found.'),
        ('invalid parameter', 'INVALID_PARAMS', 'Invalid processing parameters.'),
    ]
    
    for pattern, code, message in non_retryable_patterns:
        if pattern in error_str:
            return {
                'is_retryable': False,
                'error_code': code,
                'user_message': message,
            }
    
    # Default: unknown error, not retryable
    return {
        'is_retryable': False,
        'error_code': 'UNKNOWN_ERROR',
        'user_message': f'Processing failed: {str(error)[:200]}',
    }


def handle_job_error(
    operation_id: str,
    error: Exception,
    retry_count: int = 0,
) -> Tuple[bool, Optional[int]]:
    """
    Handle a job error and determine if retry is needed.
    
    Args:
        operation_id: UUID of the operation
        error: The exception that occurred
        retry_count: Current retry count
        
    Returns:
        Tuple of (should_retry, retry_delay_seconds)
    """
    error_info = classify_error(error)

    # Log the error
    logger.error(
        f"Job error for operation {operation_id}: {error}\n"
        f"Error code: {error_info['error_code']}, "
        f"Retryable: {error_info['is_retryable']}, "
        f"Retry count: {retry_count}"
    )
    
    # Check if we should retry
    if error_info['is_retryable'] and retry_count < MAX_RETRIES:
        retry_delay = RETRY_DELAYS[min(retry_count, len(RETRY_DELAYS) - 1)]
        logger.info(
            f"Scheduling retry for operation {operation_id} "
            f"in {retry_delay} seconds (attempt {retry_count + 1}/{MAX_RETRIES})"
        )
        return True, retry_delay
    
    return False, None


def get_worker_stats() -> Dict[str, Any]:
    """
    Get statistics about worker performance.
    
    Returns:
        Dictionary with worker statistics
    """
    from apps.operations.models import Operation
    from apps.operations.enums import OperationStatus

    now = timezone.now()
    last_hour = now - timedelta(hours=1)
    last_day = now - timedelta(days=1)

    stats = {
        'operations_processing': Operation.objects.filter(
            status=OperationStatus.PROCESSING
        ).count(),
        'operations_queued': Operation.objects.filter(
            status=OperationStatus.QUEUED
        ).count(),
        'completed_last_hour': Operation.objects.filter(
            status=OperationStatus.COMPLETED,
            completed_at__gte=last_hour
        ).count(),
        'failed_last_hour': Operation.objects.filter(
            status=OperationStatus.FAILED,
            completed_at__gte=last_hour
        ).count(),
        'completed_last_day': Operation.objects.filter(
            status=OperationStatus.COMPLETED,
            completed_at__gte=last_day
        ).count(),
        'failed_last_day': Operation.objects.filter(
            status=OperationStatus.FAILED,
            completed_at__gte=last_day
        ).count(),
        'storage_backend': 'cloudinary' if _use_cloudinary() else 'local',
    }

    # Calculate success rate
    total_completed = stats['completed_last_day'] + stats['failed_last_day']
    if total_completed > 0:
        stats['success_rate_last_day'] = round(
            (stats['completed_last_day'] / total_completed) * 100, 2
        )
    else:
        stats['success_rate_last_day'] = 100.0
    
    return stats


def cleanup_stale_operations(timeout_minutes: int = 60) -> int:
    """
    Clean up operations that have been processing for too long.
    
    This handles cases where a worker crashes without updating status.
    
    Args:
        timeout_minutes: Minutes after which a processing operation is stale
        
    Returns:
        Number of operations cleaned up
    """
    from apps.operations.models import Operation
    from apps.operations.enums import OperationStatus

    cutoff_time = timezone.now() - timedelta(minutes=timeout_minutes)

    stale_operations = Operation.objects.filter(
        status=OperationStatus.PROCESSING,
        started_at__lt=cutoff_time,
    )

    count = 0
    for operation in stale_operations:
        fail_operation(
            operation=operation,
            error_message="Operation timed out due to worker failure.",
            is_retryable=True,
            error_code="STALE_OPERATION",
        )
        count += 1
        logger.info(f"Marked stale operation {operation.id} as failed")

    if count > 0:
        logger.info(f"Cleaned up {count} stale operations")
    
    return count


check_ffmpeg_on_startup()