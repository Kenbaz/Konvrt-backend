# apps/processors/base_processor.py

"""
Base Processor class for media processing operations.

This module provides the foundation for all media processors:
- Common setup and cleanup functionality
- Temp directory management
- Error classification and handling
- Progress tracking integration
- Logging utilities
"""

import logging
import os
import shutil
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from django.conf import settings

from .exceptions import ProcessingError

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of processing errors for classification."""
    
    # Retryable errors (temporary issues)
    TEMPORARY = "temporary"  # Network, disk space, etc.
    RESOURCE = "resource"  # Memory, CPU overload
    TIMEOUT = "timeout"  # Operation timed out
    
    # Non-retryable errors (permanent issues)
    INVALID_INPUT = "invalid_input"  # Corrupted or unsupported file
    INVALID_PARAMS = "invalid_params"  # Bad parameters
    CODEC_ERROR = "codec_error"  # Unsupported codec
    PERMISSION = "permission"  # File permission issues
    NOT_FOUND = "not_found"  # File not found
    
    # Unknown errors
    UNKNOWN = "unknown"  # Unclassified error


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    output_path: Optional[str]
    output_filename: Optional[str]
    error_message: Optional[str]
    error_category: Optional[ErrorCategory]
    is_retryable: bool
    processing_time_seconds: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "output_filename": self.output_filename,
            "error_message": self.error_message,
            "error_category": self.error_category.value if self.error_category else None,
            "is_retryable": self.is_retryable,
            "processing_time_seconds": self.processing_time_seconds,
            "metadata": self.metadata,
        }


class BaseProcessor(ABC):
    """
    Abstract base class for all media processors.
    
    Provides common functionality for:
    - Temp directory management
    - Error handling and classification
    - Progress tracking
    - Logging
    
    Subclasses must implement the `process` method.
    """
    # Class-level configuration (can be overridden in subclasses)
    TEMP_DIR_PREFIX = "processor_"
    DEFAULT_TIMEOUT = 300 # 5 minutes

    def __init__(
        self,
        operation_id: UUID,
        session_key: str,
        input_path: str,
        parameters: Dict[str, Any],
        progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
    ):
        """
        Initialize the base processor.
        
        Args:
            operation_id: UUID of the operation being processed
            session_key: User's session key
            input_path: Path to the input file
            parameters: Processing parameters
            progress_callback: Optional callback for progress updates
        """
        self.operation_id = operation_id
        self.session_key = session_key
        self.input_path = input_path
        self.parameters = parameters
        self.progress_callback = progress_callback
        
        self.temp_dir: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Logging context
        self.log_prefix = f"[Op:{str(operation_id)[:8]}]"
    
    # ABSTRACT METHOD

    @abstractmethod
    def process_operation(self) -> ProcessingResult:
        """
        Perform the actual processing operation.
        
        This method must be implemented by subclasses to perform
        the specific processing logic.
        
        Returns:
            ProcessingResult with operation outcome
        """
        pass

    
    @property
    @abstractmethod
    def operation_name(self) -> str:
        """
        Get the name of this operation.
        
        Returns:
            Operation name string
        """
        pass

    # MAIN EXECUTION METHODS

    def execute_operation(self) -> ProcessingResult:
        """
        Execute the processing operation with full lifecycle management.
        
        This method:
        1. Sets up the temp directory
        2. Validates input
        3. Calls the process_operation() method
        4. Handles errors
        5. Cleans up temp files
        
        Returns:
            ProcessingResult with operation outcome
        """
        self.start_time = datetime.now()

        try:
            # Setup
            self.log_info("Starting processing")
            self._report_progress(0)

            # Validate input file exists
            if not os.path.exists(self.input_path):
                return self._create_error_result(
                    f"Input file not found: {self.input_path}",
                    ErrorCategory.NOT_FOUND,
                    is_retryable=False,
                )
            
            # Create temp directory
            self.temp_dir = self.setup_temp_directory()

            # Run the actual processing
            result = self.process_operation()

            # Ensure progress is at 100% on success
            if result.success:
                self._report_progress(100)
            
            return result
        
        except ProcessingError as e:
            self.log_error(f"Processing error: {e.message}")
            return self._create_error_result(
                e.message,
                self._classify_operation_error(e),
                is_retryable=e.is_retryable,
            )
        
        except FileNotFoundError as e:
            self.log_error(f"File not found: {e}")
            return self._create_error_result(
                str(e),
                ErrorCategory.NOT_FOUND,
                is_retryable=False,
            )
        
        except PermissionError as e:
            self.log_error(f"Permission error: {e}")
            return self._create_error_result(
                f"Permission denied: {e}",
                ErrorCategory.PERMISSION,
                is_retryable=False,
            )
            
        except MemoryError as e:
            self.log_error(f"Memory error: {e}")
            return self._create_error_result(
                "Insufficient memory for processing",
                ErrorCategory.RESOURCE,
                is_retryable=True,
            )
            
        except TimeoutError as e:
            self.log_error(f"Timeout: {e}")
            return self._create_error_result(
                f"Operation timed out: {e}",
                ErrorCategory.TIMEOUT,
                is_retryable=True,
            )
            
        except Exception as e:
            self.log_error(f"Unexpected error: {e}\n{traceback.format_exc()}")
            error_category, is_retryable = self.classify_error(e)
            return self._create_error_result(
                f"Processing failed: {str(e)}",
                error_category,
                is_retryable=is_retryable,
            )
        
        finally:
            self.end_time = datetime.now()

            # Cleanup temp directory
            if self.temp_dir:
                self.cleanup_temp_directory()
            
            # Log completion
            processing_time = self._get_processing_time()
            self.log_info(f"Processing completed in {processing_time:.2f}s")
    
    # TEMP DIRECTORY MANAGEMENT

    def setup_temp_directory(self) -> str:
        """
        Create a temporary directory for an operation.
        
        Returns:
            Path to the temp directory
        """
        base_temp_dir = os.path.join(
            getattr(settings, 'MEDIA_ROOT', '/tmp'), 'temp'
        )

        temp_dir = os.path.join(
            base_temp_dir,
            str(self.operation_id)
        )

        os.makedirs(temp_dir, exist_ok=True)
        self.log_debug(f"Created temp directory: {temp_dir}")

        return temp_dir
    

    def cleanup_temp_directory(self) -> None:
        """
        Remove the temporary directory and all its contents.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        if not self.temp_dir:
            return True
        
        if not os.path.exists(self.temp_dir):
            return True
        
        try:
            shutil.rmtree(self.temp_dir)
            self.log_debug(f"Cleaned up temp directory: {self.temp_dir}")
            return True
        except Exception as e:
            self.log_warning(f"Failed to clean up temp directory: {e}")
            return False
    

    def get_temp_file_path(self, filename: str) -> str:
        """
        Generate a path for a temporary file.
        
        Args:
            filename: Name of the temp file
            
        Returns:
            Full path to the temp file
        """
        if not self.temp_dir:
            self.temp_dir = self.setup_temp_directory()
        
        return os.path.join(self.temp_dir, filename)
    

    def ensure_temp_directory(self) -> None:
        """
        Ensure temp directory exists and return its path.
        
        Returns:
            Path to the temp directory
        """
        if not self.temp_dir or not os.path.exists(self.temp_dir):
            self.temp_dir = self.setup_temp_directory()
        return self.temp_dir
    
    # ERROR HANDLING AND CLASSIFICATION

    def classify_error(self, error: Exception) -> Tuple[ErrorCategory, bool]:
        """
        Classify an error and determine if it's retryable.
        
        Args:
            error: The exception to classify
            
        Returns:
            Tuple of (ErrorCategory, is_retryable)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # File-related errors
        if isinstance(error, FileNotFoundError):
            return ErrorCategory.NOT_FOUND, False
        
        if isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION, False
        
        if isinstance(error, MemoryError):
            return ErrorCategory.RESOURCE, True
        
        if isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT, True
        
        # Check error message for patterns
        if any(term in error_str for term in ['no space', 'disk full', 'quota']):
            return ErrorCategory.RESOURCE, True
        
        if any(term in error_str for term in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT, True
        
        if any(term in error_str for term in ['connection', 'network', 'redis']):
            return ErrorCategory.TEMPORARY, True
        
        if any(term in error_str for term in ['corrupt', 'invalid', 'malformed']):
            return ErrorCategory.INVALID_INPUT, False
        
        if any(term in error_str for term in ['codec', 'encoder', 'decoder', 'unsupported']):
            return ErrorCategory.CODEC_ERROR, False
        
        if any(term in error_str for term in ['permission', 'denied', 'access']):
            return ErrorCategory.PERMISSION, False
        
        # Default to unknown, not retryable
        return ErrorCategory.UNKNOWN, False
    

    def _classify_processing_error(self, error: ProcessingError) -> ErrorCategory:
        """
        Classify a ProcessingError.
        
        Args:
            error: ProcessingError to classify
            
        Returns:
            ErrorCategory
        """
        reason = error.reason.lower()
        
        if any(term in reason for term in ['timeout', 'timed out']):
            return ErrorCategory.TIMEOUT
        
        if any(term in reason for term in ['codec', 'encoder', 'decoder']):
            return ErrorCategory.CODEC_ERROR
        
        if any(term in reason for term in ['corrupt', 'invalid']):
            return ErrorCategory.INVALID_INPUT
        
        if any(term in reason for term in ['memory', 'resource']):
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN
    

    def _create_error_result(
        self,
        message: str,
        category: ErrorCategory,
        is_retryable: bool,
    ) -> ProcessingResult:
        """
        Create an error ProcessingResult.
        
        Args:
            message: Error message
            category: Error category
            is_retryable: Whether the error is retryable
            
        Returns:
            ProcessingResult with error details
        """
        return ProcessingResult(
            success=False,
            output_path=None,
            output_filename=None,
            error_message=message,
            error_category=category,
            is_retryable=is_retryable,
            processing_time_seconds=self._get_processing_time(),
            metadata={},
        )
    

    def create_success_result(
        self,
        output_path: str,
        output_filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessingResult:
        """
        Create a successful ProcessingResult.
        
        Args:
            output_path: Path to the output file
            output_filename: Name of the output file
            metadata: Optional additional metadata
            
        Returns:
            ProcessingResult with success details
        """
        return ProcessingResult(
            success=True,
            output_path=output_path,
            output_filename=output_filename,
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=self._get_processing_time(),
            metadata=metadata or {},
        )
    
    # PROGRESS TRACKING

    def _report_progress(
        self,
        percent: int,
        eta_seconds: Optional[float] = None,
    ) -> None:
        """
        Report progress to the callback if available.
        
        Args:
            percent: Progress percentage (0-100)
            eta_seconds: Estimated time remaining
        """
        if self.progress_callback:
            try:
                self.progress_callback(percent, eta_seconds)
            except Exception as e:
                self.log_warning(f"Progress callback failed: {e}")
    

    def update_progress(
        self,
        percent: int,
        eta_seconds: Optional[float] = None,
    ) -> None:
        """
        Update progress (public method for subclasses).
        
        Args:
            percent: Progress percentage (0-100)
            eta_seconds: Estimated time remaining
        """
        self._report_progress(percent, eta_seconds)
    
    # LOGGING UTILITIES

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        logger.debug(f"{self.log_prefix} {message}")
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        logger.info(f"{self.log_prefix} {message}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        logger.warning(f"{self.log_prefix} {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        logger.error(f"{self.log_prefix} {message}")
    
    # UTILITY METHODS

    def _get_processing_time(self) -> float:
        """
        Get the processing time in seconds.
        
        Returns:
            Processing time in seconds
        """
        if not self.start_time:
            return 0.0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    

    def get_input_filename(self) -> str:
        """
        Get the filename of the input file.
        
        Returns:
            Input filename without path
        """
        return os.path.basename(self.input_path)
    

    def get_input_extension(self) -> str:
        """
        Get the extension of the input file.
        
        Returns:
            Extension without the dot
        """
        _, ext = os.path.splitext(self.input_path)
        return ext.lstrip('.').lower()
    

    def generate_output_filename(
        self,
        suffix: str = "",
        extension: Optional[str] = None,
    ) -> str:
        """
        Generate an output filename based on input filename.
        
        Args:
            suffix: Suffix to add before extension (e.g., "_compressed")
            extension: Output extension (uses input extension if not provided)
            
        Returns:
            Generated output filename
        """
        base_name = os.path.splitext(self.get_input_filename())[0]
        ext = extension or self.get_input_extension()

        # sanitize base name
        base_name = base_name[:100] # limit length

        return f"{base_name}{suffix}.{ext}"
    

    def validate_output_file_creation(self, output_path: str) -> bool:
        """
        Validate that an output file was created successfully.
        
        Args:
            output_path: Path to the output file
            
        Returns:
            True if file exists and has content
        """
        if not os.path.exists(output_path):
            self.log_error(f"Output file not created: {output_path}")
            return False
        
        if os.path.getsize(output_path) == 0:
            self.log_error(f"Output file is empty: {output_path}")
            return False
        
        return True


class VideoProcessor(BaseProcessor):
    """
    Base class for video processing operations.
    
    Provides video-specific utilities on top of BaseProcessor.
    """
    DEFAULT_TIMEOUT = 1800  # 30 minutes

    def get_video_info(self):
        """
        Get information about the input video.
        
        Returns:
            VideoInfo object
        """
        from .utils.ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        return ffmpeg.get_video_info(self.input_path)
    

    def get_duration(self) -> float:
        """
        Get the duration of the input video.
        
        Returns:
            Duration in seconds
        """
        return self.get_video_info().duration


class ImageProcessor(BaseProcessor):
    """
    Base class for image processing operations.
    
    Provides image-specific utilities on top of BaseProcessor.
    """
    DEFAULT_TIMEOUT = 60 # 1 minute

    def open_image(self):
        """
        Open the input image with Pillow.
        
        Returns:
            PIL.Image object
        """
        from PIL import Image
        with Image.open(self.input_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
            }


class AudioProcessor(BaseProcessor):
    """
    Base class for audio processing operations.
    
    Provides audio-specific utilities on top of BaseProcessor.
    """
    DEFAULT_TIMEOUT = 300 # 5 minutes

    def get_audio_info(self):
        """
        Get information about the input audio.
        
        Returns:
            AudioInfo object
        """
        from .utils.ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        return ffmpeg.get_audio_info(self.input_path)
    

    def get_duration(self) -> float:
        """
        Get the duration of the input audio.
        
        Returns:
            Duration in seconds
        """
        return self.get_audio_info().duration