# apps/processors/video_processing.py

"""
Video processing operations.

This module implements video processing operations:
- video_compress: Compress video using H.264 codec with configurable quality
- video_convert: Convert video to different formats (mp4, webm, mov)

All operations are registered with the operation registry and can be
executed through the worker system.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from .base_processor import ProcessingResult, VideoProcessor, ErrorCategory
from .registry import (
    MediaType,
    ParameterSchema,
    ParameterType,
    get_registry,
)
from .utils.ffmpeg import get_ffmpeg_wrapper
from .utils.track_progress import (
    ThrottledProgressCallback,
)

logger = logging.getLogger(__name__)


# Video format to codec mappings
VIDEO_CODECS = {
    'mp4': 'libx264',
    'webm': 'libvpx-vp9',
    'mov': 'libx264',
    'mkv': 'libx264',
    'avi': 'libx264',
}

AUDIO_CODECS = {
    'mp4': 'aac',
    'webm': 'libopus',
    'mov': 'aac',
    'mkv': 'aac',
    'avi': 'mp3',
}

# Encoding presets for compression
ENCODING_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']


class VideoCompressProcessor(VideoProcessor):
    """
    Processor for video compression using H.264 codec.
    
    Compresses video files using the libx264 codec with configurable
    CRF (Constant Rate Factor) quality setting.
    """
    @property
    def operation_name(self) -> str:
        return "video_compress"
    
    def process_operation(self) -> ProcessingResult:
        """
        Execute video compression.
        
        Uses FFmpeg with libx264 codec and CRF quality control.
        
        Returns:
            ProcessingResult with output details
        """
        # Extract parameters with defaults
        quality = self.parameters.get('quality', 23)
        preset = self.parameters.get('preset', 'medium')
        audio_bitrate = self.parameters.get('audio_bitrate', '128k')
        
        self.log_info(f"Starting video compression with quality={quality}, preset={preset}")
        
        # Get video infor for duration (needed for progress tracking)
        try:
            video_info = self.get_video_info()
            duration = video_info.duration
            self.log_info(f"Video duration: {duration:.2f}s, resolution: {video_info.width}x{video_info.height}")
        except Exception as e:
            self.log_error(f"Failed to get video info: {e}")
            return self._create_error_result(
                f"Failed to analyze input video: {e}",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Validate duration
        if duration <= 0:
            return self._create_error_result(
                "Video has invalid duration",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Generate output filename and path
        output_filename = self.generate_output_filename(
            suffix="_compressed",
            extension="mp4"
        )
        output_path = self.get_temp_file_path(output_filename)

        # Build FFmpeg command
        ffmpeg = get_ffmpeg_wrapper()
        cmd_args = ffmpeg.build_compress_command(
            input_path=self.input_path,
            output_path=output_path,
            quality=quality,
            preset=preset,
            audio_bitrate=audio_bitrate
        )

        self.log_debug(f"FFmpeg command args: {cmd_args}")

        # Create progress callback
        def progress_handler(percent: int, eta_seconds: Optional[float]) -> None:
            self.update_progress(percent, eta_seconds)
        
        throttled_callback = ThrottledProgressCallback(
            callback=progress_handler,
            min_interval=5.0,
            min_percent_change=5
        )

        # Execute FFmpeg
        try:
            result = ffmpeg.execute(
                args=cmd_args,
                input_path=self.input_path,
                output_path=output_path,
                progress_callback=throttled_callback,
                timeout=self.DEFAULT_TIMEOUT,
                total_duration=duration
            )
        except TimeoutError as e:
            return self._create_error_result(
                f"Video compression timed out after {self.DEFAULT_TIMEOUT}s",
                ErrorCategory.TIMEOUT,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"FFmpeg execution failed: {e}")
            return self._create_error_result(
                f"Video compression failed: {e}",
                ErrorCategory.UNKNOWN,
                is_retryable=False,
            )
        
        # Check FFmpeg result
        if not result.success:
            logger.error(f"FFmpeg failed with return code: {result.return_code}")
            logger.error(f"FFmpeg stderr (raw): {result.stderr[-2000:]}")  # Last 2000 chars
            logger.error(f"FFmpeg stdout (last 500): {result.stdout[-500:]}")
            
            # Check if output file exists anyway
            import os
            if output_path:
                file_exists = os.path.exists(output_path)
                file_size = os.path.getsize(output_path) if file_exists else 0
                logger.error(f"Output file exists: {file_exists}, size: {file_size} bytes")
            error_msg = self._parse_ffmpeg_error(result.stderr)
            error_category = self._categorize_ffmpeg_error(result.stderr)
            return self._create_error_result(
                error_msg,
                error_category,
                is_retryable=error_category in (ErrorCategory.TEMPORARY, ErrorCategory.RESOURCE),
            )
        
        # Verify output file was created
        if not self.validate_output_file_creation(output_path):
            return self._create_error_result(
                "Video compression completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)
        compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        
        self.log_info(
            f"Compression complete. Input: {input_size / (1024*1024):.2f}MB, "
            f"Output: {output_size / (1024*1024):.2f}MB, "
            f"Reduction: {compression_ratio:.1f}%"
        )

        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'input_size': input_size,
                'output_size': output_size,
                'compression_ratio': round(compression_ratio, 2),
                'quality': quality,
                'preset': preset,
                'duration': duration,
                'codec': 'h264',
            }
        )
    

    def _parse_ffmpeg_error(self, stderr: str) -> str:
        """
        Parse FFmpeg stderr to extract a user-friendly error message.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            User-friendly error message
        """
        stderr_lower = stderr.lower()
        
        if 'no such file' in stderr_lower:
            return "Input file not found or inaccessible"
        elif 'invalid data' in stderr_lower or 'corrupt' in stderr_lower:
            return "Input video file appears to be corrupted"
        elif 'codec not found' in stderr_lower or 'unknown encoder' in stderr_lower:
            return "Required video codec is not available"
        elif 'permission denied' in stderr_lower:
            return "Permission denied when accessing files"
        elif 'no space left' in stderr_lower:
            return "Not enough disk space for output file"
        elif 'out of memory' in stderr_lower:
            return "Not enough memory to process video"
        else:
            # Return last meaningful line from stderr
            lines = [l.strip() for l in stderr.split('\n') if l.strip()]
            if lines:
                return f"Video compression failed: {lines[-1][:200]}"
            return "Video compression failed with unknown error"
    

    def _categorize_ffmpeg_error(self, stderr: str) -> ErrorCategory:
        """
        Categorize FFmpeg error for retry logic.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            ErrorCategory
        """
        stderr_lower = stderr.lower()
        
        if 'no space left' in stderr_lower:
            return ErrorCategory.RESOURCE
        elif 'out of memory' in stderr_lower:
            return ErrorCategory.RESOURCE
        elif 'timeout' in stderr_lower:
            return ErrorCategory.TIMEOUT
        elif 'invalid data' in stderr_lower or 'corrupt' in stderr_lower:
            return ErrorCategory.INVALID_INPUT
        elif 'codec not found' in stderr_lower:
            return ErrorCategory.CODEC_ERROR
        elif 'permission denied' in stderr_lower:
            return ErrorCategory.PERMISSION
        elif 'no such file' in stderr_lower:
            return ErrorCategory.NOT_FOUND
        else:
            return ErrorCategory.UNKNOWN


class VideoConvertProcessor(VideoProcessor):
    """
    Processor for video format conversion.
    
    Converts video files to different formats (mp4, webm, mov)
    with appropriate codecs for each format.
    """

    @property
    def operation_name(self) -> str:
        return "video_convert"
    
    def process_operation(self) -> ProcessingResult:
        """
        Execute video format conversion.
        
        Converts video to the specified output format with
        appropriate codec selection.
        
        Returns:
            ProcessingResult with output details
        """
        # Extract parameters
        output_format = self.parameters.get('output_format', 'mp4')
        quality = self.parameters.get('quality', 23)

        self.log_info(f"Starting video conversion to {output_format} format with quality={quality}")
        
        # Get video info for duration
        try:
            video_info = self.get_video_info()
            duration = video_info.duration
            self.log_info(f"Video duration: {duration:.2f}s, resolution: {video_info.width}x{video_info.height}")
        except Exception as e:
            self.log_error(f"Failed to get video info: {e}")
            return self._create_error_result(
                f"Failed to analyze input video: {e}",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Validate duration
        if duration <= 0:
            return self._create_error_result(
                "Video has invalid duration",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Determine codecs baed on output format
        video_codec = VIDEO_CODECS.get(output_format, 'libx264')
        audio_codec = AUDIO_CODECS.get(output_format, 'aac')

        self.log_debug(f"Using video codec: {video_codec}, audio codec: {audio_codec}")
        
        # Generate output filename and path
        output_filename = self.generate_output_filename(suffix="_converted", extension=output_format)
        output_path = self.get_temp_file_path(output_filename)
        
        # Build FFmpeg command
        ffmpeg = get_ffmpeg_wrapper()
        cmd_args = ffmpeg.build_convert_command(
            input_path=self.input_path,
            output_path=output_path,
            video_codec=video_codec,
            audio_codec=audio_codec,
            quality=quality,
        )

        self.log_debug(f"FFmpeg command args: {cmd_args}")

        # Create progress callback
        def progress_handler(percent: int, eta_seconds: Optional[float]) -> None:
            self.update_progress(percent, eta_seconds)
        
        throttled_callback = ThrottledProgressCallback(
            callback=progress_handler,
            min_interval=5.0,
            min_percent_change=5,
        )

        # Execute FFmpeg
        try:
            result = ffmpeg.execute(
                args=cmd_args,
                input_path=self.input_path,
                output_path=output_path,
                progress_callback=throttled_callback,
                timeout=self.DEFAULT_TIMEOUT,
                total_duration=duration,
            )
        except TimeoutError as e:
            return self._create_error_result(
                f"Video conversion timed out after {self.DEFAULT_TIMEOUT}s",
                ErrorCategory.TIMEOUT,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"FFmpeg execution failed: {e}")
            return self._create_error_result(
                f"Video conversion failed: {e}",
                ErrorCategory.UNKNOWN,
                is_retryable=False,
            )
        
        # Check FFmpeg result
        if not result.success:
            error_msg = self._parse_ffmpeg_error(result.stderr)
            error_category = self._categorize_ffmpeg_error(result.stderr)
            return self._create_error_result(
                error_msg,
                error_category,
                is_retryable=error_category in (ErrorCategory.TEMPORARY, ErrorCategory.RESOURCE),
            )
        
        # Verify output file was created
        if not self.validate_output_file_creation(output_path):
            return self._create_error_result(
                "Video conversion completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)
        input_ext = self.get_input_extension()
        
        self.log_info(
            f"Conversion complete. Format: {input_ext} -> {output_format}, "
            f"Input: {input_size / (1024*1024):.2f}MB, "
            f"Output: {output_size / (1024*1024):.2f}MB"
        )
        
        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'input_format': input_ext,
                'output_format': output_format,
                'input_size': input_size,
                'output_size': output_size,
                'video_codec': video_codec,
                'audio_codec': audio_codec,
                'quality': quality,
                'duration': duration,
            }
        )
    

    def _parse_ffmpeg_error(self, stderr: str) -> str:
        """
        Parse FFmpeg stderr to extract a user-friendly error message.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            User-friendly error message
        """
        stderr_lower = stderr.lower()
        
        if 'no such file' in stderr_lower:
            return "Input file not found or inaccessible"
        elif 'invalid data' in stderr_lower or 'corrupt' in stderr_lower:
            return "Input video file appears to be corrupted"
        elif 'codec not found' in stderr_lower or 'encoder' in stderr_lower:
            return "Required video codec is not available"
        elif 'permission denied' in stderr_lower:
            return "Permission denied when accessing files"
        elif 'no space left' in stderr_lower:
            return "Not enough disk space for output file"
        elif 'out of memory' in stderr_lower:
            return "Not enough memory to process video"
        elif 'unsupported' in stderr_lower:
            return "Video format or codec is not supported"
        else:
            lines = [l.strip() for l in stderr.split('\n') if l.strip()]
            if lines:
                return f"Video conversion failed: {lines[-1][:200]}"
            return "Video conversion failed with unknown error"
    

    def _categorize_ffmpeg_error(self, stderr: str) -> ErrorCategory:
        """
        Categorize FFmpeg error for retry logic.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            ErrorCategory
        """
        stderr_lower = stderr.lower()
        
        if 'no space left' in stderr_lower:
            return ErrorCategory.RESOURCE
        elif 'out of memory' in stderr_lower:
            return ErrorCategory.RESOURCE
        elif 'timeout' in stderr_lower:
            return ErrorCategory.TIMEOUT
        elif 'invalid data' in stderr_lower or 'corrupt' in stderr_lower:
            return ErrorCategory.INVALID_INPUT
        elif 'codec not found' in stderr_lower or 'unsupported' in stderr_lower:
            return ErrorCategory.CODEC_ERROR
        elif 'permission denied' in stderr_lower:
            return ErrorCategory.PERMISSION
        elif 'no such file' in stderr_lower:
            return ErrorCategory.NOT_FOUND
        else:
            return ErrorCategory.UNKNOWN
    
# Factory functions for creating processors

def create_video_compress_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> VideoCompressProcessor:
    """
    Create a VideoCompressProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        VideoCompressProcessor instance
    """
    return VideoCompressProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    

def create_video_convert_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> VideoConvertProcessor:
    """
    Create a VideoConvertProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        VideoConvertProcessor instance
    """
    return VideoConvertProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )

# Handler functions for registry registration

def video_compress_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for video compression operation.
    
    This function is registered with the operation registry and called
    by the worker to execute video compression.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters (quality, preset, audio_bitrate)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_video_compress_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    return processor.execute_operation()


def video_convert_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for video format conversion operation.
    
    This function is registered with the operation registry and called
    by the worker to execute video format conversion.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters (output_format, quality)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_video_convert_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    return processor.execute_operation()


# Register operations with the registry
def register_video_operations() -> None:
    """
    Register all video processing operations with the global registry.
    
    This function should be called during application startup to ensure
    all video operations are available.
    """
    registry = get_registry()

    # Register video compression operation
    if not registry.is_registered('video_compress'):
        registry.register_operation(
            operation_name='video_compress',
            media_type=MediaType.VIDEO,
            handler=video_compress_handler,
            parameters=[
                ParameterSchema(
                    param_name='quality',
                    param_type=ParameterType.INTEGER,
                    required=False,
                    default=23,
                    description='CRF quality value (18-28, lower is better quality)',
                    min_value=18,
                    max_value=28,
                ),
                ParameterSchema(
                    param_name='preset',
                    param_type=ParameterType.CHOICE,
                    required=False,
                    default='medium',
                    description='Encoding speed preset (faster = larger file)',
                    choices=ENCODING_PRESETS,
                ),
                ParameterSchema(
                    param_name='audio_bitrate',
                    param_type=ParameterType.STRING,
                    required=False,
                    default='128k',
                    description='Audio bitrate (e.g., 128k, 192k, 256k)',
                ),
            ],
            description='Compress video using H.264 codec with configurable quality',
            input_formats=['mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv'],
            output_formats=['mp4'],
        )
        logger.info("Registered video_compress operation")

    # Register video_convert operation

    if not registry.is_registered('video_convert'):
        registry.register_operation(
            operation_name='video_convert',
            media_type=MediaType.VIDEO,
            handler=video_convert_handler,
            parameters=[
                ParameterSchema(
                    param_name='output_format',
                    param_type=ParameterType.CHOICE,
                    required=False,
                    default='mp4',
                    description='Output video format',
                    choices=['mp4', 'webm', 'mov'],
                ),
                ParameterSchema(
                    param_name='quality',
                    param_type=ParameterType.INTEGER,
                    required=False,
                    default=23,
                    description='Quality value (18-28, lower is better)',
                    min_value=18,
                    max_value=28,
                ),
            ],
            description='Convert video to different formats (MP4, WebM, MOV)',
            input_formats=['mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv'],
            output_formats=['mp4', 'webm', 'mov'],
        )
        logger.info("Registered video_convert operation")
    
# Ensure operations are registered at module load
register_video_operations()