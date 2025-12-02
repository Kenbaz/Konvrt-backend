# apps/processors/utils/__init__.py

"""
Utilities for media processing operations.

This package provides:
- ffmpeg: FFmpeg and FFprobe wrapper for video/audio processing
- progress: Progress tracking and throttled callbacks
- validation: File validation utilities
"""

from .ffmpeg import (
    FFmpegWrapper,
    FFmpegResult,
    VideoInfo,
    AudioInfo,
    ProgressCallback,
    get_ffmpeg_wrapper,
)

from .track_progress import (
    ProgressParser,
    ProgressInfo,
    ThrottledProgressCallback,
    ProgressTracker,
    create_operation_progress_callback,
    pillow_progress_callback,
    estimate_processing_time,
)

from .validation import (
    ValidationResult,
    validate_video_file,
    validate_audio_file,
    validate_image_file,
    validate_media_file,
    validate_output_path,
    quick_validate_file,
    get_file_duration,
    is_video_file,
    is_audio_file,
    is_image_file,
    SUPPORTED_VIDEO_CODECS,
    SUPPORTED_AUDIO_CODECS,
    SUPPORTED_IMAGE_FORMATS,
)

__all__ = [
    # FFmpeg
    'FFmpegWrapper',
    'FFmpegResult',
    'VideoInfo',
    'AudioInfo',
    'ProgressCallback',
    'get_ffmpeg_wrapper',
    
    # Progress
    'ProgressParser',
    'ProgressInfo',
    'ThrottledProgressCallback',
    'ProgressTracker',
    'create_operation_progress_callback',
    'pillow_progress_callback',
    'estimate_processing_time',
    
    # Validation
    'ValidationResult',
    'validate_video_file',
    'validate_audio_file',
    'validate_image_file',
    'validate_media_file',
    'validate_output_path',
    'quick_validate_file',
    'get_file_duration',
    'is_video_file',
    'is_audio_file',
    'is_image_file',
    'SUPPORTED_VIDEO_CODECS',
    'SUPPORTED_AUDIO_CODECS',
    'SUPPORTED_IMAGE_FORMATS',
]