"""
File validation utilities for media processing.

This module provides utilities for validating media files before processing:
- Video validation using FFprobe
- Image validation using Pillow
- Audio validation using FFprobe
- General file validation helpers
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a file validation operation."""
    is_valid: bool
    error_message: Optional[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# Supported formats for each media type
SUPPORTED_VIDEO_CODECS = {
    'h264', 'hevc', 'h265', 'vp8', 'vp9', 'av1',
    'mpeg4', 'mpeg2video', 'mpeg1video', 'mjpeg',
    'wmv2', 'wmv3', 'prores', 'dnxhd', 'theora',
}

SUPPORTED_AUDIO_CODECS = {
    'aac', 'mp3', 'vorbis', 'opus', 'flac', 'pcm_s16le', 'pcm_s24le',
    'pcm_s32le', 'pcm_f32le', 'alac', 'wma', 'wmav2', 'ac3', 'eac3',
}

SUPPORTED_IMAGE_FORMATS = {
    'jpeg', 'jpg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif', 'ico',
}

# Maximum dimensions for images
MAX_IMAGE_DIMENSION = 16384  # 16K
MIN_IMAGE_DIMENSION = 1

# Maximum duration for media files (in seconds)
MAX_VIDEO_DURATION = 3600 * 4  # 4 hours
MAX_AUDIO_DURATION = 3600 * 8  # 8 hours


def validate_video_file(
    file_path: str,
    check_codec: bool = True,
    check_duration: bool = True,
    max_duration: Optional[float] = None,
) -> ValidationResult:
    """
    Validate a video file using FFprobe.
    
    Checks:
    - File exists and is readable
    - File contains a valid video stream
    - Video codec is supported (optional)
    - Duration is within limits (optional)
    
    Args:
        file_path: Path to the video file
        check_codec: Whether to verify codec is supported
        check_duration: Whether to check duration limits
        max_duration: Maximum allowed duration in seconds
        
    Returns:
        ValidationResult with validation details
    """
    warnings = []
    metadata = {}
    
    # Check file exists
    if not os.path.exists(file_path):
        return ValidationResult(
            is_valid=False,
            error_message=f"File not found: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Check file is readable
    if not os.access(file_path, os.R_OK):
        return ValidationResult(
            is_valid=False,
            error_message=f"File is not readable: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Get file size
    file_size = os.path.getsize(file_path)
    metadata['file_size'] = file_size
    
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File is empty",
            warnings=[],
            metadata=metadata,
        )
    
    # Use FFprobe to analyze the file
    try:
        from .ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        
        video_info = ffmpeg.get_video_info(file_path)
        
        metadata['duration'] = video_info.duration
        metadata['width'] = video_info.width
        metadata['height'] = video_info.height
        metadata['codec'] = video_info.codec
        metadata['fps'] = video_info.fps
        metadata['bitrate'] = video_info.bitrate
        metadata['has_audio'] = video_info.has_audio
        metadata['format'] = video_info.format_name
        
    except FileNotFoundError as e:
        return ValidationResult(
            is_valid=False,
            error_message=str(e),
            warnings=[],
            metadata=metadata,
        )
    except ValueError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Invalid video file: {e}",
            warnings=[],
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        return ValidationResult(
            is_valid=False,
            error_message=f"Failed to analyze video file: {e}",
            warnings=[],
            metadata=metadata,
        )
    
    # Check video dimensions
    if video_info.width <= 0 or video_info.height <= 0:
        return ValidationResult(
            is_valid=False,
            error_message="Video has invalid dimensions",
            warnings=warnings,
            metadata=metadata,
        )
    
    # Check codec is supported
    if check_codec:
        codec_lower = video_info.codec.lower()
        if codec_lower not in SUPPORTED_VIDEO_CODECS:
            warnings.append(
                f"Video codec '{video_info.codec}' may not be fully supported. "
                "Processing might fail or produce unexpected results."
            )
    
    # Check duration
    if check_duration:
        max_dur = max_duration or MAX_VIDEO_DURATION
        if video_info.duration > max_dur:
            return ValidationResult(
                is_valid=False,
                error_message=f"Video duration ({video_info.duration:.1f}s) exceeds maximum allowed ({max_dur}s)",
                warnings=warnings,
                metadata=metadata,
            )
        
        if video_info.duration <= 0:
            return ValidationResult(
                is_valid=False,
                error_message="Video has no duration or duration could not be determined",
                warnings=warnings,
                metadata=metadata,
            )
    
    # Check for potentially problematic videos
    if video_info.fps <= 0:
        warnings.append("Video frame rate could not be determined")
    elif video_info.fps > 120:
        warnings.append(f"Video has very high frame rate ({video_info.fps} fps)")
    
    if video_info.width > 7680 or video_info.height > 4320:
        warnings.append(
            f"Video has very high resolution ({video_info.width}x{video_info.height}). "
            "Processing may be slow."
        )
    
    if not video_info.has_audio:
        warnings.append("Video does not contain an audio track")
    
    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        metadata=metadata,
    )


def validate_audio_file(
    file_path: str,
    check_codec: bool = True,
    check_duration: bool = True,
    max_duration: Optional[float] = None,
) -> ValidationResult:
    """
    Validate an audio file using FFprobe.
    
    Checks:
    - File exists and is readable
    - File contains a valid audio stream
    - Audio codec is supported (optional)
    - Duration is within limits (optional)
    
    Args:
        file_path: Path to the audio file
        check_codec: Whether to verify codec is supported
        check_duration: Whether to check duration limits
        max_duration: Maximum allowed duration in seconds
        
    Returns:
        ValidationResult with validation details
    """
    warnings = []
    metadata = {}
    
    # Check file exists
    if not os.path.exists(file_path):
        return ValidationResult(
            is_valid=False,
            error_message=f"File not found: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Check file is readable
    if not os.access(file_path, os.R_OK):
        return ValidationResult(
            is_valid=False,
            error_message=f"File is not readable: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Get file size
    file_size = os.path.getsize(file_path)
    metadata['file_size'] = file_size
    
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File is empty",
            warnings=[],
            metadata=metadata,
        )
    
    # Use FFprobe to analyze the file
    try:
        from .ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        
        audio_info = ffmpeg.get_audio_info(file_path)
        
        metadata['duration'] = audio_info.duration
        metadata['codec'] = audio_info.codec
        metadata['channels'] = audio_info.channels
        metadata['sample_rate'] = audio_info.sample_rate
        metadata['bitrate'] = audio_info.bitrate
        metadata['format'] = audio_info.format_name
        
    except FileNotFoundError as e:
        return ValidationResult(
            is_valid=False,
            error_message=str(e),
            warnings=[],
            metadata=metadata,
        )
    except ValueError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Invalid audio file: {e}",
            warnings=[],
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        return ValidationResult(
            is_valid=False,
            error_message=f"Failed to analyze audio file: {e}",
            warnings=[],
            metadata=metadata,
        )
    
    # Check codec is supported
    if check_codec:
        codec_lower = audio_info.codec.lower()
        if codec_lower not in SUPPORTED_AUDIO_CODECS:
            warnings.append(
                f"Audio codec '{audio_info.codec}' may not be fully supported. "
                "Processing might fail or produce unexpected results."
            )
    
    # Check duration
    if check_duration:
        max_dur = max_duration or MAX_AUDIO_DURATION
        if audio_info.duration > max_dur:
            return ValidationResult(
                is_valid=False,
                error_message=f"Audio duration ({audio_info.duration:.1f}s) exceeds maximum allowed ({max_dur}s)",
                warnings=warnings,
                metadata=metadata,
            )
        
        if audio_info.duration <= 0:
            return ValidationResult(
                is_valid=False,
                error_message="Audio has no duration or duration could not be determined",
                warnings=warnings,
                metadata=metadata,
            )
    
    # Check audio properties
    if audio_info.channels <= 0:
        warnings.append("Audio channel count could not be determined")
    elif audio_info.channels > 8:
        warnings.append(f"Audio has many channels ({audio_info.channels})")
    
    if audio_info.sample_rate <= 0:
        warnings.append("Audio sample rate could not be determined")
    elif audio_info.sample_rate > 192000:
        warnings.append(f"Audio has very high sample rate ({audio_info.sample_rate} Hz)")
    
    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        metadata=metadata,
    )


def validate_image_file(
    file_path: str,
    check_format: bool = True,
    max_dimension: Optional[int] = None,
) -> ValidationResult:
    """
    Validate an image file using Pillow.
    
    Checks:
    - File exists and is readable
    - File is a valid image
    - Image format is supported (optional)
    - Image dimensions are within limits
    
    Args:
        file_path: Path to the image file
        check_format: Whether to verify format is supported
        max_dimension: Maximum allowed dimension (width or height)
        
    Returns:
        ValidationResult with validation details
    """
    warnings = []
    metadata = {}
    
    # Check file exists
    if not os.path.exists(file_path):
        return ValidationResult(
            is_valid=False,
            error_message=f"File not found: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Check file is readable
    if not os.access(file_path, os.R_OK):
        return ValidationResult(
            is_valid=False,
            error_message=f"File is not readable: {file_path}",
            warnings=[],
            metadata={},
        )
    
    # Get file size
    file_size = os.path.getsize(file_path)
    metadata['file_size'] = file_size
    
    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            error_message="File is empty",
            warnings=[],
            metadata=metadata,
        )
    
    # Try to open with Pillow
    try:
        from PIL import Image
        
        with Image.open(file_path) as img:
            # Verify the image can be read
            img.verify()
        
        # Re-open to get actual data (verify() leaves file in unusable state)
        with Image.open(file_path) as img:
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            
            # Get additional info
            if hasattr(img, 'info'):
                if 'dpi' in img.info:
                    metadata['dpi'] = img.info['dpi']
                if 'exif' in img.info:
                    metadata['has_exif'] = True
            
            # Check if animated
            try:
                img.seek(1)
                metadata['is_animated'] = True
                metadata['frame_count'] = getattr(img, 'n_frames', 2)
            except EOFError:
                metadata['is_animated'] = False
                metadata['frame_count'] = 1
                
    except ImportError:
        return ValidationResult(
            is_valid=False,
            error_message="Pillow library is not installed",
            warnings=[],
            metadata=metadata,
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Invalid image file: {e}",
            warnings=[],
            metadata=metadata,
        )
    
    # Check format is supported
    if check_format:
        img_format = metadata.get('format', '').lower()
        if img_format and img_format not in SUPPORTED_IMAGE_FORMATS:
            warnings.append(
                f"Image format '{img_format}' may not be fully supported"
            )
    
    # Check dimensions
    width = metadata.get('width', 0)
    height = metadata.get('height', 0)
    
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return ValidationResult(
            is_valid=False,
            error_message=f"Image dimensions ({width}x{height}) are too small",
            warnings=warnings,
            metadata=metadata,
        )
    
    max_dim = max_dimension or MAX_IMAGE_DIMENSION
    if width > max_dim or height > max_dim:
        return ValidationResult(
            is_valid=False,
            error_message=f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_dim})",
            warnings=warnings,
            metadata=metadata,
        )
    
    # Check for potentially problematic images
    if width > 8192 or height > 8192:
        warnings.append(
            f"Image has very high resolution ({width}x{height}). "
            "Processing may be slow and memory-intensive."
        )
    
    if metadata.get('is_animated', False):
        warnings.append(
            f"Image is animated with {metadata.get('frame_count', 'multiple')} frames. "
            "Some operations may only process the first frame."
        )
    
    # Check color mode
    mode = metadata.get('mode', '')
    if mode == 'P':
        warnings.append("Image uses palette mode. Some operations may convert it to RGB.")
    elif mode == 'CMYK':
        warnings.append("Image uses CMYK color mode. Some operations may convert it to RGB.")
    elif mode in ('LA', 'PA'):
        warnings.append("Image uses grayscale with alpha. Some operations may convert color mode.")
    
    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        metadata=metadata,
    )


def validate_media_file(
    file_path: str,
    media_type: str,
    **kwargs,
) -> ValidationResult:
    """
    Validate a media file based on its type.
    
    This is a convenience function that dispatches to the appropriate
    validation function based on media type.
    
    Args:
        file_path: Path to the file
        media_type: Type of media ('video', 'image', 'audio')
        **kwargs: Additional arguments passed to the specific validator
        
    Returns:
        ValidationResult with validation details
    """
    validators = {
        'video': validate_video_file,
        'image': validate_image_file,
        'audio': validate_audio_file,
    }
    
    validator = validators.get(media_type.lower())
    
    if not validator:
        return ValidationResult(
            is_valid=False,
            error_message=f"Unknown media type: {media_type}",
            warnings=[],
            metadata={},
        )
    
    return validator(file_path, **kwargs)


def quick_validate_file(file_path: str) -> Tuple[bool, str, Optional[str]]:
    """
    Quickly validate a file and detect its type.
    
    This is a fast validation that just checks if the file is valid
    and detects its media type without full analysis.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (is_valid, media_type, error_message)
        media_type will be 'video', 'image', 'audio', or 'unknown'
    """
    if not os.path.exists(file_path):
        return False, 'unknown', f"File not found: {file_path}"
    
    if os.path.getsize(file_path) == 0:
        return False, 'unknown', "File is empty"
    
    # Try image validation first (fastest)
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True, 'image', None
    except Exception:
        pass
    
    # Try FFprobe for video/audio
    try:
        from .ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        
        # Try as video first
        try:
            ffmpeg.get_video_info(file_path)
            return True, 'video', None
        except ValueError:
            pass
        
        # Try as audio
        try:
            ffmpeg.get_audio_info(file_path)
            return True, 'audio', None
        except ValueError:
            pass
        
    except FileNotFoundError:
        # FFmpeg not available, can't validate video/audio
        pass
    except Exception as e:
        logger.warning(f"FFprobe validation failed: {e}")
    
    return False, 'unknown', "File type could not be determined or is not supported"


def get_file_duration(file_path: str) -> Optional[float]:
    """
    Get the duration of a media file.
    
    Works for both video and audio files.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        Duration in seconds, or None if not available
    """
    try:
        from .ffmpeg import FFmpegWrapper
        ffmpeg = FFmpegWrapper()
        return ffmpeg.get_duration(file_path)
    except Exception as e:
        logger.error(f"Failed to get file duration: {e}")
        return None


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a valid video file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a valid video
    """
    result = validate_video_file(file_path, check_codec=False, check_duration=False)
    return result.is_valid


def is_audio_file(file_path: str) -> bool:
    """
    Check if a file is a valid audio file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a valid audio
    """
    result = validate_audio_file(file_path, check_codec=False, check_duration=False)
    return result.is_valid


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is a valid image file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a valid image
    """
    result = validate_image_file(file_path, check_format=False)
    return result.is_valid


def validate_output_path(
    output_path: str,
    create_directory: bool = True,
) -> ValidationResult:
    """
    Validate an output path for writing.
    
    Checks:
    - Parent directory exists or can be created
    - Path is writable
    - File doesn't already exist (warning only)
    
    Args:
        output_path: Path where output will be written
        create_directory: Whether to create parent directory if missing
        
    Returns:
        ValidationResult
    """
    warnings = []
    metadata = {'output_path': output_path}
    
    # Get parent directory
    parent_dir = os.path.dirname(output_path)
    
    if parent_dir and not os.path.exists(parent_dir):
        if create_directory:
            try:
                os.makedirs(parent_dir, exist_ok=True)
                metadata['directory_created'] = True
            except OSError as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Cannot create output directory: {e}",
                    warnings=[],
                    metadata=metadata,
                )
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Output directory does not exist: {parent_dir}",
                warnings=[],
                metadata=metadata,
            )
    
    # Check directory is writable
    check_dir = parent_dir if parent_dir else '.'
    if not os.access(check_dir, os.W_OK):
        return ValidationResult(
            is_valid=False,
            error_message=f"Output directory is not writable: {check_dir}",
            warnings=[],
            metadata=metadata,
        )
    
    # Check if file already exists
    if os.path.exists(output_path):
        warnings.append(f"Output file already exists and will be overwritten: {output_path}")
    
    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        metadata=metadata,
    )