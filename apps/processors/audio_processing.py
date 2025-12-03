# apps/processors/audio_processing.py

"""
Audio processing operations.

This module implements audio processing operations:
- audio_convert: Convert audio files to different formats (mp3, wav, aac, ogg, flac)
- audio_extract: Extract audio track from video files

All operations are registered with the operation registry and can be
executed through the worker system.
"""

import logging
import os
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from .base_processor import ProcessingResult, AudioProcessor, ErrorCategory
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


# Audio format configurations
AUDIO_FORMATS = {
    'mp3': {
        'extension': 'mp3',
        'codec': 'libmp3lame',
        'default_bitrate': '192k',
        'supports_quality': True,
        'mime_type': 'audio/mpeg',
    },
    'wav': {
        'extension': 'wav',
        'codec': 'pcm_s16le',
        'default_bitrate': None,
        'supports_quality': False,
        'mime_type': 'audio/wav',
    },
    'aac': {
        'extension': 'aac',
        'codec': 'aac',
        'default_bitrate': '192k',
        'supports_quality': True,
        'mime_type': 'audio/aac',
    },
    'm4a': {
        'extension': 'm4a',
        'codec': 'aac',
        'default_bitrate': '192k',
        'supports_quality': True,
        'mime_type': 'audio/mp4',
    },
    'ogg': {
        'extension': 'ogg',
        'codec': 'libvorbis',
        'default_bitrate': '192k',
        'supports_quality': True,
        'mime_type': 'audio/ogg',
    },
    'flac': {
        'extension': 'flac',
        'codec': 'flac',
        'default_bitrate': None,
        'supports_quality': False,
        'mime_type': 'audio/flac',
    },
    'opus': {
        'extension': 'opus',
        'codec': 'libopus',
        'default_bitrate': '128k',
        'supports_quality': True,
        'mime_type': 'audio/opus',
    },
}

# Valid bitrate values
VALID_BITRATES = [
    '64k', '96k', '128k', '160k', '192k', '224k', '256k', '320k'
]

# Valid sample rates
VALID_SAMPLE_RATES = [8000, 11025, 22050, 44100, 48000, 96000]

# Valid channel configurations
VALID_CHANNELS = [1, 2]  # Mono, Stereo


class AudioConvertProcessor(AudioProcessor):
    """
    Processor for audio format conversion.
    
    Converts audio files to different formats (mp3, wav, aac, ogg, flac, opus)
    with configurable bitrate, sample rate, and channel settings.
    """
    @property
    def operation_name(self) -> str:
        return "audio_convert"
    

    def process_operation(self) -> ProcessingResult:
        """
        Execute audio format conversion.
        
        Uses FFmpeg to convert audio files between formats with
        configurable quality settings.
        
        Returns:
            ProcessingResult with output details
        """

        # Extract parameters with defaults
        output_format = self.parameters.get('output_format', 'mp3')
        bitrate = self.parameters.get('bitrate', '192k')
        sample_rate = self.parameters.get('sample_rate')
        channels = self.parameters.get('channels')

        self.log_info(
            f"Starting audio conversion: format={output_format}, "
            f"bitrate={bitrate}, sample_rate={sample_rate}, channels={channels}"
        )

        # Get format configuration
        format_config = AUDIO_FORMATS.get(output_format)
        if not format_config:
            return self._create_error_result(
                f"Unsupported output format: {output_format}",
                ErrorCategory.INVALID_PARAMS,
                is_retryable=False,
            )
        
        # Get audio info for duration
        try:
            audio_info = self.get_audio_info()
            duration = audio_info.duration
            self.log_info(
                f"Audio duration: {duration:.2f}s, codec: {audio_info.codec}, "
                f"sample_rate: {audio_info.sample_rate}, channels: {audio_info.channels}"
            )
        except Exception as e:
            self.log_error(f"Failed to get audio info: {e}")
            return self._create_error_result(
                f"Failed to analyze input audio: {e}",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Validate duration
        if duration <= 0:
            return self._create_error_result(
                "Audio duration is zero or invalid.",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Generate output filename and path
        output_filename = self.generate_output_filename(
            suffix="_converted",
            extension=format_config['extension']
        )
        output_path = self.get_temp_file_path(output_filename)

        # Determine bitrate to use (use default for lossless formats)
        effective_bitrate = bitrate
        if not format_config['supports_quality']:
            effective_bitrate = None
        
        # Build FFmpeg command
        ffmpeg = get_ffmpeg_wrapper()
        cmd_args = ffmpeg.build_audio_convert_command(
            input_path=self.input_path,
            output_path=output_path,
            audio_codec=format_config['codec'],
            bitrate=effective_bitrate if effective_bitrate else '192k',
            sample_rate=sample_rate,
            channels=channels,
        )

        self.log_debug(f"FFmpeg command: {cmd_args}")

        # Create progress callback
        def progress_handler(percent: int, eta_seconds: Optional[float]) -> None:
            self.update_progress(percent, eta_seconds)
        
        throttled_progress = ThrottledProgressCallback(
            callback=progress_handler,
            min_interval=5.0,
            min_percent_change=5
        )

        # Execute FFmpeg command
        try:
            result = ffmpeg.execute(
                args=cmd_args,
                input_path=self.input_path,
                output_path=output_path,
                progress_callback=throttled_progress,
                timeout=self.DEFAULT_TIMEOUT,
                total_duration=duration,
            )
        except TimeoutError:
            return self._create_error_result(
                f"Audio conversion timed out after {self.DEFAULT_TIMEOUT}s",
                ErrorCategory.TIMEOUT,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"FFmpeg execution failed: {e}")
            return self._create_error_result(
                f"Audio conversion failed: {e}",
                ErrorCategory.UNKNOWN,
                is_retryable=False,
            )
        
        # Check ffmpeg result
        if not result.success:
            error_msg = self._parse_ffmpeg_error(result.stderr)
            error_category = self._categorize_ffmpeg_error(error_msg)
            return self._create_error_result(
                error_msg,
                error_category,
                is_retryable=error_category in (ErrorCategory.TEMPORARY, ErrorCategory.RESOURCE),
            )
        
        # Verify output file was created
        if not self.validate_output_file_creation(output_path):
            return self._create_error_result(
                "Audio conversion completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)

        # Try to get output audio info
        try:
            output_audio_info = ffmpeg.get_audio_info(output_path)
            output_duration = output_audio_info.duration
            output_sample_rate = output_audio_info.sample_rate
            output_channels = output_audio_info.channels
            output_codec = output_audio_info.codec
        except Exception as e:
            self.log_warning(f"Could not get output audio info: {e}")
            output_duration = duration
            output_sample_rate = sample_rate or audio_info.sample_rate
            output_channels = channels or audio_info.channels
            output_codec = format_config['codec']
        
        self.log_info(
            f"Conversion complete. Input: {input_size / 1024:.1f}KB, "
            f"Output: {output_size / 1024:.1f}KB, "
            f"Format: {output_format}"
        )
        
        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'input_size': input_size,
                'output_size': output_size,
                'input_format': audio_info.format_name,
                'output_format': output_format,
                'duration': output_duration,
                'bitrate': bitrate,
                'sample_rate': output_sample_rate,
                'channels': output_channels,
                'codec': output_codec,
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
            return "Input audio file appears to be corrupted"
        elif 'codec not found' in stderr_lower or 'encoder' in stderr_lower:
            return "Required audio codec is not available"
        elif 'permission denied' in stderr_lower:
            return "Permission denied when accessing files"
        elif 'no space left' in stderr_lower:
            return "Not enough disk space for output file"
        elif 'out of memory' in stderr_lower:
            return "Not enough memory to process audio"
        elif 'invalid sample rate' in stderr_lower:
            return "Invalid sample rate specified"
        elif 'invalid channel' in stderr_lower:
            return "Invalid channel configuration specified"
        else:
            # Return last meaningful line from stderr
            lines = [l.strip() for l in stderr.split('\n') if l.strip()]
            if lines:
                # Find error lines
                for line in reversed(lines):
                    if 'error' in line.lower():
                        return f"FFmpeg error: {line[:200]}"
                return f"Audio conversion failed: {lines[-1][:200]}"
            return "Audio conversion failed with unknown error"
    

    def _categorize_ffmpeg_error(self, stderr: str) -> ErrorCategory:
        """
        Categorize an FFmpeg error based on stderr output.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            ErrorCategory for the error
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


class AudioExtractProcessor(AudioProcessor):
    """
    Processor for extracting audio from video files.
    
    Extracts the audio track from video files and saves it
    in the specified format.
    """
    
    @property
    def operation_name(self) -> str:
        return "audio_extract"
    
    def process_operation(self) -> ProcessingResult:
        """
        Execute audio extraction from video.
        
        Uses FFmpeg to extract audio track from video files
        without re-encoding video.
        
        Returns:
            ProcessingResult with output details
        """
        # Exract parameters with defaults
        output_format = self.parameters.get('output_format', 'mp3')
        bitrate = self.parameters.get('bitrate', '192k')
        sample_rate = self.parameters.get('sample_rate')
        channels = self.parameters.get('channels')

        self.log_info(
            f"Starting audio extraction: format={output_format}, "
            f"bitrate={bitrate}"
        )

        # Get format configuration
        format_config = AUDIO_FORMATS.get(output_format)
        if not format_config:
            return self._create_error_result(
                f"Unsupported output format: {output_format}",
                ErrorCategory.INVALID_PARAMS,
                is_retryable=False,
            )
        
        # Get video infor for duration and to verify it has audio
        ffmpeg = get_ffmpeg_wrapper()
        try:
            video_info = ffmpeg.get_video_info(self.input_path)
            duration = video_info.duration
            has_audio = video_info.has_audio

            self.log_info(
                f"Video duration: {duration:.2f}s, has_audio: {has_audio}, "
                f"resolution: {video_info.width}x{video_info.height}"
            )
        except Exception as e:
            self.log_error(f"Failed to get video info: {e}")
            return self._create_error_result(
                f"Failed to analyze input video: {e}",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Validate video has audio
        if not has_audio:
            return self._create_error_result(
                "Video does not contain an audio track.",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Validate duration
        if duration <= 0:
            return self._create_error_result(
                "Video duration is zero or invalid.",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        
        # Generate output filename and path
        output_filename = self.generate_output_filename(
            suffix="_audio",
            extension=format_config['extension']
        )
        output_path = self.get_temp_file_path(output_filename)
        
        # Determine bitrate to use
        effective_bitrate = bitrate
        if not format_config['supports_quality']:
            effective_bitrate = '192k'  # Default for command building
        
        # Build FFmpeg command for audio extraction
        cmd_args = ffmpeg.build_audio_extract_command(
            input_path=self.input_path,
            output_path=output_path,
            audio_codec=format_config['codec'],
            bitrate=effective_bitrate,
        )

        # Add sample rate and channels if specified
        if sample_rate or channels:
            # Rebuild command with additional audio options
            cmd_args = ffmpeg.build_audio_convert_command(
                input_path=self.input_path,
                output_path=output_path,
                audio_codec=format_config['codec'],
                bitrate=effective_bitrate,
                sample_rate=sample_rate,
                channels=channels,
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

        # Execute FFmpeg command
        try:
            result = ffmpeg.execute(
                args=cmd_args,
                input_path=self.input_path,
                output_path=output_path,
                progress_callback=throttled_callback,
                timeout=self.DEFAULT_TIMEOUT,
                total_duration=duration
            )
        except TimeoutError:
            return self._create_error_result(
                f"Audio extraction timed out after {self.DEFAULT_TIMEOUT}s",
                ErrorCategory.TIMEOUT,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"FFmpeg execution failed: {e}")
            return self._create_error_result(
                f"Audio extraction failed: {e}",
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
                "Audio extraction completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)
        
        # Try to get output audio info
        try:
            output_audio_info = ffmpeg.get_audio_info(output_path)
            output_duration = output_audio_info.duration
            output_sample_rate = output_audio_info.sample_rate
            output_channels = output_audio_info.channels
            output_codec = output_audio_info.codec
            output_bitrate = output_audio_info.bitrate
        except Exception as e:
            self.log_warning(f"Could not get output audio info: {e}")
            output_duration = duration
            output_sample_rate = sample_rate
            output_channels = channels
            output_codec = format_config['codec']
            output_bitrate = None
        
        self.log_info(
            f"Extraction complete. Video size: {input_size / (1024*1024):.2f}MB, "
            f"Audio size: {output_size / 1024:.1f}KB, "
            f"Duration: {output_duration:.2f}s, Format: {output_format}"
        )
        
        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'input_size': input_size,
                'output_size': output_size,
                'source_format': video_info.format_name,
                'output_format': output_format,
                'duration': output_duration,
                'bitrate': output_bitrate or bitrate,
                'sample_rate': output_sample_rate,
                'channels': output_channels,
                'codec': output_codec,
                'source_video_resolution': f"{video_info.width}x{video_info.height}",
            }
        )
    

    def _build_extract_command_with_options(
        self,
        input_path: str,
        output_path: str,
        codec: str,
        bitrate: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> list:
        """
        Build FFmpeg command for audio extraction with additional options.
        
        Args:
            input_path: Path to input video
            output_path: Path for output audio
            codec: Audio codec to use
            bitrate: Audio bitrate
            sample_rate: Output sample rate
            channels: Number of audio channels
            
        Returns:
            List of command arguments
        """
        args = [
            '-i', input_path,
            '-vn',
            '-c:a', codec,
            '-b:a', bitrate,
        ]

        if sample_rate:
            args.extend(['-ar', str(sample_rate)])
        
        if channels:
            args.extend(['-ac', str(channels)])
        
        args.append(['-y', output_path])

        return args
    

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
        elif 'does not contain any stream' in stderr_lower or 'no audio' in stderr_lower:
            return "Input video does not contain an audio track"
        elif 'codec not found' in stderr_lower or 'encoder' in stderr_lower:
            return "Required audio codec is not available"
        elif 'permission denied' in stderr_lower:
            return "Permission denied when accessing files"
        elif 'no space left' in stderr_lower:
            return "Not enough disk space for output file"
        elif 'out of memory' in stderr_lower:
            return "Not enough memory to process audio"
        else:
            # Return last meaningful line from stderr
            lines = [l.strip() for l in stderr.split('\n') if l.strip()]
            if lines:
                for line in reversed(lines):
                    if 'error' in line.lower():
                        return f"FFmpeg error: {line[:200]}"
                return f"Audio extraction failed: {lines[-1][:200]}"
            return "Audio extraction failed with unknown error"
    

    def _categorize_ffmpeg_error(self, stderr: str) -> ErrorCategory:
        """
        Categorize an FFmpeg error based on stderr output.
        
        Args:
            stderr: FFmpeg stderr output
            
        Returns:
            ErrorCategory for the error
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
        elif 'no audio' in stderr_lower or 'does not contain' in stderr_lower:
            return ErrorCategory.INVALID_INPUT
        elif 'codec not found' in stderr_lower or 'unsupported' in stderr_lower:
            return ErrorCategory.CODEC_ERROR
        elif 'permission denied' in stderr_lower:
            return ErrorCategory.PERMISSION
        elif 'no such file' in stderr_lower:
            return ErrorCategory.NOT_FOUND
        else:
            return ErrorCategory.UNKNOWN


# Functions for creating processor instances

def create_audio_convert_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> AudioConvertProcessor:
    """
    Create an AudioConvertProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input audio
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        AudioConvertProcessor instance
    """
    return AudioConvertProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )


def create_audio_extract_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> AudioExtractProcessor:
    """
    Create an AudioExtractProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        AudioExtractProcessor instance
    """
    return AudioExtractProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )


# Handler functions for registry registration

def audio_convert_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for audio format conversion operation.
    
    This function is registered with the operation registry and called
    by the worker to execute audio format conversion.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input audio
        parameters: Processing parameters (output_format, bitrate, sample_rate, channels)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_audio_convert_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    return processor.execute_operation()


def audio_extract_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for audio extraction operation.
    
    This function is registered with the operation registry and called
    by the worker to extract audio from video files.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input video
        parameters: Processing parameters (output_format, bitrate, sample_rate, channels)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_audio_extract_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )
    return processor.execute_operation()


# Register operations with the registry
def register_audio_operations() -> None:
    """
    Register all audio processing operations with the global registry.
    
    This function should be called during application startup to ensure
    all audio operations are available. It is idempotent - calling it
    multiple times has no effect if operations are already registered.
    """
    registry = get_registry()
    
    # Register audio_convert operation
    try:
        if not registry.is_registered('audio_convert'):
            registry.register_operation(
                operation_name='audio_convert',
                media_type=MediaType.AUDIO,
                handler=audio_convert_handler,
                parameters=[
                    ParameterSchema(
                        param_name='output_format',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='mp3',
                        description='Output audio format',
                        choices=['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'],
                    ),
                    ParameterSchema(
                        param_name='bitrate',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='192k',
                        description='Audio bitrate (ignored for lossless formats like WAV and FLAC)',
                        choices=VALID_BITRATES,
                    ),
                    ParameterSchema(
                        param_name='sample_rate',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default=None,
                        description='Output sample rate in Hz (uses source rate if not specified)',
                        choices=VALID_SAMPLE_RATES,
                    ),
                    ParameterSchema(
                        param_name='channels',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default=None,
                        description='Number of audio channels (1=mono, 2=stereo)',
                        choices=VALID_CHANNELS,
                    ),
                ],
                description='Convert audio to different formats with configurable quality settings',
                input_formats=['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'wma', 'opus', 'aiff'],
                output_formats=['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'],
            )
            logger.info("Registered audio_convert operation")
    except Exception as e:
        # Already registered or other error - log and continue
        logger.debug(f"audio_convert registration skipped: {e}")
    
    # Register audio_extract operation
    try:
        if not registry.is_registered('audio_extract'):
            registry.register_operation(
                operation_name='audio_extract',
                media_type=MediaType.VIDEO,  # Input is video
                handler=audio_extract_handler,
                parameters=[
                    ParameterSchema(
                        param_name='output_format',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='mp3',
                        description='Output audio format',
                        choices=['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'],
                    ),
                    ParameterSchema(
                        param_name='bitrate',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='192k',
                        description='Audio bitrate (ignored for lossless formats)',
                        choices=VALID_BITRATES,
                    ),
                    ParameterSchema(
                        param_name='sample_rate',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default=None,
                        description='Output sample rate in Hz',
                        choices=VALID_SAMPLE_RATES,
                    ),
                    ParameterSchema(
                        param_name='channels',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default=None,
                        description='Number of audio channels (1=mono, 2=stereo)',
                        choices=VALID_CHANNELS,
                    ),
                ],
                description='Extract audio track from video files',
                input_formats=['mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv', 'm4v'],
                output_formats=['mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'opus'],
            )
            logger.info("Registered audio_extract operation")
    except Exception as e:
        # Already registered or other error - log and continue
        logger.debug(f"audio_extract registration skipped: {e}")


# Register operations at module load
register_audio_operations()