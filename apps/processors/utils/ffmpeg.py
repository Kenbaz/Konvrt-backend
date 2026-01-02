# apps/processors/utils/ffmpeg.py

"""
FFmpeg Wrapper for media processing operations.

This module provides a wrapper around FFmpeg and FFprobe commands,
handling execution, progress parsing, and metadata extraction.
"""

import json
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Information about a video file."""
    duration: float
    width: int
    height: int
    codec: str
    bitrate: Optional[int]
    fps: float
    frame_count: Optional[int]
    pixel_format: Optional[str]
    has_audio: bool
    file_size: int
    format_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "pixel_format": self.pixel_format,
            "has_audio": self.has_audio,
            "file_size": self.file_size,
            "format_name": self.format_name,
        }


@dataclass
class AudioInfo:
    """Information about an audio file."""
    duration: float
    codec: str
    bitrate: Optional[int]
    sample_rate: int
    channels: int
    file_size: int
    format_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "duration": self.duration,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "file_size": self.file_size,
            "format_name": self.format_name,
        }
    

@dataclass
class FFmpegResult:
    """Result of an FFmpeg command operation"""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    output_path: Optional[str]
    duration_seconds: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_path": self.output_path,
            "duration_seconds": self.duration_seconds,
        }

# Progress callback type
ProgressCallback = Callable[[int, Optional[float]], None] # (progress_percent, eta_seconds)


class FFmpegWrapper:
    """
    Wrapper class for FFmpeg and FFprobe operations.
    
    Provides methods for:
    - Executing FFmpeg commands with progress tracking
    - Extracting video/audio metadata with FFprobe
    - Extracting frames from videos
    - Building common FFmpeg command patterns
    """

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ):
        """
        Initialize the FFmpeg wrapper.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable (uses settings if not provided)
            ffprobe_path: Path to FFprobe executable (uses settings if not provided)
        """
        self.ffmpeg_path = ffmpeg_path or getattr(settings, 'FFMPEG_PATH', 'ffmpeg')
        self.ffprobe_path = ffprobe_path or getattr(settings, 'FFPROBE_PATH', 'ffprobe')

        # Verify executables are available
        self._verify_ffmpeg_availability()
    

    def _verify_ffmpeg_availability(self) -> None:
        """
        Verify that FFmpeg and FFprobe are available.
        
        Raises:
            FileNotFoundError: If FFmpeg or FFprobe cannot be found
        """
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"FFmpeg return error: {result.stderr}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Please install FFmpeg or configure FFMPEG_PATH in settings."
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("FFmpeg availability check timed out.")
        
        try:
            result = subprocess.run(
                [self.ffprobe_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"FFprobe return error: {result.stderr}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FFprobe not found at '{self.ffprobe_path}'. "
                "Please install FFmpeg or configure FFPROBE_PATH in settings."
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("FFprobe availability check timed out.")
    

    # COMMAND EXECUTION
    
    def execute(
        self,
        args: List[str],
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        timeout: Optional[int] = None,
        total_duration: Optional[float] = None
    ) -> FFmpegResult:
        """
        Execute an FFmpeg command.
        
        Args:
            args: FFmpeg arguments (without ffmpeg executable)
            input_path: Path to input file (for validation)
            output_path: Path to output file
            progress_callback: Optional callback for progress updates
            timeout: Command timeout in seconds
            total_duration: Total duration for progress calculation
            
        Returns:
            FFmpegResult with execution details
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            subprocess.TimeoutExpired: If command times out
        """
        # Validate input file exists
        if input_path and not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
        
        # Build full command
        cmd = [self.ffmpeg_path] + args

        logger.debug(f"Executing FFmpeg command: {' '.join(cmd)}")

        # If there is a progress callback, parse stderr in real-time
        if progress_callback and total_duration:
            return self._execute_with_progress(
                cmd=cmd,
                output_path=output_path,
                progress_callback=progress_callback,
                timeout=timeout,
                total_duration=total_duration
            )
        else:
            return self._execute_simple(
                cmd=cmd,
                output_path=output_path,
                timeout=timeout
            )
    

    def _execute_simple(
        self,
        cmd: List[str],
        output_path: Optional[str],
        timeout: Optional[int]
    ) -> FFmpegResult:
        """
        Execute FFmpeg command without progress tracking.
        
        Args:
            cmd: Full command list
            output_path: Expected output path
            timeout: Command timeout
            
        Returns:
            FFmpegResult
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            success = result.returncode == 0

            # Verify output was created if expected
            if success and output_path and not os.path.exists(output_path):
                success = False
                logger.error(f"FFmpeg succeeded but output file '{output_path}' not found")
            
            return FFmpegResult(
                success=success,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_path=output_path if success else None,
                duration_seconds=None
            )
        
        except subprocess.TimeoutExpired as e:
            logger.error(f"FFmpeg command timed out after {timeout}s")
            return FFmpegResult(
                success=False,
                return_code=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or f"Command timed out after {timeout} seconds",
                output_path=None,
                duration_seconds=None
            )
    

    def _execute_with_progress(
        self,
        cmd: List[str],
        output_path: Optional[str],
        progress_callback: ProgressCallback,
        timeout: Optional[int],
        total_duration: float
    ) -> FFmpegResult:
        """
        Execute FFmpeg command with real-time progress tracking.
        
        Args:
            cmd: Full command list
            output_path: Expected output path
            progress_callback: Callback for progress updates
            timeout: Command timeout
            total_duration: Total duration for progress calculation
            
        Returns:
            FFmpegResult
        """
        if '-progress' not in cmd:
            progress_args = ['-progress', 'pipe:1', '-nostats']
            # Find position before output 
            cmd = cmd[:-1] + progress_args + [cmd[-1]]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_lines = []
        stderr_lines = []
        current_time = 0.0


        def read_stderr():
            """Read stderr in a separate thread."""
            for line in process.stderr:
                stderr_lines.append(line)
                # Parse stderr for time info (fallback)
                time_match = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', line)
                if time_match:
                    nonlocal current_time
                    h, m, s = time_match.groups()
                    current_time = int(h) * 3600 + int(m) * 60 + float(s)
        
        # Start stderr reading thread
        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        try:
            # Read stdout for progress info
            for line in process.stdout:
                stdout_lines.append(line)

                # Parse progress output
                if line.startswith('out_time_ms='):
                    try:
                        time_ms = int(line.split('=')[1].strip())
                        current_time = time_ms / 1_000_000 # Convert to seconds

                        if total_duration > 0:
                            progress = min(99, int((current_time / total_duration) * 100))
                            remaining = total_duration - current_time
                            progress_callback(progress, remaining)
                    except (ValueError, IndexError):
                        pass
                
                elif line.startswith('progress=end'):
                    progress_callback(100, 0)
            
            # wait for process to complete
            return_code = process.wait(timeout=timeout)
            stderr_thread.join(timeout=5)

            success = return_code == 0

            # Verify output was created if expected
            if success and output_path and not os.path.exists(output_path):
                success = False
                logger.error(f"FFmpeg succeeded but output file '{output_path}' not found")

            return FFmpegResult(
                success=success,
                return_code=return_code,
                stdout=''.join(stdout_lines),
                stderr=''.join(stderr_lines),
                output_path=output_path if success else None,
                duration_seconds=current_time if success else None
            )
        
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            logger.error(f"FFmpeg command timed out after {timeout}s")
            return FFmpegResult(
                success=False,
                return_code=-1,
                stdout=''.join(stdout_lines),
                stderr=''.join(stderr_lines) + f"\nCommand timed out after {timeout} seconds",
                output_path=None,
                duration_seconds=None
            )
    
    # METADATA EXTRACTION

    def get_video_info(self, file_path: str) -> VideoInfo:
        """
        Extract video file information using FFprobe.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            VideoInfo object with video details
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid video
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        
        # Run ffprobe to get JSON metadata
        cmd = [
            self.ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ValueError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse FFprobe output: {e}")
        except subprocess.TimeoutExpired:
            raise TimeoutError("FFprobe timed out while extracting video info.")
        
        # Find video stream
        video_stream = None
        audio_stream = None

        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video' and video_stream is None:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and audio_stream is None:
                audio_stream = stream
            
        if not video_stream:
            raise ValueError(f"No video stream found in file: {file_path}")
        
        # Extract format info
        format_info = data.get('format', {})

        # Parse duration
        duration = float(format_info.get('duration', 0))
        if duration == 0:
            duration = float(video_stream.get('duration', 0))
        
        # Parse dimensions
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))

        # Parse FPS
        fps = 0.0
        fps_str = video_stream.get('r_frame_rate', '0/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            if int(den) > 0:
                fps = float(num) / float(den)
        
        # Parse bitrate
        bitrate = None
        if format_info.get('bit_rate'):
            try:
                bitrate = int(format_info['bit_rate'])
            except (ValueError, TypeError):
                pass
        
        # Parse frame count
        frame_count = None
        if video_stream.get('nb_frames'):
            try:
                frame_count = int(video_stream['nb_frames'])
            except (ValueError, TypeError):
                pass
        
        file_size = int(format_info.get('size', os.path.getsize(file_path)))

        return VideoInfo(
            duration=duration,
            width=width,
            height=height,
            codec=video_stream.get('codec_name', 'unknown'),
            bitrate=bitrate,
            fps=round(fps, 2),
            frame_count=frame_count,
            pixel_format=video_stream.get('pix_fmt'),
            has_audio=audio_stream is not None,
            file_size=file_size,
            format_name=format_info.get('format_name', 'unknown')
        )
    

    def get_audio_info(self, file_path: str) -> AudioInfo:
        """
        Extract audio file information using FFprobe.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioInfo object with audio details
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid audio file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        cmd = [
            self.ffprobe_path,
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ValueError(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse FFprobe output: {e}")
        except subprocess.TimeoutExpired:
            raise TimeoutError("FFprobe timed out while extracting audio info.")
        
        # Find audio stream
        audio_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                audio_stream = stream
                break
        
        if not audio_stream:
            raise ValueError(f"No audio stream found in file: {file_path}")
        
        # Extract format info
        format_info = data.get('format', {})

        # Parse duration
        duration = float(format_info.get('duration', 0))
        if duration == 0:
            duration = float(audio_stream.get('duration', 0))
        
        # Parse bitrate
        bitrate = None
        if format_info.get('bit_rate'):
            try:
                bitrate = int(format_info['bit_rate'])
            except (ValueError, TypeError):
                pass
        elif audio_stream.get('bit_rate'):
            try:
                bitrate = int(audio_stream['bit_rate'])
            except (ValueError, TypeError):
                pass
        
        # Get file size
        file_size = int(format_info.get('size', os.path.getsize(file_path)))

        return AudioInfo(
            duration=duration,
            codec=audio_stream.get('codec_name', 'unknown'),
            bitrate=bitrate,
            channels=int(audio_stream.get('channels', 0)),
            sample_rate=int(audio_stream.get('sample_rate', 0)),
            file_size=file_size,
            format_name=format_info.get('format_name', 'unknown'),
        )
    

    def get_duration(self, file_path: str) -> float:
        """
        Get the duration of a media file in seconds.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Duration in seconds
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If duration cannot be determined
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Media file not found: {file_path}")
        
        cmd = [
            self.ffprobe_path,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise ValueError(f"FFprobe failed: {result.stderr}")
            
            duration = float(result.stdout.strip())
            return duration
        
        except (ValueError, subprocess.TimeoutExpired) as e:
            raise ValueError(f"Failed to get duration: {e}")
    
    # FRAME EXTRACTION

    def extract_frames(
        self,
        video_path: str,
        output_path: str,
        timestamp: float = 0.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bool:
        """
        Extract a single frame from a video.
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted frame (jpg/png)
            timestamp: Time in seconds to extract frame from
            width: Optional output width (maintains aspect ratio if height not set)
            height: Optional output height
            
        Returns:
            True if extraction successful
            
        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Build command
        args = [
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-y', # Overwrite output
        ]

        # Add scaling if specified
        if width or height:
            if width and height:
                scale = f'scale={width}:{height}'
            elif width:
                scale = f'scale={width}:-1'
            else:
                scale = f'scale=-1:{height}'
            args.extend(['-vf', scale])
        
        args.append(output_path)

        result = self.execute(args, input_path=video_path, output_path=output_path)

        return result.success
    

    def extract_thumbnail(
        self,
        video_path: str,
        output_path: str,
        size: int = 320,
    ) -> bool:
        """
        Extract a thumbnail from a video (frame at 10% duration).
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the thumbnail
            size: Maximum dimension size (width or height)
            
        Returns:
            True if extraction successful
        """
        try:
            # Get video duration
            duration = self.get_duration(video_path)

            # Extract frame at 10% or 1 second duration
            timestamp = min(duration * 0.1, 1.0)

            return self.extract_frames(
                video_path=video_path,
                output_path=output_path,
                timestamp=timestamp,
                width=size,
            )
        except Exception as e:
            logger.error(f"Failed to extract thumbnail: {e}")
            return False
    
    # COMMAND BUILDERS

    def build_compress_command(
        self,
        input_path: str,
        output_path: str,
        quality: int = 23,
        preset: str = 'medium',
        audio_bitrate: str = '128k'
    ) -> List[str]:
        """
        Build FFmpeg command for video compression.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            quality: CRF quality (18-28, lower is better)
            preset: Encoding preset (ultrafast, fast, medium, slow)
            audio_bitrate: Audio bitrate
            
        Returns:
            List of command arguments
        """
        return [
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(quality),
            '-preset', preset,
            '-c:a', 'aac',
            '-b:a', audio_bitrate,
            '-movflags', '+faststart',
            '-y', # Overwrite output
            output_path
        ]
    
    def build_convert_command(
        self,
        input_path: str,
        output_path: str,
        video_codec: Optional[str] = None,
        audio_codec: Optional[str] = None,
        quality: int = 23,
    ) -> List[str]:
        """
        Build FFmpeg command for format conversion.
        
        Args:
            input_path: Path to input file
            output_path: Path for output file
            video_codec: Video codec to use (auto-detected if None)
            audio_codec: Audio codec to use (auto-detected if None)
            quality: Quality setting (CRF for x264/x265)
            
        Returns:
            List of command arguments
        """
        # Determine output format from extension
        ext = os.path.splitext(output_path)[1].lower()

        # Default codecs based on output format
        if video_codec is None:
            video_codec = {
                '.mp4': 'libx264',
                '.mkv': 'libx264',
                '.webm': 'libvpx-vp9',
                '.avi': 'libx264',
                '.mov': 'libx264',
            }.get(ext, 'libx264')
        
        if audio_codec is None:
            audio_codec = {
                '.mp4': 'aac',
                '.webm': 'libopus',
                '.mov': 'aac',
                '.mkv': 'aac',
                '.avi': 'mp3',
            }.get(ext, 'aac')
        
        args = [
            '-i', input_path,
            '-c:v', video_codec,
        ]

        # Add quality setting based on codec 
        if video_codec in ('libx264', 'libx265'):
            args.extend(['-crf', str(quality)])
        elif video_codec == 'libvpx-vp9':
            args.extend(['-b:v', '0', '-crf', str(quality)])
        
        args.extend([
            '-c:a', audio_codec,
            '-y', # Overwrite output
            output_path
        ])

        return args
    

    def build_audio_extract_command(
        self,
        input_path: str,
        output_path: str,
        audio_codec: Optional[str] = None,
        bitrate: str = '192k'
    ) -> List[str]:
        """
        Build FFmpeg command to extract audio from video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output audio
            audio_codec: Audio codec (auto-detected if None)
            bitrate: Audio bitrate
            
        Returns:
            List of command arguments
        """
        ext = os.path.splitext(output_path)[1].lower()

        if audio_codec is None:
            audio_codec = {
                '.mp3': 'libmp3lame',
                '.aac': 'aac',
                '.m4a': 'aac',
                '.ogg': 'libvorbis',
                '.wav': 'pcm_s16le',
                '.flac': 'flac',
            }.get(ext, 'aac')
        
        return [
            '-i', input_path,
            '-vn',  # No video
            '-c:a', audio_codec,
            '-b:a', bitrate,
            '-y',
            output_path,
        ]
    

    def build_audio_convert_command(
        self,
        input_path: str,
        output_path: str,
        audio_codec: Optional[str] = None,
        bitrate: str = '192k',
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> List[str]:
        """
        Build FFmpeg command for audio format conversion.
        
        Args:
            input_path: Path to input audio
            output_path: Path for output audio
            audio_codec: Audio codec (auto-detected if None)
            bitrate: Audio bitrate
            sample_rate: Output sample rate (e.g., 44100, 48000)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Returns:
            List of command arguments
        """
        ext = os.path.splitext(output_path)[1].lower()
        
        if audio_codec is None:
            audio_codec = {
                '.mp3': 'libmp3lame',
                '.aac': 'aac',
                '.m4a': 'aac',
                '.ogg': 'libvorbis',
                '.wav': 'pcm_s16le',
                '.flac': 'flac',
            }.get(ext, 'aac')
        
        args = [
            '-i', input_path,
            '-c:a', audio_codec,
            '-b:a', bitrate,
        ]
        
        if sample_rate:
            args.extend(['-ar', str(sample_rate)])
        
        if channels:
            args.extend(['-ac', str(channels)])
        
        args.extend(['-y', output_path])
        
        return args
    
    # UTILITY METHODS

    def get_ffmpeg_version(self) -> str:
        """
        Get FFmpeg version string.
        
        Returns:
            Version string
        """
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Extract version from first line
            first_line = result.stdout.split('\n')[0]
            return first_line.strip()
        
        except Exception as e:
            return f"Unknown (error: {e})"
    

    def is_valid_media_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is a valid media file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        
        cmd = [
            self.ffprobe_path,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False, f"Invalid media file: {result.stderr.strip()}"
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "File analysis timed out"
        except Exception as e:
            return False, f"Error analyzing file: {e}"


# Module-level convenience instance
_ffmpeg_wrapper: Optional[FFmpegWrapper] = None


def get_ffmpeg_wrapper() -> FFmpegWrapper:
    """
    Get the global FFmpeg wrapper instance.
    
    Returns:
        FFmpegWrapper instance
    """
    global _ffmpeg_wrapper
    if _ffmpeg_wrapper is None:
        _ffmpeg_wrapper = FFmpegWrapper()
    return _ffmpeg_wrapper