"""
Unit tests for processor utilities.

Tests cover:
- FFmpeg wrapper
- Progress parser and throttled callbacks
- File validation utilities
- Base processor class
"""

import os
import tempfile
import time
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest


# PROGRESS PARSER TESTS

class TestProgressParser:
    """Tests for ProgressParser class."""
    
    def test_parse_time_from_stderr_format(self):
        """Test parsing time from standard FFmpeg stderr format."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        line = "frame=  100 fps= 25 q=28.0 size=    1024kB time=00:00:10.00 bitrate=1000.0kbits/s speed=2.5x"
        result = parser.parse_line(line)
        
        assert result is not None
        assert result.current_time == 10.0
        assert result.percent == 10
    
    def test_parse_time_from_progress_pipe_format(self):
        """Test parsing time from progress pipe format."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=60.0)
        
        line = "out_time_ms=30000000"
        result = parser.parse_line(line)
        
        assert result is not None
        assert result.current_time == 30.0
        assert result.percent == 50
    
    def test_parse_time_out_time_format(self):
        """Test parsing out_time format."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=3600.0)
        
        line = "out_time=00:30:00.00"
        result = parser.parse_line(line)
        
        assert result is not None
        assert result.current_time == 1800.0
        assert result.percent == 50
    
    def test_parse_additional_fields(self):
        """Test parsing additional FFmpeg fields."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        line = "frame=  500 fps= 30.5 q=28.0 size=    5120kB time=00:00:20.00 bitrate=2000.0kbits/s speed=1.5x"
        result = parser.parse_line(line)
        
        assert result is not None
        assert result.frame == 500
        assert result.fps == 30.5
        assert result.bitrate == "2000.0kbits/s"
        assert result.speed == 1.5
    
    def test_parse_empty_line(self):
        """Test parsing empty line returns None."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        assert parser.parse_line("") is None
        assert parser.parse_line("   ") is None
    
    def test_parse_irrelevant_line(self):
        """Test parsing line without time info returns None."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        result = parser.parse_line("Some random FFmpeg output")
        assert result is None
    
    def test_progress_capped_at_99(self):
        """Test that progress doesn't exceed 99% until complete."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        # Even at 99.9 seconds, should be 99%
        line = "time=00:01:39.90"
        result = parser.parse_line(line)
        
        assert result.percent == 99
    
    def test_mark_complete(self):
        """Test marking operation as complete."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        result = parser.mark_complete()
        
        assert result.percent == 100
        assert result.current_time == 100.0
        assert result.eta_seconds == 0
    
    def test_get_current_progress(self):
        """Test getting current progress without parsing."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        # Parse some progress first
        parser.parse_line("time=00:00:50.00")
        
        # Get current progress
        result = parser.get_current_progress()
        
        assert result.current_time == 50.0
        assert result.percent == 50
    
    def test_eta_calculation(self):
        """Test ETA calculation with speed info."""
        from apps.processors.utils.track_progress import ProgressParser
        
        parser = ProgressParser(total_duration=100.0)
        
        line = "time=00:00:50.00 speed=2.0x"
        result = parser.parse_line(line)
        
        # 50 seconds remaining at 2x speed = 25 seconds ETA
        assert result is not None
        assert result.eta_seconds is not None
        assert 24 <= result.eta_seconds <= 26  # Allow small variance
    
    def test_progress_info_to_dict(self):
        """Test ProgressInfo to_dict method."""
        from apps.processors.utils.track_progress import ProgressInfo
        
        info = ProgressInfo(
            percent=50,
            current_time=30.0,
            total_duration=60.0,
            speed=1.5,
            eta_seconds=20.0,
            frame=100,
            fps=30.0,
            bitrate="1000kbits/s",
            size=1024,
        )
        
        result = info.to_dict()
        
        assert result["percent"] == 50
        assert result["current_time"] == 30.0
        assert result["speed"] == 1.5


class TestThrottledProgressCallback:
    """Tests for ThrottledProgressCallback class."""
    
    def test_immediate_call_for_zero_percent(self):
        """Test that 0% is always called immediately."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock()
        throttled = ThrottledProgressCallback(callback_mock, min_interval=5.0)
        
        throttled(0, None)
        
        callback_mock.assert_called_once_with(0, None)
    
    def test_immediate_call_for_100_percent(self):
        """Test that 100% is always called immediately."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock()
        throttled = ThrottledProgressCallback(callback_mock, min_interval=5.0)
        
        throttled(100, 0)
        
        callback_mock.assert_called_once_with(100, 0)
    
    def test_throttle_rapid_calls(self):
        """Test that rapid calls are throttled."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock()
        throttled = ThrottledProgressCallback(
            callback_mock,
            min_interval=1.0,
            min_percent_change=10
        )
        
        # Initial call at 0%
        throttled(0, None)
        
        # Rapid calls with small changes
        throttled(1, None)
        throttled(2, None)
        throttled(3, None)
        
        # Should only have called for 0%
        assert callback_mock.call_count == 1
    
    def test_allow_significant_percent_change(self):
        """Test that significant percent changes bypass throttle."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock()
        throttled = ThrottledProgressCallback(
            callback_mock,
            min_interval=10.0,  # Long interval
            min_percent_change=5
        )
        
        throttled(0, None)
        throttled(10, None)  # 10% change should bypass throttle
        
        assert callback_mock.call_count == 2
    
    def test_force_update(self):
        """Test force_update bypasses throttling."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock()
        throttled = ThrottledProgressCallback(callback_mock, min_interval=100.0)
        
        throttled(0, None)
        throttled(1, None)  # Throttled
        throttled.force_update(1, None)  # Forced
        
        assert callback_mock.call_count == 2
    
    def test_callback_exception_handled(self):
        """Test that exceptions in callback are handled gracefully."""
        from apps.processors.utils.track_progress import ThrottledProgressCallback
        
        callback_mock = MagicMock(side_effect=Exception("Callback error"))
        throttled = ThrottledProgressCallback(callback_mock)
        
        # Should not raise
        throttled(0, None)


class TestSimpleProgressTracker:
    """Tests for SimpleProgressTracker class."""
    
    def test_set_progress(self):
        """Test setting progress directly."""
        from apps.processors.utils.track_progress import ProgressTracker
        
        operation_id = uuid4()
        
        with patch('apps.operations.services.operations_manager.OperationsManager') as mock_manager:
            tracker = ProgressTracker(operation_id, min_interval=0)
            tracker.set_progress(50)
            
            mock_manager.update_operation_progress.assert_called_with(operation_id, 50)
    
    def test_phase_based_progress(self):
        """Test progress tracking with phases."""
        from apps.processors.utils.track_progress import ProgressTracker
        
        operation_id = uuid4()
        phases = ['load', 'process', 'save']
        
        with patch('apps.operations.services.operations_manager.OperationsManager') as mock_manager:
            tracker = ProgressTracker(operation_id, phases=phases, min_interval=0)
            
            # Start first phase
            tracker.start_phase('load')
            tracker.update_phase_progress(100)
            
            # Should be ~33% (first phase complete)
            last_call = mock_manager.update_operation_progress.call_args
            assert last_call[0][1] >= 30  # At least 30%
    
    def test_complete(self):
        """Test completing the tracker."""
        from apps.processors.utils.track_progress import ProgressTracker
        
        operation_id = uuid4()
        
        with patch('apps.operations.services.operations_manager.OperationsManager') as mock_manager:
            tracker = ProgressTracker(operation_id, min_interval=0)
            tracker.complete()
            
            mock_manager.update_operation_progress.assert_called_with(operation_id, 100)


# VALIDATION TESTS

class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_boolean_conversion(self):
        """Test boolean conversion of ValidationResult."""
        from apps.processors.utils.validation import ValidationResult
        
        valid = ValidationResult(
            is_valid=True,
            error_message=None,
            warnings=[],
            metadata={}
        )
        invalid = ValidationResult(
            is_valid=False,
            error_message="Error",
            warnings=[],
            metadata={}
        )
        
        assert bool(valid) is True
        assert bool(invalid) is False
    
    def test_to_dict(self):
        """Test to_dict method."""
        from apps.processors.utils.validation import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            error_message=None,
            warnings=["Warning 1"],
            metadata={"key": "value"}
        )
        
        data = result.to_dict()
        
        assert data["is_valid"] is True
        assert data["warnings"] == ["Warning 1"]
        assert data["metadata"]["key"] == "value"


class TestFileValidation:
    """Tests for file validation functions."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp()
        os.write(fd, b"Test content")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)
    
    def test_validate_video_file_not_found(self):
        """Test validation of non-existent file."""
        from apps.processors.utils.validation import validate_video_file
        
        result = validate_video_file("/nonexistent/path/video.mp4")
        
        assert result.is_valid is False
        assert "not found" in result.error_message.lower()
    
    def test_validate_audio_file_not_found(self):
        """Test validation of non-existent audio file."""
        from apps.processors.utils.validation import validate_audio_file
        
        result = validate_audio_file("/nonexistent/path/audio.mp3")
        
        assert result.is_valid is False
        assert "not found" in result.error_message.lower()
    
    def test_validate_image_file_not_found(self):
        """Test validation of non-existent image file."""
        from apps.processors.utils.validation import validate_image_file
        
        result = validate_image_file("/nonexistent/path/image.jpg")
        
        assert result.is_valid is False
        assert "not found" in result.error_message.lower()
    
    def test_validate_empty_file(self, temp_file):
        """Test validation of empty file."""
        from apps.processors.utils.validation import validate_video_file
        
        # Truncate file to empty
        with open(temp_file, 'w') as f:
            f.truncate(0)
        
        result = validate_video_file(temp_file)
        
        assert result.is_valid is False
        assert "empty" in result.error_message.lower()
    
    def test_validate_media_file_dispatcher(self):
        """Test that validate_media_file dispatches correctly."""
        from apps.processors.utils.validation import validate_media_file
        
        result = validate_media_file("/fake/path.mp4", "unknown_type")
        
        assert result.is_valid is False
        assert "unknown media type" in result.error_message.lower()
    
    def test_quick_validate_file_not_found(self):
        """Test quick validation of non-existent file."""
        from apps.processors.utils.validation import quick_validate_file
        
        is_valid, media_type, error = quick_validate_file("/nonexistent/file.mp4")
        
        assert is_valid is False
        assert media_type == 'unknown'
        assert error is not None
    
    def test_validate_output_path_success(self, tmp_path):
        """Test validating a valid output path."""
        from apps.processors.utils.validation import validate_output_path
        
        output_path = str(tmp_path / "subdir" / "output.mp4")
        result = validate_output_path(output_path, create_directory=True)
        
        assert result.is_valid is True
        assert os.path.exists(os.path.dirname(output_path))
    
    def test_validate_output_path_no_create(self):
        """Test validating output path without creating directory."""
        from apps.processors.utils.validation import validate_output_path
        
        result = validate_output_path(
            "/nonexistent/directory/output.mp4",
            create_directory=False
        )
        
        assert result.is_valid is False
        assert "does not exist" in result.error_message


class TestValidationHelpers:
    """Tests for validation helper functions."""
    
    def test_is_video_file_with_mock(self):
        """Test is_video_file with mocked validation."""
        from apps.processors.utils.validation import is_video_file, ValidationResult
        
        with patch('apps.processors.utils.validation.validate_video_file') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                error_message=None,
                warnings=[],
                metadata={}
            )
            
            assert is_video_file("/fake/video.mp4") is True
    
    def test_is_audio_file_with_mock(self):
        """Test is_audio_file with mocked validation."""
        from apps.processors.utils.validation import is_audio_file, ValidationResult
        
        with patch('apps.processors.utils.validation.validate_audio_file') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=False,
                error_message="Invalid",
                warnings=[],
                metadata={}
            )
            
            assert is_audio_file("/fake/audio.mp3") is False
    
    def test_is_image_file_with_mock(self):
        """Test is_image_file with mocked validation."""
        from apps.processors.utils.validation import is_image_file, ValidationResult
        
        with patch('apps.processors.utils.validation.validate_image_file') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                error_message=None,
                warnings=[],
                metadata={}
            )
            
            assert is_image_file("/fake/image.jpg") is True


# BASE PROCESSOR TESTS

class TestErrorCategory:
    """Tests for ErrorCategory enum."""
    
    def test_error_categories_exist(self):
        """Test that all expected error categories exist."""
        from apps.processors.base_processor import ErrorCategory
        
        assert hasattr(ErrorCategory, 'TEMPORARY')
        assert hasattr(ErrorCategory, 'RESOURCE')
        assert hasattr(ErrorCategory, 'TIMEOUT')
        assert hasattr(ErrorCategory, 'INVALID_INPUT')
        assert hasattr(ErrorCategory, 'CODEC_ERROR')
        assert hasattr(ErrorCategory, 'UNKNOWN')


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""
    
    def test_to_dict(self):
        """Test ProcessingResult to_dict method."""
        from apps.processors.base_processor import ProcessingResult
        
        result = ProcessingResult(
            success=True,
            output_path="/output/file.mp4",
            output_filename="file.mp4",
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=10.5,
            metadata={"key": "value"},
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["output_path"] == "/output/file.mp4"
        assert data["processing_time_seconds"] == 10.5
    
    def test_to_dict_with_error(self):
        """Test ProcessingResult to_dict with error."""
        from apps.processors.base_processor import ProcessingResult, ErrorCategory
        
        result = ProcessingResult(
            success=False,
            output_path=None,
            output_filename=None,
            error_message="Processing failed",
            error_category=ErrorCategory.TIMEOUT,
            is_retryable=True,
            processing_time_seconds=30.0,
            metadata={},
        )
        
        data = result.to_dict()
        
        assert data["success"] is False
        assert data["error_category"] == "timeout"
        assert data["is_retryable"] is True


class TestBaseProcessor:
    """Tests for BaseProcessor class."""
    
    @pytest.fixture
    def temp_input_file(self, tmp_path):
        """Create a temporary input file."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test content")
        return str(input_file)
    
    @pytest.fixture
    def mock_processor(self, temp_input_file):
        """Create a mock processor for testing."""
        from apps.processors.base_processor import BaseProcessor, ProcessingResult
        
        class MockProcessor(BaseProcessor):
            @property
            def operation_name(self) -> str:
                return "mock_operation"
            
            def process_operation(self) -> ProcessingResult:
                return self.create_success_result(
                    output_path="/output/file.txt",
                    output_filename="file.txt",
                )
        
        return MockProcessor(
            operation_id=uuid4(),
            session_key="test-session",
            input_path=temp_input_file,
            parameters={},
            progress_callback=None,
        )
    
    def test_setup_temp_directory(self, mock_processor, tmp_path):
        """Test temp directory setup."""
        with patch.object(mock_processor, 'setup_temp_directory') as mock_setup:
            mock_setup.return_value = str(tmp_path / "temp")
            
            temp_dir = mock_processor.setup_temp_directory()
            
            assert temp_dir is not None
    
    def test_cleanup_temp_directory(self, mock_processor, tmp_path):
        """Test temp directory cleanup."""
        temp_dir = tmp_path / "processor_temp"
        temp_dir.mkdir()
        (temp_dir / "temp_file.txt").write_text("temp")
        
        mock_processor.temp_dir = str(temp_dir)
        result = mock_processor.cleanup_temp_directory()
        
        assert result is True
        assert not temp_dir.exists()
    
    def test_cleanup_nonexistent_directory(self, mock_processor):
        """Test cleanup of non-existent directory."""
        mock_processor.temp_dir = "/nonexistent/directory"
        result = mock_processor.cleanup_temp_directory()
        
        assert result is True
    
    def test_get_temp_file_path(self, mock_processor, tmp_path):
        """Test temp file path generation."""
        mock_processor.temp_dir = str(tmp_path)
        
        path = mock_processor.get_temp_file_path("output.mp4")
        
        assert "output.mp4" in path
        assert str(tmp_path) in path
    
    def test_classify_error_file_not_found(self, mock_processor):
        """Test error classification for FileNotFoundError."""
        from apps.processors.base_processor import ErrorCategory
        
        error = FileNotFoundError("File not found")
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.NOT_FOUND
        assert retryable is False
    
    def test_classify_error_permission(self, mock_processor):
        """Test error classification for PermissionError."""
        from apps.processors.base_processor import ErrorCategory
        
        error = PermissionError("Permission denied")
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.PERMISSION
        assert retryable is False
    
    def test_classify_error_memory(self, mock_processor):
        """Test error classification for MemoryError."""
        from apps.processors.base_processor import ErrorCategory
        
        error = MemoryError()
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.RESOURCE
        assert retryable is True
    
    def test_classify_error_timeout(self, mock_processor):
        """Test error classification for TimeoutError."""
        from apps.processors.base_processor import ErrorCategory
        
        error = TimeoutError("Operation timed out")
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.TIMEOUT
        assert retryable is True
    
    def test_classify_error_disk_space(self, mock_processor):
        """Test error classification for disk space errors."""
        from apps.processors.base_processor import ErrorCategory
        
        error = Exception("No space left on device")
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.RESOURCE
        assert retryable is True
    
    def test_classify_error_codec(self, mock_processor):
        """Test error classification for codec errors."""
        from apps.processors.base_processor import ErrorCategory
        
        error = Exception("Unsupported codec h265")
        category, retryable = mock_processor.classify_error(error)
        
        assert category == ErrorCategory.CODEC_ERROR
        assert retryable is False
    
    def test_create_success_result(self, mock_processor):
        """Test creating a success result."""
        result = mock_processor.create_success_result(
            output_path="/output/file.mp4",
            output_filename="file.mp4",
            metadata={"duration": 10.0}
        )
        
        assert result.success is True
        assert result.output_path == "/output/file.mp4"
        assert result.metadata["duration"] == 10.0
    
    def test_get_input_filename(self, mock_processor):
        """Test getting input filename."""
        # Assuming input_path is set to something like "/path/to/input.txt"
        filename = mock_processor.get_input_filename()
        
        assert filename == "input.txt"
    
    def test_get_input_extension(self, mock_processor):
        """Test getting input extension."""
        extension = mock_processor.get_input_extension()
        
        assert extension == "txt"
    
    def test_generate_output_filename(self, mock_processor):
        """Test generating output filename."""
        filename = mock_processor.generate_output_filename(
            suffix="_processed",
            extension="mp4"
        )
        
        assert "_processed" in filename
        assert filename.endswith(".mp4")
    
    def test_validate_output_file_exists(self, mock_processor, tmp_path):
        """Test validating existing output file."""
        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"Video content")
        
        result = mock_processor.validate_output_file_creation(str(output_file))
        
        assert result is True
    
    def test_validate_output_file_not_exists(self, mock_processor):
        """Test validating non-existent output file."""
        result = mock_processor.validate_output_file_creation("/nonexistent/output.mp4")
        
        assert result is False
    
    def test_validate_output_file_empty(self, mock_processor, tmp_path):
        """Test validating empty output file."""
        output_file = tmp_path / "output.mp4"
        output_file.write_bytes(b"")
        
        result = mock_processor.validate_output_file_creation(str(output_file))
        
        assert result is False
    
    def test_execute_with_missing_input(self, mock_processor):
        """Test execute with missing input file."""
        mock_processor.input_path = "/nonexistent/input.mp4"
        
        result = mock_processor.execute_operation()
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
    
    def test_progress_reporting(self, mock_processor):
        """Test progress reporting."""
        callback_mock = MagicMock()
        mock_processor.progress_callback = callback_mock
        
        mock_processor.update_progress(50, 10.0)
        
        callback_mock.assert_called_with(50, 10.0)
    
    def test_logging_methods(self, mock_processor):
        """Test logging methods don't raise exceptions."""
        
        mock_processor.log_debug("Debug message")
        mock_processor.log_info("Info message")
        mock_processor.log_warning("Warning message")
        mock_processor.log_error("Error message")
        
        assert True
    
    def test_logging_methods_with_mock(self, mock_processor):
        """Test logging methods call the logger correctly."""
        from unittest.mock import patch
        
        with patch('apps.processors.base_processor.logger') as mock_logger:
            mock_processor.log_debug("Debug test")
            mock_processor.log_info("Info test")
            mock_processor.log_warning("Warning test")
            mock_processor.log_error("Error test")
            
            # Verify logger methods were called
            assert mock_logger.debug.called
            assert mock_logger.info.called
            assert mock_logger.warning.called
            assert mock_logger.error.called


class TestVideoProcessor:
    """Tests for VideoProcessor base class."""
    
    def test_default_timeout(self):
        """Test that VideoProcessor has longer default timeout."""
        from apps.processors.base_processor import VideoProcessor
        
        assert VideoProcessor.DEFAULT_TIMEOUT == 1800  # 30 minutes


class TestImageProcessor:
    """Tests for ImageProcessor base class."""
    
    def test_default_timeout(self):
        """Test that ImageProcessor has shorter default timeout."""
        from apps.processors.base_processor import ImageProcessor
        
        assert ImageProcessor.DEFAULT_TIMEOUT == 60  # 1 minute


class TestAudioProcessor:
    """Tests for AudioProcessor base class."""
    
    def test_default_timeout(self):
        """Test that AudioProcessor has medium default timeout."""
        from apps.processors.base_processor import AudioProcessor
        
        assert AudioProcessor.DEFAULT_TIMEOUT == 300  # 5 minutes


# FFMPEG WRAPPER TESTS (Mocked)

class TestFFmpegWrapperMocked:
    """Tests for FFmpegWrapper with mocked subprocess calls."""
    
    def test_get_ffmpeg_version_success(self):
        """Test getting FFmpeg version."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="ffmpeg version 5.1.2 Copyright (c) 2000-2022",
                stderr=""
            )
            
            # Need to mock both ffmpeg and ffprobe verification
            wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
            wrapper.ffmpeg_path = 'ffmpeg'
            wrapper.ffprobe_path = 'ffprobe'
            
            version = wrapper.get_ffmpeg_version()
            
            assert "ffmpeg" in version.lower()
    
    def test_build_compress_command(self):
        """Test building a compress command."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        # Create wrapper without verification
        wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
        wrapper.ffmpeg_path = 'ffmpeg'
        wrapper.ffprobe_path = 'ffprobe'
        
        cmd = wrapper.build_compress_command(
            input_path="/input/video.mp4",
            output_path="/output/video.mp4",
            quality=23,
            preset='medium'
        )
        
        assert '-i' in cmd
        assert '/input/video.mp4' in cmd
        assert '-crf' in cmd
        assert '23' in cmd
        assert '-preset' in cmd
        assert 'medium' in cmd
        assert 'libx264' in cmd
    
    def test_build_convert_command_mp4(self):
        """Test building a convert command for MP4."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
        wrapper.ffmpeg_path = 'ffmpeg'
        wrapper.ffprobe_path = 'ffprobe'
        
        cmd = wrapper.build_convert_command(
            input_path="/input/video.avi",
            output_path="/output/video.mp4",
            quality=23
        )
        
        assert 'libx264' in cmd
        assert 'aac' in cmd
    
    def test_build_convert_command_webm(self):
        """Test building a convert command for WebM."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
        wrapper.ffmpeg_path = 'ffmpeg'
        wrapper.ffprobe_path = 'ffprobe'
        
        cmd = wrapper.build_convert_command(
            input_path="/input/video.mp4",
            output_path="/output/video.webm",
            quality=23
        )
        
        assert 'libvpx-vp9' in cmd
        assert 'libopus' in cmd
    
    def test_build_audio_extract_command(self):
        """Test building an audio extract command."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
        wrapper.ffmpeg_path = 'ffmpeg'
        wrapper.ffprobe_path = 'ffprobe'
        
        cmd = wrapper.build_audio_extract_command(
            input_path="/input/video.mp4",
            output_path="/output/audio.mp3",
            bitrate='192k'
        )
        
        assert '-vn' in cmd  # No video
        assert 'libmp3lame' in cmd
        assert '192k' in cmd
    
    def test_build_audio_convert_command(self):
        """Test building an audio convert command."""
        from apps.processors.utils.ffmpeg import FFmpegWrapper
        
        wrapper = FFmpegWrapper.__new__(FFmpegWrapper)
        wrapper.ffmpeg_path = 'ffmpeg'
        wrapper.ffprobe_path = 'ffprobe'
        
        cmd = wrapper.build_audio_convert_command(
            input_path="/input/audio.wav",
            output_path="/output/audio.mp3",
            bitrate='192k',
            sample_rate=44100,
            channels=2
        )
        
        assert 'libmp3lame' in cmd
        assert '-ar' in cmd
        assert '44100' in cmd
        assert '-ac' in cmd
        assert '2' in cmd


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""
    
    def test_to_dict(self):
        """Test VideoInfo to_dict method."""
        from apps.processors.utils.ffmpeg import VideoInfo
        
        info = VideoInfo(
            duration=120.5,
            width=1920,
            height=1080,
            codec='h264',
            bitrate=5000000,
            fps=30.0,
            frame_count=3615,
            pixel_format='yuv420p',
            has_audio=True,
            file_size=75000000,
            format_name='mp4',
        )
        
        data = info.to_dict()
        
        assert data['duration'] == 120.5
        assert data['width'] == 1920
        assert data['codec'] == 'h264'
        assert data['has_audio'] is True


class TestAudioInfo:
    """Tests for AudioInfo dataclass."""
    
    def test_to_dict(self):
        """Test AudioInfo to_dict method."""
        from apps.processors.utils.ffmpeg import AudioInfo
        
        info = AudioInfo(
            duration=180.0,
            codec='mp3',
            bitrate=192000,
            channels=2,
            sample_rate=44100,
            file_size=4320000,
            format_name='mp3',
        )
        
        data = info.to_dict()
        
        assert data['duration'] == 180.0
        assert data['codec'] == 'mp3'
        assert data['channels'] == 2


class TestFFmpegResult:
    """Tests for FFmpegResult dataclass."""
    
    def test_to_dict(self):
        """Test FFmpegResult to_dict method."""
        from apps.processors.utils.ffmpeg import FFmpegResult
        
        result = FFmpegResult(
            success=True,
            return_code=0,
            stdout="",
            stderr="",
            output_path="/output/video.mp4",
            duration_seconds=10.5,
        )
        
        data = result.to_dict()
        
        assert data['success'] is True
        assert data['return_code'] == 0
        assert data['output_path'] == "/output/video.mp4"


# ESTIMATE PROCESSING TIME TESTS

class TestEstimateProcessingTime:
    """Tests for estimate_processing_time function."""
    
    def test_video_compress_estimate(self):
        """Test processing time estimate for video compression."""
        from apps.processors.utils.track_progress import estimate_processing_time
        # 2 minute video
        estimate = estimate_processing_time(
            duration_seconds=120,
            operation_type='video_compress'
        )
        
        # Should be less than real-time (multiplier is 0.5)
        assert estimate < 120
        assert estimate > 0
    
    def test_image_resize_estimate(self):
        """Test processing time estimate for image resize."""
        from apps.processors.utils.track_progress import estimate_processing_time
        
        estimate = estimate_processing_time(
            duration_seconds=0,  # Images don't have duration
            operation_type='image_resize'
        )
        
        # Should be at least 1 second (minimum)
        assert estimate >= 1.0
    
    def test_estimate_with_file_size(self):
        """Test that file size affects estimate."""
        from apps.processors.utils.track_progress import estimate_processing_time
        
        # Without file size
        estimate1 = estimate_processing_time(
            duration_seconds=60,
            operation_type='video_compress'
        )
        
        # With large file size
        estimate2 = estimate_processing_time(
            duration_seconds=60,
            operation_type='video_compress',
            file_size_bytes=500 * 1024 * 1024  # 500MB
        )
        
        # Larger file should have higher estimate
        assert estimate2 > estimate1
    
    def test_estimate_capped_at_max(self):
        """Test that estimate is capped at 1 hour."""
        from apps.processors.utils.track_progress import estimate_processing_time
        
        # Very long video
        estimate = estimate_processing_time(
            duration_seconds=100000,
            operation_type='video_compress'
        )
        
        assert estimate <= 3600  # 1 hour max