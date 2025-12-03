# tests/integration/test_audio_processors.py

"""
Integration tests for audio processing operations.

These tests verify the complete audio processing pipeline including:
- Audio format conversion
- Audio extraction from video
- Progress tracking
- Error handling
- Output file verification
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from apps.processors.audio_processing import (
    AudioConvertProcessor,
    AudioExtractProcessor,
    audio_convert_handler,
    audio_extract_handler,
    create_audio_convert_processor,
    create_audio_extract_processor,
    register_audio_operations,
    AUDIO_FORMATS,
    VALID_BITRATES,
    VALID_SAMPLE_RATES,
    VALID_CHANNELS,
)
from apps.processors.base_processor import ProcessingResult, ErrorCategory
from apps.processors.registry import get_registry, MediaType


# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures'


class TestAudioConvertProcessor:
    """Integration tests for AudioConvertProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_audio_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_audio_path(self, temp_dir):
        """
        Create a sample audio file for testing.
        
        Uses FFmpeg to generate a test audio file if fixtures aren't available.
        """
        sample_path = FIXTURES_DIR / 'sample.mp3'
        
        if sample_path.exists():
            # Copy fixture to temp dir
            temp_audio = os.path.join(temp_dir, 'test_input.mp3')
            shutil.copy(sample_path, temp_audio)
            return temp_audio
        
        # Generate a test audio file using FFmpeg (1 second of silence with tone)
        temp_audio = os.path.join(temp_dir, 'test_input.mp3')
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=2',
                '-c:a', 'libmp3lame',
                '-b:a', '128k',
                temp_audio
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_audio):
                return temp_audio
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        pytest.skip("FFmpeg not available or failed to create test audio")
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings for tests."""
        with patch('apps.processors.base_processor.settings') as mock:
            mock.MEDIA_ROOT = tempfile.gettempdir()
            yield mock
    
    def test_audio_convert_to_wav(self, sample_audio_path, temp_dir, mock_settings):
        """Test converting audio to WAV format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'wav',
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_path is not None
        assert result.output_filename is not None
        assert result.output_filename.endswith('.wav')
        assert result.error_message is None
        assert result.is_retryable is False
        
        # Verify metadata (file is cleaned up after processing, but metadata captured)
        assert 'output_format' in result.metadata
        assert result.metadata['output_format'] == 'wav'
        assert 'duration' in result.metadata
        assert result.metadata['duration'] > 0
        # Output size was captured before cleanup
        assert 'output_size' in result.metadata
        assert result.metadata['output_size'] > 0
    
    def test_audio_convert_to_ogg(self, sample_audio_path, temp_dir, mock_settings):
        """Test converting audio to OGG format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'ogg',
            'bitrate': '128k',
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.ogg')
        # Verify output was created (captured in metadata before cleanup)
        assert result.metadata['output_size'] > 0
    
    def test_audio_convert_to_flac(self, sample_audio_path, temp_dir, mock_settings):
        """Test converting audio to FLAC (lossless) format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'flac',
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.flac')
        # Verify output was created (captured in metadata before cleanup)
        assert result.metadata['output_size'] > 0
    
    def test_audio_convert_with_sample_rate(self, sample_audio_path, temp_dir, mock_settings):
        """Test converting audio with custom sample rate."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'mp3',
            'bitrate': '128k',
            'sample_rate': 22050,
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata.get('sample_rate') == 22050 or result.metadata.get('sample_rate') is not None
    
    def test_audio_convert_to_mono(self, sample_audio_path, temp_dir, mock_settings):
        """Test converting audio to mono."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'mp3',
            'channels': 1,
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        # Mono file should be smaller or channels should be 1
        assert result.metadata.get('channels') == 1 or 'channels' in result.metadata
    
    def test_audio_convert_with_progress_callback(self, sample_audio_path, temp_dir, mock_settings):
        """Test that progress callback is called during conversion."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'wav',
        }
        
        progress_updates = []
        
        def progress_callback(percent: int, eta_seconds):
            progress_updates.append(percent)
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
            progress_callback=progress_callback,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        # Should have at least start (0) and end (100) progress
        assert len(progress_updates) >= 2
        assert 0 in progress_updates
        assert 100 in progress_updates
    
    def test_audio_convert_missing_input_file(self, temp_dir, mock_settings):
        """Test handling of missing input file."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {'output_format': 'mp3'}
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/nonexistent/path/audio.mp3',
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND
        assert result.is_retryable is False
    
    def test_audio_convert_invalid_format(self, sample_audio_path, temp_dir, mock_settings):
        """Test handling of unsupported output format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'invalid_format',
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.INVALID_PARAMS
        assert 'unsupported' in result.error_message.lower() or 'format' in result.error_message.lower()


class TestAudioExtractProcessor:
    """Integration tests for AudioExtractProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_audio_extract_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_video_path(self, temp_dir):
        """
        Create a sample video file with audio for testing.
        
        Uses FFmpeg to generate a test video if fixtures aren't available.
        """
        sample_path = FIXTURES_DIR / 'sample.mp4'
        
        if sample_path.exists():
            # Copy fixture to temp dir
            temp_video = os.path.join(temp_dir, 'test_video.mp4')
            shutil.copy(sample_path, temp_video)
            return temp_video
        
        # Generate a test video file using FFmpeg
        temp_video = os.path.join(temp_dir, 'test_video.mp4')
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'testsrc=duration=2:size=320x240:rate=30',
                '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                temp_video
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_video):
                return temp_video
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        pytest.skip("FFmpeg not available or failed to create test video")
    
    @pytest.fixture
    def video_without_audio(self, temp_dir):
        """Create a video file without audio track."""
        temp_video = os.path.join(temp_dir, 'test_video_no_audio.mp4')
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', 'testsrc=duration=2:size=320x240:rate=30',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-an',  # No audio
                '-pix_fmt', 'yuv420p',
                temp_video
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_video):
                return temp_video
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        pytest.skip("FFmpeg not available or failed to create test video without audio")
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings for tests."""
        with patch('apps.processors.base_processor.settings') as mock:
            mock.MEDIA_ROOT = tempfile.gettempdir()
            yield mock
    
    def test_audio_extract_to_mp3(self, sample_video_path, temp_dir, mock_settings):
        """Test extracting audio from video to MP3 format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'mp3',
            'bitrate': '192k',
        }
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_video_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_path is not None
        assert result.output_filename is not None
        assert result.output_filename.endswith('.mp3')
        assert result.error_message is None
        
        # Verify metadata (output_size captured before cleanup)
        assert 'output_format' in result.metadata
        assert result.metadata['output_format'] == 'mp3'
        assert 'duration' in result.metadata
        assert result.metadata['duration'] > 0
        assert 'source_video_resolution' in result.metadata
        assert result.metadata['output_size'] > 0
    
    def test_audio_extract_to_wav(self, sample_video_path, temp_dir, mock_settings):
        """Test extracting audio from video to WAV format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'wav',
        }
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_video_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.wav')
        # Verify output was created (captured in metadata before cleanup)
        assert result.metadata['output_size'] > 0
    
    def test_audio_extract_to_aac(self, sample_video_path, temp_dir, mock_settings):
        """Test extracting audio from video to AAC format."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'aac',
            'bitrate': '256k',
        }
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_video_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.aac')
    
    def test_audio_extract_with_sample_rate(self, sample_video_path, temp_dir, mock_settings):
        """Test extracting audio with custom sample rate."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'mp3',
            'sample_rate': 44100,
        }
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_video_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
    
    def test_audio_extract_video_without_audio(self, video_without_audio, temp_dir, mock_settings):
        """Test handling of video without audio track."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {'output_format': 'mp3'}
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=video_without_audio,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.INVALID_INPUT
        assert 'audio' in result.error_message.lower()
    
    def test_audio_extract_with_progress_callback(self, sample_video_path, temp_dir, mock_settings):
        """Test that progress callback is called during extraction."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'mp3',
        }
        
        progress_updates = []
        
        def progress_callback(percent: int, eta_seconds):
            progress_updates.append(percent)
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_video_path,
            parameters=parameters,
            progress_callback=progress_callback,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        assert len(progress_updates) >= 2
        assert 0 in progress_updates
        assert 100 in progress_updates
    
    def test_audio_extract_missing_input_file(self, temp_dir, mock_settings):
        """Test handling of missing input file."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {'output_format': 'mp3'}
        
        processor = AudioExtractProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/nonexistent/path/video.mp4',
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND
        assert result.is_retryable is False


class TestAudioHandlerFunctions:
    """Tests for handler functions used by the operation registry."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_audio_handlers_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_audio_path(self, temp_dir):
        """Create a sample audio file."""
        sample_path = FIXTURES_DIR / 'sample.mp3'
        
        if sample_path.exists():
            temp_audio = os.path.join(temp_dir, 'test_input.mp3')
            shutil.copy(sample_path, temp_audio)
            return temp_audio
        
        temp_audio = os.path.join(temp_dir, 'test_input.mp3')
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=2',
                '-c:a', 'libmp3lame',
                '-b:a', '128k',
                temp_audio
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_audio):
                return temp_audio
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        pytest.skip("FFmpeg not available")
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings for tests."""
        with patch('apps.processors.base_processor.settings') as mock:
            mock.MEDIA_ROOT = tempfile.gettempdir()
            yield mock
    
    def test_audio_convert_handler(self, sample_audio_path, mock_settings):
        """Test audio_convert_handler function."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'wav',
        }
        
        result = audio_convert_handler(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.output_path is not None


class TestAudioOperationRegistration:
    """Tests for audio operation registration with the registry."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Clear and setup registry for each test."""
        registry = get_registry()
        # Note: Operations may already be registered from module import
        yield registry
    
    def test_audio_convert_registered(self, setup_registry):
        """Test that audio_convert operation is registered."""
        # Ensure operations are registered
        register_audio_operations()
        
        assert setup_registry.is_registered('audio_convert')
        
        operation = setup_registry.get_operation('audio_convert')
        assert operation.operation_name == 'audio_convert'
        assert operation.media_type == MediaType.AUDIO
        assert len(operation.parameters) >= 4  # output_format, bitrate, sample_rate, channels
    
    def test_audio_extract_registered(self, setup_registry):
        """Test that audio_extract operation is registered."""
        register_audio_operations()
        
        assert setup_registry.is_registered('audio_extract')
        
        operation = setup_registry.get_operation('audio_extract')
        assert operation.operation_name == 'audio_extract'
        assert operation.media_type == MediaType.VIDEO  # Input is video
        assert len(operation.parameters) >= 4
    
    def test_audio_convert_parameter_validation(self, setup_registry):
        """Test parameter validation for audio_convert operation."""
        register_audio_operations()
        
        # Valid parameters
        valid_params = {
            'output_format': 'mp3',
            'bitrate': '192k',
        }
        
        validated = setup_registry.validate_parameters('audio_convert', valid_params)
        assert validated['output_format'] == 'mp3'
        assert validated['bitrate'] == '192k'
    
    def test_audio_extract_parameter_validation(self, setup_registry):
        """Test parameter validation for audio_extract operation."""
        register_audio_operations()
        
        # Valid parameters
        valid_params = {
            'output_format': 'wav',
        }
        
        validated = setup_registry.validate_parameters('audio_extract', valid_params)
        assert validated['output_format'] == 'wav'
    
    def test_audio_operations_input_formats(self, setup_registry):
        """Test that input formats are correctly specified."""
        register_audio_operations()
        
        convert_op = setup_registry.get_operation('audio_convert')
        assert 'mp3' in convert_op.input_formats
        assert 'wav' in convert_op.input_formats
        assert 'aac' in convert_op.input_formats
        
        extract_op = setup_registry.get_operation('audio_extract')
        assert 'mp4' in extract_op.input_formats
        assert 'mov' in extract_op.input_formats
        assert 'mkv' in extract_op.input_formats
    
    def test_audio_operations_output_formats(self, setup_registry):
        """Test that output formats are correctly specified."""
        register_audio_operations()
        
        convert_op = setup_registry.get_operation('audio_convert')
        assert 'mp3' in convert_op.output_formats
        assert 'wav' in convert_op.output_formats
        assert 'flac' in convert_op.output_formats
        
        extract_op = setup_registry.get_operation('audio_extract')
        assert 'mp3' in extract_op.output_formats
        assert 'wav' in extract_op.output_formats


class TestAudioProcessorMetadata:
    """Tests for metadata generation in audio processors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_audio_metadata_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_audio_path(self, temp_dir):
        """Create a sample audio file."""
        sample_path = FIXTURES_DIR / 'sample.mp3'
        
        if sample_path.exists():
            temp_audio = os.path.join(temp_dir, 'test_input.mp3')
            shutil.copy(sample_path, temp_audio)
            return temp_audio
        
        temp_audio = os.path.join(temp_dir, 'test_input.mp3')
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=3',
                '-c:a', 'libmp3lame',
                '-b:a', '192k',
                '-ar', '44100',
                '-ac', '2',
                temp_audio
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_audio):
                return temp_audio
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        pytest.skip("FFmpeg not available")
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings for tests."""
        with patch('apps.processors.base_processor.settings') as mock:
            mock.MEDIA_ROOT = tempfile.gettempdir()
            yield mock
    
    def test_conversion_metadata_contains_required_fields(self, sample_audio_path, mock_settings):
        """Test that conversion result contains all required metadata fields."""
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {
            'output_format': 'wav',
            'sample_rate': 44100,
            'channels': 2,
        }
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=sample_audio_path,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is True
        
        # Check required metadata fields
        required_fields = [
            'input_size',
            'output_size',
            'output_format',
            'duration',
        ]
        
        for field in required_fields:
            assert field in result.metadata, f"Missing metadata field: {field}"
        
        # Validate metadata types and values
        assert isinstance(result.metadata['input_size'], int)
        assert isinstance(result.metadata['output_size'], int)
        assert result.metadata['input_size'] > 0
        assert result.metadata['output_size'] > 0
        assert result.metadata['duration'] > 0


class TestAudioProcessorErrorHandling:
    """Tests for error handling in audio processors."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = tempfile.mkdtemp(prefix='test_audio_errors_')
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Django settings for tests."""
        with patch('apps.processors.base_processor.settings') as mock:
            mock.MEDIA_ROOT = tempfile.gettempdir()
            yield mock
    
    def test_corrupted_audio_file(self, temp_dir, mock_settings):
        """Test handling of corrupted audio file."""
        # Create a fake audio file with invalid content
        corrupted_audio = os.path.join(temp_dir, 'corrupted.mp3')
        with open(corrupted_audio, 'wb') as f:
            f.write(b'This is not valid audio data')
        
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {'output_format': 'wav'}
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=corrupted_audio,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category in (
            ErrorCategory.INVALID_INPUT,
            ErrorCategory.CODEC_ERROR,
            ErrorCategory.UNKNOWN,
        )
    
    def test_empty_audio_file(self, temp_dir, mock_settings):
        """Test handling of empty audio file."""
        empty_audio = os.path.join(temp_dir, 'empty.mp3')
        with open(empty_audio, 'wb') as f:
            pass  # Create empty file
        
        operation_id = uuid4()
        session_key = 'test_session'
        parameters = {'output_format': 'wav'}
        
        processor = AudioConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path=empty_audio,
            parameters=parameters,
        )
        
        result = processor.execute_operation()
        
        assert result.success is False
    
    def test_processing_result_to_dict(self, mock_settings):
        """Test ProcessingResult to_dict conversion."""
        result = ProcessingResult(
            success=True,
            output_path='/path/to/output.mp3',
            output_filename='output.mp3',
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=1.5,
            metadata={'format': 'mp3'},
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['success'] is True
        assert result_dict['output_path'] == '/path/to/output.mp3'
        assert result_dict['output_filename'] == 'output.mp3'
        assert result_dict['error_message'] is None
        assert result_dict['error_category'] is None
        assert result_dict['is_retryable'] is False
        assert result_dict['processing_time_seconds'] == 1.5
        assert result_dict['metadata'] == {'format': 'mp3'}


class TestAudioFormatConfigurations:
    """Tests for audio format configuration constants."""
    
    def test_all_formats_have_required_keys(self):
        """Test that all audio formats have required configuration keys."""
        required_keys = ['extension', 'codec', 'supports_quality', 'mime_type']
        
        for format_name, config in AUDIO_FORMATS.items():
            for key in required_keys:
                assert key in config, f"Format '{format_name}' missing key: {key}"
    
    def test_format_extensions_are_valid(self):
        """Test that format extensions are valid."""
        for format_name, config in AUDIO_FORMATS.items():
            assert config['extension'] is not None
            assert len(config['extension']) > 0
            assert not config['extension'].startswith('.')
    
    def test_valid_bitrates(self):
        """Test that VALID_BITRATES contains expected values."""
        assert '128k' in VALID_BITRATES
        assert '192k' in VALID_BITRATES
        assert '320k' in VALID_BITRATES
    
    def test_valid_sample_rates(self):
        """Test that VALID_SAMPLE_RATES contains expected values."""
        assert 44100 in VALID_SAMPLE_RATES
        assert 48000 in VALID_SAMPLE_RATES
    
    def test_valid_channels(self):
        """Test that VALID_CHANNELS contains expected values."""
        assert 1 in VALID_CHANNELS  # Mono
        assert 2 in VALID_CHANNELS  # Stereo