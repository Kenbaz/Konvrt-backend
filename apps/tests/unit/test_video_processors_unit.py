# tests/unit/test_video_processors.py

"""
Unit tests for video processing operations.

These tests verify the video processor logic using mocks,
without requiring actual FFmpeg execution.
"""

import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def operation_id():
    """Generate a unique operation ID for testing."""
    return uuid.uuid4()


@pytest.fixture
def session_key():
    """Generate a test session key."""
    return 'test_session_12345'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix='test_video_unit_')
    yield temp_path
    # Cleanup
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_video_info():
    """Create a mock VideoInfo object."""
    mock_info = MagicMock()
    mock_info.duration = 60.0  # 1 minute
    mock_info.width = 1920
    mock_info.height = 1080
    mock_info.codec = 'h264'
    mock_info.fps = 30.0
    mock_info.bitrate = 5000000
    mock_info.has_audio = True
    mock_info.format_name = 'mp4'
    return mock_info


@pytest.fixture
def mock_ffmpeg_result():
    """Create a mock FFmpegResult object."""
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.return_code = 0
    mock_result.stdout = ''
    mock_result.stderr = ''
    mock_result.output_path = '/tmp/output.mp4'
    return mock_result


class TestVideoCompressProcessorUnit:
    """Unit tests for VideoCompressProcessor."""
    
    def test_processor_initialization(self, operation_id, session_key):
        """Test processor initializes correctly."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'quality': 23},
        )
        
        assert processor.operation_id == operation_id
        assert processor.session_key == session_key
        assert processor.input_path == '/test/input.mp4'
        assert processor.parameters == {'quality': 23}
        assert processor.operation_name == 'video_compress'
    
    def test_operation_name_property(self, operation_id, session_key):
        """Test operation_name property returns correct value."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={},
        )
        
        assert processor.operation_name == 'video_compress'
    
    def test_default_timeout(self, operation_id, session_key):
        """Test default timeout is set for video processing."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={},
        )
        
        # Video processors should have longer timeout (30 minutes)
        assert processor.DEFAULT_TIMEOUT == 1800
    
    @patch('apps.processors.video_processing.get_ffmpeg_wrapper')
    @patch('apps.processors.base_processor.settings')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_compress_success_flow(
        self,
        mock_getsize,
        mock_exists,
        mock_settings,
        mock_get_ffmpeg,
        operation_id,
        session_key,
        temp_dir,
        mock_video_info,
        mock_ffmpeg_result,
    ):
        """Test successful compression flow."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        # Setup mocks
        mock_settings.MEDIA_ROOT = temp_dir
        mock_exists.return_value = True
        mock_getsize.side_effect = lambda p: 10000000 if 'input' in p else 5000000  # 10MB -> 5MB
        
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.get_video_info.return_value = mock_video_info
        mock_ffmpeg.build_compress_command.return_value = ['-i', 'input', 'output']
        mock_ffmpeg.execute.return_value = mock_ffmpeg_result
        mock_get_ffmpeg.return_value = mock_ffmpeg
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'quality': 23, 'preset': 'fast'},
        )
        
        # Mock methods
        processor.get_video_info = MagicMock(return_value=mock_video_info)
        processor.get_temp_file_path = MagicMock(return_value='/tmp/test_output.mp4')
        processor.validate_output_file_creation = MagicMock(return_value=True)
        
        result = processor.process_operation()
        
        assert result.success is True
        assert result.metadata['codec'] == 'h264'
        mock_ffmpeg.build_compress_command.assert_called_once()
        mock_ffmpeg.execute.assert_called_once()
    
    def test_error_categorization(self, operation_id, session_key):
        """Test error categorization logic."""
        from apps.processors.video_processing import VideoCompressProcessor
        from apps.processors.base_processor import ErrorCategory
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={},
        )
        
        # Test various error patterns
        test_cases = [
            ('Invalid data found when processing input', ErrorCategory.INVALID_INPUT),
            ('File appears corrupt', ErrorCategory.INVALID_INPUT),
            ('No such file or directory', ErrorCategory.NOT_FOUND),
            ('Permission denied', ErrorCategory.PERMISSION),
            ('No space left on device', ErrorCategory.RESOURCE),
            ('Out of memory', ErrorCategory.RESOURCE),
            ('Encoder or codec not found', ErrorCategory.CODEC_ERROR),
            ('Some random unknown error', ErrorCategory.UNKNOWN),
        ]
        
        for stderr, expected_category in test_cases:
            category = processor._categorize_ffmpeg_error(stderr)
            assert category == expected_category, f"Failed for stderr: {stderr}"
    
    def test_error_message_parsing(self, operation_id, session_key):
        """Test error message parsing logic."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        processor = VideoCompressProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={},
        )
        
        # Test parsing various FFmpeg error outputs
        msg = processor._parse_ffmpeg_error('No such file or directory')
        assert 'not found' in msg.lower() or 'inaccessible' in msg.lower()
        
        msg = processor._parse_ffmpeg_error('Invalid data found')
        assert 'corrupt' in msg.lower()
        
        msg = processor._parse_ffmpeg_error('Permission denied')
        assert 'permission' in msg.lower()


class TestVideoConvertProcessorUnit:
    """Unit tests for VideoConvertProcessor."""
    
    def test_processor_initialization(self, operation_id, session_key):
        """Test processor initializes correctly."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        processor = VideoConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'output_format': 'webm'},
        )
        
        assert processor.operation_id == operation_id
        assert processor.session_key == session_key
        assert processor.parameters['output_format'] == 'webm'
        assert processor.operation_name == 'video_convert'
    
    def test_operation_name_property(self, operation_id, session_key):
        """Test operation_name property returns correct value."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        processor = VideoConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={},
        )
        
        assert processor.operation_name == 'video_convert'
    
    @patch('apps.processors.video_processing.get_ffmpeg_wrapper')
    @patch('apps.processors.base_processor.settings')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_convert_to_webm_codecs(
        self,
        mock_getsize,
        mock_exists,
        mock_settings,
        mock_get_ffmpeg,
        operation_id,
        session_key,
        temp_dir,
        mock_video_info,
        mock_ffmpeg_result,
    ):
        """Test that WebM conversion uses correct codecs."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        mock_settings.MEDIA_ROOT = temp_dir
        mock_exists.return_value = True
        mock_getsize.return_value = 5000000
        
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.get_video_info.return_value = mock_video_info
        mock_ffmpeg.build_convert_command.return_value = ['-i', 'input', 'output']
        mock_ffmpeg.execute.return_value = mock_ffmpeg_result
        mock_get_ffmpeg.return_value = mock_ffmpeg
        
        processor = VideoConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'output_format': 'webm'},
        )
        
        processor.get_video_info = MagicMock(return_value=mock_video_info)
        processor.get_temp_file_path = MagicMock(return_value='/tmp/output.webm')
        processor.validate_output_file_creation = MagicMock(return_value=True)
        processor.get_input_extension = MagicMock(return_value='mp4')
        
        result = processor.process_operation()
        
        # Verify correct codec was requested
        mock_ffmpeg.build_convert_command.assert_called_once()
        call_args = mock_ffmpeg.build_convert_command.call_args
        assert call_args.kwargs.get('video_codec') == 'libvpx-vp9'
        assert call_args.kwargs.get('audio_codec') == 'libopus'


class TestVideoCodecMappings:
    """Tests for codec and format mappings."""
    
    def test_video_codecs_mapping(self):
        """Test VIDEO_CODECS mapping is correct."""
        from apps.processors.video_processing import VIDEO_CODECS
        
        assert VIDEO_CODECS['mp4'] == 'libx264'
        assert VIDEO_CODECS['webm'] == 'libvpx-vp9'
        assert VIDEO_CODECS['mov'] == 'libx264'
    
    def test_audio_codecs_mapping(self):
        """Test AUDIO_CODECS mapping is correct."""
        from apps.processors.video_processing import AUDIO_CODECS
        
        assert AUDIO_CODECS['mp4'] == 'aac'
        assert AUDIO_CODECS['webm'] == 'libopus'
        assert AUDIO_CODECS['mov'] == 'aac'
    
    def test_encoding_presets(self):
        """Test ENCODING_PRESETS list is valid."""
        from apps.processors.video_processing import ENCODING_PRESETS
        
        expected_presets = [
            'ultrafast', 'superfast', 'veryfast', 'faster',
            'fast', 'medium', 'slow', 'slower', 'veryslow'
        ]
        assert ENCODING_PRESETS == expected_presets


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_video_compress_processor(self, operation_id, session_key):
        """Test create_video_compress_processor factory."""
        from apps.processors.video_processing import (
            create_video_compress_processor,
            VideoCompressProcessor,
        )
        
        processor = create_video_compress_processor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'quality': 20},
        )
        
        assert isinstance(processor, VideoCompressProcessor)
        assert processor.parameters['quality'] == 20
    
    def test_create_video_convert_processor(self, operation_id, session_key):
        """Test create_video_convert_processor factory."""
        from apps.processors.video_processing import (
            create_video_convert_processor,
            VideoConvertProcessor,
        )
        
        processor = create_video_convert_processor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'output_format': 'webm'},
        )
        
        assert isinstance(processor, VideoConvertProcessor)
        assert processor.parameters['output_format'] == 'webm'


class TestHandlerFunctions:
    """Tests for handler functions."""
    
    @patch('apps.processors.video_processing.create_video_compress_processor')
    def test_video_compress_handler_creates_processor(
        self,
        mock_create,
        operation_id,
        session_key,
    ):
        """Test video_compress_handler creates processor and calls execute."""
        from apps.processors.video_processing import video_compress_handler
        from apps.processors.base_processor import ProcessingResult
        
        mock_processor = MagicMock()
        mock_processor.execute_operation.return_value = ProcessingResult(
            success=True,
            output_path='/tmp/output.mp4',
            output_filename='output.mp4',
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=10.0,
            metadata={},
        )
        mock_create.return_value = mock_processor
        
        result = video_compress_handler(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'quality': 23},
        )
        
        mock_create.assert_called_once_with(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'quality': 23},
            progress_callback=None,
        )
        mock_processor.execute_operation.assert_called_once()
        assert result.success is True
    
    @patch('apps.processors.video_processing.create_video_convert_processor')
    def test_video_convert_handler_creates_processor(
        self,
        mock_create,
        operation_id,
        session_key,
    ):
        """Test video_convert_handler creates processor and calls execute."""
        from apps.processors.video_processing import video_convert_handler
        from apps.processors.base_processor import ProcessingResult
        
        mock_processor = MagicMock()
        mock_processor.execute_operation.return_value = ProcessingResult(
            success=True,
            output_path='/tmp/output.webm',
            output_filename='output.webm',
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=15.0,
            metadata={},
        )
        mock_create.return_value = mock_processor
        
        result = video_convert_handler(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.mp4',
            parameters={'output_format': 'webm'},
        )
        
        mock_create.assert_called_once()
        assert result.success is True


class TestRegistrationFunction:
    """Tests for the registration function."""
    
    def test_register_video_operations_idempotent(self):
        """Test that register_video_operations can be called multiple times."""
        from apps.processors.video_processing import register_video_operations
        from apps.processors.registry import get_registry
        
        # Clear and re-register
        registry = get_registry()
        
        # First registration
        register_video_operations()
        
        assert registry.is_registered('video_compress')
        assert registry.is_registered('video_convert')
        
        # Second registration should not raise
        register_video_operations()
        
        # Still registered
        assert registry.is_registered('video_compress')
        assert registry.is_registered('video_convert')
    
    def test_video_compress_operation_definition(self):
        """Test video_compress operation definition is correct."""
        from apps.processors.video_processing import register_video_operations
        from apps.processors.registry import get_registry, MediaType
        
        register_video_operations()
        registry = get_registry()
        
        operation = registry.get_operation('video_compress')
        
        assert operation.operation_name == 'video_compress'
        assert operation.media_type == MediaType.VIDEO
        assert len(operation.parameters) == 3
        
        # Check parameter names
        param_names = [p.param_name for p in operation.parameters]
        assert 'quality' in param_names
        assert 'preset' in param_names
        assert 'audio_bitrate' in param_names
        
        # Check input/output formats
        assert 'mp4' in operation.input_formats
        assert 'mov' in operation.input_formats
        assert 'mp4' in operation.output_formats
    
    def test_video_convert_operation_definition(self):
        """Test video_convert operation definition is correct."""
        from apps.processors.video_processing import register_video_operations
        from apps.processors.registry import get_registry, MediaType
        
        register_video_operations()
        registry = get_registry()
        
        operation = registry.get_operation('video_convert')
        
        assert operation.operation_name == 'video_convert'
        assert operation.media_type == MediaType.VIDEO
        assert len(operation.parameters) == 2
        
        # Check parameter names
        param_names = [p.param_name for p in operation.parameters]
        assert 'output_format' in param_names
        assert 'quality' in param_names
        
        # Check output formats
        assert 'mp4' in operation.output_formats
        assert 'webm' in operation.output_formats
        assert 'mov' in operation.output_formats