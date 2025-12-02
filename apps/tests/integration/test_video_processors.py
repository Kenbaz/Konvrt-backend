# apps/tests/integration/test_video_processors.py

"""
Integration tests for video processing operations.

These tests verify that the video processors work correctly with
real FFmpeg operations on sample video files.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Skip tests if FFmpeg is not available
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
    FFMPEG_AVAILABLE = result.returncode == 0
except Exception:
    FFMPEG_AVAILABLE = False


# Test fixture directory path
FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix='test_video_')
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_video_path(temp_dir):
    """
    Create a sample video file for testing.
    
    If a fixtures directory with sample.mp4 exists, use that.
    Otherwise, generate a minimal test video using FFmpeg.
    """
    # Check for existing fixture
    fixture_path = FIXTURES_DIR / 'sample.mp4'
    if fixture_path.exists():
        # Copy to temp directory
        dest_path = os.path.join(temp_dir, 'sample.mp4')
        shutil.copy(str(fixture_path), dest_path)
        return dest_path
    
    # Generate a minimal test video (5 seconds, 640x480)
    output_path = os.path.join(temp_dir, 'sample.mp4')
    
    if FFMPEG_AVAILABLE:
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'testsrc=duration=5:size=640x480:rate=30',
            '-f', 'lavfi',
            '-i', 'sine=frequency=440:duration=5',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to generate test video: {e.stderr}")
    
    pytest.skip("FFmpeg not available for generating test video")


@pytest.fixture
def operation_id():
    """Generate a unique operation ID for testing."""
    return uuid.uuid4()


@pytest.fixture
def session_key():
    """Generate a test session key."""
    return 'test_session_key_12345'


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback."""
    callback = MagicMock()
    progress_values = []
    
    def track_progress(percent: int, eta: Optional[float] = None):
        progress_values.append((percent, eta))
        callback(percent, eta)
    
    track_progress.mock = callback
    track_progress.values = progress_values
    return track_progress


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestVideoCompressProcessor:
    """Tests for VideoCompressProcessor."""
    
    def test_compress_video_default_quality(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video compression with default quality settings."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        # Patch settings for temp directory
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_path is not None
        assert result.output_filename is not None
        assert result.output_filename.endswith('.mp4')
        # Note: temp directory is cleaned up after processing, 
        # so we check the result metadata instead
        assert result.metadata.get('output_size', 0) > 0
        
        # Verify metadata
        assert 'compression_ratio' in result.metadata
        assert 'codec' in result.metadata
        assert result.metadata['codec'] == 'h264'
    
    def test_compress_video_high_quality(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video compression with high quality (low CRF)."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'quality': 18,  # High quality
                    'preset': 'fast',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['quality'] == 18
        assert result.metadata['preset'] == 'fast'
    
    def test_compress_video_low_quality(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video compression with low quality (high CRF) for smaller file."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'quality': 28,  # Low quality, small file
                    'preset': 'ultrafast',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['quality'] == 28
        # Lower quality should generally produce smaller files
    
    def test_compress_video_with_progress_tracking(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
        mock_progress_callback,
    ):
        """Test that progress is tracked during compression."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={},
                progress_callback=mock_progress_callback,
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        
        # Check progress was reported
        assert len(mock_progress_callback.values) > 0
        
        # First progress should be 0%
        assert mock_progress_callback.values[0][0] == 0
        
        # Last progress should be 100%
        assert mock_progress_callback.values[-1][0] == 100
    
    def test_compress_nonexistent_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test compression fails gracefully for non-existent input."""
        from apps.processors.video_processing import VideoCompressProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path='/nonexistent/path/video.mp4',
                parameters={},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND
        assert result.is_retryable is False
    
    def test_compress_invalid_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test compression fails gracefully for invalid video file."""
        from apps.processors.video_processing import VideoCompressProcessor
        from apps.processors.base_processor import ErrorCategory
        
        # Create an invalid "video" file
        invalid_path = os.path.join(temp_dir, 'invalid.mp4')
        with open(invalid_path, 'w') as f:
            f.write('This is not a valid video file')
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=invalid_path,
                parameters={},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_message is not None


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestVideoConvertProcessor:
    """Tests for VideoConvertProcessor."""
    
    def test_convert_to_mp4(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video conversion to MP4 format."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'mp4',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.mp4')
        assert result.metadata['output_format'] == 'mp4'
        assert result.metadata['video_codec'] == 'libx264'
        assert result.metadata.get('output_size', 0) > 0
    
    def test_convert_to_webm(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video conversion to WebM format."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'webm',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.webm')
        assert result.metadata['output_format'] == 'webm'
        assert result.metadata['video_codec'] == 'libvpx-vp9'
        assert result.metadata.get('output_size', 0) > 0
    
    def test_convert_to_mov(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video conversion to MOV format."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'mov',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.mov')
        assert result.metadata['output_format'] == 'mov'
        assert result.metadata.get('output_size', 0) > 0
    
    def test_convert_with_quality_setting(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video conversion with custom quality setting."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'mp4',
                    'quality': 20,
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['quality'] == 20
    
    def test_convert_with_progress_tracking(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
        mock_progress_callback,
    ):
        """Test that progress is tracked during conversion."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'mp4',
                },
                progress_callback=mock_progress_callback,
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert len(mock_progress_callback.values) > 0
    
    def test_convert_nonexistent_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test conversion fails gracefully for non-existent input."""
        from apps.processors.video_processing import VideoConvertProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path='/nonexistent/path/video.mp4',
                parameters={
                    'output_format': 'mp4',
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestVideoHandlerFunctions:
    """Tests for the handler functions used by the registry."""
    
    def test_video_compress_handler(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video_compress_handler function."""
        from apps.processors.video_processing import video_compress_handler
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            result = video_compress_handler(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'quality': 23,
                },
            )
        
        assert result.success is True
        assert result.metadata.get('output_size', 0) > 0
    
    def test_video_convert_handler(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test video_convert_handler function."""
        from apps.processors.video_processing import video_convert_handler
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            result = video_convert_handler(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={
                    'output_format': 'webm',
                },
            )
        
        assert result.success is True
        assert result.output_filename.endswith('.webm')


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestVideoOperationRegistration:
    """Tests for operation registration."""
    
    def test_video_compress_registered(self):
        """Test video_compress operation is registered."""
        from apps.processors.registry import get_registry
        from apps.processors.video_processing import register_video_operations
        
        # Ensure operations are registered
        register_video_operations()
        
        registry = get_registry()
        assert registry.is_registered('video_compress')
        
        operation = registry.get_operation('video_compress')
        assert operation.operation_name == 'video_compress'
        assert operation.media_type.value == 'video'
        assert len(operation.parameters) == 3  # quality, preset, audio_bitrate
    
    def test_video_convert_registered(self):
        """Test video_convert operation is registered."""
        from apps.processors.registry import get_registry
        from apps.processors.video_processing import register_video_operations
        
        # Ensure operations are registered
        register_video_operations()
        
        registry = get_registry()
        assert registry.is_registered('video_convert')
        
        operation = registry.get_operation('video_convert')
        assert operation.operation_name == 'video_convert'
        assert operation.media_type.value == 'video'
        assert len(operation.parameters) == 2  # output_format, quality
    
    def test_parameter_validation(self):
        """Test parameter validation for video operations."""
        from apps.processors.registry import get_registry
        from apps.processors.video_processing import register_video_operations
        from apps.processors.exceptions import InvalidParametersError
        
        register_video_operations()
        registry = get_registry()
        
        # Valid parameters should pass
        validated = registry.validate_parameters(
            'video_compress',
            {'quality': 23, 'preset': 'fast'}
        )
        assert validated['quality'] == 23
        assert validated['preset'] == 'fast'
        
        # Invalid quality should fail
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'video_compress',
                {'quality': 50}  # Out of range (max is 28)
            )
        
        # Invalid preset choice should fail
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'video_compress',
                {'preset': 'invalid_preset'}
            )
    
    def test_convert_parameter_validation(self):
        """Test parameter validation for video_convert operation."""
        from apps.processors.registry import get_registry
        from apps.processors.video_processing import register_video_operations
        from apps.processors.exceptions import InvalidParametersError
        
        register_video_operations()
        registry = get_registry()
        
        # Valid output_format
        validated = registry.validate_parameters(
            'video_convert',
            {'output_format': 'webm'}
        )
        assert validated['output_format'] == 'webm'
        
        # Invalid output_format
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'video_convert',
                {'output_format': 'invalid_format'}
            )


class TestVideoProcessorErrorHandling:
    """Tests for error handling in video processors."""
    
    def test_error_categorization_invalid_input(self):
        """Test error categorization for invalid input."""
        from apps.processors.video_processing import VideoCompressProcessor
        from apps.processors.base_processor import ErrorCategory
        
        processor = VideoCompressProcessor(
            operation_id=uuid.uuid4(),
            session_key='test',
            input_path='/test/path',
            parameters={},
        )
        
        # Test various error messages
        assert processor._categorize_ffmpeg_error('Invalid data found') == ErrorCategory.INVALID_INPUT
        assert processor._categorize_ffmpeg_error('File is corrupted') == ErrorCategory.INVALID_INPUT
        assert processor._categorize_ffmpeg_error('No such file') == ErrorCategory.NOT_FOUND
        assert processor._categorize_ffmpeg_error('Permission denied') == ErrorCategory.PERMISSION
        assert processor._categorize_ffmpeg_error('Out of memory') == ErrorCategory.RESOURCE
        assert processor._categorize_ffmpeg_error('No space left') == ErrorCategory.RESOURCE
        assert processor._categorize_ffmpeg_error('Codec not found') == ErrorCategory.CODEC_ERROR
        assert processor._categorize_ffmpeg_error('Unknown error') == ErrorCategory.UNKNOWN
    
    def test_error_message_parsing(self):
        """Test error message parsing."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        processor = VideoCompressProcessor(
            operation_id=uuid.uuid4(),
            session_key='test',
            input_path='/test/path',
            parameters={},
        )
        
        # Test various error scenarios
        assert 'corrupted' in processor._parse_ffmpeg_error('Invalid data found, file is corrupt').lower()
        assert 'not found' in processor._parse_ffmpeg_error('No such file or directory').lower()
        assert 'permission' in processor._parse_ffmpeg_error('Permission denied').lower()
        assert 'disk space' in processor._parse_ffmpeg_error('No space left on device').lower()
        assert 'memory' in processor._parse_ffmpeg_error('Out of memory').lower()


class TestVideoProcessorMetadata:
    """Tests for metadata generation."""
    
    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
    def test_compress_metadata_fields(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test that compression result contains expected metadata."""
        from apps.processors.video_processing import VideoCompressProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoCompressProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={'quality': 23},
            )
            
            result = processor.execute_operation()
        
        if result.success:
            metadata = result.metadata
            assert 'input_size' in metadata
            assert 'output_size' in metadata
            assert 'compression_ratio' in metadata
            assert 'quality' in metadata
            assert 'preset' in metadata
            assert 'duration' in metadata
            assert 'codec' in metadata
            
            assert isinstance(metadata['input_size'], int)
            assert isinstance(metadata['output_size'], int)
            assert isinstance(metadata['compression_ratio'], float)
            assert metadata['codec'] == 'h264'
    
    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
    def test_convert_metadata_fields(
        self,
        sample_video_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test that conversion result contains expected metadata."""
        from apps.processors.video_processing import VideoConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = VideoConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_video_path,
                parameters={'output_format': 'mp4'},
            )
            
            result = processor.execute_operation()
        
        if result.success:
            metadata = result.metadata
            assert 'input_format' in metadata
            assert 'output_format' in metadata
            assert 'input_size' in metadata
            assert 'output_size' in metadata
            assert 'video_codec' in metadata
            assert 'audio_codec' in metadata
            assert 'quality' in metadata
            assert 'duration' in metadata
            
            assert metadata['output_format'] == 'mp4'