"""
Integration tests for image processing operations.

These tests verify that the image processors work correctly with
real Pillow operations on sample image files.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Test fixture directory path
FIXTURES_DIR = Path(__file__).parent.parent / 'fixtures'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix='test_image_')
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_path(temp_dir):
    """
    Create a sample image file for testing.
    
    If a fixtures directory with sample.jpg exists, use that.
    Otherwise, generate a test image using Pillow.
    """
    # Check for existing fixture
    for ext in ['jpg', 'jpeg', 'png']:
        fixture_path = FIXTURES_DIR / f'sample.{ext}'
        if fixture_path.exists():
            dest_path = os.path.join(temp_dir, f'sample.{ext}')
            shutil.copy(str(fixture_path), dest_path)
            return dest_path
    
    # Generate a test image (640x480, RGB)
    output_path = os.path.join(temp_dir, 'sample.png')
    
    # Create a gradient image for testing
    img = Image.new('RGB', (640, 480))
    pixels = img.load()
    
    for i in range(640):
        for j in range(480):
            pixels[i, j] = (
                int(i * 255 / 640),  # Red gradient
                int(j * 255 / 480),  # Green gradient
                128,                  # Blue constant
            )
    
    img.save(output_path, 'PNG')
    return output_path


@pytest.fixture
def sample_rgba_image_path(temp_dir):
    """Create a sample RGBA image with transparency for testing."""
    output_path = os.path.join(temp_dir, 'sample_rgba.png')
    
    # Create an image with transparency
    img = Image.new('RGBA', (400, 300), (0, 0, 0, 0))
    pixels = img.load()
    
    for i in range(400):
        for j in range(300):
            # Create a circle with transparency
            dist = ((i - 200) ** 2 + (j - 150) ** 2) ** 0.5
            if dist < 100:
                alpha = 255
            elif dist < 120:
                alpha = int(255 * (120 - dist) / 20)
            else:
                alpha = 0
            pixels[i, j] = (255, 100, 50, alpha)
    
    img.save(output_path, 'PNG')
    return output_path


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


class TestImageResizeProcessor:
    """Tests for ImageResizeProcessor."""
    
    def test_resize_with_width_only(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resizing image with only width specified."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'width': 320},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['new_width'] == 320
        # Aspect ratio should be maintained
        assert result.metadata['new_height'] == 240  # 480 * (320/640)
        assert result.metadata['maintain_aspect_ratio'] is True
    
    def test_resize_with_height_only(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resizing image with only height specified."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'height': 240},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['new_height'] == 240
        # Aspect ratio should be maintained
        assert result.metadata['new_width'] == 320  # 640 * (240/480)
    
    def test_resize_with_both_dimensions(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resizing with both dimensions (fit within bounds)."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={
                    'width': 200,
                    'height': 200,
                    'maintain_aspect_ratio': True,
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        # Should fit within 200x200 while maintaining aspect ratio
        assert result.metadata['new_width'] <= 200
        assert result.metadata['new_height'] <= 200
    
    def test_resize_without_aspect_ratio(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resizing without maintaining aspect ratio."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={
                    'width': 300,
                    'height': 300,
                    'maintain_aspect_ratio': False,
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['new_width'] == 300
        assert result.metadata['new_height'] == 300
        assert result.metadata['maintain_aspect_ratio'] is False
    
    def test_resize_with_different_resampling(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resizing with different resampling methods."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        resampling_methods = ['lanczos', 'bicubic', 'bilinear', 'nearest']
        
        for method in resampling_methods:
            with patch('apps.processors.base_processor.settings') as mock_settings:
                mock_settings.MEDIA_ROOT = temp_dir
                
                processor = ImageResizeProcessor(
                    operation_id=operation_id,
                    session_key=session_key,
                    input_path=sample_image_path,
                    parameters={
                        'width': 320,
                        'resampling': method,
                    },
                )
                
                result = processor.execute_operation()
            
            assert result.success is True, f"Failed for resampling method: {method}"
            assert result.metadata['resampling'] == method
    
    def test_resize_with_progress_tracking(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
        mock_progress_callback,
    ):
        """Test that progress is tracked during resize."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'width': 320},
                progress_callback=mock_progress_callback,
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert len(mock_progress_callback.values) > 0
        
        # Check progress includes 0% and 100%
        percents = [p[0] for p in mock_progress_callback.values]
        assert 0 in percents
        assert 100 in percents
    
    def test_resize_nonexistent_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resize fails gracefully for non-existent input."""
        from apps.processors.image_processing import ImageResizeProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path='/nonexistent/path/image.jpg',
                parameters={'width': 320},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND
    
    def test_resize_invalid_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resize fails gracefully for invalid image file."""
        from apps.processors.image_processing import ImageResizeProcessor
        from apps.processors.base_processor import ErrorCategory
        
        # Create an invalid "image" file
        invalid_path = os.path.join(temp_dir, 'invalid.jpg')
        with open(invalid_path, 'w') as f:
            f.write('This is not a valid image file')
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=invalid_path,
                parameters={'width': 320},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.INVALID_INPUT
    
    def test_resize_no_dimensions_specified(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test resize fails when no dimensions are specified."""
        from apps.processors.image_processing import ImageResizeProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.INVALID_PARAMS


class TestImageConvertProcessor:
    """Tests for ImageConvertProcessor."""
    
    def test_convert_to_jpg(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test converting image to JPEG format."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'jpg'},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.jpg')
        assert result.metadata['output_format'] == 'jpg'
    
    def test_convert_to_png(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test converting image to PNG format."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'png'},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.png')
        assert result.metadata['output_format'] == 'png'
    
    def test_convert_to_webp(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test converting image to WebP format."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'webp'},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.output_filename.endswith('.webp')
        assert result.metadata['output_format'] == 'webp'
    
    def test_convert_with_quality_setting(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test converting with custom quality setting."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={
                    'output_format': 'jpg',
                    'quality': 50,
                },
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert result.metadata['quality'] == 50
    
    def test_convert_rgba_to_jpg(
        self,
        sample_rgba_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test converting RGBA image to JPEG (handles transparency)."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_rgba_image_path,
                parameters={'output_format': 'jpg'},
            )
            
            result = processor.execute_operation()
        
        # Should succeed even though JPEG doesn't support transparency
        assert result.success is True
        assert result.output_filename.endswith('.jpg')
    
    def test_convert_with_progress_tracking(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
        mock_progress_callback,
    ):
        """Test that progress is tracked during conversion."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'webp'},
                progress_callback=mock_progress_callback,
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        assert len(mock_progress_callback.values) > 0
    
    def test_convert_invalid_format(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test conversion fails for invalid output format."""
        from apps.processors.image_processing import ImageConvertProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'invalid_format'},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.INVALID_PARAMS
    
    def test_convert_nonexistent_file(
        self,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test conversion fails for non-existent input."""
        from apps.processors.image_processing import ImageConvertProcessor
        from apps.processors.base_processor import ErrorCategory
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path='/nonexistent/path/image.png',
                parameters={'output_format': 'jpg'},
            )
            
            result = processor.execute_operation()
        
        assert result.success is False
        assert result.error_category == ErrorCategory.NOT_FOUND


class TestImageHandlerFunctions:
    """Tests for the handler functions used by the registry."""
    
    def test_image_resize_handler(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test image_resize_handler function."""
        from apps.processors.image_processing import image_resize_handler
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            result = image_resize_handler(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'width': 320},
            )
        
        assert result.success is True
        assert result.metadata.get('output_size', 0) > 0
    
    def test_image_convert_handler(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test image_convert_handler function."""
        from apps.processors.image_processing import image_convert_handler
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            result = image_convert_handler(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'webp'},
            )
        
        assert result.success is True
        assert result.output_filename.endswith('.webp')


class TestImageOperationRegistration:
    """Tests for operation registration."""
    
    def test_image_resize_registered(self):
        """Test image_resize operation is registered."""
        from apps.processors.registry import get_registry
        # Import triggers auto-registration
        # from apps.processors import image  # noqa
        
        registry = get_registry()
        assert registry.is_registered('image_resize')
        
        operation = registry.get_operation('image_resize')
        assert operation.operation_name == 'image_resize'
        assert operation.media_type.value == 'image'
        assert len(operation.parameters) == 4  # width, height, maintain_aspect_ratio, resampling
    
    def test_image_convert_registered(self):
        """Test image_convert operation is registered."""
        from apps.processors.registry import get_registry
        # Import triggers auto-registration
        # from apps.processors import image  # noqa
        
        registry = get_registry()
        assert registry.is_registered('image_convert')
        
        operation = registry.get_operation('image_convert')
        assert operation.operation_name == 'image_convert'
        assert operation.media_type.value == 'image'
        assert len(operation.parameters) == 2  # output_format, quality
    
    def test_resize_parameter_validation(self):
        """Test parameter validation for image_resize."""
        from apps.processors.registry import get_registry
        # Import triggers auto-registration
        # from apps.processors import image  # noqa
        from apps.processors.exceptions import InvalidParametersError
        
        registry = get_registry()
        
        # Valid parameters
        validated = registry.validate_parameters(
            'image_resize',
            {'width': 800, 'height': 600}
        )
        assert validated['width'] == 800
        assert validated['height'] == 600
        
        # Invalid width (too large)
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'image_resize',
                {'width': 20000}  # Exceeds max of 16384
            )
    
    def test_convert_parameter_validation(self):
        """Test parameter validation for image_convert."""
        from apps.processors.registry import get_registry
        # Import triggers auto-registration
        # from apps.processors import image  # noqa
        from apps.processors.exceptions import InvalidParametersError
        
        registry = get_registry()
        
        # Valid output format
        validated = registry.validate_parameters(
            'image_convert',
            {'output_format': 'webp'}
        )
        assert validated['output_format'] == 'webp'
        
        # Invalid output format
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'image_convert',
                {'output_format': 'tiff'}  # Not in allowed choices
            )
        
        # Invalid quality
        with pytest.raises(InvalidParametersError):
            registry.validate_parameters(
                'image_convert',
                {'quality': 150}  # Exceeds max of 100
            )


class TestImageProcessorMetadata:
    """Tests for metadata generation."""
    
    def test_resize_metadata_fields(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test that resize result contains expected metadata."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageResizeProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'width': 320},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        metadata = result.metadata
        
        assert 'original_width' in metadata
        assert 'original_height' in metadata
        assert 'new_width' in metadata
        assert 'new_height' in metadata
        assert 'maintain_aspect_ratio' in metadata
        assert 'resampling' in metadata
        assert 'input_size' in metadata
        assert 'output_size' in metadata
        assert 'format' in metadata
    
    def test_convert_metadata_fields(
        self,
        sample_image_path,
        temp_dir,
        operation_id,
        session_key,
    ):
        """Test that convert result contains expected metadata."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        with patch('apps.processors.base_processor.settings') as mock_settings:
            mock_settings.MEDIA_ROOT = temp_dir
            
            processor = ImageConvertProcessor(
                operation_id=operation_id,
                session_key=session_key,
                input_path=sample_image_path,
                parameters={'output_format': 'jpg', 'quality': 80},
            )
            
            result = processor.execute_operation()
        
        assert result.success is True
        metadata = result.metadata
        
        assert 'input_format' in metadata
        assert 'output_format' in metadata
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'quality' in metadata
        assert 'input_size' in metadata
        assert 'output_size' in metadata
        assert 'size_change_percent' in metadata