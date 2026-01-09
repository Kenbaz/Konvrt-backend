"""
Unit tests for image processing operations.

These tests verify the image processor logic using mocks,
without requiring actual image processing.
"""

import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch, PropertyMock

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
    temp_path = tempfile.mkdtemp(prefix='test_image_unit_')
    yield temp_path
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


class TestImageResizeProcessorUnit:
    """Unit tests for ImageResizeProcessor."""
    
    def test_processor_initialization(self, operation_id, session_key):
        """Test processor initializes correctly."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={'width': 800},
        )
        
        assert processor.operation_id == operation_id
        assert processor.session_key == session_key
        assert processor.input_path == '/test/input.jpg'
        assert processor.parameters == {'width': 800}
        assert processor.operation_name == 'image_resize'
    
    def test_operation_name_property(self, operation_id, session_key):
        """Test operation_name property returns correct value."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        assert processor.operation_name == 'image_resize'
    
    def test_default_timeout(self, operation_id, session_key):
        """Test default timeout is set for image processing."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        # Image processors should have 1 minute timeout
        assert processor.DEFAULT_TIMEOUT == 60
    
    def test_calculate_dimensions_width_only(self, operation_id, session_key):
        """Test dimension calculation with only width specified."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        new_width, new_height = processor._calculate_dimensions(
            original_width=1000,
            original_height=500,
            target_width=500,
            target_height=None,
            maintain_aspect_ratio=True,
        )
        
        assert new_width == 500
        assert new_height == 250  # Maintains 2:1 ratio
    
    def test_calculate_dimensions_height_only(self, operation_id, session_key):
        """Test dimension calculation with only height specified."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        new_width, new_height = processor._calculate_dimensions(
            original_width=1000,
            original_height=500,
            target_width=None,
            target_height=250,
            maintain_aspect_ratio=True,
        )
        
        assert new_width == 500  # Maintains 2:1 ratio
        assert new_height == 250
    
    def test_calculate_dimensions_both_fit_width(self, operation_id, session_key):
        """Test dimension calculation fitting within bounds (width constrained)."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        # Wide image (2:1) fitting into square (1:1) bounds
        new_width, new_height = processor._calculate_dimensions(
            original_width=1000,
            original_height=500,
            target_width=400,
            target_height=400,
            maintain_aspect_ratio=True,
        )
        
        assert new_width == 400  # Constrained by width
        assert new_height == 200  # Maintains 2:1 ratio
    
    def test_calculate_dimensions_both_fit_height(self, operation_id, session_key):
        """Test dimension calculation fitting within bounds (height constrained)."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        # Tall image (1:2) fitting into square (1:1) bounds
        new_width, new_height = processor._calculate_dimensions(
            original_width=500,
            original_height=1000,
            target_width=400,
            target_height=400,
            maintain_aspect_ratio=True,
        )
        
        assert new_width == 200  # Maintains 1:2 ratio
        assert new_height == 400  # Constrained by height
    
    def test_calculate_dimensions_no_aspect_ratio(self, operation_id, session_key):
        """Test dimension calculation without maintaining aspect ratio."""
        from apps.processors.image_processing import ImageResizeProcessor
        
        processor = ImageResizeProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={},
        )
        
        new_width, new_height = processor._calculate_dimensions(
            original_width=1000,
            original_height=500,
            target_width=300,
            target_height=300,
            maintain_aspect_ratio=False,
        )
        
        assert new_width == 300
        assert new_height == 300  # Ignores aspect ratio


class TestImageConvertProcessorUnit:
    """Unit tests for ImageConvertProcessor."""
    
    def test_processor_initialization(self, operation_id, session_key):
        """Test processor initializes correctly."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        processor = ImageConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={'output_format': 'jpg'},
        )
        
        assert processor.operation_id == operation_id
        assert processor.session_key == session_key
        assert processor.parameters['output_format'] == 'jpg'
        assert processor.operation_name == 'image_convert'
    
    def test_operation_name_property(self, operation_id, session_key):
        """Test operation_name property returns correct value."""
        from apps.processors.image_processing import ImageConvertProcessor
        
        processor = ImageConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={},
        )
        
        assert processor.operation_name == 'image_convert'
    
    def test_convert_color_mode_rgba_to_rgb(self, operation_id, session_key):
        """Test color mode conversion from RGBA to RGB."""
        from apps.processors.image_processing import ImageConvertProcessor
        from PIL import Image
        
        processor = ImageConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={},
        )
        
        # Create an RGBA image
        rgba_img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        
        # Convert to RGB
        rgb_img = processor._convert_color_mode(rgba_img, 'RGB')
        
        assert rgb_img.mode == 'RGB'
    
    def test_convert_color_mode_same_mode(self, operation_id, session_key):
        """Test that same mode conversion returns original."""
        from apps.processors.image_processing import ImageConvertProcessor
        from PIL import Image
        
        processor = ImageConvertProcessor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={},
        )
        
        # Create an RGB image
        rgb_img = Image.new('RGB', (100, 100), (255, 0, 0))
        
        # Convert to same mode
        result_img = processor._convert_color_mode(rgb_img, 'RGB')
        
        assert result_img.mode == 'RGB'
        assert result_img is rgb_img  # Should be same object


class TestImageFormatMappings:
    """Tests for image format mappings."""
    
    def test_image_formats_mapping(self):
        """Test IMAGE_FORMATS mapping contains expected formats."""
        from apps.processors.image_processing import IMAGE_FORMATS
        
        assert 'jpg' in IMAGE_FORMATS
        assert 'jpeg' in IMAGE_FORMATS
        assert 'png' in IMAGE_FORMATS
        assert 'webp' in IMAGE_FORMATS
        assert 'gif' in IMAGE_FORMATS
        assert 'bmp' in IMAGE_FORMATS
    
    def test_jpg_format_config(self):
        """Test JPG format configuration."""
        from apps.processors.image_processing import IMAGE_FORMATS
        
        jpg_config = IMAGE_FORMATS['jpg']
        assert jpg_config['extension'] == 'jpg'
        assert jpg_config['pillow_format'] == 'JPEG'
        assert jpg_config['mode'] == 'RGB'
        assert jpg_config['supports_quality'] is True
    
    def test_png_format_config(self):
        """Test PNG format configuration."""
        from apps.processors.image_processing import IMAGE_FORMATS
        
        png_config = IMAGE_FORMATS['png']
        assert png_config['extension'] == 'png'
        assert png_config['pillow_format'] == 'PNG'
        assert png_config['mode'] == 'RGBA'
        assert png_config['supports_quality'] is False
    
    def test_webp_format_config(self):
        """Test WebP format configuration."""
        from apps.processors.image_processing import IMAGE_FORMATS
        
        webp_config = IMAGE_FORMATS['webp']
        assert webp_config['extension'] == 'webp'
        assert webp_config['pillow_format'] == 'WEBP'
        assert webp_config['supports_quality'] is True
    
    def test_resampling_methods(self):
        """Test RESAMPLING_METHODS mapping."""
        from apps.processors.image_processing import RESAMPLING_METHODS
        from PIL import Image
        
        assert 'lanczos' in RESAMPLING_METHODS
        assert 'bicubic' in RESAMPLING_METHODS
        assert 'bilinear' in RESAMPLING_METHODS
        assert 'nearest' in RESAMPLING_METHODS
        
        assert RESAMPLING_METHODS['lanczos'] == Image.Resampling.LANCZOS
        assert RESAMPLING_METHODS['bicubic'] == Image.Resampling.BICUBIC


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_image_resize_processor(self, operation_id, session_key):
        """Test create_image_resize_processor factory."""
        from apps.processors.image_processing import (
            create_image_resize_processor,
            ImageResizeProcessor,
        )
        
        processor = create_image_resize_processor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={'width': 800},
        )
        
        assert isinstance(processor, ImageResizeProcessor)
        assert processor.parameters['width'] == 800
    
    def test_create_image_convert_processor(self, operation_id, session_key):
        """Test create_image_convert_processor factory."""
        from apps.processors.image_processing import (
            create_image_convert_processor,
            ImageConvertProcessor,
        )
        
        processor = create_image_convert_processor(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={'output_format': 'webp'},
        )
        
        assert isinstance(processor, ImageConvertProcessor)
        assert processor.parameters['output_format'] == 'webp'


class TestHandlerFunctions:
    """Tests for handler functions."""
    
    @patch('apps.processors.image_processing.create_image_resize_processor')
    def test_image_resize_handler_creates_processor(
        self,
        mock_create,
        operation_id,
        session_key,
    ):
        """Test image_resize_handler creates processor and calls execute."""
        from apps.processors.image_processing import image_resize_handler
        from apps.processors.base_processor import ProcessingResult
        
        mock_processor = MagicMock()
        mock_processor.execute_operation.return_value = ProcessingResult(
            success=True,
            output_path='/tmp/output.jpg',
            output_filename='output.jpg',
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=0.5,
            metadata={},
        )
        mock_create.return_value = mock_processor
        
        result = image_resize_handler(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={'width': 800},
        )
        
        mock_create.assert_called_once_with(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.jpg',
            parameters={'width': 800},
            progress_callback=None,
        )
        mock_processor.execute_operation.assert_called_once()
        assert result.success is True
    
    @patch('apps.processors.image_processing.create_image_convert_processor')
    def test_image_convert_handler_creates_processor(
        self,
        mock_create,
        operation_id,
        session_key,
    ):
        """Test image_convert_handler creates processor and calls execute."""
        from apps.processors.image_processing import image_convert_handler
        from apps.processors.base_processor import ProcessingResult
        
        mock_processor = MagicMock()
        mock_processor.execute_operation.return_value = ProcessingResult(
            success=True,
            output_path='/tmp/output.webp',
            output_filename='output.webp',
            error_message=None,
            error_category=None,
            is_retryable=False,
            processing_time_seconds=0.3,
            metadata={},
        )
        mock_create.return_value = mock_processor
        
        result = image_convert_handler(
            operation_id=operation_id,
            session_key=session_key,
            input_path='/test/input.png',
            parameters={'output_format': 'webp'},
        )
        
        mock_create.assert_called_once()
        assert result.success is True


class TestRegistrationFunction:
    """Tests for the registration function."""
    
    def test_register_image_operations_idempotent(self):
        """Test that register_image_operations can be called multiple times."""
        from apps.processors.image_processing import register_image_operations
        from apps.processors.registry import get_registry
        
        registry = get_registry()
        
        # First registration
        register_image_operations()
        
        assert registry.is_registered('image_resize')
        assert registry.is_registered('image_convert')
        
        # Second registration should not raise
        register_image_operations()
        
        # Still registered
        assert registry.is_registered('image_resize')
        assert registry.is_registered('image_convert')
    
    def test_image_resize_operation_definition(self):
        """Test image_resize operation definition is correct."""
        from apps.processors.image_processing import register_image_operations
        from apps.processors.registry import get_registry, MediaType
        
        register_image_operations()
        registry = get_registry()
        
        operation = registry.get_operation('image_resize')
        
        assert operation.operation_name == 'image_resize'
        assert operation.media_type == MediaType.IMAGE
        assert len(operation.parameters) == 4
        
        # Check parameter names
        param_names = [p.param_name for p in operation.parameters]
        assert 'width' in param_names
        assert 'height' in param_names
        assert 'maintain_aspect_ratio' in param_names
        assert 'resampling' in param_names
    
    def test_image_convert_operation_definition(self):
        """Test image_convert operation definition is correct."""
        from apps.processors.image_processing import register_image_operations
        from apps.processors.registry import get_registry, MediaType
        
        register_image_operations()
        registry = get_registry()
        
        operation = registry.get_operation('image_convert')
        
        assert operation.operation_name == 'image_convert'
        assert operation.media_type == MediaType.IMAGE
        assert len(operation.parameters) == 2
        
        # Check parameter names
        param_names = [p.param_name for p in operation.parameters]
        assert 'output_format' in param_names
        assert 'quality' in param_names