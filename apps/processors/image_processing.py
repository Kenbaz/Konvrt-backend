# apps/processors/image_processing.py

"""
Image processing operations.

This module implements image processing operations:
- image_resize: Resize images with optional aspect ratio preservation
- image_convert: Convert images to different formats (jpg, png, webp)

All operations are registered with the operation registry and can be
executed through the worker system.
"""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from PIL import Image, ImageOps, UnidentifiedImageError

from .base_processor import ProcessingResult, ImageProcessor, ErrorCategory
from .registry import (
    MediaType,
    ParameterSchema,
    ParameterType,
    get_registry,
)
from .exceptions import ProcessingError

logger = logging.getLogger(__name__)


# Supported image formats and their settings
IMAGE_FORMATS = {
    'jpg': {
        'extension': 'jpg',
        'pillow_format': 'JPEG',
        'mode': 'RGB',
        'supports_quality': True,
        'default_quality': 85,
    },
    'jpeg': {
        'extension': 'jpg',
        'pillow_format': 'JPEG',
        'mode': 'RGB',
        'supports_quality': True,
        'default_quality': 85,
    },
    'png': {
        'extension': 'png',
        'pillow_format': 'PNG',
        'mode': 'RGBA',
        'supports_quality': False,
        'default_quality': None,
    },
    'webp': {
        'extension': 'webp',
        'pillow_format': 'WEBP',
        'mode': 'RGBA',
        'supports_quality': True,
        'default_quality': 85,
    },
    'gif': {
        'extension': 'gif',
        'pillow_format': 'GIF',
        'mode': 'P',
        'supports_quality': False,
        'default_quality': None,
    },
    'bmp': {
        'extension': 'bmp',
        'pillow_format': 'BMP',
        'mode': 'RGB',
        'supports_quality': False,
        'default_quality': None,
    },
}

# Resampling methods
RESAMPLING_METHODS = {
    'lanczos': Image.Resampling.LANCZOS,
    'bicubic': Image.Resampling.BICUBIC,
    'bilinear': Image.Resampling.BILINEAR,
    'nearest': Image.Resampling.NEAREST,
}


class ImageResizeProcessor(ImageProcessor):
    """
    Processor for resizing images.
    
    Resizes images to specified dimensions with optional aspect ratio
    preservation using high-quality LANCZOS resampling.
    """
    @property
    def operation_name(self) -> str:
        return 'image_resize'
    

    def process_operation(self) -> ProcessingResult:
        """
        Execute image resize operation.
        
        Uses Pillow to resize images with configurable dimensions
        and aspect ratio preservation.
        
        Returns:
            ProcessingResult with output details
        """
        # Extract parameters
        target_width = self.parameters.get('width')
        target_height = self.parameters.get('height')
        maintain_aspect_ratio = self.parameters.get('maintain_aspect_ratio', True)
        resampling = self.parameters.get('resampling', 'lanczos')

        self.log_info(
            f"Starting image resize: width={target_width}, height={target_height}, "
            f"maintain_aspect_ratio={maintain_aspect_ratio}"
        )

        # Validate at least one dimension is provided
        if target_width is None and target_height is None:
            return self._create_error_result(
                "At least one of 'width' or 'height' must be specified",
                ErrorCategory.INVALID_PARAMS,
                is_retryable=False,
            )
        
        # Report initial progress
        self.update_progress(0)

        # Open and process the image
        try:
            with Image.open(self.input_path) as img:
                original_width, original_height = img.size
                original_mode = img.mode
                original_format = img.format

                self.log_info(
                    f"Original image: {original_width}x{original_height}, "
                    f"mode={original_mode}, format={original_format}"
                )

                # Calculate target dimensions
                new_width, new_height = self._calculate_dimensions(
                    original_width=original_width,
                    original_height=original_height,
                    target_width=target_width,
                    target_height=target_height,
                    maintain_aspect_ratio=maintain_aspect_ratio,
                )
                
                self.log_debug(f"Calculated dimensions: {new_width}x{new_height}")
                
                # Report progress
                self.update_progress(25)

                # Get resampling method
                resample_method = RESAMPLING_METHODS.get(
                    resampling.lower(),
                    Image.Resampling.LANCZOS
                )

                # Resize the image
                resized_img = img.resize((new_width, new_height), resample=resample_method)

                # Report progress
                self.update_progress(50)

                # Handle EXIF orientation
                resized_img = ImageOps.exif_transpose(resized_img) or resized_img

                # Generate output path
                output_filename = self.generate_output_filename(
                    suffix=f"_{new_width}x{new_height}",
                    extension=self._get_output_extension(original_format)
                )
                output_path = self.get_temp_file_path(output_filename)

                # Report progress
                self.update_progress(75)

                # Save the resized image
                save_kwargs = self._get_save_kwargs(original_format, original_mode)
                resized_img.save(output_path, **save_kwargs)
        
        except UnidentifiedImageError as e:
            self.log_error(f"Cannot identify image file: {e}")
            return self._create_error_result(
                "The file is not a valid image or the format is not supported",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        except PermissionError as e:
            self.log_error(f"Permission error: {e}")
            return self._create_error_result(
                f"Permission denied: {e}",
                ErrorCategory.PERMISSION,
                is_retryable=False,
            )
        except MemoryError as e:
            self.log_error(f"Memory error during resize: {e}")
            return self._create_error_result(
                "Not enough memory to process the image. Try a smaller image.",
                ErrorCategory.RESOURCE,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"Error during image resize: {e}")
            return self._create_error_result(
                f"Failed to resize image: {e}",
                ErrorCategory.UNKNOWN,
                is_retryable=False,
            )
        
        # Verify output was created
        if not self.validate_output_file_creation(output_path):
            return self._create_error_result(
                "Image resize completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)

        self.log_info(
            f"Resize complete: {original_width}x{original_height} -> {new_width}x{new_height}, "
            f"Input: {input_size / 1024:.1f}KB, Output: {output_size / 1024:.1f}KB"
        )
        
        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'original_width': original_width,
                'original_height': original_height,
                'new_width': new_width,
                'new_height': new_height,
                'maintain_aspect_ratio': maintain_aspect_ratio,
                'resampling': resampling,
                'input_size': input_size,
                'output_size': output_size,
                'format': original_format,
            }
        )
    

    def _calculate_dimensions(
        self,
        original_width: int,
        original_height: int,
        target_width: Optional[int],
        target_height: Optional[int],
        maintain_aspect_ratio: bool
    ) -> Tuple[int, int]:
        """
        Calculate the target dimensions for resizing.
        
        Args:
            original_width: Original image width
            original_height: Original image height
            target_width: Desired width (optional)
            target_height: Desired height (optional)
            maintain_aspect_ratio: Whether to preserve aspect ratio
            
        Returns:
            Tuple of (new_width, new_height)
        """
        if not maintain_aspect_ratio:
            # Use exact dimensions, defaulting to original if not specified
            return (
                target_width or original_width,
                target_height or original_height,
            )
        
        # Maintain aspect ratio
        aspect_ratio = original_width / original_height

        if target_width is not None and target_height is not None:
            # Both dimensions specified - fit within bounds
            target_ratio = target_width / target_height

            if aspect_ratio > target_ratio:
                # Image is wider - constrain by width
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Image is taller - constrain by height
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        
        elif target_width is not None:
            # Only width specified
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        
        else:
            # Only height specified
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions are at least 1 pixel
        return max(1, new_width), max(1, new_height)


    def _get_output_extension(self, original_format: Optional[str]) -> str:
        """Get the output file extension based on original format."""
        if original_format:
            format_lower = original_format.lower()
            if format_lower in IMAGE_FORMATS:
                return IMAGE_FORMATS[format_lower]['extension']
        return 'png'  # Default to PNG
    

    def _get_save_kwargs(
        self, 
        original_format: Optional[str],
        original_mode: str,
    ) -> Dict[str, Any]:
        """Get the save parameters for the output format."""
        kwargs = {}
        
        if original_format:
            format_lower = original_format.lower()
            if format_lower in IMAGE_FORMATS:
                format_info = IMAGE_FORMATS[format_lower]
                kwargs['format'] = format_info['pillow_format']
                
                if format_info['supports_quality']:
                    kwargs['quality'] = format_info['default_quality']
        
        return kwargs


class ImageConvertProcessor(ImageProcessor):
    """
    Processor for converting image formats.
    
    Converts images between different formats (JPEG, PNG, WebP, etc.)
    with configurable quality settings.
    """
    
    @property
    def operation_name(self) -> str:
        return "image_convert"
    

    def process_operation(self) -> ProcessingResult:
        """
        Execute image format conversion.
        
        Converts images to the specified format with optional
        quality settings.
        
        Returns:
            ProcessingResult with output details
        """
        # Extract parameters
        output_format = self.parameters.get('output_format', 'jpg').lower()
        quality = self.parameters.get('quality', 85)
        
        self.log_info(f"Starting image conversion to {output_format} with quality={quality}")
        
        # Validate output format
        if output_format not in IMAGE_FORMATS:
            return self._create_error_result(
                f"Unsupported output format: {output_format}. "
                f"Supported formats: {', '.join(IMAGE_FORMATS.keys())}",
                ErrorCategory.INVALID_PARAMS,
                is_retryable=False,
            )
        
        format_info = IMAGE_FORMATS[output_format]

        # Report initial progress
        self.update_progress(0)

        # Open and convert the image
        try:
            with Image.open(self.input_path) as img:
                original_width, original_height = img.size
                original_mode = img.mode
                original_format = img.format

                self.log_info(
                    f"Original image: {original_width}x{original_height}, "
                    f"mode={original_mode}, format={original_format}"
                )

                # Report progress
                self.update_progress(25)

                # Handle EXIF orientation
                img = ImageOps.exif_transpose(img) or img

                # Convert color mode if necessary
                converted_img = self._convert_color_mode(img, format_info['mode'])

                # Report progress
                self.update_progress(50)

                # Generate output filename and path
                output_filename = self.generate_output_filename(
                    suffix=f"_converted",
                    extension=format_info['extension']
                )
                output_path = self.get_temp_file_path(output_filename)

                # Report progress
                self.update_progress(75)

                # Build save kwargs
                save_kwargs = {
                    'format': format_info['pillow_format'],
                }
                
                if format_info['supports_quality']:
                    save_kwargs['quality'] = quality
                
                # Special handling for specific formats
                if output_format in ('jpg', 'jpeg'):
                    save_kwargs['optimize'] = True
                elif output_format == 'png':
                    save_kwargs['optimize'] = True
                elif output_format == 'webp':
                    save_kwargs['method'] = 4  # Good balance of speed and compression
                
                # Save the converted image
                converted_img.save(output_path, **save_kwargs)
                
        except UnidentifiedImageError as e:
            self.log_error(f"Cannot identify image file: {e}")
            return self._create_error_result(
                "The file is not a valid image or the format is not supported",
                ErrorCategory.INVALID_INPUT,
                is_retryable=False,
            )
        except PermissionError as e:
            self.log_error(f"Permission error: {e}")
            return self._create_error_result(
                f"Permission denied: {e}",
                ErrorCategory.PERMISSION,
                is_retryable=False,
            )
        except MemoryError as e:
            self.log_error(f"Memory error during conversion: {e}")
            return self._create_error_result(
                "Not enough memory to process the image. Try a smaller image.",
                ErrorCategory.RESOURCE,
                is_retryable=True,
            )
        except Exception as e:
            self.log_error(f"Error during image conversion: {e}")
            return self._create_error_result(
                f"Failed to convert image: {e}",
                ErrorCategory.UNKNOWN,
                is_retryable=False,
            )
        
        # Verify output was created
        if not self.validate_output_file_creation(output_path):
            return self._create_error_result(
                "Image conversion completed but output file was not created",
                ErrorCategory.UNKNOWN,
                is_retryable=True,
            )
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        input_size = os.path.getsize(self.input_path)
        size_change = ((output_size - input_size) / input_size * 100) if input_size > 0 else 0
        
        self.log_info(
            f"Conversion complete: {original_format} -> {output_format.upper()}, "
            f"Input: {input_size / 1024:.1f}KB, Output: {output_size / 1024:.1f}KB "
            f"({size_change:+.1f}%)"
        )
        
        return self.create_success_result(
            output_path=output_path,
            output_filename=output_filename,
            metadata={
                'input_format': original_format,
                'output_format': output_format,
                'width': original_width,
                'height': original_height,
                'quality': quality if format_info['supports_quality'] else None,
                'input_size': input_size,
                'output_size': output_size,
                'size_change_percent': round(size_change, 2),
            }
        )
    

    def _convert_color_mode(self, img: Image.Image, target_mode: str) -> Image.Image:
        """
        Convert image to the target color mode.
        
        Args:
            img: PIL Image object
            target_mode: Target color mode (RGB, RGBA, P, L)
            
        Returns:
            Converted image
        """
        current_mode = img.mode
        
        if current_mode == target_mode:
            return img
        
        # Handle conversion based on target mode
        if target_mode == 'RGB':
            # Converting to RGB (JPEG)
            if current_mode == 'RGBA':
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # Use alpha as mask
                return background
            elif current_mode == 'P':
                return img.convert('RGB')
            elif current_mode == 'L':
                return img.convert('RGB')
            elif current_mode == 'LA':
                return img.convert('RGBA').convert('RGB')
            else:
                return img.convert('RGB')
                
        elif target_mode == 'RGBA':
            # Converting to RGBA (PNG, WebP)
            if current_mode == 'P':
                # Preserve transparency in palette mode
                if 'transparency' in img.info:
                    return img.convert('RGBA')
                return img.convert('RGB').convert('RGBA')
            elif current_mode == 'L':
                return img.convert('RGBA')
            elif current_mode == 'LA':
                return img.convert('RGBA')
            else:
                return img.convert('RGBA')
                
        elif target_mode == 'P':
            # Converting to palette mode (GIF)
            if current_mode == 'RGBA':
                # Quantize with transparency
                return img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            return img.convert('P')
            
        else:
            # Default conversion
            return img.convert(target_mode)


# Factory functions for creating processors
def create_image_resize_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ImageResizeProcessor:
    """
    Create an ImageResizeProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input image
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        ImageResizeProcessor instance
    """
    return ImageResizeProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )


def create_image_convert_processor(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ImageConvertProcessor:
    """
    Create an ImageConvertProcessor instance.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input image
        parameters: Processing parameters
        progress_callback: Optional progress callback
        
    Returns:
        ImageConvertProcessor instance
    """
    return ImageConvertProcessor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )


# Handler functions for registry registration
def image_resize_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for image resize operation.
    
    This function is registered with the operation registry and called
    by the worker to execute image resizing.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input image
        parameters: Processing parameters (width, height, maintain_aspect_ratio)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_image_resize_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )

    return processor.execute_operation()


def image_convert_handler(
    operation_id: UUID,
    session_key: str,
    input_path: str,
    parameters: Dict[str, Any],
    progress_callback: Optional[Callable[[int, Optional[float]], None]] = None,
) -> ProcessingResult:
    """
    Handler function for image format conversion operation.
    
    This function is registered with the operation registry and called
    by the worker to execute image format conversion.
    
    Args:
        operation_id: UUID of the operation
        session_key: User's session key
        input_path: Path to input image
        parameters: Processing parameters (output_format, quality)
        progress_callback: Optional progress callback
        
    Returns:
        ProcessingResult with operation outcome
    """
    processor = create_image_convert_processor(
        operation_id=operation_id,
        session_key=session_key,
        input_path=input_path,
        parameters=parameters,
        progress_callback=progress_callback,
    )

    return processor.execute_operation()


# Register operations with the registry
def register_image_operations() -> None:
    """
    Register all image processing operations with the global registry.
    
    This function should be called during application startup to ensure
    all image operations are available. It is idempotent - calling it
    multiple times has no effect if operations are already registered.
    """
    registry = get_registry()
    
    # Register image_resize operation
    try:
        if not registry.is_registered('image_resize'):
            registry.register_operation(
                operation_name='image_resize',
                media_type=MediaType.IMAGE,
                handler=image_resize_handler,
                parameters=[
                    ParameterSchema(
                        param_name='width',
                        param_type=ParameterType.INTEGER,
                        required=False,
                        default=None,
                        description='Target width in pixels',
                        min_value=1,
                        max_value=16384,
                    ),
                    ParameterSchema(
                        param_name='height',
                        param_type=ParameterType.INTEGER,
                        required=False,
                        default=None,
                        description='Target height in pixels',
                        min_value=1,
                        max_value=16384,
                    ),
                    ParameterSchema(
                        param_name='maintain_aspect_ratio',
                        param_type=ParameterType.BOOLEAN,
                        required=False,
                        default=True,
                        description='Maintain original aspect ratio when resizing',
                    ),
                    ParameterSchema(
                        param_name='resampling',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='lanczos',
                        description='Resampling algorithm to use',
                        choices=['lanczos', 'bicubic', 'bilinear', 'nearest'],
                    ),
                ],
                description='Resize image to specified dimensions with optional aspect ratio preservation',
                input_formats=['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff'],
                output_formats=['jpg', 'png', 'webp', 'gif', 'bmp'],
            )
            logger.info("Registered image_resize operation")
    except Exception as e:
        # Already registered or other error - log and continue
        logger.debug(f"image_resize registration skipped: {e}")
    
    
    # Register image_convert operation
    try:
        if not registry.is_registered('image_convert'):
            registry.register_operation(
                operation_name='image_convert',
                media_type=MediaType.IMAGE,
                handler=image_convert_handler,
                parameters=[
                    ParameterSchema(
                        param_name='output_format',
                        param_type=ParameterType.CHOICE,
                        required=False,
                        default='jpg',
                        description='Output image format',
                        choices=['jpg', 'png', 'webp', 'gif', 'bmp'],
                    ),
                    ParameterSchema(
                        param_name='quality',
                        param_type=ParameterType.INTEGER,
                        required=False,
                        default=85,
                        description='Output quality (1-100, applicable to JPEG and WebP)',
                        min_value=1,
                        max_value=100,
                    ),
                ],
                description='Convert image to different formats with configurable quality',
                input_formats=['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'tiff'],
                output_formats=['jpg', 'png', 'webp', 'gif', 'bmp'],
            )
            logger.info("Registered image_convert operation")
    except Exception as e:
        # Already registered or other error - log and continue
        logger.debug(f"image_convert registration skipped: {e}")


# Register operations at module load
register_image_operations()