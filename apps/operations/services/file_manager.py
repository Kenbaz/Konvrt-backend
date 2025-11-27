# apps/operations/services/file_manager.py

"""
File Manager Service for handling file operations.

This module provides centralized file handling including:
- Saving uploaded files
- Validating files (size, type, format)
- Moving files between directories
- Cleaning up expired files
- Deleting operation files
"""

import logging
import os
import re
import shutil
import uuid
from typing import Optional, Tuple

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone

from ..exceptions import (
    FileTooLargeError,
    UnsupportedFileFormatError,
    FileNotFoundError as CustomFileNotFoundError,
    StorageError,
)

logger = logging.getLogger(__name__)


# MIME type to media type mapping
MIME_TYPE_MAP = {
    # Video MIME types
    "video/mp4": "video",
    "video/mpeg": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/x-matroska": "video",
    "video/webm": "video",
    "video/x-ms-wmv": "video",
    "video/x-flv": "video",
    "video/3gpp": "video",

    # Image MIME types
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/tiff": "image",
    "image/svg+xml": "image",

    # Audio MIME types
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/ogg": "audio",
    "audio/aac": "audio",
    "audio/flac": "audio",
    "audio/x-flac": "audio",
    "audio/mp4": "audio",
    "audio/x-m4a": "audio",
}


# Extension to MIME type mapping for common formats
EXTENSION_MIME_MAPPING = {
    # Video
    "mp4": "video/mp4",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "wmv": "video/x-ms-wmv",
    "flv": "video/x-flv",
    "3gp": "video/3gpp",

    # Image
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "svg": "image/svg+xml",

    # Audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/x-m4a",
    "wma": "audio/x-ms-wma",
}


class FileManager:
    """
    Service class for managing file operations.

    All methods are static as this is a stateless service
    """

    @staticmethod
    def save_uploaded_file(
        uploaded_file: UploadedFile,
        operation_id: str,
        session_key: str,
    ) -> dict:
        """
        Save an uploaded file to disk and return file information.
        
        Args:
            uploaded_file: The Django UploadedFile object
            operation_id: UUID of the operation
            session_key: User's session key
            
        Returns:
            Dictionary containing file information:
            - file_path: Relative path to saved file
            - file_name: Original filename (sanitized)
            - file_size: Size in bytes
            - mime_type: Detected MIME type
            - media_type: Category (video/image/audio)
            
        Raises:
            FileTooLargeError: If file exceeds size limit
            UnsupportedFileFormatError: If format not supported
            StorageError: If unable to save file
        """

        # Sanitize filename
        original_filename = uploaded_file.name
        sanitized_filename = FileManager.sanitize_filename(original_filename)
        extension = FileManager.get_file_extension(sanitized_filename)

        # Detect MIME type
        mime_type = FileManager.detect_mime_type(uploaded_file, extension)
        media_type = FileManager.get_media_type_from_mime_type(mime_type)

        if media_type is None:
            raise UnsupportedFileFormatError(
                filename=original_filename,
                extension=extension,
                media_type="unknown",
                supported_formats=list(EXTENSION_MIME_MAPPING.keys()),
            )
        
        # Get file size
        file_size = uploaded_file.size

        # Validate file
        FileManager.validate_file(
            filename=original_filename,
            file_size=file_size,
            extension=extension,
            media_type=media_type,
        )

        # Generate file path
        file_path, full_path = FileManager.get_uploaded_file_path(
            session_key=session_key,
            operation_id=operation_id,
            filename=sanitized_filename
        )

        # Ensure directory exists
        directory = os.path.dirname(full_path)
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise StorageError(
                operation="create_file_directory",
                path=directory,
                reason=str(e),
            )
        
        # Save file to disk
        try:
            with open(full_path, 'wb+') as destination:
                for chunck in uploaded_file.chunks():
                    destination.write(chunck)
        except (IOError, OSError) as e:
            raise StorageError(
                operation="save_file",
                path=full_path,
                reason=str(e),
            )
        
        logger.info(
            f"Saved uploaded file: {sanitized_filename} "
            f"(Operation={operation_id}, size={file_size}, type={mime_type})"
        )

        return {
            "file_path": file_path,
            "file_name": sanitized_filename,
            "file_size": file_size,
            "mime_type": mime_type,
            "media_type": media_type,
        }


    @staticmethod
    def validate_file(
        filename: str,
        file_size: int,
        extension: str,
        media_type: str
    ) -> None:
        """
        Validate a file against size and format constraints.
            
        Args:
            filename: Original filename
            file_size: Size in bytes
            extension: File extension (without dot)
            media_type: Media type category (video/image/audio)
                
        Raises:
            FileTooLargeError: If file exceeds size limit
            UnsupportedFileFormatError: If format not supported
        """
        
        # Get max file size for media type
        max_size = FileManager._get_max_file_size(media_type)

        # Check file size
        if file_size > max_size:
            raise FileTooLargeError(
                filename=filename,
                file_size=file_size,
                max_size=max_size,
                media_type=media_type,
            )
        
        # Get supported formats for media type
        supported_formats = FileManager._get_supported_formats(media_type)

        # Check if extension is supported
        if extension.lower() not in supported_formats:
            raise UnsupportedFileFormatError(
                filename=filename,
                extension=extension,
                media_type=media_type,
                supported_formats=supported_formats,
            )


    @staticmethod
    def detect_mime_type(uploaded_file: UploadedFile, extension: str) -> str:
        """
            Detect the MIME type of an uploaded file.
            
            First tries to use python-magic for accurate detection,
            falls back to extension-based detection if magic is unavailable.
            
            Args:
                uploaded_file: The uploaded file object
                extension: File extension
                
            Returns:
                Detected MIME type string
        """

        # First try content_type from upload
        content_type = getattr(uploaded_file, 'content_type', None)

        # Try usisng python-magic if available
        try:
            import magic

            # Read a small portion of the file for magic detection
            uploaded_file.seek(0)
            file_header = uploaded_file.read(8192)
            uploaded_file.seek(0)

            detected_type = magic.from_buffer(file_header, mime=True)
            if detected_type and detected_type != 'application/octet-stream':
                return detected_type
        except ImportError:
            logger.debug("python-magic not available, using fallback detection")
        except Exception as e:
            logger.warning(f"Magic detection failed: {e}")
        
        # Fall back to content type from upload headers
        if content_type and content_type in MIME_TYPE_MAP:
            return content_type
        
        # Fall back to extension-based detection
        ext = extension.lower()
        if ext in EXTENSION_MIME_MAPPING:
            return EXTENSION_MIME_MAPPING[ext]
        
        # Return a generic type based on extension
        return f"application/{ext}" if extension else "application/octet-stream"


    @staticmethod
    def get_media_type_from_mime_type(mime_type: str) -> Optional[str]:
        """
        Get the media type category from a MIME type.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            Media type category (video/image/audio) or None if unknown
        """

        # Direct lookup
        if mime_type in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[mime_type]
        
        # Try to infer MIME type prefix
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("audio/"):
            return "audio"
        
        return None
    

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a filename to remove dangerous characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem operations
        """

        if not filename:
            return f"file_{uuid.uuid4().hex[:8]}"
        
        # Get extension first
        name, ext = os.path.splitext(filename)

        # Remove path separators and null bytes
        name = name.replace('/', '_').replace('\\', '_').replace('\x00', '')

        # Replace special characters with underscore
        name = re.sub(r'[<>:"|?*]', '_', name)

        # Replace consecutive underscores
        name = re.sub(r'_+', '_', name)

        # Remove leading/trailing dots and underscores
        name = name.strip('_.')

        # Limit length (255 - extension length - 1 for the dot)
        max_name_length = 200 
        if len(name) > max_name_length:
            name = name[:max_name_length]
        
        # Ensure valid name
        if not name:
            name = f"file_{uuid.uuid4().hex[:8]}"
        
        # Clean extension
        ext = ext.lower()
        if ext:
            ext = re.sub(r'[^a-z0-9.]', '', ext)
        
        return f"{name}{ext}"
    

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Extract and normalize the file extension.
        
        Args:
            filename: Filename to extract extension from
            
        Returns:
            Lowercase extension without the dot
        """
        if not filename:
            return ""
        
        _, ext = os.path.splitext(filename)
        return ext.lstrip('.').lower()
    

    @staticmethod
    def get_uploaded_file_path(
        session_key: str,
        operation_id: str,
        filename: str
    ) -> Tuple[str, str]:
        """
        Generate paths for uploaded files.
        
        Args:
            session_key: User's session key
            operation_id: Operation UUID
            filename: Sanitized filename
            
        Returns:
            Tuple of (relative_path, full_path)
        """
        # Structure: uploads/{session_key}/{operation_id}/{filename}
        relative_path = os.path.join(
            "uploads",
            session_key[:32], # Truncate key for path safety
            str(operation_id),
            filename
        )

        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

        return relative_path, full_path
    

    @staticmethod
    def get_output_path(
        session_key: str,
        operation_id: str,
        filename: str
    ) -> Tuple[str, str]:
        """
        Generate paths for output files.
        
        Args:
            session_key: User's session key
            operation_id: Operation UUID
            filename: Output filename
            
        Returns:
            Tuple of (relative_path, full_path)
        """
        # Structure: outputs/{session_key}/{operation_id}/{filename}
        relative_path = os.path.join(
            "outputs",
            session_key[:32],
            str(operation_id),
            filename
        )

        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)

        return relative_path, full_path
    
    @staticmethod
    def move_to_output(
        temp_path: str,
        operation_id: str,
        session_key: str,
        output_filename: str,
    ) -> dict:
        """
        Move a processed file from temp to output directory.
        
        Args:
            temp_path: Full path to temporary file
            operation_id: Operation UUID
            session_key: User's session key
            output_filename: Desired output filename
            
        Returns:
            Dictionary containing output file information:
            - file_path: Relative path to output file
            - file_name: Output filename
            - file_size: Size in bytes
            - mime_type: MIME type
            
        Raises:
            FileNotFoundError: If temp file doesn't exist
            StorageError: If move operation fails
        """
        # Verify temp file exists
        if not os.path.exists(temp_path):
            raise CustomFileNotFoundError(temp_path)
        
        # Generate output path
        output_path, full_output_path = FileManager.get_output_path(
            session_key=session_key,
            operation_id=operation_id,
            filename=output_filename
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(full_output_path)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise StorageError(
                operation="create_directory",
                path=output_dir,
                reason=str(e),
            )
        
        # Move file
        try:
            shutil.move(temp_path, full_output_path)
        except (IOError, OSError, shutil.Error) as e:
            raise StorageError(
                operation="move_file",
                path=temp_path,
                reason=str(e),
            )
        
        # Get file info
        file_size = os.path.getsize(full_output_path)
        extension = FileManager.get_file_extension(output_filename)
        mime_type = EXTENSION_MIME_MAPPING.get(extension, "application/octet-stream")
        
        logger.info(
            f"Moved output file: {output_filename} "
            f"(operation={operation_id}, size={file_size})"
        )
        
        return {
            "file_path": output_path,
            "file_name": output_filename,
            "file_size": file_size,
            "mime_type": mime_type,
        }
    

    @staticmethod
    def delete_operation_files(operation_id: str, session_key: str) -> int:
        """
        Delete all files associated with an operation.
        
        Args:
            operation_id: Operation UUID
            session_key: User's session key
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0

        # Delete upload directory
        upload_dir = os.path.join(
            settings.MEDIA_ROOT,
            "uploads",
            session_key[:32],
            str(operation_id)
        )
        if os.path.exists(upload_dir):
            try:
                shutil.rmtree(upload_dir)
                deleted_count += 1
                logger.debug(f"Deleted upload directory: {upload_dir}")
            except OSError as e:
                logger.error(f"Failed to delete upload directory: {e}")
        
        # Delete output directory
        output_dir = os.path.join(
            settings.MEDIA_ROOT,
            "outputs",
            session_key[:32],
            str(operation_id)
        )
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                deleted_count += 1
                logger.debug(f"Deleted output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Failed to delete output directory: {e}")
        
        # Delete temp directory
        temp_dir = os.path.join(
            settings.MEDIA_ROOT,
            "temp",
            session_key[:32],
            str(operation_id)
        )
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                deleted_count += 1
                logger.debug(f"Deleted temp directory: {temp_dir}")
            except OSError as e:
                logger.error(f"Failed to delete temp directory: {e}")
        
        return deleted_count
    

    @staticmethod
    def cleanup_expired_files(dry_run: bool = False) -> dict:
        """
        Clean up files for expired operations.
        
        This method is called periodically (e.g., via cron job)
        to remove files for operations that have expired.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with cleanup statistics:
            - operations_cleaned: Number of operations processed
            - files_deleted: Number of file directories deleted
            - bytes_freed: Estimated bytes freed
            - errors: List of any errors encountered
        """
        from ..models import Operation
        from ..enums import OperationStatus

        stats = {
            "operations_cleaned": 0,
            "files_deleted": 0,
            "bytes_freed": 0,
            "errors": [],
        }

        # Find expired operations that havent been soft-deleted
        expired_operations = Operation.objects.filter(
            expires_at__lt=timezone.now(),
            is_deleted=False,
        ).select_related()

        for operation in expired_operations:
            try:
                # Calculate storage used before deletion
                bytes_used = FileManager._calculate_operation_storage(operation)

                if not dry_run:
                    deleted = FileManager.delete_operation_files(
                        operation_id=str(operation.id),
                        session_key=operation.session_key or "anonymous",
                    )

                    # Mark operation as deleted
                    operation.is_deleted = True
                    operation.save(update_fields=['is_deleted'])

                    stats["files_deleted"] += deleted

                stats["operations_cleaned"] += 1
                stats["bytes_freed"] += bytes_used
            
            except Exception as e:
                error_msg = f"Error cleaning operation {operation.id}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        logger.info(
            f"Cleanup complete: {stats['operations_cleaned']} operations, "
            f"{stats['bytes_freed'] / (1024*1024):.2f} MB freed"
        )

        return stats
    

    @staticmethod
    def get_file_full_path(file_path: str) -> str:
        """
        Get the full filesystem path for a relative file path.
        
        Args:
            file_path: Relative path from MEDIA_ROOT
            
        Returns:
            Full filesystem path
        """
        return os.path.join(settings.MEDIA_ROOT, file_path)
    
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Check if a file exists at the given relative path.
        
        Args:
            file_path: Relative path from MEDIA_ROOT
            
        Returns:
            True if file exists, False otherwise
        """
        full_path = FileManager.get_file_full_path(file_path)
        return os.path.exists(full_path)
    

    @staticmethod
    def get_temp_path(operation_id: str, filename: str) -> Tuple[str, str]:
        """
        Generate paths for temporary files during processing.
        
        Args:
            operation_id: Operation UUID
            filename: Temporary filename
            
        Returns:
            Tuple of (relative_path, full_path)
        """
        relative_path = os.path.join(
            "temp",
            str(operation_id),
            filename,
        )
        
        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        
        return relative_path, full_path
    

    @staticmethod
    def ensure_temp_directory(operation_id: str) -> str:
        """
        Ensure the temp directory exists for an operation.
        
        Args:
            operation_id: Operation UUID
            
        Returns:
            Full path to temp directory
        """
        temp_dir = os.path.join(
            settings.MEDIA_ROOT,
            "temp",
            str(operation_id),
        )
        
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    

    @staticmethod
    def cleanup_temp_directory(operation_id: str) -> bool:
        """
        Clean up the temp directory for an operation.
        
        Args:
            operation_id: Operation UUID
            
        Returns:
            True if cleaned successfully, False otherwise
        """
        temp_dir = os.path.join(
            settings.MEDIA_ROOT,
            "temp",
            str(operation_id),
        )
        
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                return True
            except OSError as e:
                logger.error(f"Failed to cleanup temp directory: {e}")
                return False
        
        return True
    

    @staticmethod
    def _get_max_file_size(media_type: str) -> int:
        """
        Get the maximum file size for a media type.
        
        Args:
            media_type: Media type category
            
        Returns:
            Maximum file size in bytes
        """
        max_sizes = getattr(settings, 'MAX_FILE_SIZE', {})
        
        defaults = {
            'video': 524288000,  # 500MB
            'image': 52428800,   # 50MB
            'audio': 104857600,  # 100MB
        }
        
        return max_sizes.get(media_type, defaults.get(media_type, 52428800))
    

    @staticmethod
    def _get_supported_formats(media_type: str) -> list:
        """
        Get the supported formats for a media type.
        
        Args:
            media_type: Media type category
            
        Returns:
            List of supported format extensions
        """
        supported = getattr(settings, 'SUPPORTED_FORMATS', {})
        
        defaults = {
            'video': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
            'image': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
            'audio': ['mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'],
        }
        
        return supported.get(media_type, defaults.get(media_type, []))
    

    @staticmethod
    def _calculate_operation_storage(operation) -> int:
        """
        Calculate total storage used by an operation.
        
        Args:
            operation: Operation model instance
            
        Returns:
            Total bytes used by operation files
        """
        total_bytes = 0
        
        for file_record in operation.files.all():
            if file_record.file_size:
                total_bytes += file_record.file_size
        
        return total_bytes