# apps/operations/services/cloudinary_storage.py

"""
Cloudinary Storage Service for handling cloud file operations.

This module provides centralized cloud storage handling including:
- Uploading files to Cloudinary (with chunked upload for large files)
- Downloading files from Cloudinary to local temp
- Deleting files from Cloudinary
- Generating secure download URLs
- Checking file existence
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple, BinaryIO
from urllib.parse import urlparse

import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from ..exceptions import StorageError


class CloudinaryStorageError(Exception):
    def __init__(self, operation: str, path: str, reason: str):
        self.operation = operation
        self.path = path
        self.reason = reason
        self.message = f"Storage {operation} failed for '{path}': {reason}"
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary representation."""
        return {
            "code": "STORAGE_ERROR",
            "message": self.message,
            "details": {
                "operation": self.operation,
                "path": self.path,
                "reason": self.reason,
            }
        }

CloudinaryStorageError = StorageError

logger = logging.getLogger(__name__)

def _get_user_friendly_error_message(error: Exception) -> str:
    """
    Convert technical error messages to user-friendly messages
    """
    error_str = str(error).lower()
    
    error_mappings = [
        # File size errors
        ('file size too large', 'File is too large. Please upload a smaller file.'),
        ('maximum is', 'File exceeds the maximum allowed size. Please upload a smaller file.'),
        
        # Network/Connection errors
        ('ssl', 'Connection error occurred. Please check your internet and try again.'),
        ('timeout', 'Upload timed out. Please try again.'),
        ('connection', 'Connection failed. Please check your internet and try again.'),
        ('network', 'Network error occurred. Please try again.'),
        ('eof occurred', 'Connection was interrupted. Please try again.'),
        ('broken pipe', 'Connection was lost. Please try again.'),
        ('max retries exceeded', 'Upload failed after multiple attempts. Please try again later.'),
        
        # Authentication/Permission errors
        ('invalid api', 'Storage service configuration error. Please contact support.'),
        ('unauthorized', 'Storage service authentication failed. Please contact support.'),
        ('forbidden', 'Access denied to storage service. Please contact support.'),
        
        # Resource errors
        ('not found', 'File not found. It may have been deleted.'),
        ('does not exist', 'The requested file does not exist.'),
        
        # Rate limiting
        ('rate limit', 'Too many requests. Please wait a moment and try again.'),
        ('too many', 'Too many requests. Please wait a moment and try again.'),
        
        # Storage errors
        ('quota', 'Storage quota exceeded. Please contact support.'),
        ('disk', 'Storage error occurred. Please try again later.'),
    ]
    
    for pattern, friendly_message in error_mappings:
        if pattern in error_str:
            return friendly_message
    
    # Default message for unknown errors
    return 'An error occurred while processing your file. Please try again.'

@dataclass
class CloudinaryUploadResult:
    public_id: str
    secure_url: str
    resource_type: str
    format: str
    bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None

@dataclass 
class CloudinaryDownloadResult:
    local_path: str
    file_size: int


class CloudinaryStorageService:
    """
    Service class for managing Cloudinary storage operations.
    
    Handles uploads, downloads, deletions, and URL generation for
    media files stored in Cloudinary.
    """

    RESOURCE_TYPE_MAP = {
        'image': 'image',
        'video': 'video',
        'audio': 'video', 
    }

    # Upload configuration
    CHUNK_SIZE = 6 * 1024 * 1024  # 6MB
    LARGE_FILE_THRESHOLD = 20 * 1024 * 1024  # 20MB
    MAX_RETRIES = 5 
    RETRY_BASE_DELAY = 2
    UPLOAD_TIMEOUT = 300 # 5 minutes

    @staticmethod
    def _get_root_folder() -> str:
        return settings.CLOUDINARY_STORAGE.get('ROOT_FOLDER', 'mediaprocessor')
    
    @staticmethod
    def _get_resource_type(media_type: str) -> str:
        resource_map = settings.CLOUDINARY_STORAGE.get(
            'RESOURCE_TYPE_MAP', CloudinaryStorageService.RESOURCE_TYPE_MAP
        )
        return resource_map.get(media_type, 'auto')
    
    @staticmethod
    def _build_public_id(
        folder_type: str,
        session_key: str,
        operation_id: str,
        filename: str
    ) -> str:
        root_folder = CloudinaryStorageService._get_root_folder()

        name_without_ext = os.path.splitext(filename)[0]

        safe_session = session_key[:32] if session_key else 'anonymous'

        public_id = f"{root_folder}/{folder_type}/{safe_session}/{operation_id}/{name_without_ext}"

        return public_id
    
    @staticmethod
    def upload_file(
        file_source: UploadedFile | str | BinaryIO,
        folder_type: str,
        session_key: str,
        operation_id: str,
        filename: str,
        media_type: str,
    ) -> CloudinaryUploadResult:
        if not settings.USE_CLOUDINARY:
            raise StorageError(
                operation="upload",
                path=filename,
                reason="Cloudinary storage is disabled in settings."
            )
        
        file_size = CloudinaryStorageService._get_file_size(file_source)
        max_file_size = CloudinaryStorageService._get_max_file_size()
        
        if file_size > max_file_size:
            max_mb = max_file_size / (1024 * 1024)
            file_mb = file_size / (1024 * 1024)
            raise StorageError(
                operation="upload",
                path=filename,
                reason=(
                    f"File is too large ({file_mb:.1f} MB). "
                    f"Maximum allowed size is {max_mb:.0f} MB. "
                    f"Please upload a smaller file"
                )
            )
        
        public_id = CloudinaryStorageService._build_public_id(
            folder_type=folder_type,
            session_key=session_key,
            operation_id=operation_id,
            filename=filename
        )

        resource_type = CloudinaryStorageService._get_resource_type(media_type)

        use_chunked = file_size > CloudinaryStorageService.LARGE_FILE_THRESHOLD

        if use_chunked:
            logger.info(
                f"Using chunked upload for large file: {filename} "
                f"({file_size / (1024 * 1024):.2f} MB)"
            )
            return CloudinaryStorageService._upload_large_file(
                file_source=file_source,
                public_id=public_id,
                resource_type=resource_type,
                filename=filename,
                file_size=file_size
            )
        else:
            return CloudinaryStorageService._upload_with_retry(
                file_source=file_source,
                public_id=public_id,
                resource_type=resource_type,
                filename=filename
            )

    @staticmethod
    def _get_file_size(file_source: UploadedFile | str | BinaryIO) -> int:
        """Get the size of a file from various source types."""
        if isinstance(file_source, str):
            if os.path.exists(file_source):
                return os.path.getsize(file_source)
            return 0
        elif isinstance(file_source, UploadedFile):
            return file_source.size
        elif hasattr(file_source, 'seek') and hasattr(file_source, 'tell'):
            # For file-like objects, get size by seeking to end
            current_pos = file_source.tell()
            file_source.seek(0, 2)  # Seek to end
            size = file_source.tell()
            file_source.seek(current_pos)  # Restore position
            return size
        return 0

    @staticmethod
    def _upload_with_retry(
        file_source: UploadedFile | str | BinaryIO,
        public_id: str,
        resource_type: str,
        filename: str
    ) -> CloudinaryUploadResult:
        """Upload a file with exponential backoff retry logic."""
        
        upload_options = {
            'public_id': public_id,
            'resource_type': resource_type,
            'overwrite': True,
            'invalidate': True,
            'unique_filename': False,
            'use_filename': False,
            'timeout': CloudinaryStorageService.UPLOAD_TIMEOUT,
        }

        last_error = None
        
        for attempt in range(CloudinaryStorageService.MAX_RETRIES):
            try:
                # Reset file position for retries
                if hasattr(file_source, 'seek'):
                    file_source.seek(0)

                # Handle different file source types
                if isinstance(file_source, str):
                    if not os.path.exists(file_source):
                        raise StorageError(
                            operation="upload",
                            path=file_source,
                            reason="Source file does not exist"
                        )
                    result = cloudinary.uploader.upload(file_source, **upload_options)
                elif isinstance(file_source, UploadedFile):
                    file_source.seek(0)
                    result = cloudinary.uploader.upload(file_source, **upload_options)
                else:
                    result = cloudinary.uploader.upload(file_source, **upload_options)

                logger.info(
                    f"Uploaded file to Cloudinary: {public_id} "
                    f"(resource_type={resource_type}, bytes={result.get('bytes', 0)})"
                )

                return CloudinaryUploadResult(
                    public_id=result['public_id'],
                    secure_url=result['secure_url'],
                    resource_type=result['resource_type'],
                    format=result.get('format', ''),
                    bytes=result.get('bytes', 0),
                    width=result.get('width'),
                    height=result.get('height'),
                    duration=result.get('duration'),
                )

            except (cloudinary.exceptions.Error, Exception) as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a retryable error
                is_retryable = any(err_type in error_str.lower() for err_type in [
                    'ssl', 'timeout', 'connection', 'eof', 'retry', 'network',
                    'reset', 'broken pipe', 'temporary'
                ])
                
                if not is_retryable:
                    logger.error(f"Non-retryable Cloudinary upload error for {public_id}: {e}")
                    user_message = _get_user_friendly_error_message(e)
                    raise StorageError(
                        operation="upload",
                        path=filename,
                        reason=user_message
                    )
                
                if attempt < CloudinaryStorageService.MAX_RETRIES - 1:
                    # Calculate exponential backoff delay
                    delay = CloudinaryStorageService.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Cloudinary upload attempt {attempt + 1} failed for {public_id}, "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Cloudinary upload failed after {CloudinaryStorageService.MAX_RETRIES} "
                        f"attempts for {public_id}: {e}"
                    )

        # All retries exhausted - use user-friendly message
        user_message = _get_user_friendly_error_message(last_error)
        raise StorageError(
            operation="upload",
            path=filename,
            reason=user_message
        )

    @staticmethod
    def _upload_large_file(
        file_source: UploadedFile | str | BinaryIO,
        public_id: str,
        resource_type: str,
        filename: str,
        file_size: int
    ) -> CloudinaryUploadResult:
        """
        Upload large files using Cloudinary's chunked upload.
        
        Chunked upload is more resilient to network issues as each chunk
        is uploaded separately and can be retried independently.
        """
        
        upload_options = {
            'public_id': public_id,
            'resource_type': resource_type,
            'overwrite': True,
            'invalidate': True,
            'unique_filename': False,
            'use_filename': False,
            'chunk_size': CloudinaryStorageService.CHUNK_SIZE,
            'timeout': CloudinaryStorageService.UPLOAD_TIMEOUT,
        }

        last_error = None
        
        for attempt in range(CloudinaryStorageService.MAX_RETRIES):
            try:
                # Reset file position for retries
                if hasattr(file_source, 'seek'):
                    file_source.seek(0)

                if isinstance(file_source, str):
                    if not os.path.exists(file_source):
                        raise StorageError(
                            operation="upload",
                            path=file_source,
                            reason="Source file does not exist"
                        )
                    # Upload from file path - Cloudinary handles chunking
                    result = cloudinary.uploader.upload_large(
                        file_source, 
                        **upload_options
                    )
                elif isinstance(file_source, UploadedFile):
                    # For Django UploadedFile, save to temp file first for reliable chunked upload
                    result = CloudinaryStorageService._upload_uploaded_file_chunked(
                        file_source, upload_options
                    )
                else:
                    # For file-like objects, try direct upload
                    result = cloudinary.uploader.upload_large(
                        file_source, 
                        **upload_options
                    )

                logger.info(
                    f"Uploaded large file to Cloudinary (chunked): {public_id} "
                    f"(resource_type={resource_type}, bytes={result.get('bytes', 0)})"
                )

                return CloudinaryUploadResult(
                    public_id=result['public_id'],
                    secure_url=result['secure_url'],
                    resource_type=result['resource_type'],
                    format=result.get('format', ''),
                    bytes=result.get('bytes', 0),
                    width=result.get('width'),
                    height=result.get('height'),
                    duration=result.get('duration'),
                )

            except (cloudinary.exceptions.Error, Exception) as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a retryable error
                is_retryable = any(err_type in error_str.lower() for err_type in [
                    'ssl', 'timeout', 'connection', 'eof', 'retry', 'network',
                    'reset', 'broken pipe', 'temporary'
                ])
                
                if not is_retryable:
                    logger.error(f"Non-retryable Cloudinary chunked upload error for {public_id}: {e}")
                    user_message = _get_user_friendly_error_message(e)
                    raise StorageError(
                        operation="upload",
                        path=filename,
                        reason=user_message
                    )
                
                if attempt < CloudinaryStorageService.MAX_RETRIES - 1:
                    delay = CloudinaryStorageService.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Cloudinary chunked upload attempt {attempt + 1} failed for {public_id}, "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Cloudinary chunked upload failed after {CloudinaryStorageService.MAX_RETRIES} "
                        f"attempts for {public_id}: {e}"
                    )

        user_message = _get_user_friendly_error_message(last_error)
        raise StorageError(
            operation="upload",
            path=filename,
            reason=user_message
        )

    @staticmethod
    def _upload_uploaded_file_chunked(
        uploaded_file: UploadedFile,
        upload_options: dict
    ) -> dict:
        """
        Handle chunked upload for Django UploadedFile objects.
        """
        temp_file_path = None
        try:
            # Create a temp file to store the uploaded file
            suffix = os.path.splitext(uploaded_file.name)[1] if uploaded_file.name else ''
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name
                uploaded_file.seek(0)
                
                # Write in chunks to handle large files efficiently
                for chunk in uploaded_file.chunks(chunk_size=8192):
                    temp_file.write(chunk)
            
            # Upload from temp file
            result = cloudinary.uploader.upload_large(
                temp_file_path,
                **upload_options
            )
            
            return result
            
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")
    
    @staticmethod
    def upload_from_path(
        file_path: str,
        folder_type: str,
        session_key: str,
        operation_id: str,
        media_type: str,
        filename: str
    ) -> CloudinaryUploadResult:
        """Upload a file from local path to Cloudinary storage"""

        return CloudinaryStorageService.upload_file(
            file_source=file_path,
            folder_type=folder_type,
            session_key=session_key,
            operation_id=operation_id,
            filename=filename,
            media_type=media_type,
        )
    
    @staticmethod
    def download_file(
        public_id: str,
        resource_type: str,
        destination_dir: Optional[str] = None
    ) -> CloudinaryDownloadResult:
        """Download a file from cloudinary to local storage"""

        import requests

        if not settings.USE_CLOUDINARY:
            raise StorageError(
                operation="download",
                path=public_id,
                reason="Cloudinary storage is disabled in settings."
            )
        
        try:
            # Generate download url
            url, _ = cloudinary_url(
                public_id,
                resource_type=resource_type,
                secure=True
            )

            if not url:
                raise StorageError(
                    operation="download",
                    path=public_id,
                    reason="Failed to generate download URL"
                )
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Determine file extension
            content_type = response.headers.get('Content-Type', '')
            extension = CloudinaryStorageService._get_extension_from_content_type(content_type)
            if not extension:
                parsed_url = urlparse(url)
                extension = os.path.splitext(parsed_url.path)[1] or '.tmp'
            
            # Create destination path
            if destination_dir:
                os.makedirs(destination_dir, exist_ok=True)
                filename = os.path.basename(public_id) + extension
                local_path = os.path.join(destination_dir, filename)
            else:
                # Use temp file
                fd, local_path = tempfile.mkstemp(suffix=extension)
                os.close(fd)
            
            # Write file to disk
            file_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        file_size += len(chunk)
            logger.info(f"Downloaded file from Cloudinary: {public_id} to {local_path} ({file_size} bytes)")

            return CloudinaryDownloadResult(
                local_path=local_path,
                file_size=file_size
            )
        
        except requests.RequestException as e:
            logger.error(f"Failed to download from Cloudinary: {public_id}: {e}")
            raise StorageError(
                operation="download",
                path=public_id,
                reason=f"Download failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error downloading from Cloudinary: {e}")
            raise StorageError(
                operation="download",
                path=public_id,
                reason=f"Download failed: {str(e)}"
            )
    
    @staticmethod
    def delete_file(public_id: str, resource_type: str) -> bool:
        if not settings.USE_CLOUDINARY:
            raise StorageError(
                operation="delete",
                path=public_id,
                reason="Cloudinary storage is disabled in settings."
            )
        
        try:
            result = cloudinary.uploader.destroy(
                public_id,
                resource_type=resource_type,
                invalidate=True
            )

            success = result.get('result') == 'ok'

            if success:
                logger.info(f"Deleted file from Cloudinary: {public_id}")
            else:
                logger.warning(f"Failed to delete file from Cloudinary: {public_id}, result: {result}")
            
            return success
        
        except cloudinary.exceptions.Error as e:
            logger.error(f"Cloudinary delete failed for {public_id}: {e}")
            raise StorageError(
                operation="delete",
                path=public_id,
                reason=f"Cloudinary delete failed: {str(e)}"
            )
    
    @staticmethod
    def delete_folder(folder_path: str) -> int:
        if not settings.USE_CLOUDINARY:
            raise StorageError(
                operation="delete_folder",
                path=folder_path,
                reason="Cloudinary storage is disabled in settings."
            )
        
        deleted_count = 0
        try:
            for resource_type in ['image', 'video', 'raw']:
                try:
                    result = cloudinary.api.delete_resource_by_prefix(
                        folder_path,
                        resource_type=resource_type,
                        invalidate=True
                    )
                    deleted = result.get('deleted', {})
                    deleted_count += len(deleted)
                except cloudinary.exceptions.Error:
                    pass
            
            try:
                cloudinary.api.delete_folder(folder_path)
            except cloudinary.exceptions.Error:
                pass

            logger.info(f"Deleted {deleted_count} files from Cloudinary folder: {folder_path}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting Cloudinary folder {folder_path}: {e}")
            return deleted_count
    
    @staticmethod
    def get_download_url(
        public_id: str,
        resource_type: str,
        attachment: bool = True,
        expires_in: Optional[int] = None
    ) -> str:
        """Generate a secure download URL for a Cloudinary file"""

        if not settings.USE_CLOUDINARY:
            return ""
        
        try:
            options = {
                'resource_type': resource_type,
                'secure': True,
            }

            if attachment:
                options['flags'] = 'attachment'
            
            if expires_in:
                options['sign_url'] = True
                options['type'] = 'authenticated'
            
            url, _ = cloudinary_url(public_id, **options)

            return url or ""
        
        except Exception as e:
            logger.error(f"Error generating Cloudinary URL for {public_id}: {e}")
            return ""
    
    @staticmethod
    def get_secure_url(
        public_id: str,
        resource_type: str
    ) -> str:
        """Generate a secure URL for a cloudinary file"""
        return CloudinaryStorageService.get_download_url(
            public_id=public_id,
            resource_type=resource_type,
            attachment=False,
            expires_in=None
        )
    
    @staticmethod
    def file_exists(public_id: str, resource_type: str) -> bool:
        """
        Check if a file exists in Cloudinary
        """
        if not settings.USE_CLOUDINARY:
            return False
        
        try:
            result = cloudinary.api.resource(
                public_id,
                resource_type=resource_type,
            )
            return bool(result.get('public_id'))
        except cloudinary.exceptions.NotFound:
            return False
        except cloudinary.exceptions.Error as e:
            logger.warning(f"Error checking Cloudinary file existence for {public_id}: {e}")
            return False
    
    @staticmethod
    def get_file_info(public_id: str, resource_type: str) -> Optional[dict]:
        """
        Get detailed information about a file in Cloudinary
        """
        if not settings.USE_CLOUDINARY:
            return None
        
        try:
            result = cloudinary.api.resource(
                public_id,
                resource_type=resource_type,
            )
            return {
                'public_id': result.get('public_id'),
                'format': result.get('format'),
                'resource_type': result.get('resource_type'),
                'bytes': result.get('bytes', 0),
                'width': result.get('width'),
                'height': result.get('height'),
                'duration': result.get('duration'),
                'created_at': result.get('created_at'),
                'secure_url': result.get('secure_url'),
            }
        except cloudinary.exceptions.NotFound:
            return None
        except cloudinary.exceptions.Error as e:
            logger.warning(f"Error getting Cloudinary file info for {public_id}: {e}")
            return None
    
    @staticmethod
    def check_connection() -> Tuple[bool, str]:
        """
        Check if Cloudinary connection is working
        """
        if not settings.USE_CLOUDINARY:
            return True, "Cloudinary disabled (local storage mode)"
        
        if not settings.CLOUDINARY_CLOUD_NAME:
            return False, "Cloudinary cloud name not configured"
        
        if not settings.CLOUDINARY_API_KEY:
            return False, "Cloudinary API key not configured"
        
        if not settings.CLOUDINARY_API_SECRET:
            return False, "Cloudinary API secret not configured"
        
        try:
            # Try to ping Cloudinary by fetching account info
            result = cloudinary.api.ping()
            if result.get('status') == 'ok':
                return True, "Cloudinary connection OK"
            return False, f"Cloudinary ping returned: {result}"
        except cloudinary.exceptions.Error as e:
            return False, f"Cloudinary connection failed: {str(e)}"
        except Exception as e:
            return False, f"Cloudinary check failed: {str(e)}"
    
    @staticmethod
    def _get_extension_from_content_type(content_type: str) -> str:
        """
        Get file extension from content type
        """
        content_type_map = {
            'video/mp4': '.mp4',
            'video/quicktime': '.mov',
            'video/x-msvideo': '.avi',
            'video/x-matroska': '.mkv',
            'video/webm': '.webm',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/bmp': '.bmp',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav',
            'audio/aac': '.aac',
            'audio/ogg': '.ogg',
            'audio/flac': '.flac',
            'audio/x-m4a': '.m4a',
            'audio/mp4': '.m4a',
        }
        return content_type_map.get(content_type, '')