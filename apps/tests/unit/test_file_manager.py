import os
import pytest
from unittest.mock import patch, MagicMock

from apps.operations.services.file_manager import (
    FileManager,
)
from apps.operations.exceptions import (
    FileTooLargeError,
    UnsupportedFileFormatError,
    FileNotFoundError as CustomFileNotFoundError,
)


class MockUploadedFile:
    """Mock Django UploadedFile for testing."""
    
    def __init__(
        self,
        name: str = "test_file.mp4",
        size: int = 1024,
        content_type: str = "video/mp4",
        content: bytes = None,
    ):
        self.name = name
        self.size = size
        self.content_type = content_type
        self._content = content or b"fake file content" * (size // 17 + 1)
        self._content = self._content[:size]
        self._position = 0
    
    def read(self, size: int = None) -> bytes:
        if size is None:
            data = self._content[self._position:]
            self._position = len(self._content)
        else:
            data = self._content[self._position:self._position + size]
            self._position += size
        return data
    
    def seek(self, position: int) -> None:
        self._position = position
    
    def chunks(self, chunk_size: int = 8192):
        self.seek(0)
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk


class TestSanitizeFilename:
    """Tests for filename sanitization."""
    
    def test_sanitize_normal_filename(self):
        """Test sanitizing a normal filename."""
        result = FileManager.sanitize_filename("video.mp4")
        assert result == "video.mp4"
    
    def test_sanitize_filename_with_spaces(self):
        """Test sanitizing filename with spaces."""
        result = FileManager.sanitize_filename("my video file.mp4")
        assert result == "my video file.mp4"
    
    def test_sanitize_filename_with_path_separators(self):
        """Test sanitizing filename with path separators."""
        result = FileManager.sanitize_filename("../../etc/passwd")
        assert "/" not in result
        assert "\\" not in result
    
    def test_sanitize_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        result = FileManager.sanitize_filename('file<>:"|?*.mp4')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert '"' not in result
        assert "|" not in result
        assert "?" not in result
        assert "*" not in result
    
    def test_sanitize_empty_filename(self):
        """Test sanitizing empty filename generates a valid name."""
        result = FileManager.sanitize_filename("")
        assert result.startswith("file_")
        assert len(result) > 0
    
    def test_sanitize_none_filename(self):
        """Test sanitizing None filename generates a valid name."""
        result = FileManager.sanitize_filename(None)
        assert result.startswith("file_")
    
    def test_sanitize_long_filename(self):
        """Test that long filenames are truncated."""
        long_name = "a" * 300 + ".mp4"
        result = FileManager.sanitize_filename(long_name)
        assert len(result) <= 205  # 200 chars name + 4 for .mp4 + 1 buffer
    
    def test_sanitize_preserves_extension(self):
        """Test that extension is preserved and lowercased."""
        result = FileManager.sanitize_filename("VIDEO.MP4")
        assert result.endswith(".mp4")
    
    def test_sanitize_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        result = FileManager.sanitize_filename("file___name.mp4")
        assert "___" not in result


class TestGetFileExtension:
    """Tests for file extension extraction."""
    
    def test_get_extension_normal(self):
        """Test extracting normal extension."""
        assert FileManager.get_file_extension("video.mp4") == "mp4"
    
    def test_get_extension_uppercase(self):
        """Test that extension is lowercased."""
        assert FileManager.get_file_extension("VIDEO.MP4") == "mp4"
    
    def test_get_extension_double(self):
        """Test extracting from double extension."""
        assert FileManager.get_file_extension("archive.tar.gz") == "gz"
    
    def test_get_extension_no_extension(self):
        """Test file with no extension."""
        assert FileManager.get_file_extension("filename") == ""
    
    def test_get_extension_empty_filename(self):
        """Test empty filename."""
        assert FileManager.get_file_extension("") == ""
    
    def test_get_extension_dot_only(self):
        """Test file with dot but no extension."""
        assert FileManager.get_file_extension("file.") == ""


class TestDetectMimeType:
    """Tests for MIME type detection."""
    
    def test_detect_from_content_type_in_mapping(self):
        """Test detection from content type header when in mapping."""
        uploaded_file = MockUploadedFile(
            name="video.mp4",
            content_type="video/mp4",
        )
        
        # Mock magic to raise ImportError so we fall back to content_type
        with patch.dict('sys.modules', {'magic': None}):
            with patch('apps.operations.services.file_manager.logger'):
                mime_type = FileManager.detect_mime_type(uploaded_file, "mp4")
                # Should return video/mp4 from content_type or extension
                assert mime_type == "video/mp4"
    
    def test_detect_fallback_to_extension(self):
        """Test fallback to extension-based detection."""
        uploaded_file = MockUploadedFile(
            name="video.mp4",
            content_type="application/octet-stream",
        )
        
        # Mock magic to raise ImportError so we fall back to extension
        with patch.dict('sys.modules', {'magic': None}):
            mime_type = FileManager.detect_mime_type(uploaded_file, "mp4")
            # Should fall back to extension-based detection
            assert mime_type == "video/mp4"
    
    def test_detect_image_extensions(self):
        """Test detection for various image extensions."""
        test_cases = [
            ("jpg", "image/jpeg"),
            ("jpeg", "image/jpeg"),
            ("png", "image/png"),
            ("gif", "image/gif"),
            ("webp", "image/webp"),
        ]
        
        for ext, expected_mime in test_cases:
            uploaded_file = MockUploadedFile(
                name=f"image.{ext}",
                content_type="application/octet-stream",
            )
            # Force extension-based detection by using unknown content type
            with patch.dict('sys.modules', {'magic': None}):
                mime_type = FileManager.detect_mime_type(uploaded_file, ext)
                assert mime_type == expected_mime, f"Failed for extension {ext}"
    
    def test_detect_audio_extensions(self):
        """Test detection for audio extensions."""
        test_cases = [
            ("mp3", "audio/mpeg"),
            ("wav", "audio/wav"),
            ("ogg", "audio/ogg"),
        ]
        
        for ext, expected_mime in test_cases:
            uploaded_file = MockUploadedFile(
                name=f"audio.{ext}",
                content_type="application/octet-stream",
            )
            with patch.dict('sys.modules', {'magic': None}):
                mime_type = FileManager.detect_mime_type(uploaded_file, ext)
                assert mime_type == expected_mime, f"Failed for extension {ext}"


class TestGetMediaTypeFromMime:
    """Tests for media type categorization."""
    
    def test_video_mime_types(self):
        """Test video MIME types return 'video'."""
        video_mimes = [
            "video/mp4",
            "video/mpeg",
            "video/quicktime",
            "video/x-msvideo",
            "video/webm",
        ]
        
        for mime in video_mimes:
            assert FileManager.get_media_type_from_mime_type(mime) == "video"
    
    def test_image_mime_types(self):
        """Test image MIME types return 'image'."""
        image_mimes = [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
        
        for mime in image_mimes:
            assert FileManager.get_media_type_from_mime_type(mime) == "image"
    
    def test_audio_mime_types(self):
        """Test audio MIME types return 'audio'."""
        audio_mimes = [
            "audio/mpeg",
            "audio/wav",
            "audio/ogg",
            "audio/aac",
        ]
        
        for mime in audio_mimes:
            assert FileManager.get_media_type_from_mime_type(mime) == "audio"
    
    def test_unknown_mime_type_with_video_prefix(self):
        """Test unknown video MIME type returns 'video'."""
        result = FileManager.get_media_type_from_mime_type("video/unknown-format")
        assert result == "video"
    
    def test_completely_unknown_mime_type(self):
        """Test completely unknown MIME type returns None."""
        result = FileManager.get_media_type_from_mime_type("application/unknown")
        assert result is None


class TestValidateFile:
    """Tests for file validation."""
    
    def test_validate_valid_file(self):
        """Test validating a valid file."""
        with patch.object(FileManager, '_get_max_file_size', return_value=100 * 1024 * 1024):
            with patch.object(FileManager, '_get_supported_formats', return_value=["mp4", "avi", "mov"]):
                # Should not raise
                FileManager.validate_file(
                    filename="video.mp4",
                    file_size=10 * 1024 * 1024,  # 10MB
                    extension="mp4",
                    media_type="video",
                )
    
    def test_validate_file_too_large(self):
        """Test validation fails for file too large."""
        with patch.object(FileManager, '_get_max_file_size', return_value=10 * 1024 * 1024):
            with patch.object(FileManager, '_get_supported_formats', return_value=["mp4"]):
                with pytest.raises(FileTooLargeError) as exc_info:
                    FileManager.validate_file(
                        filename="large_video.mp4",
                        file_size=100 * 1024 * 1024,  # 100MB
                        extension="mp4",
                        media_type="video",
                    )
                
                assert exc_info.value.filename == "large_video.mp4"
                assert exc_info.value.media_type == "video"
    
    def test_validate_unsupported_format(self):
        """Test validation fails for unsupported format."""
        with patch.object(FileManager, '_get_max_file_size', return_value=100 * 1024 * 1024):
            with patch.object(FileManager, '_get_supported_formats', return_value=["mp4", "avi"]):
                with pytest.raises(UnsupportedFileFormatError) as exc_info:
                    FileManager.validate_file(
                        filename="video.wmv",
                        file_size=1024,
                        extension="wmv",
                        media_type="video",
                    )
                
                assert exc_info.value.extension == "wmv"
                assert "mp4" in exc_info.value.supported_formats


class TestSaveUpload:
    """Tests for file upload saving."""
    
    @pytest.fixture
    def temp_media_root(self, tmp_path):
        """Create a temporary media root directory."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        return str(media_root)
    
    def test_save_upload_success(self, temp_media_root):
        """Test successful file upload save."""
        uploaded_file = MockUploadedFile(
            name="test_video.mp4",
            size=1024,
            content_type="video/mp4",
        )
        
        # Create a mock settings object
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = temp_media_root
        mock_settings.MAX_FILE_SIZE = {'video': 100*1024*1024, 'image': 50*1024*1024, 'audio': 100*1024*1024}
        mock_settings.SUPPORTED_FORMATS = {'video': ['mp4', 'avi'], 'image': ['jpg', 'png'], 'audio': ['mp3', 'wav']}
        
        # Mock detect_mime_type to return expected value directly
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            with patch.object(FileManager, 'detect_mime_type', return_value='video/mp4'):
                result = FileManager.save_uploaded_file(
                    uploaded_file=uploaded_file,
                    operation_id="test-op-id",
                    session_key="test-session",
                )
        
        assert result["file_name"] == "test_video.mp4"
        assert result["file_size"] == 1024
        assert result["mime_type"] == "video/mp4"
        assert result["media_type"] == "video"
        assert "file_path" in result
        
        # Verify file was actually created
        full_path = os.path.join(temp_media_root, result["file_path"])
        assert os.path.exists(full_path)
    
    def test_save_upload_unsupported_format(self, temp_media_root):
        """Test upload fails for completely unsupported format."""
        uploaded_file = MockUploadedFile(
            name="document.xyz",
            size=1024,
            content_type="application/xyz",
        )
        
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = temp_media_root
        mock_settings.MAX_FILE_SIZE = {'video': 100*1024*1024, 'image': 50*1024*1024, 'audio': 100*1024*1024}
        mock_settings.SUPPORTED_FORMATS = {'video': ['mp4'], 'image': ['jpg'], 'audio': ['mp3']}
        
        # Return a mime type that won't map to a known media type
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            with patch.object(FileManager, 'detect_mime_type', return_value='application/xyz'):
                with pytest.raises(UnsupportedFileFormatError):
                    FileManager.save_uploaded_file(
                        uploaded_file=uploaded_file,
                        operation_id="test-op-id",
                        session_key="test-session",
                    )


class TestGetUploadPath:
    """Tests for upload path generation."""
    
    def test_get_upload_path_structure(self, tmp_path):
        """Test upload path has correct structure."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            relative, full = FileManager.get_uploaded_file_path(
                session_key="abc123",
                operation_id="op-456",
                filename="video.mp4",
            )
        
        assert "uploads" in relative
        assert "abc123" in relative
        assert "op-456" in relative
        assert "video.mp4" in relative
    
    def test_get_upload_path_truncates_session_key(self, tmp_path):
        """Test that long session keys are truncated."""
        long_session = "a" * 100
        
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            relative, full = FileManager.get_uploaded_file_path(
                session_key=long_session,
                operation_id="op-id",
                filename="file.mp4",
            )
        
        # Session key should be truncated to 32 chars
        assert "a" * 100 not in relative
        assert "a" * 32 in relative


class TestGetOutputPath:
    """Tests for output path generation."""
    
    def test_get_output_path_structure(self, tmp_path):
        """Test output path has correct structure."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            relative, full = FileManager.get_output_path(
                session_key="abc123",
                operation_id="op-456",
                filename="output.mp4",
            )
        
        assert "outputs" in relative
        assert "abc123" in relative
        assert "op-456" in relative
        assert "output.mp4" in relative


class TestMoveToOutput:
    """Tests for moving files to output directory."""
    
    @pytest.fixture
    def temp_media_root(self, tmp_path):
        """Create temp media root with test file."""
        media_root = tmp_path / "media"
        media_root.mkdir()
        
        # Create a test temp file
        temp_dir = media_root / "temp" / "test-op"
        temp_dir.mkdir(parents=True)
        test_file = temp_dir / "processed.mp4"
        test_file.write_bytes(b"processed content")
        
        return str(media_root)
    
    def test_move_to_output_success(self, temp_media_root):
        """Test successful move to output directory."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = temp_media_root
        
        temp_path = os.path.join(temp_media_root, "temp", "test-op", "processed.mp4")
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            result = FileManager.move_to_output(
                temp_path=temp_path,
                operation_id="test-op",
                session_key="test-session",
                output_filename="final_output.mp4",
            )
        
        assert result["file_name"] == "final_output.mp4"
        assert "outputs" in result["file_path"]
        
        # Verify file was moved
        output_full_path = os.path.join(temp_media_root, result["file_path"])
        assert os.path.exists(output_full_path)
        assert not os.path.exists(temp_path)  # Original should be gone
    
    def test_move_to_output_file_not_found(self, temp_media_root):
        """Test move fails if source file doesn't exist."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = temp_media_root
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            with pytest.raises(CustomFileNotFoundError):
                FileManager.move_to_output(
                    temp_path="/nonexistent/path/file.mp4",
                    operation_id="test-op",
                    session_key="test-session",
                    output_filename="output.mp4",
                )


class TestDeleteOperationFiles:
    """Tests for deleting operation files."""
    
    @pytest.fixture
    def temp_media_with_files(self, tmp_path):
        """Create temp media root with test files."""
        media_root = tmp_path / "media"
        
        # Create upload directory
        upload_dir = media_root / "uploads" / "test-session" / "test-op"
        upload_dir.mkdir(parents=True)
        (upload_dir / "input.mp4").write_bytes(b"input")
        
        # Create output directory
        output_dir = media_root / "outputs" / "test-session" / "test-op"
        output_dir.mkdir(parents=True)
        (output_dir / "output.mp4").write_bytes(b"output")
        
        # Create temp directory
        temp_dir = media_root / "temp" / "test-op"
        temp_dir.mkdir(parents=True)
        (temp_dir / "temp.mp4").write_bytes(b"temp")
        
        return str(media_root)
    
    def test_delete_operation_files_success(self, temp_media_with_files):
        """Test successful deletion of operation files."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = temp_media_with_files
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            deleted_count = FileManager.delete_operation_files(
                operation_id="test-op",
                session_key="test-session",
            )
        
        # Should have deleted at least 2 directories (upload and output)
        assert deleted_count >= 2
        
        # Verify directories are gone
        upload_dir = os.path.join(temp_media_with_files, "uploads", "test-session", "test-op")
        assert not os.path.exists(upload_dir)
        
        output_dir = os.path.join(temp_media_with_files, "outputs", "test-session", "test-op")
        assert not os.path.exists(output_dir)
    
    def test_delete_nonexistent_files(self, tmp_path):
        """Test deletion of nonexistent files doesn't raise."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            # Should not raise, just return 0
            deleted_count = FileManager.delete_operation_files(
                operation_id="nonexistent-op",
                session_key="nonexistent-session",
            )
        
        assert deleted_count == 0


class TestTempDirectory:
    """Tests for temp directory management."""
    
    def test_ensure_temp_directory_creates(self, tmp_path):
        """Test ensuring temp directory creates it if needed."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            temp_dir = FileManager.ensure_temp_directory("test-op-123")
        
        assert os.path.exists(temp_dir)
        assert "test-op-123" in temp_dir
    
    def test_cleanup_temp_directory(self, tmp_path):
        """Test cleaning up temp directory."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        # Create temp directory with file
        temp_dir = tmp_path / "temp" / "test-op"
        temp_dir.mkdir(parents=True)
        (temp_dir / "temp_file.mp4").write_bytes(b"temp content")
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            result = FileManager.cleanup_temp_directory("test-op")
        
        assert result is True
        assert not os.path.exists(str(temp_dir))
    
    def test_cleanup_nonexistent_temp_directory(self, tmp_path):
        """Test cleanup of nonexistent temp directory returns True."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            result = FileManager.cleanup_temp_directory("nonexistent-op")
        
        assert result is True


class TestGetTempPath:
    """Tests for temp path generation."""
    
    def test_get_temp_path_structure(self, tmp_path):
        """Test temp path has correct structure."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            relative, full = FileManager.get_temp_path(
                operation_id="op-123",
                filename="processing.mp4",
            )
        
        assert "temp" in relative
        assert "op-123" in relative
        assert "processing.mp4" in relative


class TestFileExists:
    """Tests for file existence check."""
    
    def test_file_exists_true(self, tmp_path):
        """Test file_exists returns True for existing file."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            assert FileManager.file_exists("test.txt") is True
    
    def test_file_exists_false(self, tmp_path):
        """Test file_exists returns False for nonexistent file."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = str(tmp_path)
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            assert FileManager.file_exists("nonexistent.txt") is False


class TestGetFileFullPath:
    """Tests for full path generation."""
    
    def test_get_file_full_path(self):
        """Test getting full path from relative path."""
        mock_settings = MagicMock()
        mock_settings.MEDIA_ROOT = "/var/media"
        
        with patch('apps.operations.services.file_manager.settings', mock_settings):
            full_path = FileManager.get_file_full_path("uploads/session/op/file.mp4")
        
        # Normalize paths for cross-platform comparison
        expected = os.path.join("/var/media", "uploads", "session", "op", "file.mp4")
        assert os.path.normpath(full_path) == os.path.normpath(expected)






