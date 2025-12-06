# apps/api/serializers.py

"""
Serializers for the API layer.

This module provides serializers for converting between Django models
and JSON representations for API requests and responses.
"""

import json
import logging
from typing import Any, Dict

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.utils import timezone
from rest_framework import serializers

logger = logging.getLogger(__name__)


class FileSerializer(serializers.Serializer):
    """
    Serializer for File model.
    
    Used for representing input and output files in API responses.
    """
    id = serializers.UUIDField(read_only=True)
    file_type = serializers.CharField(read_only=True)
    file_name = serializers.CharField(read_only=True)
    file_size = serializers.IntegerField(read_only=True)
    mime_type = serializers.CharField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)

    # Computing fields
    file_size_formated = serializers.SerializerMethodField()
    download_url = serializers.SerializerMethodField()

    def get_file_size_formated(self, obj) -> str:
        """Format file size in human-readable form."""
        size = obj.file_size
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.2f} GB"
    
    def get_download_url(self, obj) -> str:
        """Generate the download URL for the file"""
        request = self.context.get('request')
        if request and hasattr(obj, 'file_url'):
            return request.build_absolute_uri(obj.file_url)
        elif hasattr(obj, 'file_url'):
            return obj.file_url
        return ""  # Fallback if no URL is available


class OperationSerializer(serializers.Serializer):
    """
    Serializer for Operation model (full details).
    
    Used for detailed operation responses including all fields.
    """
    id = serializers.UUIDField(read_only=True)
    operation = serializers.CharField(read_only=True)
    status = serializers.CharField(read_only=True)
    progress = serializers.IntegerField(read_only=True)
    parameters = serializers.JSONField(read_only=True)
    error_message = serializers.CharField(read_only=True, allow_null=True)
    
    # Timestamps
    created_at = serializers.DateTimeField(read_only=True)
    started_at = serializers.DateTimeField(read_only=True, allow_null=True)
    completed_at = serializers.DateTimeField(read_only=True, allow_null=True)
    expires_at = serializers.DateTimeField(read_only=True, allow_null=True)
    
    # Computed fields
    processing_time = serializers.SerializerMethodField()
    processing_time_formatted = serializers.SerializerMethodField()
    is_expired = serializers.SerializerMethodField()
    is_processing = serializers.SerializerMethodField()
    can_be_deleted = serializers.SerializerMethodField()
    can_be_retried = serializers.SerializerMethodField()
    
    # Related files
    input_file = serializers.SerializerMethodField()
    output_file = serializers.SerializerMethodField()
    
    def get_processing_time(self, obj) -> float:
        """Get processing time in seconds."""
        if hasattr(obj, 'processing_time') and obj.processing_time is not None:
            return obj.processing_time
        return None
    
    def get_processing_time_formatted(self, obj) -> str:
        """Get processing time in human-readable format."""
        processing_time = self.get_processing_time(obj)
        if processing_time is None:
            return None
        
        if processing_time < 1:
            return f"{processing_time * 1000:.0f}ms"
        elif processing_time < 60:
            return f"{processing_time:.1f}s"
        elif processing_time < 3600:
            minutes = int(processing_time // 60)
            seconds = int(processing_time % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(processing_time // 3600)
            minutes = int((processing_time % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_is_expired(self, obj) -> bool:
        """Check if operation has expired."""
        if hasattr(obj, 'is_expired'):
            return obj.is_expired
        return False
    
    def get_is_processing(self, obj) -> bool:
        """Check if operation is currently processing."""
        if hasattr(obj, 'is_processing'):
            return obj.is_processing
        return obj.status in ['queued', 'processing']
    
    def get_can_be_deleted(self, obj) -> bool:
        """Check if operation can be deleted."""
        if hasattr(obj, 'can_be_deleted'):
            return obj.can_be_deleted()
        return obj.status in ['completed', 'failed']
    
    def get_can_be_retried(self, obj) -> bool:
        """Check if operation can be retried."""
        if hasattr(obj, 'can_be_retried'):
            return obj.can_be_retried()
        return obj.status == 'failed'
    
    def get_input_file(self, obj) -> Dict[str, Any]:
        """Get the input file for this operation."""
        if hasattr(obj, 'files'):
            input_file = obj.files.filter(file_type='input').first()
            if input_file:
                return FileSerializer(input_file, context=self.context).data
        return None
    
    def get_output_file(self, obj) -> Dict[str, Any]:
        """Get the output file for this operation."""
        if hasattr(obj, 'files'):
            output_file = obj.files.filter(file_type='output').first()
            if output_file:
                return FileSerializer(output_file, context=self.context).data
        return None


class OperationListSerializer(serializers.Serializer):
    """
    Lightweight serializer for operation listings.
    
    Used for list endpoints where full details aren't needed.
    """
    id = serializers.UUIDField(read_only=True)
    operation = serializers.CharField(read_only=True)
    status = serializers.CharField(read_only=True)
    progress = serializers.IntegerField(read_only=True)
    created_at = serializers.DateTimeField(read_only=True)
    completed_at = serializers.DateTimeField(read_only=True, allow_null=True)
    
    # Computed fields
    is_expired = serializers.SerializerMethodField()
    has_output = serializers.SerializerMethodField()
    
    def get_is_expired(self, obj) -> bool:
        """Check if operation has expired."""
        if hasattr(obj, 'is_expired'):
            return obj.is_expired
        return False
    
    def get_has_output(self, obj) -> bool:
        """Check if operation has an output file."""
        if hasattr(obj, 'files'):
            return obj.files.filter(file_type='output').exists()
        return False


class OperationStatusSerializer(serializers.Serializer):
    """
    Lightweight serializer for polling operation status.
    
    Used for the status endpoint to minimize payload size
    during frequent polling.
    """
    id = serializers.UUIDField(read_only=True)
    status = serializers.CharField(read_only=True)
    progress = serializers.IntegerField(read_only=True)
    error_message = serializers.CharField(read_only=True, allow_null=True)
    
    # Computed fields
    eta_seconds = serializers.SerializerMethodField()
    is_complete = serializers.SerializerMethodField()
    has_output = serializers.SerializerMethodField()
    
    def get_eta_seconds(self, obj) -> int:
        """
        Estimate time remaining in seconds.
        
        This is a rough estimate based on progress and elapsed time.
        Returns None if estimation is not possible.
        """
        if obj.status != 'processing' or obj.progress <= 0:
            return None
        
        if not obj.started_at:
            return None
        
        elapsed = (timezone.now() - obj.started_at).total_seconds()
        if elapsed <= 0:
            return None
        
        # Estimate total time based on current progress
        progress_rate = obj.progress / elapsed  # percent per second
        if progress_rate <= 0:
            return None
        
        remaining_progress = 100 - obj.progress
        eta = remaining_progress / progress_rate
        
        return max(0, int(eta))
    
    def get_is_complete(self, obj) -> bool:
        """Check if operation is complete (success or failure)."""
        return obj.status in ['completed', 'failed']
    
    def get_has_output(self, obj) -> bool:
        """Check if operation has an output file ready for download."""
        if obj.status != 'completed':
            return False
        if hasattr(obj, 'files'):
            return obj.files.filter(file_type='output').exists()
        return False


class OperationCreateSerializer(serializers.Serializer):
    """
    Serializer for creating new operations.
    
    Handles validation of operation name, parameters, and uploaded file.
    """
    operation = serializers.CharField(
        required=True,
        help_text="Name of the operation to perform (e.g., 'video_compress)"
    )
    parameters = serializers.JSONField(
        required=False,
        default=dict,
        help_text="Operation-specific parameters as JSON object"
    )
    file = serializers.FileField(
        required=True,
        help_text="The file to process"
    )


    def validate_operation(self, value: str) -> str:
        """Validate the operation exists in the registry"""
        from apps.processors.registry import get_registry

        registry = get_registry()
        if not registry.is_registered(value):
            raise serializers.ValidationError(
                f"Operation '{value}' is not available. "
                "Use GET /api/v1/operations/ to see available operations."
            )
        return value
    

    def validate_parameters(self, value: Any) -> Dict[str, Any]:
        """Validate and normalize parameters."""
        # Handle string JSON
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise serializers.ValidationError(
                    "Parameters must be a valid JSON object."
                )
        
        # Ensure it's a dictionary
        if not isinstance(value, dict):
            raise serializers.ValidationError(
                "Parameters must be an object/dictionary."
            )
        
        return value
    

    def validate_file(self, value: UploadedFile) -> UploadedFile:
        """Validate the uploaded file."""
        if not value:
            raise serializers.ValidationError("No file was uploaded.")
        
        # Check file size
        if value.size <= 0:
            raise serializers.ValidationError("Uploaded file is empty.")
        
        # Get file extension
        filename = value.name or ""
        extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ""
        
        if not extension:
            raise serializers.ValidationError(
                "Could not determine file type. Please ensure the file has an extension."
            )
        
        return value
    

    def validate(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-field validation.
        
        Validates parameters against the operation's schema and
        checks file type compatibility.
        """
        operation_name = attrs.get('operation')
        parameters = attrs.get('parameters', {})
        uploaded_file = attrs.get('file')
        
        if not operation_name:
            return attrs
        
        from apps.processors.registry import get_registry
        from apps.processors.exceptions import InvalidParametersError

        registry = get_registry()

        # Validate parameters against operation schema
        try:
            validated_params = registry.validate_parameters(
                operation_name,
                parameters
            )
            attrs['parameters'] = validated_params
        except InvalidParametersError as e:
            raise serializers.ValidationError({
                'parameters': e.errors if hasattr(e, 'errors') else [str(e)]
            })
        
        # Validate file type for operation
        if uploaded_file:
            operation_def = registry.get_operation(operation_name)
            media_type = operation_def.media_type.value

            # Get file extension
            filename = uploaded_file.name or ""
            extension = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ""

            # Check against supported formats
            supported_formats = settings.SUPPORTED_FORMATS.get(media_type, [])
            if extension and extension not in supported_formats:
                raise serializers.ValidationError({
                    'file': f"File type '.{extension}' is not supported for {media_type} operations. "
                            f"Supported formats: {', '.join(supported_formats)}"
                })
            
            # Check file size
            max_size = settings.MAX_FILE_SIZE.get(media_type, 0)
            if max_size and uploaded_file.size > max_size:
                max_size_mb = max_size / (1024 * 1024)
                file_size_mb = uploaded_file.size / (1024 * 1024)
                raise serializers.ValidationError({
                    'file': f"File size ({file_size_mb:.1f} MB) exceeds maximum "
                            f"allowed size ({max_size_mb:.0f} MB) for {media_type} files."
                })
        
        return attrs
    

    def create(self, validated_data: Dict[str, Any]) -> Any:
        """
        Create and queue a new operation.
        
        This method:
        1. Creates the operation record
        2. Saves the uploaded file
        3. Queues the operation for processing
        """
        from apps.operations.services.operations_manager import OperationsManager

        request = self.context.get('request')
        session_key = request.session.session_key if request else None

        if not session_key:
            raise serializers.ValidationError(
                "Session is required. Please ensure cookies are enabled."
            )
        
        # Create the operation
        operation = OperationsManager.create_operation(
            session_key=session_key,
            operation_name=validated_data['operation'],
            parameters=validated_data['parameters'],
            uploaded_file=validated_data['file'],
            user=request.user if request and request.user.is_authenticated else None
        )

        # Queue the operation for processing
        OperationsManager.queue_operation(operation.id)

        logger.info(
            f"Created and queued operation {operation.id} "
            f"(operation={validated_data['operation']}, session={session_key[:8]}...)"
        )
        
        return operation


class ParameterSchemaSerializer(serializers.Serializer):
    """
    Serializer for operation parameter schemas.
    
    Used to describe available parameters for an operation.
    """
    param_name = serializers.CharField(read_only=True)
    type = serializers.CharField(source='param_type.value', read_only=True)
    required = serializers.BooleanField(read_only=True)
    default = serializers.JSONField(read_only=True, allow_null=True)
    description = serializers.CharField(read_only=True)
    min = serializers.FloatField(source='min_value', read_only=True, allow_null=True)
    max = serializers.FloatField(source='max_value', read_only=True, allow_null=True)
    choices = serializers.ListField(read_only=True, allow_null=True)


class OperationDefinitionSerializer(serializers.Serializer):
    """
    Serializer for operation definitions from the registry.
    
    Used to describe available operations and their parameters.
    """
    operation_name = serializers.CharField(read_only=True)
    media_type = serializers.SerializerMethodField()
    description = serializers.CharField(read_only=True)
    parameters = serializers.SerializerMethodField()
    input_formats = serializers.ListField(
        child=serializers.CharField(),
        read_only=True
    )
    output_formats = serializers.ListField(
        child=serializers.CharField(),
        read_only=True
    )
    
    def get_media_type(self, obj) -> str:
        """Get media type as string."""
        if hasattr(obj.media_type, 'value'):
            return obj.media_type.value
        return str(obj.media_type)
    
    def get_parameters(self, obj) -> list:
        """Serialize parameter schemas."""
        if hasattr(obj, 'parameters'):
            return [p.to_dict() for p in obj.parameters]
        return []


class OperationDefinitionListSerializer(serializers.Serializer):
    """
    Lightweight serializer for listing available operations.
    """
    operation_name = serializers.CharField(read_only=True)
    media_type = serializers.SerializerMethodField()
    description = serializers.CharField(read_only=True)
    
    def get_media_type(self, obj) -> str:
        """Get media type as string."""
        if hasattr(obj.media_type, 'value'):
            return obj.media_type.value
        return str(obj.media_type)


class HealthCheckSerializer(serializers.Serializer):
    """
    Serializer for health check response.
    """
    status = serializers.CharField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True)
    version = serializers.CharField(read_only=True)
    
    # Component statuses
    database = serializers.DictField(read_only=True)
    redis = serializers.DictField(read_only=True)
    storage = serializers.DictField(read_only=True)
    queues = serializers.DictField(read_only=True)


class QueueStatsSerializer(serializers.Serializer):
    """
    Serializer for queue statistics.
    """
    queue_name = serializers.CharField(read_only=True)
    queued = serializers.IntegerField(read_only=True)
    started = serializers.IntegerField(read_only=True)
    failed = serializers.IntegerField(read_only=True)
    workers = serializers.IntegerField(read_only=True, default=0)