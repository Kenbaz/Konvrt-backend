#  apps/operations/admin.py
from django.contrib import admin
from .models import Operation, File


@admin.register(Operation)
class OperationAdmin(admin.ModelAdmin):
    """Admin interface for Operation model."""

    list_display = [
        'id', 
        'operation', 
        'status', 
        'progress', 
        'session_key', 
        'created_at',
        'is_expired'
    ]

    list_filter = ['status', 'operation', 'created_at', 'is_deleted']
    search_fields = ['id', 'session_key', 'operation', 'user__username']

    readonly_fields = [
        'id',
        'created_at',
        'started_at',
        'completed_at',
        'processing_time',
        'is_expired'
    ]

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'operation', 'status', 'progress')
        }),
        ('User Information', {
            'fields': ('user', 'session_key')
        }),
        ('Processing Details', {
            'fields': ('parameters', 'error_message')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'started_at', 'completed_at', 'processing_time', 'expires_at', 'is_expired')
        }),
        ('Metadata', {
            'fields': ('is_deleted',)
        }),
    )

    date_hierarchy = 'created_at'

    def get_queryset(self, request):
        """soft-deleted operations are visible in admin."""
        qs = super().get_queryset(request)
        return qs
    

@admin.register(File)
class FileAdmin(admin.ModelAdmin):
    """Admin interface for File model."""

    list_display = [
        'id',
        'file_name',
        'file_type',
        'file_size',
        'mime_type',
        'operation',
        'created_at'
    ]
    
    list_filter = [
        'file_type',
        'mime_type',
        'created_at'
    ]
    
    search_fields = [
        'id',
        'file_name',
        'operation__id'
    ]
    
    readonly_fields = [
        'id',
        'created_at',
        'file_url'
    ]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'operation', 'file_type')
        }),
        ('File Details', {
            'fields': ('file_name', 'file_path', 'file_size', 'mime_type', 'file_url')
        }),
        ('Metadata', {
            'fields': ('metadata', 'created_at')
        }),
    )
    
    date_hierarchy = 'created_at'