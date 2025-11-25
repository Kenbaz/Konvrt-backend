"""
Enumerations for the operations app.
"""
from django.db import models


class OperationStatus(models.TextChoices):
    """Status of an operation in the processing pipeline."""
    PENDING = 'pending', 'Pending'
    QUEUED = 'queued', 'Queued'
    PROCESSING = 'processing', 'Processing'
    COMPLETED = 'completed', 'Completed'
    FAILED = 'failed', 'Failed'


class FileType(models.TextChoices):
    """Type of file in relation to operations."""
    INPUT = 'input', 'Input'
    OUTPUT = 'output', 'Output'