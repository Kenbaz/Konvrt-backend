# config/settings/rq_settings.py

"""
Redis Queue (RQ) configuration for media processing workers.

This module provides configuration for django-rq including:
- Queue definitions for different media types
- Timeout settings per media type
- Retry configuration
- Connection settings
"""

import os

# Redis connection settings
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)
REDIS_URL = os.environ.get('REDIS_URL', None)

# Build Redis connection config
if REDIS_URL:
    REDIS_CONNECTION = {'URL': REDIS_URL}
else:
    REDIS_CONNECTION = {
        'HOST': REDIS_HOST,
        'PORT': REDIS_PORT,
        'DB': REDIS_DB,
    }
    if REDIS_PASSWORD:
        REDIS_CONNECTION['PASSWORD'] = REDIS_PASSWORD


# Queue timeout settings (in seconds)
QUEUE_TIMEOUTS = {
    'video_queue': 1800,    # 30 minutes for video processing
    'image_queue': 60,      # 1 minute for image processing
    'audio_queue': 300,     # 5 minutes for audio processing
    'default': 600,         # 10 minutes default
}


# RQ Queues configuration for django-rq
RQ_QUEUES = {
    'default': {
        **REDIS_CONNECTION,
        'DEFAULT_TIMEOUT': QUEUE_TIMEOUTS['default'],
    },
    'video_queue': {
        **REDIS_CONNECTION,
        'DEFAULT_TIMEOUT': QUEUE_TIMEOUTS['video_queue'],
    },
    'image_queue': {
        **REDIS_CONNECTION,
        'DEFAULT_TIMEOUT': QUEUE_TIMEOUTS['image_queue'],
    },
    'audio_queue': {
        **REDIS_CONNECTION,
        'DEFAULT_TIMEOUT': QUEUE_TIMEOUTS['audio_queue'],
    },
}


# Show RQ admin link in Django admin
RQ_SHOW_ADMIN_LINK = True


# Retry configuration
RQ_RETRY_MAX_TIMES = 2
RQ_RETRY_DELAYS = [60, 300]  # 1 minute, 5 minutes


# Worker configuration
WORKER_CONFIG = {
    'video': {
        'queue': 'video_queue',
        'timeout': QUEUE_TIMEOUTS['video_queue'],
        'count': int(os.environ.get('VIDEO_WORKERS', 1)),
        'worker_ttl': 600,  # 10 minutes
    },
    'image': {
        'queue': 'image_queue',
        'timeout': QUEUE_TIMEOUTS['image_queue'],
        'count': int(os.environ.get('IMAGE_WORKERS', 2)),
        'worker_ttl': 180,  # 3 minutes
    },
    'audio': {
        'queue': 'audio_queue',
        'timeout': QUEUE_TIMEOUTS['audio_queue'],
        'count': int(os.environ.get('AUDIO_WORKERS', 1)),
        'worker_ttl': 360,  # 6 minutes
    },
}


# Operation expiration (days)
OPERATION_EXPIRATION_DAYS = int(os.environ.get('OPERATION_EXPIRATION_DAYS', 7))


def get_queue_for_media_type(media_type: str) -> str:
    """
    Get the queue name for a given media type.
    
    Args:
        media_type: 'video', 'image', or 'audio'
        
    Returns:
        Queue name string
    """
    queue_mapping = {
        'video': 'video_queue',
        'image': 'image_queue',
        'audio': 'audio_queue',
    }
    return queue_mapping.get(media_type, 'default')


def get_timeout_for_media_type(media_type: str) -> int:
    """
    Get the timeout for a given media type.
    
    Args:
        media_type: 'video', 'image', or 'audio'
        
    Returns:
        Timeout in seconds
    """
    timeout_mapping = {
        'video': QUEUE_TIMEOUTS['video_queue'],
        'image': QUEUE_TIMEOUTS['image_queue'],
        'audio': QUEUE_TIMEOUTS['audio_queue'],
    }
    return timeout_mapping.get(media_type, QUEUE_TIMEOUTS['default'])