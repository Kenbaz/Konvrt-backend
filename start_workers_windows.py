#!/usr/bin/env python
"""
Windows-compatible RQ worker startup script.

RQ's default worker uses os.fork() which is not available on Windows.
This script uses SimpleWorker which works on Windows.

Usage:
    python start_workers_windows.py [queue_name]
    
Examples:
    python start_workers_windows.py              # Start all queues
    python start_workers_windows.py image_queue  # Start image queue only
    python start_workers_windows.py video_queue  # Start video queue only
    python start_workers_windows.py audio_queue  # Start audio queue only
"""

import os
import sys
import logging

# Setup Django before importing anything else
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

import django
django.setup()

from redis import Redis
from rq import SimpleWorker, Queue
from django.conf import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_redis_connection():
    """Get Redis connection from Django settings."""
    rq_queues = getattr(settings, 'RQ_QUEUES', {})
    
    # Try to get connection info from any queue config
    queue_config = rq_queues.get('default', rq_queues.get('image_queue', {}))
    
    if 'URL' in queue_config:
        return Redis.from_url(queue_config['URL'])
    
    return Redis(
        host=queue_config.get('HOST', 'localhost'),
        port=queue_config.get('PORT', 6379),
        db=queue_config.get('DB', 0),
        password=queue_config.get('PASSWORD', None),
    )


def start_worker(queue_names: list[str]):
    """
    Start a SimpleWorker for the specified queues.
    
    Args:
        queue_names: List of queue names to process
    """
    try:
        connection = get_redis_connection()
        
        # Test Redis connection
        connection.ping()
        logger.info("Connected to Redis successfully")
        
        # Create queue objects
        queues = [Queue(name, connection=connection) for name in queue_names]
        
        logger.info(f"Starting SimpleWorker for queues: {', '.join(queue_names)}")
        logger.info("Press Ctrl+C to stop the worker")
        
        # Create and start SimpleWorker (works on Windows)
        worker = SimpleWorker(queues, connection=connection)
        worker.work(with_scheduler=False)
        
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise


def main():
    """Main entry point."""
    # Default queues
    all_queues = ['image_queue', 'video_queue', 'audio_queue']
    
    # Check command line arguments
    if len(sys.argv) > 1:
        queue_name = sys.argv[1]
        if queue_name in all_queues:
            queue_names = [queue_name]
        elif queue_name == 'all':
            queue_names = all_queues
        else:
            print(f"Unknown queue: {queue_name}")
            print(f"Available queues: {', '.join(all_queues)}")
            print("Usage: python start_workers_windows.py [queue_name|all]")
            sys.exit(1)
    else:
        # Default to all queues
        queue_names = all_queues
    
    print("=" * 60)
    print("RQ Worker for Windows (SimpleWorker)")
    print("=" * 60)
    print(f"Queues: {', '.join(queue_names)}")
    print("=" * 60)
    
    start_worker(queue_names)


if __name__ == '__main__':
    main()
