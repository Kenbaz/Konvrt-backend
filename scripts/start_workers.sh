#!/bin/bash

# scripts/start_workers.sh

# Script to start RQ workers for processing media operations.

set -e

# CONFIGURATION
DJANGO_SETTINGS_MODULE="${DJANGO_SETTINGS_MODULE:-config.settings.development}"
VIDEO_WORKERS="${VIDEO_WORKERS:-1}"
IMAGE_WORKERS="${IMAGE_WORKERS:-2}"
AUDIO_WORKERS="${AUDIO_WORKERS:-1}"

# WORKER TIMEOUTS
VIDEO_TIMEOUT=1800  # 30 minutes
IMAGE_TIMEOUT=60   # 1 minute
AUDIO_TIMEOUT=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Redis is running
check_redis() {
    if ! redis-cli ping > /dev/null 2>&1; then
        log_error "Redis server is not running. Please start Redis and try again."
        exit 1
    fi
    log_info "Redis server is running."
}

# Check if django-rq is installed
check_dependencies() {
    if ! python -c "import django_rq" 2>/dev/null; then
        log_error "django-rq is not installed. Run: pip install django-rq"
        exit 1
    fi
    log_info "All dependencies are satisfied."
}

# Start video workers
start_video_workers() {
    log_info "Starting ${VIDEO_WORKERS} video worker(s)..."
    for i in $(seq 1 $VIDEO_WORKERS); do
        python manage.py rqworker video_queue \
            --worker-ttl 600 \
            --with-scheduler \
            --name "video-worker-$i" &
        log_info "Started video worker $i (PID: $!)"
    done
}

# Start image workers
start_image_workers() {
    log_info "Starting ${IMAGE_WORKERS} image worker(s)..."
    for i in $(seq 1 $IMAGE_WORKERS); do
        python manage.py rqworker image_queue \
            --worker-ttl 180 \
            --name "image-worker-$i" &
        log_info "Started image worker $i (PID: $!)"
    done
}

# Start audio workers
start_audio_workers() {
    log_info "Starting ${AUDIO_WORKERS} audio worker(s)..."
    for i in $(seq 1 $AUDIO_WORKERS); do
        python manage.py rqworker audio_queue \
            --worker-ttl 360 \
            --name "audio-worker-$i" &
        log_info "Started audio worker $i (PID: $!)"
    done
}

# Start all workers
start_all_workers() {
    log_info "Starting all workers..."
    start_video_workers
    start_image_workers
    start_audio_workers
    log_info "All workers started successfully"
}

# Stop all workers
stop_all_workers() {
    log_info "Stopping all workers..."
    
    # Find and kill all rqworker processes
    pkill -f "rqworker" 2>/dev/null || true
    
    log_info "All workers stopped"
}

# show worker status
show_status() {
    log_info "Worker Status:"
    echo ""
    
    # Check for running workers
    local video_count=$(pgrep -f "rqworker video_queue" | wc -l)
    local image_count=$(pgrep -f "rqworker image_queue" | wc -l)
    local audio_count=$(pgrep -f "rqworker audio_queue" | wc -l)
    
    echo "  Video workers: $video_count running"
    echo "  Image workers: $image_count running"
    echo "  Audio workers: $audio_count running"
    echo ""

    # Show queue stats if python is available
    if command -v python &> \dev/null; then
        python -c "
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', '$DJANGO_SETTINGS_MODULE')
import django
django.setup()

try:
    import django_rq
    
    queues = ['video_queue', 'image_queue', 'audio_queue']
    print('Queue Status:')
    for queue_name in queues:
        try:
            queue = django_rq.get_queue(queue_name)
            print(f'  {queue_name}: {queue.count} jobs queued')
        except Exception as e:
            print(f'  {queue_name}: Error - {e}')
except Exception as e:
    print(f'Could not get queue stats: {e}')
" 2>/dev/null || log_warn "Could not retrieve queue stats"
    fi
}

# Main entry point
main() {
    export DJANGO_SETTINGS_MODULE

    case "${1:-all}" in
        video)
            check_redis
            check_dependencies
            start_video_workers
            ;;
        image)
            check_redis
            check_dependencies
            start_image_workers
            ;;
        audio)
            check_redis
            check_dependencies
            start_audio_workers
            ;;
        all)
            check_redis
            check_dependencies
            start_all_workers
            ;;
        stop)
            stop_all_workers
            ;;
        status)
            show_status
            ;;
        restart)
            stop_all_workers
            sleep 2
            check_redis
            check_dependencies
            start_all_workers
            ;;
        *)
            echo "Usage: $0 {video|image|audio|all|stop|status|restart}"
            echo ""
            echo "Commands:"
            echo "  video   - Start video processing workers"
            echo "  image   - Start image processing workers"
            echo "  audio   - Start audio processing workers"
            echo "  all     - Start all workers (default)"
            echo "  stop    - Stop all workers"
            echo "  status  - Show worker and queue status"
            echo "  restart - Stop and restart all workers"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

# Wait for all background jobs if starting workers
if [[ "${1:-all}" != "stop" && "${1:-all}" != "status" ]]; then
    log_info "Workers are running in background. Press Ctrl+C to stop."
    wait
fi