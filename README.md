# Konvrt Backend

A comprehensive media processing API built with Django REST Framework that enables video compression, format conversion, image resizing, audio extraction, and more.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Supported Operations](#supported-operations)
- [Database Models](#database-models)
- [Queue System](#queue-system)
- [Storage](#storage)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Deployment](#deployment)

---

## Overview

Konvrt is a media processing platform that provides a REST API for processing video, image, and audio files. Users can upload files, configure processing parameters, and receive processed results asynchronously with real-time progress tracking.

**Key Features:**
- Asynchronous job processing with Redis Queue (RQ)
- Real-time progress tracking with throttled updates
- Support for video compression, conversion, image resizing, and audio processing
- Cloudinary integration for cloud storage
- Session-based user tracking (no authentication required)
- Comprehensive error handling with user-friendly messages
- Automatic file expiration and cleanup

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│   Django API     │────▶│   PostgreSQL    │
│  (Frontend) │     │   (DRF Views)    │     │   (Database)    │
└─────────────┘     └────────┬─────────┘     └─────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Redis Queue    │
                    │   (Job Queue)    │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │   Video    │ │   Image    │ │   Audio    │
       │   Worker   │ │   Worker   │ │   Worker   │
       └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │   Cloudinary     │
                    │   (File Storage) │
                    └──────────────────┘
```

**Request Flow:**
1. Client uploads file and parameters to the API
2. API validates input, creates Operation record, stores file
3. Job is queued to the appropriate Redis queue (video/image/audio)
4. Worker picks up the job, downloads input, processes file
5. Output is uploaded to storage, Operation status updated
6. Client polls for status and downloads result when complete

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Django 4.2+ |
| API | Django REST Framework 3.14+ |
| Database | PostgreSQL |
| Queue | Redis + django-rq |
| Video/Audio Processing | FFmpeg |
| Image Processing | Pillow |
| Cloud Storage | Cloudinary |
| WSGI Server | Gunicorn |
| Static Files | WhiteNoise |

---

## Project Structure

```
backend/
├── config/
│   ├── settings/
│   │   ├── base.py            # Shared settings
│   │   ├── development.py     # Development settings
│   │   ├── production.py      # Production settings (Railway)
│   │   └── rq_settings.py     # Redis Queue configuration
│   ├── urls.py                # Root URL configuration
│   └── wsgi.py                # WSGI entry point
│
├── apps/
│   ├── api/                   # REST API layer
│   │   ├── views.py           # API ViewSets and Views
│   │   ├── serializers.py     # Request/Response serializers
│   │   ├── urls.py            # API URL routing
│   │   ├── pagination.py      # Pagination classes
│   │   ├── throttling.py      # Rate limiting
│   │   ├── permissions.py     # Permission classes
│   │   ├── validators.py      # Input validators
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── utils.py           # API utilities
│   │
│   ├── core/                  # Shared utilities
│   │   ├── middleware.py      # Session middleware
│   │   └── utils.py           # Core utilities
│   │
│   ├── operations/            # Operation management
│   │   ├── models.py          # Operation & File models
│   │   ├── enums.py           # Status & FileType enums
│   │   └── services/
│   │       ├── operations_manager.py  # Operation lifecycle
│   │       ├── queue_manager.py       # Job queue management
│   │       ├── file_manager.py        # File handling
│   │       └── cloudinary_storage.py  # Cloudinary integration
│   │
│   └── processors/            # Media processing
│       ├── registry.py        # Operation registry
│       ├── base_processor.py  # Base processor classes
│       ├── video_processing.py    # Video operations
│       ├── image_processing.py    # Image operations
│       ├── audio_processing.py    # Audio operations
│       ├── rq_workers.py      # RQ worker functions
│       ├── exceptions.py      # Processing exceptions
│       └── utils/
│           ├── ffmpeg.py      # FFmpeg wrapper
│           ├── track_progress.py  # Progress tracking
│           └── validation.py  # File validation
│
├── scripts/
│   ├── start_workers.sh       # Worker startup script (Unix)
│   └── start_workers_windows.py  # Worker startup (Windows)
│
├── requirements/
│   ├── base.txt               # Core dependencies
│   ├── development.txt        # Dev dependencies
│   └── production.txt         # Prod dependencies
│
├── manage.py
└── requirements.txt           # Combined requirements
```

---

## API Endpoints

All endpoints are prefixed with `/api/v1/`

### Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/operations/` | Create a new operation (multipart/form-data) |
| `GET` | `/operations/` | List user's operations |
| `GET` | `/operations/{id}/` | Get operation details |
| `DELETE` | `/operations/{id}/` | Delete operation and files |
| `GET` | `/operations/{id}/status/` | Lightweight status for polling |
| `GET` | `/operations/{id}/download/` | Download output file |
| `POST` | `/operations/{id}/retry/` | Retry failed operation |
| `POST` | `/operations/{id}/cancel/` | Cancel queued operation |

### Operation Definitions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/operation-definitions/` | List available operations |
| `GET` | `/operation-definitions/{name}/` | Get operation details and parameters |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health/` | Health check (DB, Redis, Storage, Cloudinary) |
| `GET` | `/queues/` | Queue statistics |
| `GET` | `/session/` | Current session info |

### Request/Response Examples

**Create Operation:**
```http
POST /api/v1/operations/
Content-Type: multipart/form-data

operation: video_compress
parameters: {"quality": 23, "preset": "medium"}
file: [binary file data]
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "operation": "video_compress",
    "status": "queued",
    "progress": 0,
    "parameters": {"quality": 23, "preset": "medium"},
    "created_at": "2025-01-09T10:30:00.000Z",
    "input_file": {
      "file_name": "video.mp4",
      "file_size": 52428800,
      "mime_type": "video/mp4"
    }
  },
  "message": "Operation created and queued for processing"
}
```

**Poll Status:**
```http
GET /api/v1/operations/{id}/status/
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "processing",
    "progress": 45,
    "eta_seconds": 120
  }
}
```

---

## Supported Operations

### Video Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `video_compress` | Compress video using H.264 codec | `quality` (18-28), `preset`, `audio_bitrate` |
| `video_convert` | Convert video format | `output_format`, `quality` |

**Input Formats:** mp4, mov, avi, mkv, webm, wmv, flv  
**Output Formats:** mp4, webm, mov

### Image Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `image_resize` | Resize image with aspect ratio control | `width`, `height`, `maintain_aspect_ratio` |
| `image_convert` | Convert image format | `output_format`, `quality` |

**Input Formats:** jpg, jpeg, png, webp, gif, bmp  
**Output Formats:** jpg, png, webp

### Audio Operations

| Operation | Description | Parameters |
|-----------|-------------|------------|
| `audio_convert` | Convert audio format | `output_format`, `bitrate`, `sample_rate`, `channels` |
| `audio_extract` | Extract audio from video | `output_format`, `bitrate` |

**Input Formats:** mp3, wav, aac, m4a, ogg, flac, opus  
**Output Formats:** mp3, wav, aac, ogg, flac

---

## Database Models

### Operation

Represents a media processing job.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `session_key` | CharField | User session identifier |
| `operation` | CharField | Operation name (e.g., "video_compress") |
| `status` | CharField | pending, queued, processing, completed, failed |
| `progress` | IntegerField | Progress percentage (0-100) |
| `parameters` | JSONField | Operation parameters |
| `error_message` | TextField | Error details if failed |
| `created_at` | DateTimeField | Creation timestamp |
| `started_at` | DateTimeField | Processing start time |
| `completed_at` | DateTimeField | Completion time |
| `expires_at` | DateTimeField | Expiration date (7 days after completion) |
| `is_deleted` | BooleanField | Soft delete flag |

### File

Represents input or output files.

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key |
| `operation` | ForeignKey | Associated operation |
| `file_type` | CharField | "input" or "output" |
| `file_path` | CharField | Local file path |
| `file_name` | CharField | Original filename |
| `file_size` | BigIntegerField | Size in bytes |
| `mime_type` | CharField | MIME type |
| `cloudinary_public_id` | CharField | Cloudinary ID |
| `cloudinary_url` | URLField | Cloudinary URL |
| `metadata` | JSONField | Additional file metadata |

---

## Queue System

The application uses Redis Queue (RQ) with separate queues for different media types:

| Queue | Timeout | Worker Count | Use Case |
|-------|---------|--------------|----------|
| `video_queue` | 30 min | 1 | Video compression/conversion |
| `image_queue` | 1 min | 2 | Image resize/conversion |
| `audio_queue` | 5 min | 1 | Audio conversion/extraction |

### Worker Configuration

Workers are configured in `config/settings/rq_settings.py`:

```python
WORKER_CONFIG = {
    'video': {
        'queue': 'video_queue',
        'timeout': 1800,  # 30 minutes
        'count': 1,
        'worker_ttl': 600,
    },
    'image': {
        'queue': 'image_queue',
        'timeout': 60,  # 1 minute
        'count': 2,
        'worker_ttl': 180,
    },
    'audio': {
        'queue': 'audio_queue',
        'timeout': 300,  # 5 minutes
        'count': 1,
        'worker_ttl': 360,
    },
}
```

### Error Handling & Retries

- **Retryable errors:** Cloudinary timeouts, temporary network issues
- **Non-retryable errors:** Invalid input, codec errors, permission issues
- **Max retries:** 2
- **Retry delays:** 1 minute, then 5 minutes (exponential backoff)

---

## Storage

### Cloudinary (Production)

When `USE_CLOUDINARY=True`, files are stored in Cloudinary:

```python
CLOUDINARY_STORAGE = {
    'ROOT_FOLDER': 'mediaprocessor',
    'UPLOADS_FOLDER': 'uploads',
    'OUTPUTS_FOLDER': 'outputs',
}
```

### Local Storage (Development)

Files are stored in the local filesystem:

```
storage/
├── uploads/     # Input files
├── outputs/     # Processed files
└── temp/        # Temporary processing files
```

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Redis 6+
- FFmpeg 5+ (with libx264, libvpx-vp9, aac codecs)

### Setup

1. **Clone and create virtual environment:**
   ```bash
   git clone <repository>
   cd konvrt-backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Create static files directory:**
   ```bash
   python manage.py collectstatic
   ```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Django secret key | (required) |
| `DEBUG` | Debug mode | `False` |
| `ALLOWED_HOSTS` | Comma-separated hosts | `localhost,127.0.0.1` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `USE_CLOUDINARY` | Enable Cloudinary storage | `False` |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary cloud name | - |
| `CLOUDINARY_API_KEY` | Cloudinary API key | - |
| `CLOUDINARY_API_SECRET` | Cloudinary API secret | - |
| `CORS_ALLOWED_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |

### File Size Limits

Configured in `config/settings/base.py`:

```python
MAX_FILE_SIZE = {
    'video': 500 * 1024 * 1024,  # 500 MB
    'image': 25 * 1024 * 1024,   # 25 MB
    'audio': 100 * 1024 * 1024,  # 100 MB
}
```

### Rate Limiting

```python
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_RATES': {
        'anon_burst': '60/minute',
        'anon_sustained': '1000/day',
        'uploads': '10/hour',
        'status_checks': '120/minute',
    },
}
```

---

## Running the Application

### Development

1. **Start the Django development server:**
   ```bash
   python manage.py runserver
   ```

2. **Start Redis (if not running):**
   ```bash
   redis-server
   ```

3. **Start workers:**
   ```bash
   # Unix/Linux/Mac
   ./scripts/start_workers.sh all

   # Windows
   python scripts/start_workers_windows.py all
   ```

### Worker Commands

```bash
# Start all workers
./scripts/start_workers.sh all

# Start specific worker type
./scripts/start_workers.sh video
./scripts/start_workers.sh image
./scripts/start_workers.sh audio

# Check status
./scripts/start_workers.sh status

# Stop all workers
./scripts/start_workers.sh stop

# Restart workers
./scripts/start_workers.sh restart
```

---

## Deployment

### Railway Deployment

The backend is configured for Railway deployment with:

- **HTTP Service:** Gunicorn serving the Django API
- **Worker Service:** RQ workers processing jobs
- **PostgreSQL:** Managed database
- **Redis:** Managed Redis instance

**Procfile:**
```
web: gunicorn config.wsgi:application --bind 0.0.0.0:$PORT
worker: python manage.py rqworker video_queue image_queue audio_queue --with-scheduler
```

**Environment Variables (Railway):**
```
DJANGO_SETTINGS_MODULE=config.settings.production
DATABASE_URL=<provided by Railway>
REDIS_URL=<provided by Railway>
USE_CLOUDINARY=True
CLOUDINARY_CLOUD_NAME=<cloud name>
CLOUDINARY_API_KEY=<api key>
CLOUDINARY_API_SECRET=<api secret>
CORS_ALLOWED_ORIGINS=https://frontend-domain.vercel.app
```

### Health Check

The `/api/v1/health/` endpoint checks:
- Database connectivity
- Redis connectivity
- Local storage availability
- Cloudinary connectivity (when enabled)

Returns `200 OK` if healthy, `503 Service Unavailable` if degraded.

---

## License

MIT License

## Author

Kenneth Bassey