"""
Base settings for mediaprocessor project.
These settings are shared across all environments.
"""

import os
from pathlib import Path
import environ
import secrets
import dj_database_url
import cloudinary

# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, False)
)

ENVIRONMENT = os.getenv('DJANGO_ENVIRONMENT', 'development')


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Read .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))


SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third party apps
    'rest_framework',
    'django_rq',
    'django_filters',
    'corsheaders',
    
    # Local apps
    'apps.core',
    'apps.operations',
    'apps.processors',
    'apps.api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'apps.core.middleware.SessionMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# Database
# This will be overridden in development.py and production.py
if ENVIRONMENT == 'production':
    DATABASES = {
        'default': dj_database_url.config(
            default=os.getenv('DB_PRODUCTION_URL'),
            conn_max_age=600,
            conn_health_checks=True,
        )
    }

    # Ensure SSL is required for PostgreSQL in production
    if DATABASES['default']:
        DATABASES['default']['OPTIONS'] = {
            'sslmode': 'require',
        }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME'),
            'USER': os.getenv('DB_USER'),
            'PASSWORD': os.getenv('DB_PASSWORD'),
            'HOST': os.getenv('DB_HOST'),
            'PORT': os.getenv('DB_PORT'),
        }
    }

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = Path(os.getenv('MEDIA_ROOT', BASE_DIR / 'storage'))

# Custom storage paths
UPLOAD_DIR = MEDIA_ROOT / 'uploads'
OUTPUT_DIR = MEDIA_ROOT / 'outputs'
TEMP_DIR = MEDIA_ROOT / 'temp'

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
CLOUDINARY_ROOT_FOLDER = os.getenv('CLOUDINARY_ROOT_FOLDER')

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

# Cloudinary storage settings 
CLOUDINARY_STORAGE = {
    'ROOT_FOLDER': CLOUDINARY_ROOT_FOLDER,

    'UPLOADS_FOLDER': 'uploads',
    'OUTPUTS_FOLDER': 'outputs',

    'DOWNLOAD_URL_EXPIRY': 3600,

    'RESOURCE_TYPE_MAP': {
        'video': 'video',
        'image': 'image',
        'audio': 'video',
    }
}

USE_CLOUDINARY = env.bool('USE_CLOUDINARY', default=True)

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760 #10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760 #10MB

# Media type configurations
MAX_FILE_SIZE = {
    'video': 524288000,  # 500MB
    'image': 10485760,   # 10MB
    'audio': 52428800,   # 50MB
}

SUPPORTED_FORMATS = {
    'video': ['mp4', 'avi', 'mov', 'mkv', 'webm'],
    'image': ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
    'audio': ['mp3', 'wav', 'aac', 'ogg', 'flac', 'm4a'],
}

# MIME type to media type mapping
MIME_TYPE_MAPPING = {
    # Video
    'video/mp4': 'video',
    'video/x-msvideo': 'video',
    'video/quicktime': 'video',
    'video/x-matroska': 'video',
    'video/webm': 'video',
    # Image
    'image/jpeg': 'image',
    'image/png': 'image',
    'image/gif': 'image',
    'image/webp': 'image',
    'image/bmp': 'image',
    # Audio
    'audio/mpeg': 'audio',
    'audio/wav': 'audio',
    'audio/x-wav': 'audio',
    'audio/aac': 'audio',
    'audio/ogg': 'audio',
    'audio/flac': 'audio',
    'audio/x-m4a': 'audio',
    'audio/mp4': 'audio',
}

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Redis Queue Configuration
RQ_QUEUES = {
    'video_queue': {
        'HOST': env('REDIS_HOST', default='localhost'),
        'PORT': env('REDIS_PORT', default=6379),
        'DB': env('REDIS_DB', default=0),
        'DEFAULT_TIMEOUT': 1800,  # 30 minutes
    },
    'image_queue': {
        'HOST': env('REDIS_HOST', default='localhost'),
        'PORT': env('REDIS_PORT', default=6379),
        'DB': env('REDIS_DB', default=0),
        'DEFAULT_TIMEOUT': 60,  # 1 minute
    },
    'audio_queue': {
        'HOST': env('REDIS_HOST', default='localhost'),
        'PORT': env('REDIS_PORT', default=6379),
        'DB': env('REDIS_DB', default=0),
        'DEFAULT_TIMEOUT': 300,  # 5 minutes
    },
}

# Django REST Framework
REST_FRAMEWORK = {
    # Pagination
    'DEFAULT_PAGINATION_CLASS': 'apps.api.pagination.StandardLimitOffsetPagination',
    'PAGE_SIZE': 50,
    
    # Renderers - JSON only for API
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    
    # Parsers - Support JSON, form data, and multipart (file uploads)
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    
    # Authentication - Session-based for anonymous users
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    
    # Permissions - Allow any for anonymous access
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    
    # Throttling - Rate limiting for abuse prevention
    'DEFAULT_THROTTLE_CLASSES': [
        'apps.api.throttling.AnonBurstRateThrottle',
        'apps.api.throttling.AnonSustainedRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon_burst': '60/minute',
        'anon_sustained': '1000/day',
        'uploads': '10/hour',
        'status_checks': '120/minute',
    },
    
    # Filtering
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.OrderingFilter',
    ],
    
    # Exception handling - Use custom handler
    'EXCEPTION_HANDLER': 'apps.api.exceptions.custom_exception_handler',
    
    # Date/time formatting
    'DATETIME_FORMAT': '%Y-%m-%dT%H:%M:%S.%fZ',
    'DATE_FORMAT': '%Y-%m-%d',
    'TIME_FORMAT': '%H:%M:%S',
    
    # Content negotiation
    'DEFAULT_CONTENT_NEGOTIATION_CLASS': 'rest_framework.negotiation.DefaultContentNegotiation',
    
    # Metadata
    'DEFAULT_METADATA_CLASS': 'rest_framework.metadata.SimpleMetadata',
    
    # Versioning (for future API versions)
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'ALLOWED_VERSIONS': ['v1'],
    'DEFAULT_VERSION': 'v1',
    
    # Test request defaults
    'TEST_REQUEST_DEFAULT_FORMAT': 'json',
}

# CORS Configuration
CORS_ALLOWED_ORIGINS = env.list('CORS_ALLOWED_ORIGINS', default=[
    'http://localhost:3000',
    'http://127.0.0.1:3000',
])
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# FFmpeg Configuration
FFMPEG_PATH = env('FFMPEG_PATH', default='ffmpeg')
FFPROBE_PATH = env('FFPROBE_PATH', default='ffprobe')

# Session Configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_SAVE_EVERY_REQUEST = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'None'
SESSION_COOKIE_SECURE = True

# API-specific settings
API_VERSION = 'v1'
API_TITLE = 'Media Processor API'
API_DESCRIPTION = 'REST API for media processing operations'

# Operation expiration settings
OPERATION_EXPIRY_DAYS = 7

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} {module} {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'mediaprocessor.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps.operations': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'apps.processors': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'apps.api': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}