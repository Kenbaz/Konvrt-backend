"""
Base settings for mediaprocessor project.
These settings are shared across all environments.
"""

import os
from pathlib import Path
import environ

# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, False)
)

# Build paths inside the project
# BASE_DIR is three levels up from this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Read .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env('SECRET_KEY', default='django-insecure-j20zyakxx*kh_!68m-6$v_cb-i99mof5dp@fq^diq=9x9g!$qe')

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
    
    # Local apps
    'apps.core',
    'apps.operations',
    'apps.processors',
    'apps.api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
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
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME', default='mediaprocessor'),
        'USER': env('DB_USER', default='postgres'),
        'PASSWORD': env('DB_PASSWORD', default=''),
        'HOST': env('DB_HOST', default='localhost'),
        'PORT': env('DB_PORT', default='5432'),
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

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'storage'

# Custom storage paths
UPLOAD_DIR = MEDIA_ROOT / 'uploads'
OUTPUT_DIR = MEDIA_ROOT / 'outputs'
TEMP_DIR = MEDIA_ROOT / 'temp'

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
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# FFmpeg Configuration
FFMPEG_PATH = env('FFMPEG_PATH', default='ffmpeg')
FFPROBE_PATH = env('FFPROBE_PATH', default='ffprobe')

# Session Configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_SAVE_EVERY_REQUEST = True

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
        'apps.jobs': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'apps.processors': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}