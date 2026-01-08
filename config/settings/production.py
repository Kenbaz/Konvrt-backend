"""
Production settings for mediaprocessor project.
Optimized for Railway deployment.
"""

import os
import dj_database_url

from .base import *
from .rq_settings import (
    RQ_QUEUES,
    RQ_SHOW_ADMIN_LINK,
    RQ_RETRY_MAX_TIMES,
    RQ_RETRY_DELAYS,
    WORKER_CONFIG,
    OPERATION_EXPIRATION_DAYS,
)


DEBUG = False

ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])

ALLOWED_HOSTS.extend(['.railway.app', '.up.railway.app'])


RAILWAY_PUBLIC_DOMAIN = os.environ.get('RAILWAY_PUBLIC_DOMAIN')
if RAILWAY_PUBLIC_DOMAIN:
    ALLOWED_HOSTS.append(RAILWAY_PUBLIC_DOMAIN)


DATABASE_URL = os.getenv('DATABASE_URL')

if DATABASE_URL:
    DATABASES = {
        'default': dj_database_url.config(
            default=DATABASE_URL,
            conn_max_age=600,
            conn_health_checks=True,
            ssl_require=True,
        )
    }
else:
    
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME', 'mediaprocessor'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PASSWORD': os.getenv('DB_PASSWORD', ''),
            'HOST': os.getenv('DB_HOST', 'localhost'),
            'PORT': os.getenv('DB_PORT', '5432'),
            'CONN_MAX_AGE': 600,
        }
    }


SECURE_SSL_REDIRECT = env.bool('SECURE_SSL_REDIRECT', default=True)
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_SAMESITE = 'None'
SESSION_COOKIE_HTTPONLY = True

CSRF_COOKIE_SECURE = True
CSRF_COOKIE_SAMESITE = 'None'
CSRF_COOKIE_HTTPONLY = True

SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# HSTS settings
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Trust Railway's proxy
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')


CSRF_TRUSTED_ORIGINS = env.list('CSRF_TRUSTED_ORIGINS', default=[])

# Add Railway domains
CSRF_TRUSTED_ORIGINS.extend([
    'https://*.railway.app',
    'https://*.up.railway.app',
])

if RAILWAY_PUBLIC_DOMAIN:
    CSRF_TRUSTED_ORIGINS.append(f'https://{RAILWAY_PUBLIC_DOMAIN}')


CORS_ALLOWED_ORIGINS = env.list('CORS_ALLOWED_ORIGINS', default=[])


STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {name} {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps.operations': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps.processors': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'apps.api': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'rq.worker': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}


STORAGE_ROOT = Path(os.environ.get('STORAGE_ROOT', str(BASE_DIR / 'storage')))
MEDIA_ROOT = STORAGE_ROOT
UPLOAD_DIR = STORAGE_ROOT / 'uploads'
OUTPUT_DIR = STORAGE_ROOT / 'outputs'
TEMP_DIR = STORAGE_ROOT / 'temp'


for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)