"""
Production settings for mediaprocessor project.
"""

from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# ALLOWED_HOSTS must be set via environment variable
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])

# Database - PostgreSQL for production
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': env('DB_NAME'),
        'USER': env('DB_USER'),
        'PASSWORD': env('DB_PASSWORD'),
        'HOST': env('DB_HOST'),
        'PORT': env('DB_PORT', default='5432'),
        'CONN_MAX_AGE': 600,
    }
}

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# HSTS settings
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# Logging - Less verbose in production
LOGGING['root']['level'] = 'INFO'
LOGGING['loggers']['apps.operations']['level'] = 'INFO'
LOGGING['loggers']['apps.processors']['level'] = 'INFO'

STORAGE_ROOT = Path(env('STORAGE_ROOT', default=str(BASE_DIR / 'storage')))
MEDIA_ROOT = STORAGE_ROOT
UPLOAD_DIR = STORAGE_ROOT / 'uploads'
OUTPUT_DIR = STORAGE_ROOT / 'outputs'
TEMP_DIR = STORAGE_ROOT / 'temp'

for directory in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)