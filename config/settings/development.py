"""
Development settings for mediaprocessor project.
"""

from .base import *
import dj_database_url

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ENVIRONMENT = os.getenv('DJANGO_ENVIRONMENT', 'development')

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Add django-debug-toolbar for development
INSTALLED_APPS += [
    'debug_toolbar',
]

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

MIDDLEWARE += [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

# Debug toolbar configuration
INTERNAL_IPS = [
    '127.0.0.1',
]

# Logging - More verbose in development
LOGGING['root']['level'] = 'DEBUG'
LOGGING['loggers']['apps.operations']['level'] = 'DEBUG'
LOGGING['loggers']['apps.processors']['level'] = 'DEBUG'

# Email backend for development
# EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'