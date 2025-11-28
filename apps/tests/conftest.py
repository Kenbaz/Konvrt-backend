# tests/conftest.py
"""
Pytest configuration for Django tests.

This file configures Django to use the development settings before any tests run.
"""

import os
import sys
import django


def pytest_configure(config):
    """
    Configure Django settings for pytest.
    
    This function runs before any tests are collected.
    It tells Django to use our existing development settings.
    """
    # Set the Django settings module to use our development settings
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
    
    # Setup Django
    django.setup()