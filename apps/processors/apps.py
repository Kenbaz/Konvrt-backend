# apps/processors/apps.py

from django.apps import AppConfig
from rq_workers import check_ffmpeg_on_startup


class ProcessorsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.processors"
    verbose_name = "Media Processors"

    def ready(self):
        """
        Called when the app is ready.
        
        Import all processor modules to trigger operation registration.
        """
        # Import processors to register operations with the registry
        from . import video_processing  # noqa: F401
        from . import image_processing  # noqa: F401
        from . import audio_processing  # noqa: F401

        # Check FFmpeg availability on startup
        check_ffmpeg_on_startup()