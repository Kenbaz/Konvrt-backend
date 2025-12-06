# apps/api/apps.py
from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.api"
    verbose_name = "API"

    def ready(self):
        """
        Perform initialization when the app is ready.
        
        This is called once when Django starts.
        """
        # Import signals or perform other initialization if needed
        pass
