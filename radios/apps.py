from django.apps import AppConfig


class RadiosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'radios'

    def ready(self):
        import radios.signals  # noqa: F401 — register FTS sync signals
