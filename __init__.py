"""Package root. Backwards-compatible re-exports to the new app layout."""

from .app.main import app  # re-export for compatibility

__all__ = ["app"]
