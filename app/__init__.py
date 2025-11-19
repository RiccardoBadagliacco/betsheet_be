"""App package root with lazy FastAPI import."""
from importlib import import_module
from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        # Lazy import to avoid pulling FastAPI stack when not needed
        module = import_module("app.main")
        return module.app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
