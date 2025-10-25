"""Application settings.

This module is compatible with both pydantic v1 and v2+ (which moved
BaseSettings to the separate `pydantic-settings` package).
"""

from typing import List

# BaseSettings moved to pydantic-settings in pydantic v2. Try to import
# from there first, otherwise fall back to pydantic (v1).
try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover - fallback for older/newer environments
    from pydantic import BaseSettings

# AnyHttpUrl can be imported from pydantic (works for most versions)
try:
    from pydantic import AnyHttpUrl
except Exception:
    # last resort: import from networks module
    from pydantic.networks import AnyHttpUrl


class Settings(BaseSettings):
    APP_NAME: str = "BetSheet API"
    DEBUG: bool = False
    # Support both sqlite and postgres urls
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./data/bets.db"
    FOOTBALL_DATABASE_URL: str = "sqlite:///./data/football_dataset.db"
    # Comma-separated origins or list
    CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:4200", "http://127.0.0.1:8000"]
    SECRET_KEY: str = "replace-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    class Config:
        env_file = ".env"


settings = Settings()
