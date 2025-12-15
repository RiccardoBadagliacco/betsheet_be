"""Application settings.

This module is compatible with both pydantic v1 and v2+ (which moved
BaseSettings to the separate `pydantic-settings` package).
"""

from pathlib import Path
from typing import List, Optional

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


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "BetSheet API"
    DEBUG: bool = False
    # Support both sqlite and postgres urls
    SQLALCHEMY_DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'bets.db'}"
    FOOTBALL_DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'football_dataset.db'}"
    # Comma-separated origins or list
    CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:4200", "http://127.0.0.1:8000","http://localhost:3000"]
    SECRET_KEY: str = "replace-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    # Telegram bot token (for local/dev). Can also be set via env var TELEGRAM_BOT_TOKEN
    # NOTE: this embeds a token in source — remove before sharing publicly if needed.
    TELEGRAM_BOT_TOKEN: Optional[str] = "8168882419:AAGJAutgGoERpvNV6x45DY3J1CjzUyYsiZI"

    class Config:
        env_file = ".env"


settings = Settings()
