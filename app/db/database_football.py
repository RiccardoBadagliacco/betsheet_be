"""
Football database connection module.
Manages connection to the separate football dataset database.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.settings import settings

# Create engine for football database
football_engine = create_engine(
    settings.FOOTBALL_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.FOOTBALL_DATABASE_URL else {}
)

# Create SessionLocal for football database
FootballSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=football_engine)

# Create base for football models
FootballBase = declarative_base()

def get_football_db():
    """Dependency to get football database session."""
    db = FootballSessionLocal()
    try:
        yield db
    finally:
        db.close()