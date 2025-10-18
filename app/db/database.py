from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.settings import settings

# Use settings.SQLALCHEMY_DATABASE_URL
SQLALCHEMY_DATABASE_URL = settings.SQLALCHEMY_DATABASE_URL

# SQLite needs check_same_thread
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency that yields a DB session and ensures close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
