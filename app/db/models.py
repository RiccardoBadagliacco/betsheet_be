from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text
from sqlalchemy import JSON as SA_JSON
from datetime import datetime
from app.db.database import Base
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, nullable=False, index=True)
    revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)


class Backroll(Base):
    __tablename__ = "backroll"

    id = Column(String, primary_key=True, index=True)
    backroll = Column(Float, nullable=True)
    cassa = Column(Float, nullable=True)
    name = Column(String, nullable=True)

    # relationship to bets
    bets = relationship("Bet", back_populates="backroll", cascade="all, delete-orphan")


class Bet(Base):
    __tablename__ = "bets"

    id = Column(String, primary_key=True, index=True)
    backroll_id = Column(String, ForeignKey("backroll.id"), nullable=True, index=True)
    data = Column(String, nullable=True)
    backroll_iniziale = Column(Float, nullable=True)
    cassa = Column(Float, nullable=True)
    quota = Column(Float, nullable=True)
    stake = Column(Float, nullable=True)
    importo = Column(Float, nullable=True)
    vincita = Column(Float, nullable=True)
    backroll_finale = Column(Float, nullable=True)
    profitto_totale = Column(Float, nullable=True)
    profitto = Column(Float, nullable=True)
    esito = Column(String, nullable=True)
    partite = Column(SA_JSON, nullable=True)
    tipo_giocata = Column(String, nullable=True)
    link = Column(Text, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    backroll = relationship("Backroll", back_populates="bets")
