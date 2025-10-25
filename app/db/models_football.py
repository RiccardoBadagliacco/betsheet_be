"""
Modelli per il database strutturato dei dati calcistici
"""

from sqlalchemy import Column, Integer, String, Date, Boolean, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
import uuid
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from app.db.database_football import FootballBase

class Country(FootballBase):
    __tablename__ = "countries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(3), unique=True, nullable=False, index=True)  # es. "ITA", "ENG"
    name = Column(String(100), nullable=False)  # es. "Italy", "England"
    flag_url = Column(String(255), nullable=True)  # URL icona bandiera
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relazioni
    leagues = relationship("League", back_populates="country")

class League(FootballBase):
    """Modello per le leghe"""
    __tablename__ = "leagues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(10), unique=True, nullable=False, index=True)  # es. "I1", "E0"
    name = Column(String(100), nullable=False)  # es. "Serie A", "Premier League"
    logo_url = Column(String(255), nullable=True)  # URL logo lega
    country_id = Column(UUID(as_uuid=True), ForeignKey("countries.id"), nullable=False)
    tier = Column(Integer, nullable=True)  # 1 per serie A, 2 per serie B, etc.
    active = Column(Boolean, default=True)  # Se la lega è ancora attiva
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relazioni
    country = relationship("Country", back_populates="leagues")
    seasons = relationship("Season", back_populates="league")

class Season(FootballBase):
    """Modello per le stagioni"""
    __tablename__ = "seasons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id = Column(UUID(as_uuid=True), ForeignKey("leagues.id"), nullable=False)
    name = Column(String(20), nullable=False)  # es. "2023/2024"
    code = Column(String(10), nullable=False)  # es. "2324"
    start_date = Column(Date, nullable=True)  # Data prima partita
    end_date = Column(Date, nullable=True)    # Data ultima partita (null se non conclusa)
    is_completed = Column(Boolean, default=False, nullable=False)  # True se stagione conclusa
    total_matches = Column(Integer, default=0)  # Numero totale partite
    processed_matches = Column(Integer, default=0)  # Partite elaborate nel DB
    csv_file_path = Column(String(255), nullable=True)  # Percorso file CSV originale
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relazioni
    league = relationship("League", back_populates="seasons")
    matches = relationship("Match", back_populates="season")

class Team(FootballBase):
    """Modello per le squadre"""
    __tablename__ = "teams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, index=True)
    normalized_name = Column(String(100), nullable=False, index=True)  # Nome normalizzato per matching
    country_id = Column(UUID(as_uuid=True), ForeignKey("countries.id"), nullable=True)
    logo_url = Column(String(255), nullable=True)
    founded_year = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relazioni
    country = relationship("Country")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")

class Match(FootballBase):
    """Modello per le partite"""
    __tablename__ = "matches"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    season_id = Column(UUID(as_uuid=True), ForeignKey("seasons.id"), nullable=False, index=True)
    home_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=False)
    
    # Dati temporali
    match_date = Column(Date, nullable=False, index=True)
    match_time = Column(String(10), nullable=True)  # es. "15:00"
    
    # Risultati tempo pieno
    home_goals_ft = Column(Integer, nullable=True)  # FTHG
    away_goals_ft = Column(Integer, nullable=True)  # FTAG
    
    # Risultati primo tempo
    home_goals_ht = Column(Integer, nullable=True)  # HTHG  
    away_goals_ht = Column(Integer, nullable=True)  # HTAG
    
    # Statistiche partita
    home_shots = Column(Integer, nullable=True)       # HS
    away_shots = Column(Integer, nullable=True)       # AS
    home_shots_target = Column(Integer, nullable=True)  # HST
    away_shots_target = Column(Integer, nullable=True)  # AST
    
    # Quote medie
    avg_home_odds = Column(Float, nullable=True)      # AvgH
    avg_draw_odds = Column(Float, nullable=True)      # AvgD  
    avg_away_odds = Column(Float, nullable=True)      # AvgA
    avg_over_25_odds = Column(Float, nullable=True)   # Avg>2.5
    avg_under_25_odds = Column(Float, nullable=True)  # Avg<2.5
    
    # Metadati
    csv_row_number = Column(Integer, nullable=True)   # Numero riga nel CSV originale
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relazioni
    season = relationship("Season", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")


class Fixture(FootballBase):
    """Modello per le partite in programma (fixtures)"""
    __tablename__ = "fixtures"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    season_id = Column(UUID(as_uuid=True), ForeignKey("seasons.id"), nullable=True, index=True)
    home_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    away_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"), nullable=True)
    
    # Dati temporali
    match_date = Column(Date, nullable=False, index=True)
    match_time = Column(String(10), nullable=True)  # es. "15:00"
    
    # Risultati tempo pieno (sempre NULL per fixtures)
    home_goals_ft = Column(Integer, nullable=True)  # FTHG
    away_goals_ft = Column(Integer, nullable=True)  # FTAG
    
    # Risultati primo tempo (sempre NULL per fixtures)
    home_goals_ht = Column(Integer, nullable=True)  # HTHG  
    away_goals_ht = Column(Integer, nullable=True)  # HTAG
    
    # Statistiche partita (sempre NULL per fixtures)
    home_shots = Column(Integer, nullable=True)       # HS
    away_shots = Column(Integer, nullable=True)       # AS
    home_shots_target = Column(Integer, nullable=True)  # HST
    away_shots_target = Column(Integer, nullable=True)  # AST
    
    # Quote medie
    avg_home_odds = Column(Float, nullable=True)      # AvgH
    avg_draw_odds = Column(Float, nullable=True)      # AvgD  
    avg_away_odds = Column(Float, nullable=True)      # AvgA
    avg_over_25_odds = Column(Float, nullable=True)   # Avg>2.5
    avg_under_25_odds = Column(Float, nullable=True)  # Avg<2.5
    
    # Dati grezzi per identificazione
    league_code = Column(String(10), nullable=True)      # Codice lega dal CSV (es. "E0", "I1")
    league_name = Column(String(100), nullable=True)     # Nome lega completo
    
    # Metadati
    csv_row_number = Column(Integer, nullable=True)   # Numero riga nel CSV originale
    downloaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relazioni (opzionali per fixtures)
    season = relationship("Season")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])