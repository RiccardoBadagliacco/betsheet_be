"""
Script per inizializzare il database calcistico
"""

from app.db.database_football import football_engine, FootballBase
from app.db.models_football import Country, League, Season, Team, Match

def create_football_tables():
    """Crea tutte le tabelle nel database calcistico"""
    print("Creando le tabelle del database calcistico...")
    FootballBase.metadata.create_all(bind=football_engine)
    print("Tabelle create con successo!")
    
    # Verifica le tabelle create
    from sqlalchemy import inspect
    inspector = inspect(football_engine)
    tables = inspector.get_table_names()
    print(f"Tabelle create: {tables}")

if __name__ == "__main__":
    create_football_tables()