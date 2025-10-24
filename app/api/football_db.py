"""
API endpoints per gestire il database calcistico
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from app.db.database import get_db
from app.db.database_football import get_football_db
from typing import Dict, Any
import os

router = APIRouter(prefix="/football-db", tags=["Football Database"])

@router.get("/status")
async def get_database_status(
    main_db: Session = Depends(get_db),
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Restituisce lo stato dei database (principale e calcistico)
    """
    try:
        # Controlla database principale
        main_inspector = inspect(main_db.bind)
        main_tables = main_inspector.get_table_names()
        
        # Controlla database calcistico
        football_inspector = inspect(football_db.bind)
        football_tables = football_inspector.get_table_names()
        
        # Conta record nelle tabelle principali del database calcistico
        football_counts = {}
        for table in football_tables:
            result = football_db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            football_counts[table] = count
        
        # Informazioni sui file database
        main_db_path = "./bets.db"
        football_db_path = "./football_dataset.db"
        
        main_db_size = os.path.getsize(main_db_path) if os.path.exists(main_db_path) else 0
        football_db_size = os.path.getsize(football_db_path) if os.path.exists(football_db_path) else 0
        
        return {
            "status": "success",
            "main_database": {
                "path": main_db_path,
                "size_bytes": main_db_size,
                "size_mb": round(main_db_size / (1024 * 1024), 2),
                "tables": main_tables
            },
            "football_database": {
                "path": football_db_path,
                "size_bytes": football_db_size,
                "size_mb": round(football_db_size / (1024 * 1024), 2),
                "tables": football_tables,
                "record_counts": football_counts
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel controllo database: {str(e)}")

@router.get("/tables/{table_name}")
async def get_table_info(
    table_name: str,
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Restituisce informazioni dettagliate su una tabella specifica
    """
    try:
        inspector = inspect(football_db.bind)
        tables = inspector.get_table_names()
        
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Tabella {table_name} non trovata")
        
        # Ottieni schema della tabella
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        # Conta record
        result = football_db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        record_count = result.scalar()
        
        return {
            "table_name": table_name,
            "record_count": record_count,
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero informazioni tabella: {str(e)}")

@router.delete("/clear/{table_name}")
async def clear_table(
    table_name: str,
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Svuota una tabella specifica (ATTENZIONE: operazione irreversibile)
    """
    try:
        inspector = inspect(football_db.bind)
        tables = inspector.get_table_names()
        
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Tabella {table_name} non trovata")
        
        # Conta record prima della cancellazione
        result = football_db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        records_before = result.scalar()
        
        # Svuota la tabella
        football_db.execute(text(f"DELETE FROM {table_name}"))
        football_db.commit()
        
        return {
            "status": "success",
            "table_name": table_name,
            "records_deleted": records_before,
            "message": f"Tabella {table_name} svuotata con successo"
        }
    except HTTPException:
        raise
    except Exception as e:
        football_db.rollback()
        raise HTTPException(status_code=500, detail=f"Errore nel svuotare la tabella: {str(e)}")

@router.get("/leagues/seasons", summary="Get all leagues with available seasons")
async def get_leagues_with_seasons(db: Session = Depends(get_football_db)) -> Dict[str, Any]:
    """
    Get all leagues with their available seasons from the football database.
    
    Returns a dictionary with each league and its available seasons based on actual data.
    Only shows leagues that have data in the database.
    """
    from app.constants.leagues import LEAGUES, get_formatted_league_name
    from app.db.models_football import FootballMatch
    from sqlalchemy import distinct
    
    try:
        # Query per ottenere tutte le combinazioni league/season dal database
        from app.db.models_football import Season, League
        league_seasons = db.query(
            League.code,
            Season.code
        ).join(Season, League.id == Season.league_id).distinct().order_by(
            League.code,
            Season.code.desc()
        ).all()
        
        # Raggruppa le stagioni per lega
        leagues_data = {}
        for league_code, season_code in league_seasons:
            if league_code not in leagues_data:
                # Ottieni info della lega dalle costanti
                league_info = LEAGUES.get(league_code, {
                    "name": f"Unknown League ({league_code})",
                    "country": "Unknown"
                })
                
            leagues_data[league_code] = {
                "code": league_code,
                "name": league_info["name"],
                "country": league_info["country"],
                "formatted_name": get_formatted_league_name(league_code),
                "seasons": []
            }
        
        leagues_data[league_code]["seasons"].append(season_code)        # Converti in lista e ordina per paese e nome lega
        leagues_list = list(leagues_data.values())
        leagues_list.sort(key=lambda x: (x["country"], x["name"]))
        
        # Statistiche totali
        total_leagues = len(leagues_list)
        total_seasons = sum(len(league["seasons"]) for league in leagues_list)
        
        # Conta anche i matches totali per completezza
        from app.db.models_football import Match
        total_matches = db.query(Match).count()
        
        return {
            "leagues": leagues_list,
            "statistics": {
                "total_leagues_with_data": total_leagues,
                "total_seasons": total_seasons,
                "total_matches": total_matches,
                "average_seasons_per_league": round(total_seasons / total_leagues, 1) if total_leagues > 0 else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero delle leghe e stagioni: {str(e)}")

@router.get("/leagues/all", summary="Get all defined leagues (with or without data)")
async def get_all_leagues(db: Session = Depends(get_football_db)) -> Dict[str, Any]:
    """
    Get all leagues defined in constants, showing which ones have data and which don't.
    
    Returns information about all leagues including data availability status.
    """
    from app.constants.leagues import LEAGUES, get_formatted_league_name
    from app.db.models_football import FootballMatch
    from sqlalchemy import distinct
    
    try:
        # Ottieni leghe con dati dal database
        from app.db.models_football import Season, League
        leagues_with_data = db.query(distinct(League.code)).all()
        leagues_with_data_set = {league[0] for league in leagues_with_data}
        
        # Per ogni lega con dati, ottieni anche le stagioni
        leagues_seasons_data = {}
        if leagues_with_data_set:
            league_seasons = db.query(
                League.code,
                Season.code
            ).join(Season, League.id == Season.league_id).distinct().all()
            
            for league_code, season_code in league_seasons:
                if league_code not in leagues_seasons_data:
                    leagues_seasons_data[league_code] = []
                leagues_seasons_data[league_code].append(season_code)
        
        # Costruisci lista completa delle leghe
        all_leagues = []
        for code, info in LEAGUES.items():
            has_data = code in leagues_with_data_set
            seasons = sorted(leagues_seasons_data.get(code, []), reverse=True)
            
            all_leagues.append({
                "code": code,
                "name": info["name"],
                "country": info["country"],
                "formatted_name": get_formatted_league_name(code),
                "has_data": has_data,
                "seasons_count": len(seasons),
                "seasons": seasons if has_data else []
            })
        
        # Ordina per paese e nome
        all_leagues.sort(key=lambda x: (x["country"], x["name"]))
        
        # Statistiche
        leagues_with_data_count = sum(1 for league in all_leagues if league["has_data"])
        leagues_without_data_count = len(all_leagues) - leagues_with_data_count
        total_seasons = sum(league["seasons_count"] for league in all_leagues)
        
        return {
            "leagues": all_leagues,
            "statistics": {
                "total_leagues_defined": len(all_leagues),
                "leagues_with_data": leagues_with_data_count,
                "leagues_without_data": leagues_without_data_count,
                "total_seasons": total_seasons,
                "data_coverage_percentage": round((leagues_with_data_count / len(all_leagues)) * 100, 1)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero di tutte le leghe: {str(e)}")