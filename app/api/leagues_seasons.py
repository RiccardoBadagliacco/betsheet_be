"""
API semplice per ottenere le stagioni di una lega
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database_football import get_football_db
from typing import Dict, Any, List

router = APIRouter(prefix="/leagues", tags=["Leagues & Seasons"])

@router.get("/seasons",
           summary="Get all leagues with their seasons")
async def get_all_leagues_seasons(db: Session = Depends(get_football_db)) -> Dict[str, Any]:
    """
    Ritorna tutte le leghe con le loro stagioni.
    
    Returns:
        Dizionario con tutte le leghe e le loro stagioni
    """
    from app.constants.leagues import LEAGUES
    from app.db.models_football import Season, League
    
    try:
        # Query per ottenere tutte le combinazioni league/season
        league_seasons = db.query(
            League.code,
            Season.code
        ).join(Season, League.id == Season.league_id).distinct().all()
        
        # Raggruppa per lega
        leagues_data = {}
        for league_code, season_code in league_seasons:
            if league_code not in leagues_data:
                league_info = LEAGUES.get(league_code, {
                    "name": f"League {league_code}",
                    "country": "Unknown"
                })
                
                leagues_data[league_code] = {
                    "code": league_code,
                    "name": league_info["name"],
                    "country": league_info["country"],
                    "seasons": []
                }
            
            leagues_data[league_code]["seasons"].append(season_code)
        
        # Ordina le stagioni per ogni lega (più recenti prima)
        for league_code in leagues_data:
            leagues_data[league_code]["seasons"] = sorted(
                leagues_data[league_code]["seasons"], 
                reverse=True
            )
        
        # Raggruppa per paese come si aspetta il frontend
        countries_data = {}
        for league in leagues_data.values():
            country = league["country"]
            if country not in countries_data:
                countries_data[country] = {
                    "country": country,
                    "leagues": [],
                    "leagues_count": 0
                }
            
            countries_data[country]["leagues"].append({
                "code": league["code"],
                "name": league["name"],
                "seasons": league["seasons"]
            })
            countries_data[country]["leagues_count"] += 1
        
        # Ordina paesi e leghe
        countries_list = list(countries_data.values())
        countries_list.sort(key=lambda x: x["country"])
        
        # Ordina le leghe all'interno di ogni paese per codice
        for country in countries_list:
            country["leagues"].sort(key=lambda x: x["code"])
        
        # Calcola statistiche totali
        total_leagues = sum(len(country["leagues"]) for country in countries_list)
        total_seasons = sum(
            len(league["seasons"]) 
            for country in countries_list 
            for league in country["leagues"]
        )
        
        return {
            "countries": countries_list,
            "total_countries": len(countries_list),
            "total_leagues": total_leagues,
            "total_seasons": total_seasons
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.get("/{league_code}/seasons",
           summary="Get all seasons for a specific league")
async def get_league_seasons(
    league_code: str, 
    db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Ritorna tutte le stagioni per una lega specifica.
    
    Args:
        league_code: Il codice della lega (es: I1, E0, D1)
    
    Returns:
        Lista delle stagioni per la lega specificata
    """
    from app.constants.leagues import LEAGUES
    from app.db.models_football import Season, League
    
    try:
        # Normalizza il codice lega
        league_code = league_code.upper()
        
        # Verifica se la lega esiste nelle costanti
        league_info = LEAGUES.get(league_code)
        if not league_info:
            raise HTTPException(status_code=404, detail=f"Lega {league_code} non trovata")
        
        # Ottieni stagioni dal database con tutte le informazioni
        seasons = db.query(Season).join(
            League, League.id == Season.league_id
        ).filter(
            League.code == league_code
        ).order_by(Season.code.desc()).all()
        
        # Costruisci la lista con informazioni complete
        seasons_list = []
        for season in seasons:
            season_data = {
                "code": season.code,
                "name": season.name,
                "start_date": season.start_date.isoformat() if season.start_date else None,
                "end_date": season.end_date.isoformat() if season.end_date else None,
                "is_completed": season.is_completed,
                "total_matches": season.total_matches,
                "processed_matches": season.processed_matches,
                "csv_file_path": season.csv_file_path,
                "created_at": season.created_at.isoformat() if season.created_at else None,
                "updated_at": season.updated_at.isoformat() if season.updated_at else None,
                "stats": None  # Campo stats attualmente null come richiesto
            }
            seasons_list.append(season_data)
        
        if not seasons_list:
            return {
                "league_code": league_code,
                "league_name": league_info["name"],
                "country": league_info["country"],
                "seasons": [],
                "total_seasons": 0,
                "message": f"Nessuna stagione trovata per {league_info['name']}"
            }
        
        return {
            "league_code": league_code,
            "league_name": league_info["name"],
            "country": league_info["country"],
            "seasons": seasons_list,
            "total_seasons": len(seasons_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")