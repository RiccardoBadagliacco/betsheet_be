"""
API semplice per ottenere le stagioni di una lega
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database_football import get_football_db
from typing import Dict, Any, List
from sqlalchemy import func
from app.constants.leagues import get_all_leagues

router = APIRouter(prefix="/leagues", tags=["Leagues & Seasons"])

def calculate_season_stats(db: Session, season_id: str) -> Dict[str, Any]:
    """
    Calcola le statistiche per una stagione specifica.
    
    Args:
        db: Sessione database
        season_id: ID della stagione
    
    Returns:
        Dizionario con statistiche raggruppate per categoria
    """
    from app.db.models_football import Match
    
    # Query per ottenere tutte le partite della stagione con risultati validi
    matches = db.query(Match).filter(
        Match.season_id == season_id,
        Match.home_goals_ft.isnot(None),
        Match.away_goals_ft.isnot(None)
    ).all()
    
    if not matches:
        return {
            "goals": {"over_0_5": 0, "under_0_5": 0, "over_1_5": 0, "under_1_5": 0, 
                     "over_2_5": 0, "under_2_5": 0, "over_3_5": 0, "under_3_5": 0},
            "marcatori_multigol": {"casa_1_3": 0, "casa_1_4": 0, "casa_1_5": 0,
                                  "ospite_1_3": 0, "ospite_1_4": 0, "ospite_1_5": 0},
            "gol_goal": {"gg": 0, "no_gg": 0},
            "segna_squadra": {"segna_casa_si": 0, "segna_casa_no": 0, "segna_ospite_si": 0, "segna_ospite_no": 0},
            "total_matches_analyzed": 0
        }
    
    total_matches = len(matches)
    
    # Inizializza contatori
    over_0_5 = under_0_5 = 0
    over_1_5 = under_1_5 = 0  
    over_2_5 = under_2_5 = 0
    over_3_5 = under_3_5 = 0
    
    casa_1_3 = casa_1_4 = casa_1_5 = 0
    ospite_1_3 = ospite_1_4 = ospite_1_5 = 0
    
    gg = no_gg = 0
    segna_casa_si = segna_casa_no = 0
    segna_ospite_si = segna_ospite_no = 0
    
    for match in matches:
        home_goals = match.home_goals_ft
        away_goals = match.away_goals_ft
        total_goals = home_goals + away_goals
        
        # Statistiche Goal Totali
        if total_goals > 0.5: over_0_5 += 1
        else: under_0_5 += 1
        
        if total_goals > 1.5: over_1_5 += 1
        else: under_1_5 += 1
        
        if total_goals > 2.5: over_2_5 += 1
        else: under_2_5 += 1
        
        if total_goals > 3.5: over_3_5 += 1
        else: under_3_5 += 1
        
        # Statistiche Marcatori Multigol (1-3 gol, 1-4 gol, 1-5 gol)
        if 1 <= home_goals <= 3: casa_1_3 += 1
        if 1 <= home_goals <= 4: casa_1_4 += 1
        if 1 <= home_goals <= 5: casa_1_5 += 1
        
        if 1 <= away_goals <= 3: ospite_1_3 += 1
        if 1 <= away_goals <= 4: ospite_1_4 += 1
        if 1 <= away_goals <= 5: ospite_1_5 += 1
        
        # Gol Goal (entrambe le squadre segnano)
        if home_goals > 0 and away_goals > 0: gg += 1
        else: no_gg += 1
        
        # Segna Casa
        if home_goals > 0: segna_casa_si += 1
        else: segna_casa_no += 1
        
        # Segna Ospite
        if away_goals > 0: segna_ospite_si += 1
        else: segna_ospite_no += 1
    
    # Calcola percentuali
    def calc_percentage(count: int, total: int) -> float:
        return round((count / total) * 100, 2) if total > 0 else 0.0
    
    return {
        "goals": {
            "over_0_5": calc_percentage(over_0_5, total_matches),
            "under_0_5": calc_percentage(under_0_5, total_matches),
            "over_1_5": calc_percentage(over_1_5, total_matches),
            "under_1_5": calc_percentage(under_1_5, total_matches),
            "over_2_5": calc_percentage(over_2_5, total_matches),
            "under_2_5": calc_percentage(under_2_5, total_matches),
            "over_3_5": calc_percentage(over_3_5, total_matches),
            "under_3_5": calc_percentage(under_3_5, total_matches)
        },
        "marcatori_multigol": {
            "casa_1_3": calc_percentage(casa_1_3, total_matches),
            "casa_1_4": calc_percentage(casa_1_4, total_matches), 
            "casa_1_5": calc_percentage(casa_1_5, total_matches),
            "ospite_1_3": calc_percentage(ospite_1_3, total_matches),
            "ospite_1_4": calc_percentage(ospite_1_4, total_matches),
            "ospite_1_5": calc_percentage(ospite_1_5, total_matches)
        },
        "gol_goal": {
            "gg": calc_percentage(gg, total_matches),
            "no_gg": calc_percentage(no_gg, total_matches)
        },
        "segna_squadra": {
            "segna_casa_si": calc_percentage(segna_casa_si, total_matches),
            "segna_casa_no": calc_percentage(segna_casa_no, total_matches),
            "segna_ospite_si": calc_percentage(segna_ospite_si, total_matches),
            "segna_ospite_no": calc_percentage(segna_ospite_no, total_matches)
        },
        "total_matches_analyzed": total_matches
    }

@router.get("/seasons",
           summary="Get all leagues with their seasons")
async def get_all_leagues_seasons(db: Session = Depends(get_football_db)) -> Dict[str, Any]:
    """
    Ritorna tutte le leghe con le loro stagioni.
    
    Returns:
        Dizionario con tutte le leghe e le loro stagioni
    """
    LEAGUES = get_all_leagues('all')
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
    from app.constants.leagues import get_all_leagues
    from app.db.models_football import Season, League
    
    try:
        # Normalizza il codice lega
        league_code = league_code.upper()
        
        # Verifica se la lega esiste nelle costanti
        league_info = get_all_leagues('all').get(league_code)
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
            # Calcola statistiche per la stagione
            stats = calculate_season_stats(db, season.id)
            
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
                "stats": stats
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