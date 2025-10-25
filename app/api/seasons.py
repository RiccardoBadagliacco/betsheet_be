"""
API per gestire e correggere i dati delle stagioni
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from app.db.database_football import get_football_db
from app.db.models_football import Season, League, Match
from datetime import datetime, date
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/seasons/status")
async def get_seasons_status(
    db: Session = Depends(get_football_db)
):
    """
    Mostra lo stato di tutte le stagioni nel database
    """
    try:
        seasons = db.query(Season).join(League).all()
        
        seasons_data = []
        current_year = datetime.now().year
        
        for season in seasons:
            # Determina se la stagione dovrebbe essere completata
            season_start_year = int(season.code[:2])
            if season_start_year > 50:  # 1900+
                season_start_year += 1900
            else:  # 2000+
                season_start_year += 2000
            
            # Una stagione dovrebbe essere completata SOLO se ha una end_date
            # Se non ha end_date, deve rimanere is_completed = False
            should_be_completed = season.end_date is not None
            
            # Conta le partite
            match_count = db.query(Match).filter(Match.season_id == season.id).count()
            
            seasons_data.append({
                "id": str(season.id),
                "league_code": season.league.code,
                "league_name": season.league.name,
                "season_name": season.name,
                "season_code": season.code,
                "is_completed": season.is_completed,
                "should_be_completed": should_be_completed,
                "needs_update": season.is_completed != should_be_completed,
                "match_count": match_count,
                "start_date": season.start_date.strftime("%Y-%m-%d") if season.start_date else None,
                "end_date": season.end_date.strftime("%Y-%m-%d") if season.end_date else None,
                "created_at": season.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Statistiche
        total_seasons = len(seasons_data)
        needs_update = sum(1 for s in seasons_data if s['needs_update'])
        completed_seasons = sum(1 for s in seasons_data if s['is_completed'])
        
        return {
            "success": True,
            "total_seasons": total_seasons,
            "completed_seasons": completed_seasons,
            "needs_update": needs_update,
            "seasons": seasons_data,
            "statistics": {
                "completion_rate": round(completed_seasons / total_seasons * 100, 1) if total_seasons > 0 else 0,
                "seasons_needing_update": needs_update
            }
        }
        
    except Exception as e:
        logger.error(f"Errore nel recupero status stagioni: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.post("/seasons/update-completion-status")
async def update_seasons_completion_status(
    dry_run: bool = Query(False, description="Se True, mostra solo cosa verrebbe aggiornato senza modificare"),
    db: Session = Depends(get_football_db)
):
    """
    Aggiorna lo stato is_completed delle stagioni in base all'anno
    """
    try:
        current_year = datetime.now().year
        seasons_to_update = []
        
        # Trova stagioni che dovrebbero essere completate ma non lo sono
        seasons = db.query(Season).join(League).all()
        
        for season in seasons:
            # Determina se la stagione dovrebbe essere completata
            season_start_year = int(season.code[:2])
            if season_start_year > 50:  # 1900+
                season_start_year += 1900
            else:  # 2000+
                season_start_year += 2000
            
            # Una stagione dovrebbe essere completata SOLO se ha una end_date
            # Se non ha end_date, deve rimanere is_completed = False
            should_be_completed = season.end_date is not None
            
            if season.is_completed != should_be_completed:
                seasons_to_update.append({
                    "id": season.id,
                    "league_code": season.league.code,
                    "season_name": season.name,
                    "season_code": season.code,
                    "current_status": season.is_completed,
                    "new_status": should_be_completed,
                    "season_start_year": season_start_year
                })
        
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "seasons_to_update": len(seasons_to_update),
                "changes": seasons_to_update,
                "message": f"Dry run: {len(seasons_to_update)} stagioni verrebbero aggiornate"
            }
        
        # Applica gli aggiornamenti
        updated_count = 0
        for season_info in seasons_to_update:
            season = db.query(Season).filter(Season.id == season_info["id"]).first()
            if season:
                season.is_completed = season_info["new_status"]
                season.updated_at = datetime.utcnow()
                updated_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "updated_seasons": updated_count,
            "changes": seasons_to_update,
            "message": f"Aggiornate {updated_count} stagioni"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Errore nell'aggiornamento stagioni: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.get("/seasons/old-active")
async def get_old_active_seasons(
    db: Session = Depends(get_football_db)
):
    """
    Trova stagioni vecchie che risultano ancora attive (is_completed=False)
    """
    try:
        current_year = datetime.now().year
        old_seasons = []
        
        seasons = db.query(Season).join(League).filter(
            Season.is_completed == False
        ).all()
        
        for season in seasons:
            # Determina l'anno di inizio della stagione
            season_start_year = int(season.code[:2])
            if season_start_year > 50:  # 1900+
                season_start_year += 1900
            else:  # 2000+
                season_start_year += 2000
            
            # Se la stagione non ha end_date, deve rimanere attiva (is_completed = False)
            if season.end_date is None:
                match_count = db.query(Match).filter(Match.season_id == season.id).count()
                
                old_seasons.append({
                    "id": str(season.id),
                    "league_code": season.league.code,
                    "league_name": season.league.name,
                    "season_name": season.name,
                    "season_code": season.code,
                    "season_start_year": season_start_year,
                    "match_count": match_count,
                    "years_old": current_year - season_start_year,
                    "should_be_completed": True
                })
        
        return {
            "success": True,
            "old_active_seasons": len(old_seasons),
            "current_year": current_year,
            "seasons": old_seasons,
            "message": f"Trovate {len(old_seasons)} stagioni vecchie ancora marcate come attive"
        }
        
    except Exception as e:
        logger.error(f"Errore nel recupero stagioni vecchie: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.post("/seasons/mark-completed")
async def mark_seasons_completed(
    season_codes: List[str] = Query(..., description="Lista dei codici stagione da marcare come completate"),
    db: Session = Depends(get_football_db)
):
    """
    Marca specifiche stagioni come completate
    """
    try:
        updated_seasons = []
        
        for season_code in season_codes:
            seasons = db.query(Season).filter(Season.code == season_code).all()
            
            for season in seasons:
                if not season.is_completed:
                    season.is_completed = True
                    season.updated_at = datetime.utcnow()
                    
                    updated_seasons.append({
                        "id": str(season.id),
                        "league_code": season.league.code,
                        "season_name": season.name,
                        "season_code": season.code
                    })
        
        db.commit()
        
        return {
            "success": True,
            "updated_count": len(updated_seasons),
            "updated_seasons": updated_seasons,
            "message": f"Marcate {len(updated_seasons)} stagioni come completate"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Errore nel marcare stagioni completate: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")