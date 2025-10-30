#!/usr/bin/env python3
"""
Model Management API
===================

API per gestire la creazione, salvataggio e caricamento di tutti i modelli delle leghe.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models_football import League, Season, Match
from app.api.ml_football_exact import ExactSimpleFootballPredictor, get_football_db
from app.storage.model_storage import ModelStorage, model_storage

router = APIRouter()

class LeagueModelStatus(BaseModel):
    """Status di un modello di lega"""
    league_code: str
    league_name: str
    is_trained: bool
    is_saved: bool
    is_fresh: bool
    training_data_size: int
    last_trained: Optional[str] = None
    file_size: Optional[int] = None
    accuracy_metrics: Optional[Dict] = None

class ModelGenerationRequest(BaseModel):
    """Richiesta per generazione modelli"""
    leagues: Optional[List[str]] = None  # Se None, genera per tutte le leghe
    force_retrain: bool = False  # Se True, ri-allena anche se esiste
    max_age_hours: int = 24  # Considera fresco se < 24 ore

class ModelGenerationResponse(BaseModel):
    """Risposta generazione modelli"""
    success: bool
    total_leagues: int
    generated_models: int
    skipped_models: int
    failed_models: int
    results: List[LeagueModelStatus]
    generation_time_seconds: float

# Cache in-memory dei modelli caricati
loaded_models: Dict[str, ExactSimpleFootballPredictor] = {}

@router.get("/model-status", response_model=List[LeagueModelStatus])
async def get_all_models_status(db: Session = Depends(get_football_db)):
    """Ottieni lo status di tutti i modelli disponibili"""
    
    try:
        # Ottieni tutte le leghe dal database
        leagues = db.query(League).filter(League.active == True).all()
        
        statuses = []
        for league in leagues:
            # Check se modello esiste in memoria
            is_trained = league.code in loaded_models
            
            # Check se modello esiste su disco
            is_saved = model_storage.model_exists(league.code)
            
            # Check se il modello è fresco
            is_fresh = model_storage.is_model_fresh(league.code)
            
            # Ottieni metadati se esistono
            metadata = model_storage.get_model_metadata(league.code) if is_saved else None
            
            # Conta le partite per questa lega
            training_data_size = db.query(Match).join(Season).filter(
                Season.league_id == league.id
            ).count()
            
            status = LeagueModelStatus(
                league_code=league.code,
                league_name=league.name,
                is_trained=is_trained,
                is_saved=is_saved,
                is_fresh=is_fresh,
                training_data_size=training_data_size,
                last_trained=metadata.get('saved_at') if metadata else None,
                file_size=None,  # TODO: get from disk
                accuracy_metrics=metadata.get('accuracy_metrics') if metadata else None
            )
            statuses.append(status)
        
        return statuses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-all-models", response_model=ModelGenerationResponse)
async def generate_all_models(
    request: ModelGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_football_db)
):
    """Genera e salva modelli per tutte le leghe (o quelle specificate)"""
    
    start_time = datetime.now()
    
    try:
        # Ottieni le leghe da processare
        if request.leagues:
            leagues = db.query(League).filter(
                League.code.in_(request.leagues),
                League.active == True
            ).all()
        else:
            leagues = db.query(League).filter(League.active == True).all()
        
        results = []
        generated_count = 0
        skipped_count = 0
        failed_count = 0
        
        for league in leagues:
            try:
                print(f"🔄 Processing league {league.code} ({league.name})...")
                
                # Check se dobbiamo saltare questo modello
                should_skip = (
                    not request.force_retrain and 
                    model_storage.model_exists(league.code) and 
                    model_storage.is_model_fresh(league.code, request.max_age_hours)
                )
                
                if should_skip:
                    print(f"⏭️  Skipping {league.code} - fresh model exists")
                    skipped_count += 1
                    
                    metadata = model_storage.get_model_metadata(league.code)
                    results.append(LeagueModelStatus(
                        league_code=league.code,
                        league_name=league.name,
                        is_trained=league.code in loaded_models,
                        is_saved=True,
                        is_fresh=True,
                        training_data_size=0,  # TODO: get from metadata
                        last_trained=metadata.get('saved_at') if metadata else None
                    ))
                    continue
                
                # Genera il modello
                print(f"🧠 Training model for {league.code}...")
                predictor = ExactSimpleFootballPredictor()
                print(f'[INFO] Context scoring attivo = {predictor.use_context_scoring} ({predictor.model_version})')
                
                # Carica i dati e "allena" il modello
                df = predictor.load_data(db, league.code)
                training_data_size = len(df)
                
                # Salva il modello su disco
                print(f"💾 Saving model for {league.code}...")
                model_version = getattr(predictor, 'model_version', 'EXACT_REPLICA')
                metadata = {
                    'training_matches': training_data_size,
                    'model_version': model_version,
                    'league_name': league.name
                }
                
                save_success = model_storage.save_model(predictor, league.code, metadata)
                
                if save_success:
                    # Carica il modello in memoria per uso immediato
                    loaded_models[league.code] = predictor
                    generated_count += 1
                    print(f"✅ Model {league.code} trained and saved successfully")
                    
                    results.append(LeagueModelStatus(
                        league_code=league.code,
                        league_name=league.name,
                        is_trained=True,
                        is_saved=True,
                        is_fresh=True,
                        training_data_size=training_data_size,
                        last_trained=datetime.now().isoformat()
                    ))
                else:
                    failed_count += 1
                    print(f"❌ Failed to save model {league.code}")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ Error processing league {league.code}: {e}")
                results.append(LeagueModelStatus(
                    league_code=league.code,
                    league_name=league.name,
                    is_trained=False,
                    is_saved=False,
                    is_fresh=False,
                    training_data_size=0
                ))
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        return ModelGenerationResponse(
            success=failed_count == 0,
            total_leagues=len(leagues),
            generated_models=generated_count,
            skipped_models=skipped_count,
            failed_models=failed_count,
            results=results,
            generation_time_seconds=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-model/{league_code}")
async def load_model_from_disk(league_code: str):
    """Carica un modello specifico da disco in memoria"""
    
    try:
        # Check se già caricato
        if league_code in loaded_models:
            return {
                "success": True,
                "message": f"Model {league_code} already loaded in memory",
                "loaded_from": "memory"
            }
        
        # Carica da disco
        predictor = model_storage.load_model(league_code)
        
        if predictor is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No saved model found for league {league_code}"
            )
        
        # Carica in memoria
        loaded_models[league_code] = predictor
        
        metadata = model_storage.get_model_metadata(league_code)
        
        return {
            "success": True,
            "message": f"Model {league_code} loaded successfully from disk",
            "loaded_from": "disk",
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-all-models")
async def load_all_models_from_disk():
    """Carica tutti i modelli salvati da disco in memoria"""
    
    try:
        saved_models = model_storage.list_saved_models()
        
        loaded_count = 0
        failed_count = 0
        results = {}
        
        for league_code in saved_models.keys():
            try:
                if league_code not in loaded_models:
                    predictor = model_storage.load_model(league_code)
                    if predictor:
                        loaded_models[league_code] = predictor
                        loaded_count += 1
                        results[league_code] = "loaded"
                    else:
                        failed_count += 1
                        results[league_code] = "failed"
                else:
                    results[league_code] = "already_loaded"
            except Exception as e:
                failed_count += 1
                results[league_code] = f"error: {e}"
        
        return {
            "success": failed_count == 0,
            "total_models": len(saved_models),
            "loaded_models": loaded_count,
            "failed_models": failed_count,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loaded-models")
async def get_loaded_models():
    """Ottieni la lista dei modelli attualmente caricati in memoria"""
    
    models_info = {}
    for league_code, predictor in loaded_models.items():
        models_info[league_code] = {
            "model_class": predictor.__class__.__name__,
            "data_loaded": hasattr(predictor, 'data') and predictor.data is not None,
            "training_data_size": len(predictor.data) if hasattr(predictor, 'data') and predictor.data is not None else 0
        }
    
    return {
        "loaded_models_count": len(loaded_models),
        "models": models_info
    }

def get_model_for_league(league_code: str) -> ExactSimpleFootballPredictor:
    """Ottieni un modello per una lega, caricandolo se necessario"""
    
    # Se già in memoria, restituiscilo
    if league_code in loaded_models:
        return loaded_models[league_code]
    
    # Prova a caricare da disco
    predictor = model_storage.load_model(league_code)
    if predictor:
        loaded_models[league_code] = predictor
        print(f"✅ Model {league_code} auto-loaded from disk")
        return predictor
    
    # Se non esiste, solleva errore
    raise HTTPException(
        status_code=404,
        detail=f"No model available for league {league_code}. Generate it first using /generate-all-models"
    )