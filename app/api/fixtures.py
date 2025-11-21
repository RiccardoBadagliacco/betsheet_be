"""
API semplice per scaricare il palinsesto e salvarlo nel database
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import delete
from fastapi import Query
import pandas as pd
from datetime import datetime
from app.db.database_football import get_football_db
from app.db.models_football import Fixture, Team
import logging
import json
import os
from app.constants.leagues import get_league_name_from_code
import logging
import pandas as pd
from app.services.fixtures_download_service import download_fixtures_pair
from app.services.fixtures_cleaner import clean_fixtures_data, process_additional_csv_with_db
from app.services.fixtures_db_importer import import_fixtures_dataframe
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)
router = APIRouter()

import json
from datetime import datetime
from typing import Dict
import os
import pandas as pd
from sqlalchemy.orm import Session
from app.db.models_football import Fixture,Season
from uuid import UUID
from sqlalchemy.orm import joinedload

# URL fisso del CSV fixtures
FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTURES_CSV_OTHER_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"


async def generate_all_predictions_and_save(save: bool = True):
    """
    ✅ Usa la nuova logica OverModelV1 con patch7 e soglie
    """
    try:
        db = next(get_football_db())
        try:
            logger.info("🔄 Generazione previsioni Over...")

            # ✅ chiamiamo direttamente la nuova API
            result = None

            result["generated_at"] = datetime.utcnow().isoformat()
            result["generated_by"] = "cron_fixtures_over"

            os.makedirs("data", exist_ok=True)
            json_file = "data/all_predictions.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ Predizioni Over salvate in {json_file}")

            return {
                "success": True,
                "fixtures": result.get("fixtures_count"),
                "json_file": json_file
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"❌ Errore generazione predizioni: {e}")
        return {"success": False, "error": str(e)}




def organize_predictions_by_hierarchy(fixtures: list) -> dict:
    """
    Organizza le fixture gerarchicamente: data → paese → lega → fixture
    
    Args:
        fixtures: Lista delle fixture con predizioni
        
    Returns:
        Dizionario organizzato gerarchicamente
    """
    organized = {}
    
    # Ottieni mapping league_code -> country_code dal DB
    from app.db.database_football import get_football_db
    db = next(get_football_db())
    league_to_country = {}
    try:
        from app.db.models_football import League, Country
        leagues = db.query(League).all()
        for l in leagues:
            if l.country and l.country.code:
                league_to_country[l.code] = l.country.code
    except Exception:
        pass
    finally:
        try:
            db.close()
        except Exception:
            pass

    for fixture in fixtures:
        # Skip fixtures without recommendations or with empty recommendations
        recs = fixture.get('recommendations')
        if not recs or (isinstance(recs, (list, dict)) and len(recs) == 0):
            continue
        # Normalizza la data della fixture al formato YYYY-MM-DD
        raw_md = fixture.get('match_date', None)
        match_date = 'Unknown'
        try:
            if raw_md is not None and str(raw_md).strip() != '':
                    # Parse with month-first interpretation (dayfirst=False) to
                    # ensure ambiguous dates like '11/01/2025' become 2025-11-01
                    dt = pd.to_datetime(raw_md, errors='coerce', dayfirst=False, yearfirst=False)
                    if pd.isna(dt):
                        # fallback: try dayfirst parsing if month-first fails
                        dt = pd.to_datetime(raw_md, errors='coerce', dayfirst=True)
                    if not pd.isna(dt):
                        match_date = dt.date().isoformat()
        except Exception:
            try:
                # fallback: prendi i primi 10 caratteri se sembra una ISO string
                s = str(raw_md)
                if len(s) >= 10:
                    match_date = s[:10]
            except Exception:
                match_date = 'Unknown'
        league_code = fixture.get('league_code', 'Unknown')
        country_code = league_to_country.get(league_code, 'UNK')
        league_name = get_league_name_from_code(league_code)

        if match_date not in organized:
            organized[match_date] = {}
        if country_code not in organized[match_date]:
            organized[match_date][country_code] = {}
        if league_code not in organized[match_date][country_code]:
            organized[match_date][country_code][league_code] = {
                "league_name": league_name,
                "country": country_code,
                "fixtures": []
            }

        # Normalizza e assicura che la fixture contenga anche l'ora (match_time) nel formato HH:MM
        fixture_out = dict(fixture) if isinstance(fixture, dict) else {}
        # Assicura che il campo match_date nella fixture abbia lo stesso formato YYYY-MM-DD
        fixture_out['match_date'] = match_date if match_date != 'Unknown' else None
        mt = fixture.get('match_time') or fixture.get('time') or None
        if mt is None:
            # prova a estrarre l'ora da match_date se è un datetime
            md = fixture.get('match_date')
            try:
                import pandas as _pd
                if md is not None and not _pd.isna(md):
                    parsed = _pd.to_datetime(md)
                    if parsed and (parsed.hour or parsed.minute):
                        fixture_out['match_time'] = parsed.strftime('%H:%M')
                    else:
                        fixture_out['match_time'] = None
                else:
                    fixture_out['match_time'] = None
            except Exception:
                fixture_out['match_time'] = None
        else:
            # Normalizza stringhe "HH:MM[:SS]" o datetime-like
            try:
                from datetime import datetime as _dt
                if isinstance(mt, str):
                    mt_str = mt.strip()
                    try:
                        parsed_time = _dt.strptime(mt_str, '%H:%M:%S')
                        fixture_out['match_time'] = parsed_time.strftime('%H:%M')
                    except Exception:
                        try:
                            parsed_time = _dt.strptime(mt_str, '%H:%M')
                            fixture_out['match_time'] = parsed_time.strftime('%H:%M')
                        except Exception:
                            # leave as-is trimmed
                            fixture_out['match_time'] = mt_str
                else:
                    # datetime/time-like
                    try:
                        fixture_out['match_time'] = mt.strftime('%H:%M')
                    except Exception:
                        fixture_out['match_time'] = str(mt)
            except Exception:
                fixture_out['match_time'] = str(mt)

        organized[match_date][country_code][league_code]["fixtures"].append(fixture_out)
    
    # Ordina le date
    sorted_organized = {}
    for date_key in sorted(organized.keys()):
        sorted_organized[date_key] = {}

        # Ordina i country_code alfabeticamente
        for country_code in sorted(organized[date_key].keys()):
            sorted_organized[date_key][country_code] = {}

            # Ordina le leghe alfabeticamente
            for league_key in sorted(organized[date_key][country_code].keys()):
                league_data = organized[date_key][country_code][league_key]

                # Ordina le fixture per orario se disponibile (gestisce None values)
                fixtures_sorted = sorted(
                    league_data["fixtures"],
                    key=lambda x: (x.get('match_time') or '00:00', x.get('home_team') or '')
                )

                sorted_organized[date_key][country_code][league_key] = {
                    "league_name": league_data["league_name"],
                    "country": league_data["country"],
                    "fixture_count": len(fixtures_sorted),
                    "fixtures": fixtures_sorted
                }
    return sorted_organized


def normalize_team_name(team_name: str) -> str:
    """
    Normalizza il nome della squadra per il matching
    
    Args:
        team_name: Nome originale della squadra
        
    Returns:
        Nome normalizzato
    """
    if not team_name or pd.isna(team_name):
        return ""
    
    # Converti in stringa e rimuovi spazi extra
    name = str(team_name).strip()
    
    # Converti in minuscolo per matching case-insensitive
    name = name.lower()
    
    # Rimuovi caratteri speciali comuni
    replacements = {
        'fc': '',
        'f.c.': '',
        'ac': '',
        'a.c.': '',
        'sc': '',
        's.c.': '',
        'cf': '',
        'c.f.': '',
        'united': 'utd',
        'city': '',
        'town': '',
        'rovers': '',
        'county': '',
        'albion': '',
        '.': '',
        '-': ' ',
        '_': ' ',
    }
    
    # Applica sostituzioni
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Rimuovi spazi multipli e strip
    name = ' '.join(name.split())
    
    return name


def find_or_create_team(db: Session, team_name: str, country_id: str = None) -> Team:
    """
    Trova una squadra esistente o ne crea una nuova
    
    Args:
        db: Sessione database
        team_name: Nome della squadra
        country_id: ID del paese (opzionale)
        
    Returns:
        Oggetto Team
    """
    if not team_name or pd.isna(team_name):
        return None
    
    # Normalizza il nome per il matching
    normalized_name = normalize_team_name(team_name)
    
    # Cerca squadra esistente con nome normalizzato
    existing_team = db.query(Team).filter(
        Team.normalized_name == normalized_name
    ).first()
    
    if existing_team:
        logger.debug(f"Team trovato: {existing_team.name} (ID: {existing_team.id})")
        return existing_team
    
    # Crea nuova squadra
    new_team = Team(
        name=str(team_name).strip(),
        normalized_name=normalized_name,
        country_id=country_id
    )
    
    db.add(new_team)
    db.flush()  # Per ottenere l'ID senza commit
    
    logger.info(f"Team creato: {new_team.name} (ID: {new_team.id})")
    return new_team


@router.post("/fixtures/download")
async def download_fixtures(db: Session = Depends(get_football_db)):
    try:
        # 1️⃣ Scarica entrambi i CSV in parallelo
        df_main, df_other = await download_fixtures_pair()

        # 2️⃣ Pulisce e unisce i dati
        cleaned_main = clean_fixtures_data(df_main)
        cleaned_other = process_additional_csv_with_db(df_other, db)
        combined = pd.concat([cleaned_main, cleaned_other], ignore_index=True)
        logger.info("📊 Combinato: %d righe totali", len(combined))

        if combined.empty:
            return JSONResponse({
                "success": False,
                "message": "Nessuna fixture valida trovata"
            })

        # 3️⃣ Cancella le fixtures esistenti
        db.execute(delete(Fixture))
        db.commit()
        logger.info("🧹 Fixtures esistenti cancellate")

        # 4️⃣ Importa nuove fixtures in batch
        result = import_fixtures_dataframe(combined, db)
        logger.info(f"💾 Fixtures importate: {result['saved']} salvate, {result['skipped']} saltate")

        # 5️⃣ Genera raccomandazioni (attende completamento)
        logger.info("🧠 Avvio generazione raccomandazioni...")
        rec_result = None
        logger.info("✅ Raccomandazioni completate")

        # 6️⃣ Risposta finale
        return JSONResponse({
            "success": True,
            "fixtures_saved": result["saved"],
            "fixtures_skipped": result["skipped"],
            "recommendations": rec_result,
            "message": "Fixtures e raccomandazioni generate con successo"
        })

    except Exception as e:
        logger.exception("❌ Errore /fixtures/download")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions")
async def get_all_predictions():
    """
    Ritorna tutte le predizioni organizzate gerarchicamente per data → paese → lega → fixture
    """
    try:
        json_file = "data/all_predictions.json"
        
        # Verifica se il file esiste
        if not os.path.exists(json_file):
            return {
                "success": False,
                "message": "Nessuna predizione trovata. Esegui prima il download delle fixture.",
                "predictions": None,
                "file_exists": False
            }
        
        # Leggi il file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        
        # Organizza i dati gerarchicamente: data → paese → lega → fixture
        organized_data = organize_predictions_by_hierarchy(predictions_data.get('fixtures', []))
        
        # Aggiungi informazioni sul file e metadata
        file_stats = os.stat(json_file)
        
        return {
            "success": True,
            "predictions_by_date": organized_data,
            "metadata": {
                "total_fixtures": predictions_data.get('total_fixtures', 0),
                "successful_predictions": predictions_data.get('successful_predictions', 0),
                "total_recommendations": predictions_data.get('total_recommendations', 0),
                "generated_at": predictions_data.get('generated_at'),
                "generated_by": predictions_data.get('generated_by'),
                "model_info": predictions_data.get('model_info', {}),
                "file_info": {
                    "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "file_size_kb": round(file_stats.st_size / 1024, 2)
                }
            }
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel parsing del JSON: {e}")
        raise HTTPException(status_code=500, detail="File delle predizioni corrotto")
    except Exception as e:
        logger.error(f"Errore nel recupero predizioni: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.delete("/fixtures")
async def clear_fixtures():
    """
    Cancella tutte le fixtures dal database
    """
    try:
        db: Session = next(get_football_db())
        
        try:
            result = db.execute(delete(Fixture))
            db.commit()
            
            return {
                "success": True,
                "message": f"Cancellate {result.rowcount} fixtures",
                "deleted_count": result.rowcount
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Errore nella cancellazione fixtures: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")
    


import pandas as pd

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session, joinedload


@router.get("/{fixture_id}")
async def analyze_fixture(
    fixture_id: str,
    top_n: int = Query(80, ge=10, le=200),
    min_neighbors: int = Query(20, ge=5, le=100),
    db: Session = Depends(get_football_db),
):
    """
    Analizza una fixture futura e restituisce probabilità basate sugli Affini Soft.
    """

    try:
        fixture_uuid = UUID(fixture_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="fixture_id must be a valid UUID")

    # 1) Carico fixture
    f: Fixture | None = (
        db.query(Fixture)
        .options(
            joinedload(Fixture.season).joinedload(Season.league),
            joinedload(Fixture.home_team),
            joinedload(Fixture.away_team),
        )
        .filter(Fixture.id == fixture_uuid)
        .first()
    )
    if f is None:
        raise HTTPException(status_code=404, detail="Fixture non trovata")

    league = f.season.league if f.season else None

    meta = {
        "fixture_id": f.id,
        "home_team": f.home_team.name,
        "away_team": f.away_team.name,
        "league_name": league.name if league else None,
        "league_code": league.code if league else None,
        "match_date": f.match_date.isoformat() if f.match_date else None,
        "match_time": str(f.match_time),
        "odds": f.odds or {},
    }



    # 9) Risposta finale API
    return {
        "success": True,
        "meta": meta,
    }
