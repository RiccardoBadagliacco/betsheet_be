from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
import os
import requests
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import logging
import pandas as pd
import io
from sqlalchemy.orm import Session
from app.constants.leagues import get_all_leagues, get_formatted_league_name
from app.db.database import get_db
from app.db.database_football import get_football_db
from app.services.football_data_service import FootballDataService

# Configura il logger
logger = logging.getLogger(__name__)

# Colonne da mantenere nel CSV finale (in ordine di preferenza)
REQUIRED_COLUMNS = [
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'FTHG', 'FTAG','Season',
    'AvgH', 'AvgD', 'AvgA','Avg>2.5', 'Avg<2.5'
]

# Mapping per colonne con nomi alternativi (formato: nome_target: [possibili_nomi_sorgente])
COLUMN_MAPPINGS = {
    'AvgH': ['AvgH', 'BbAvH','AvgCH'],
    'AvgD': ['AvgD', 'BbAvD', 'AvgCD'],
    'AvgA': ['AvgA', 'BbAvA', 'AvgCA'],
    'Avg>2.5': ['Avg>2.5', 'BbAv>2.5'],
    'Avg<2.5': ['Avg<2.5', 'BbAv<2.5'],
    'HomeTeam': ['HomeTeam', 'Home'],
    'AwayTeam': ['AwayTeam', 'Away'],
    'FTHG': ['FTHG', 'HG'],
    'FTAG': ['FTAG', 'AG'],
    'Season': ['Season']
    
}


def filter_csv_columns(csv_content: bytes) -> bytes:
    """
    Filtra il CSV per mantenere solo le colonne richieste, con mapping per nomi alternativi.
    
    Args:
        csv_content: Contenuto CSV originale in bytes
        
    Returns:
        CSV filtrato con solo le colonne necessarie in bytes
    """
    try:
        # Leggi il CSV in un DataFrame
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        
        # Log delle colonne disponibili
        available_columns = df.columns.tolist()
        logger.debug(f"📋 Colonne disponibili nel CSV: {available_columns}")
        
        # Costruisci il DataFrame finale con le colonne richieste
        final_columns = []
        final_data = {}
        
        for required_col in REQUIRED_COLUMNS:
            column_found = False
            
            # Prima prova il nome esatto
            if required_col in available_columns:
                final_columns.append(required_col)
                final_data[required_col] = df[required_col]
                column_found = True
                logger.debug(f"✅ Trovata colonna: {required_col}")
            
            # Se non trovata, prova i mapping alternativi
            elif required_col in COLUMN_MAPPINGS:
                for alt_name in COLUMN_MAPPINGS[required_col]:
                    if alt_name in available_columns:
                        final_columns.append(required_col)
                        final_data[required_col] = df[alt_name]
                        column_found = True
                        logger.debug(f"✅ Mappata colonna: {alt_name} -> {required_col}")
                        break
            
            if not column_found:
                # Per colonne mancanti, aggiungi una colonna vuota per mantenere la struttura
                final_columns.append(required_col)
                final_data[required_col] = pd.Series([''] * len(df), dtype=str)
                logger.warning(f"⚠️  Colonna mancante: {required_col} (aggiunta vuota)")
        
        # Crea il DataFrame finale
        filtered_df = pd.DataFrame(final_data)[final_columns]
        
        logger.info(f"✅ CSV processato: {len(final_columns)} colonne, {len(filtered_df)} righe")
        
        # Converte di nuovo in CSV bytes
        csv_string = filtered_df.to_csv(index=False)
        return csv_string.encode('utf-8')
        
    except Exception as e:
        logger.error(f"❌ Errore nel filtraggio CSV: {str(e)}")
        # In caso di errore, ritorna il contenuto originale
        return csv_content

router = APIRouter()

def validate_season_format(season: str) -> tuple[bool, str]:
    """Validate season format (e.g., '2324' for 2023/2024)."""
    if not re.match(r'^\d{4}$', season):
        return False, "Season must be 4 digits (e.g., '2324' for 2023/2024)"
    
    # Extract years
    year1 = int(season[:2])
    year2 = int(season[2:])
    
    # Validate that second year follows first year
    if year2 != (year1 + 1) % 100:
        return False, "Invalid season format. Second year must follow first year (e.g., '2324' = 2023/2024)"
    
    return True, ""

def season_code_to_years(season: str) -> str:
    """Convert season code to full year format (e.g., '2324' -> '2023_2024')."""
    year1 = int(season[:2])
    year2 = int(season[2:])
    
    # Handle century transition
    if year1 >= 50:  # Assume 1950-1999
        full_year1 = 1900 + year1
    else:  # Assume 2000-2049
        full_year1 = 2000 + year1
    
    full_year2 = full_year1 + 1
    
    return f"{full_year1}_{full_year2}"

def create_league_directory(league_code: str) -> Path:
    """Create and return the directory path for a league."""
    base_path = Path("leagues")
    league_path = base_path / league_code
    league_path.mkdir(parents=True, exist_ok=True)
    return league_path

def generate_recent_seasons(num_seasons: int = 6) -> List[str]:
    """
    Generate the most recent season codes.
    
    Args:
        num_seasons: Number of seasons to generate (default: 6)
        
    Returns:
        List of season codes in format YYZZ (e.g., ['2324', '2223', ...])
    """
    current_year = datetime.now().year
    
    # Se siamo nella seconda metà dell'anno (dopo luglio), 
    # la stagione corrente è già iniziata
    if datetime.now().month >= 7:
        current_season_start = current_year
    else:
        current_season_start = current_year - 1
    
    seasons = []
    for i in range(num_seasons):
        season_start = current_season_start - i
        season_end = season_start + 1
        # Formato YYZZ: prendi le ultime 2 cifre di ogni anno
        season_code = f"{str(season_start)[-2:]}{str(season_end)[-2:]}"
        seasons.append(season_code)
    
    return seasons


def process_csv_file(file_path: str, league_code: str, season: str, db: Session = None) -> Dict[str, Any]:
    """
    Process a CSV file for a specific league and season.

    Args:
        file_path: Path to the CSV file.
        league_code: League code (e.g., 'I1' for Serie A).
        season: Season code (e.g., '2324' for 2023/2024).
        db: Database session for saving data (optional).

    Returns:
        Dict with success status and details.
    """
    league_upper = league_code.upper()

    # Validate season format
    is_valid, error_msg = validate_season_format(season)
    if not is_valid:
        logger.error(f"❌ Invalid season format: {season} - {error_msg}")
        return {
            "success": False,
            "league_code": league_upper,
            "season": season,
            "error": error_msg
        }

    # Add check for completed seasons in process_csv_file
    if db:
        from app.db.models_football import Season, League
        from sqlalchemy import and_

        # Check if season exists and is completed
        league_obj = db.query(League).filter(League.code == league_upper).first()
        if league_obj:
            season_obj = db.query(Season).filter(
                and_(Season.league_id == league_obj.id, Season.code == season)
            ).first()

            if season_obj and season_obj.is_completed:
                logger.info(f"⏭️  Skipping {league_upper}/{season} - already completed")
                return {
                    "success": True,
                    "skipped": True,
                    "league_code": league_upper,
                    "season": season,
                    "message": f"Season {season} already completed",
                    "file_path": None
                }

    try:
        # Filtra le colonne del CSV
        logger.info(f"📝 Filtering CSV columns for {league_upper}/{season}")
        with open(file_path, "rb") as f:
            original_content = f.read()
        original_size = len(original_content)
        filtered_content = filter_csv_columns(original_content)
        filtered_size = len(filtered_content)
        

        logger.debug(f"📊 Size reduction: {original_size} → {filtered_size} bytes ({((original_size-filtered_size)/original_size*100):.1f}% smaller)")

        # Save the filtered file
        with open(file_path, "wb") as f:
            f.write(filtered_content)

        # Get file size for response (filtered file)
        file_size = filtered_size

        logger.debug(f"💾 Processed {league_upper}/{season}: {file_path} ({file_size} bytes, filtered)")

        # Popola il database strutturato se sessione DB fornita
        database_result = None
        if db is not None:
            try:
                logger.info(f"🗃️  Populating database for {league_upper}/{season}...")
                service = FootballDataService(db)
                database_result = service.process_csv_to_database(file_path, league_upper, season)
                logger.info(f"✅ Database populated: {database_result.get('matches_processed', 0)} matches")
            except Exception as e:
                logger.error(f"❌ Database population failed for {league_upper}/{season}: {str(e)}")
                database_result = {"success": False, "error": str(e)}

        result = {
            "success": True,
            "league_code": league_upper,
            "season": season,
            "filename": Path(file_path).name,
            "file_size_bytes": file_size,
            "saved_to": file_path
        }

        # Aggiungi info database se disponibili
        if database_result:
            result["database"] = database_result

        return result

    except Exception as e:
        logger.error(f"❌ Unexpected error for {league_upper}/{season}: {str(e)}")
        return {
            "success": False,
            "league_code": league_upper,
            "season": season,
            "error": str(e)
        }
        
@router.post("/download-all-leagues-centralized", tags=["CSV Download"])
async def download_all_leagues_centralized(
    n_seasons: int = Query(6, description="Numero stagioni recenti da scaricare per le leghe 'main'"),
    populate_db: bool = Query(True, description="Se True, scrive i dati nel database"),
    db: Session = Depends(get_football_db)
):
    """
    🔁 Scarica e popola il database per tutte le leghe ('main' e 'other'),
    salvando i file CSV in leagues/{league_code}/{season}.csv.
    ❗ Se il file è già presente, viene riutilizzato senza riscaricarlo.
    """
    start_time = datetime.now()
    logger.info(f"🚀 Avvio download centralizzato con caching locale (n_seasons={n_seasons})")

    service = FootballDataService(db)
    results = {"main": [], "other": []}
    total_processed = 0
    total_errors = 0

    # === MAIN LEAGUES ===
    main_leagues = get_all_leagues("main")
    recent_seasons = generate_recent_seasons(n_seasons)
    base_url = "https://www.football-data.co.uk/mmz4281"

    for league_code, info in main_leagues.items():
        league_name = f"{info['name']} ({info['country']})"
        logger.info(f"⚽ Processing MAIN league: {league_code} - {league_name}")

        league_dir = create_league_directory(league_code)

        for season in recent_seasons:
            season_filename = f"{season_code_to_years(season)}.csv"
            file_path = league_dir / season_filename

            try:
                if file_path.exists():
                    logger.info(f"⏭️ File già presente: {file_path}")
                else:
                    download_url = f"{base_url}/{season}/{league_code}.csv"
                    response = requests.get(download_url, timeout=20)
                    if response.status_code == 404:
                        logger.warning(f"⚠️ Nessun file trovato per {league_code} stagione {season}")
                        results["main"].append({
                            "league": league_name,
                            "season": season,
                            "success": False,
                            "error": "File CSV non trovato"
                        })
                        continue

                    filtered_csv = filter_csv_columns(response.content)
                    with open(file_path, "wb") as f:
                        f.write(filtered_csv)
                    logger.info(f"💾 Salvato: {file_path}")

                # Processa la stagione (solo se richiesto)
                db_result = None
                if populate_db:
                    db_result = service.process_csv_to_database(str(file_path), league_code, season)

                total_processed += 1
                results["main"].append({
                    "league": league_name,
                    "season": season,
                    "file": str(file_path),
                    "cached": file_path.exists(),
                    "success": db_result.get("success", True) if db_result else True,
                    "matches_processed": db_result.get("matches_processed") if db_result else None,
                    "errors_count": db_result.get("errors_count") if db_result else None
                })

            except Exception as e:
                total_errors += 1
                logger.error(f"❌ Errore su {league_code}/{season}: {str(e)}")
                results["main"].append({
                    "league": league_name,
                    "season": season,
                    "success": False,
                    "error": str(e)
                })

            await asyncio.sleep(0.2)

    # === OTHER LEAGUES ===
    other_leagues = get_all_leagues("other")
    url_template = "https://www.football-data.co.uk/new/{league_code}.csv"

    for league_code, info in other_leagues.items():
        league_name = f"{info['name']} ({info['country']})"
        url = url_template.format(league_code=league_code)
        logger.info(f"🌍 Processing OTHER league: {league_code} - {league_name}")

        league_dir = create_league_directory(league_code)

        try:
            # se abbiamo già file separati per stagione, salta download
            existing_files = list(league_dir.glob("*.csv"))
            if existing_files:
                logger.info(f"⏭️ CSV stagionali già presenti per {league_code}, salto download generale.")
                for f in existing_files:
                    if populate_db:
                        # determina season_name dal nome file
                        season_name = f.stem
                        service.process_csv_to_database(str(f), league_code, season_name)
                continue

            # scarica il CSV unico se non ci sono file salvati
            response = requests.get(url, timeout=25)
            response.raise_for_status()
            filtered_csv = filter_csv_columns(response.content)
            df = pd.read_csv(io.BytesIO(filtered_csv))

            if "Season" not in df.columns:
                logger.warning(f"⚠️ Nessuna colonna 'Season' trovata per {league_code}")
                continue

            df = df.sort_values(by="Season", ascending=True)
            unique_seasons = df["Season"].drop_duplicates().tail(n_seasons)

            for season in unique_seasons:
                season_str = str(season)
                formatted_season = season_str.replace("/", "_")
                file_path = league_dir / f"{formatted_season}.csv"
                
                if file_path.exists():
                    logger.info(f"⏭️ File già presente: {file_path}")
                else:
                    df[df["Season"] == season].to_csv(file_path, index=False)
                    logger.info(f"💾 Salvato: {file_path}")

                if populate_db:
                    service.process_csv_to_database(str(file_path), league_code, formatted_season)

                total_processed += 1
                results["other"].append({
                    "league": league_name,
                    "season": formatted_season,
                    "file": str(file_path),
                    "cached": file_path.exists(),
                    "success": True
                })

                await asyncio.sleep(0.2)

        except Exception as e:
            total_errors += 1
            logger.error(f"❌ Errore su {league_code}: {str(e)}")
            results["other"].append({
                "league": league_name,
                "success": False,
                "error": str(e)
            })

    # === SUMMARY ===
    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 1)

    summary = {
        "success": True,
        "processed_seasons": total_processed,
        "errors": total_errors,
        "duration_seconds": duration,
        "results": results
    }

    logger.info(f"🏁 Download centralizzato completato in {duration}s "
                f"({total_processed} stagioni processate, {total_errors} errori)")

    return JSONResponse(content=summary)
 