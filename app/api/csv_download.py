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
from app.constants.leagues import LEAGUES, get_formatted_league_name
from app.db.database import get_db
from app.db.database_football import get_football_db
from app.services.football_data_service import FootballDataService

# Configura il logger
logger = logging.getLogger(__name__)

# Colonne da mantenere nel CSV finale (in ordine di preferenza)
REQUIRED_COLUMNS = [
    'Date', 'Time', 'HomeTeam', 'AwayTeam', 'HTHG', 'HTAG', 'FTHG', 'FTAG',
    'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5', 'HS', 'AS', 'HST', 'AST'
]

# Mapping per colonne con nomi alternativi (formato: nome_target: [possibili_nomi_sorgente])
COLUMN_MAPPINGS = {
    'AvgH': ['AvgH', 'BbAvH'],
    'AvgD': ['AvgD', 'BbAvD'], 
    'AvgA': ['AvgA', 'BbAvA'],
    'Avg>2.5': ['Avg>2.5', 'BbAv>2.5'],
    'Avg<2.5': ['Avg<2.5', 'BbAv<2.5']
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

async def download_single_csv(league_code: str, season: str, db: Session = None, skip_completed: bool = True) -> Dict[str, Any]:
    """
    Download a single CSV file.
    
    Returns:
        Dict with success status and details
    """
    league_upper = league_code.upper()
    
    # Check if season is already completed and should be skipped
    if skip_completed and db:
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
                season_display = f"20{season[:2]}/20{season[2:]}" if len(season) == 4 else season
                return {
                    "success": True,
                    "skipped": True,
                    "league_code": league_upper,
                    "season": season,
                    "season_display": season_display,
                    "message": f"Season {season_display} already completed",
                    "file_path": None
                }
    
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
    
    # Build download URL
    base_url = "https://www.football-data.co.uk/mmz4281"
    download_url = f"{base_url}/{season}/{league_upper}.csv"
    
    try:
        logger.debug(f"🌐 Fetching: {download_url}")
        
        # Download the CSV file
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Filtra le colonne del CSV
        logger.info(f"📝 Filtering CSV columns for {league_upper}/{season}")
        original_size = len(response.content)
        filtered_content = filter_csv_columns(response.content)
        filtered_size = len(filtered_content)
        
        logger.debug(f"📊 Size reduction: {original_size} → {filtered_size} bytes ({((original_size-filtered_size)/original_size*100):.1f}% smaller)")
        
        # Create league directory
        league_dir = create_league_directory(league_upper)
        
        # Create filename with year format
        year_format = season_code_to_years(season)
        filename = f"{year_format}.csv"
        file_path = league_dir / filename
        
        # Save the filtered file
        with open(file_path, 'wb') as f:
            f.write(filtered_content)
        
        # Get file size for response (filtered file)
        file_size = filtered_size
        
        logger.debug(f"💾 Saved {league_upper}/{season}: {filename} ({file_size} bytes, filtered)")
        
        # Popola il database strutturato se sessione DB fornita
        database_result = None
        if db is not None:
            try:
                logger.info(f"🗃️  Populating database for {league_upper}/{season}...")
                service = FootballDataService(db)
                database_result = service.process_csv_to_database(str(file_path), league_upper, season)
                logger.info(f"✅ Database populated: {database_result.get('matches_processed', 0)} matches")
            except Exception as e:
                logger.error(f"❌ Database population failed for {league_upper}/{season}: {str(e)}")
                database_result = {"success": False, "error": str(e)}
        
        result = {
            "success": True,
            "league_code": league_upper,
            "league_name": get_formatted_league_name(league_upper),
            "season": season,
            "season_years": year_format.replace("_", "/"),
            "filename": filename,
            "file_size_bytes": file_size,
            "saved_to": str(file_path)
        }
        
        # Aggiungi info database se disponibili
        if database_result:
            result["database"] = database_result
        
        return result
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"⚠️  File not found: {league_upper}/{season} (404)")
            return {
                "success": False,
                "league_code": league_upper,
                "season": season,
                "error": f"CSV not found for season {season} and league {league_upper}"
            }
        else:
            logger.error(f"❌ HTTP error {e.response.status_code}: {league_upper}/{season}")
            return {
                "success": False,
                "league_code": league_upper,
                "season": season,
                "error": f"HTTP error: {e.response.status_code}"
            }
    
    except Exception as e:
        logger.error(f"❌ Unexpected error for {league_upper}/{season}: {str(e)}")
        return {
            "success": False,
            "league_code": league_upper,
            "season": season,
            "error": str(e)
        }

@router.get("/download-csv", tags=["CSV Download"])
async def download_football_csv(
    season: str = Query(..., description="Season in format YYZZ (e.g., '2324' for 2023/2024)"),
    league: str = Query(..., description="League code (e.g., 'I1' for Serie A)"),
    populate_db: bool = Query(True, description="Whether to populate structured database (default: True)"),
    skip_completed: bool = Query(True, description="Skip seasons that are already completed in database (default: True)"),
    db: Session = Depends(get_football_db)
):
    """
    Download CSV data from football-data.co.uk for specified season and league.
    
    Creates organized folder structure: leagues/{league_code}/{season_years}.csv
    
    Examples:
    - season='2324', league='I1' -> downloads to leagues/I1/2023_2024.csv
    - season='2223', league='E0' -> downloads to leagues/E0/2022_2023.csv
    """
    
    # Validate season format
    is_valid, error_msg = validate_season_format(season)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate league code (optional - we allow any code but warn if unknown)
    league_upper = league.upper()
    league_name = get_formatted_league_name(league_upper)
    
    # Check if season is already completed and should be skipped
    if skip_completed and populate_db:
        from app.db.models_football import Season, League
        from sqlalchemy import and_
        
        # Check if season exists and is completed
        league_obj = db.query(League).filter(League.code == league.upper()).first()
        if league_obj:
            season_obj = db.query(Season).filter(
                and_(Season.league_id == league_obj.id, Season.code == season)
            ).first()
            
            if season_obj and season_obj.is_completed:
                logger.info(f"⏭️  Skipping {league_upper}/{season} - already completed")
                season_display = f"20{season[:2]}/20{season[2:]}" if len(season) == 4 else season
                return JSONResponse(content={
                    "status": "skipped",
                    "message": f"Season {season_display} already completed in database",
                    "league": league_name,
                    "season": season_display,
                    "file_path": None,
                    "is_completed": True
                })

    # Build download URL
    base_url = "https://www.football-data.co.uk/mmz4281"
    download_url = f"{base_url}/{season}/{league_upper}.csv"
    
    try:
        # Download the CSV file
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Filtra le colonne del CSV
        logger.info(f"📝 Filtering CSV columns for {league_upper}/{season}")
        filtered_content = filter_csv_columns(response.content)
        
        # Create league directory
        league_dir = create_league_directory(league_upper)
        
        # Create filename with year format
        year_format = season_code_to_years(season)
        filename = f"{year_format}.csv"
        file_path = league_dir / filename
        
        # Save the filtered file
        with open(file_path, 'wb') as f:
            f.write(filtered_content)
        
        # Get file size for response (filtered file)
        file_size = len(filtered_content)
        
        # Popola il database strutturato se richiesto
        database_result = None
        if populate_db:
            try:
                logger.info(f"🗃️  Populating database for {league_upper}/{season}...")
                service = FootballDataService(db)
                database_result = service.process_csv_to_database(str(file_path), league_upper, season)
                logger.info(f"✅ Database populated: {database_result.get('matches_processed', 0)} matches")
            except Exception as e:
                logger.error(f"❌ Database population failed for {league_upper}/{season}: {str(e)}")
                database_result = {"success": False, "error": str(e)}
        
        response_data = {
            "success": True,
            "message": "CSV downloaded successfully",
            "details": {
                "source_url": download_url,
                "saved_to": str(file_path),
                "league_code": league_upper,
                "league_name": league_name,
                "season": season,
                "season_years": year_format.replace("_", "/"),
                "file_size_bytes": file_size,
                "filename": filename
            }
        }
        
        # Aggiungi info database se disponibili
        if database_result:
            response_data["database"] = database_result
        
        return JSONResponse(response_data)
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout - server took too long to respond")
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Connection error - unable to reach football-data.co.uk")
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404, 
                detail=f"CSV not found for season {season} and league {league_upper}. Check if the combination exists on football-data.co.uk"
            )
        else:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"HTTP error from source: {e.response.status_code}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/download-multiple-seasons", tags=["CSV Download"])
async def download_multiple_seasons_for_league(
    league: str = Query(..., description="League code (e.g., 'I1' for Serie A)"),
    seasons: Optional[int] = Query(6, description="Number of recent seasons to download (default: 6)"),
    custom_seasons: Optional[str] = Query(None, description="Comma-separated list of specific seasons (e.g., '2324,2223,2122')"),
    populate_db: bool = Query(True, description="Whether to populate structured database (default: True)"),
    skip_completed: bool = Query(True, description="Skip seasons that are already completed in database (default: True)"),
    db: Session = Depends(get_football_db)
):
    """
    Download multiple seasons for a specific league.
    
    You can either:
    - Use 'seasons' to download the N most recent seasons automatically
    - Use 'custom_seasons' to specify exact seasons (comma-separated)
    
    Examples:
    - league='I1', seasons=3 -> downloads last 3 seasons for Serie A
    - league='E0', custom_seasons='2324,2223,2122' -> downloads specific seasons for Premier League
    """
    league_upper = league.upper()
    
    # Validate league code exists in our constants
    if league_upper not in LEAGUES:
        return JSONResponse({
            "success": False,
            "error": f"Unknown league code: {league_upper}",
            "available_leagues": list(LEAGUES.keys())
        })
    
    # Determine which seasons to download
    if custom_seasons:
        seasons_list = [s.strip() for s in custom_seasons.split(',')]
        # Validate each season
        for season in seasons_list:
            is_valid, error_msg = validate_season_format(season)
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid season '{season}': {error_msg}")
    else:
        seasons_list = generate_recent_seasons(seasons)
    
    league_info = LEAGUES[league_upper]
    
    # Start downloads
    results = []
    total_downloads = len(seasons_list)
    successful_downloads = 0
    
    for season in seasons_list:
        db_session = db if populate_db else None
        result = await download_single_csv(league_upper, season, db_session, skip_completed)
        results.append(result)
        
        if result["success"]:
            successful_downloads += 1
        
        # Small delay between downloads
        await asyncio.sleep(0.5)
    
    return {
        "success": True,
        "league_code": league_upper,
        "league_name": f"{league_info['name']} ({league_info['country']})",
        "total_requested": total_downloads,
        "successful_downloads": successful_downloads,
        "failed_downloads": total_downloads - successful_downloads,
        "success_rate": round(successful_downloads / total_downloads * 100, 1) if total_downloads > 0 else 0,
        "results": results,
        "summary": {
            "seasons_downloaded": [r["season"] for r in results if r["success"]],
            "seasons_failed": [r["season"] for r in results if not r["success"]]
        }
    }

@router.post("/download-all-recent", tags=["CSV Download"])
async def download_recent_seasons_all_leagues(
    seasons: Optional[int] = Query(6, description="Number of recent seasons to download for each league (default: 6)"),
    league_filter: Optional[str] = Query(None, description="Comma-separated list of specific leagues to include (e.g., 'I1,E0,D1')"),
    populate_db: bool = Query(True, description="Whether to populate structured database (default: True)"),
    skip_completed: bool = Query(True, description="Skip seasons that are already completed in database (default: True)"),
    db: Session = Depends(get_football_db)
):
    """
    Download recent seasons for all leagues (or filtered leagues).
    
    This endpoint downloads the last N seasons for multiple leagues in parallel.
    
    Examples:
    - seasons=3 -> downloads last 3 seasons for all leagues
    - seasons=6, league_filter='I1,E0,D1' -> downloads 6 seasons for Serie A, Premier League, and Bundesliga
    """
    
    start_time = datetime.now()
    logger.info(f"🚀 Starting bulk download - seasons: {seasons}, league_filter: {league_filter}")
    
    # Determine which leagues to process
    if league_filter:
        requested_leagues = [l.strip().upper() for l in league_filter.split(',')]
        # Validate leagues
        invalid_leagues = [l for l in requested_leagues if l not in LEAGUES]
        if invalid_leagues:
            logger.error(f"❌ Invalid league codes requested: {invalid_leagues}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid league codes: {', '.join(invalid_leagues)}. Available: {', '.join(LEAGUES.keys())}"
            )
        leagues_to_process = requested_leagues
        logger.info(f"📋 Processing filtered leagues: {leagues_to_process}")
    else:
        leagues_to_process = list(LEAGUES.keys())
        logger.info(f"📋 Processing all {len(leagues_to_process)} leagues")
    
    # Generate seasons
    seasons_list = generate_recent_seasons(seasons)
    logger.info(f"📅 Seasons to download: {seasons_list}")
    
    # Prepare summary
    total_downloads = len(leagues_to_process) * len(seasons_list)
    league_results = []
    overall_successful = 0
    overall_failed = 0
    
    logger.info(f"📊 Total downloads planned: {total_downloads} ({len(leagues_to_process)} leagues × {len(seasons_list)} seasons)")
    
    # Process each league
    for idx, league_code in enumerate(sorted(leagues_to_process), 1):
        league_info = LEAGUES[league_code]
        league_name = f"{league_info['name']} ({league_info['country']})"
        
        logger.info(f"📁 [{idx}/{len(leagues_to_process)}] Starting {league_code}: {league_name}")
        
        league_successful = 0
        league_failed = 0
        league_downloads = []
        
        # Download all seasons for this league
        for season_idx, season in enumerate(seasons_list, 1):
            season_years = season_code_to_years(season).replace("_", "/")
            logger.info(f"   📥 [{season_idx}/{len(seasons_list)}] Downloading {league_code} season {season_years}...")
            
            db_session = db if populate_db else None
            result = await download_single_csv(league_code, season, db_session, skip_completed)
            league_downloads.append(result)
            
            if result["success"]:
                league_successful += 1
                overall_successful += 1
                
                if result.get("skipped"):
                    logger.info(f"   ⏭️  Skipped: {result['message']}")
                else:
                    file_size_mb = result["file_size_bytes"] / 1024 / 1024
                    logger.info(f"   ✅ Success: {result['filename']} ({file_size_mb:.1f}MB)")
            else:
                league_failed += 1
                overall_failed += 1
                logger.warning(f"   ❌ Failed: {result['error']}")
            
            # Small delay between downloads
            await asyncio.sleep(0.3)
        
        league_success_rate = round(league_successful / len(seasons_list) * 100, 1)
        logger.info(f"📈 {league_code} completed: {league_successful}/{len(seasons_list)} successes ({league_success_rate}%)")
        
        league_results.append({
            "league_code": league_code,
            "league_name": league_name,
            "successful": league_successful,
            "failed": league_failed,
            "success_rate": league_success_rate,
            "downloads": league_downloads
        })
    
    # Calculate final statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    overall_success_rate = round(overall_successful / total_downloads * 100, 1) if total_downloads > 0 else 0
    
    # Final summary logs
    logger.info(f"🏁 Bulk download completed!")
    logger.info(f"⏱️  Duration: {duration:.1f} seconds")
    logger.info(f"📊 Results: {overall_successful}/{total_downloads} successes ({overall_success_rate}%)")
    logger.info(f"📈 Success rate by league:")
    
    for result in sorted(league_results, key=lambda x: x["success_rate"], reverse=True):
        status_emoji = "✅" if result["success_rate"] == 100.0 else "⚠️" if result["success_rate"] >= 50.0 else "❌"
        logger.info(f"   {status_emoji} {result['league_code']}: {result['success_rate']}% ({result['successful']}/{result['successful']+result['failed']})")
    
    if overall_failed > 0:
        logger.warning(f"⚠️  {overall_failed} downloads failed - check individual league results for details")
    
    # Calculate statistics
    most_successful = sorted(league_results, key=lambda x: x["success_rate"], reverse=True)[:5]
    leagues_with_failures = [r for r in league_results if r["failed"] > 0]
    
    logger.info(f"💾 Files saved to: ./leagues/ directory structure")
    
    return {
        "success": True,
        "execution_time_seconds": round(duration, 1),
        "summary": {
            "leagues_processed": len(leagues_to_process),
            "seasons_per_league": len(seasons_list),
            "total_downloads_attempted": total_downloads,
            "overall_successful": overall_successful,
            "overall_failed": overall_failed,
            "overall_success_rate": overall_success_rate,
            "seasons_requested": seasons_list,
            "started_at": start_time.isoformat(),
            "completed_at": end_time.isoformat()
        },
        "league_results": league_results,
        "statistics": {
            "most_successful_leagues": [
                {"league": r["league_code"], "success_rate": r["success_rate"]} 
                for r in most_successful
            ],
            "leagues_with_failures": [
                {"league": r["league_code"], "failed_count": r["failed"]} 
                for r in leagues_with_failures
            ]
        }
    }
    """
    Get comprehensive help and examples for all CSV download endpoints.
    """
    return {
        "csv_download_api": {
            "description": "API for downloading football CSV data from football-data.co.uk",
            "base_url": "/csv",
            "endpoints": {
                "GET /leagues": {
                    "description": "List all supported league codes and names",
                    "example": "curl http://localhost:8000/csv/leagues"
                },
                "GET /download-csv": {
                    "description": "Download a single CSV file for specific league and season",
                    "parameters": {
                        "league": "League code (e.g., 'I1' for Serie A)",
                        "season": "Season in format YYZZ (e.g., '2324' for 2023/2024)"
                    },
                    "example": "curl 'http://localhost:8000/csv/download-csv?league=I1&season=2324'"
                },
                "POST /download-multiple-seasons": {
                    "description": "Download multiple seasons for a specific league",
                    "parameters": {
                        "league": "League code (required)",
                        "seasons": "Number of recent seasons (default: 6)",
                        "custom_seasons": "Comma-separated specific seasons (optional)"
                    },
                    "examples": [
                        "curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=I1&seasons=3'",
                        "curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=E0&custom_seasons=2324,2223,2122'"
                    ]
                },
                "POST /download-all-recent": {
                    "description": "Download recent seasons for multiple leagues",
                    "parameters": {
                        "seasons": "Number of recent seasons per league (default: 6)",
                        "league_filter": "Comma-separated league codes to include (optional)"
                    },
                    "examples": [
                        "curl -X POST 'http://localhost:8000/csv/download-all-recent?seasons=6'",
                        "curl -X POST 'http://localhost:8000/csv/download-all-recent?seasons=3&league_filter=I1,E0,D1'"
                    ]
                },
                "GET /leagues/{league_code}/files": {
                    "description": "List downloaded files for a specific league",
                    "example": "curl http://localhost:8000/csv/leagues/I1/files"
                }
            },
            "common_use_cases": {
                "download_all_recent": {
                    "description": "Download last 6 seasons for all leagues",
                    "command": "curl -X POST 'http://localhost:8000/csv/download-all-recent'"
                },
                "download_top_5_leagues": {
                    "description": "Download recent seasons for top 5 European leagues",
                    "command": "curl -X POST 'http://localhost:8000/csv/download-all-recent?league_filter=E0,D1,I1,SP1,F1'"
                },
                "download_serie_a_history": {
                    "description": "Download last 10 Serie A seasons",
                    "command": "curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=I1&seasons=10'"
                },
                "download_specific_seasons": {
                    "description": "Download specific seasons for Premier League",
                    "command": "curl -X POST 'http://localhost:8000/csv/download-multiple-seasons?league=E0&custom_seasons=2324,2223,2122,2021'"
                }
            },
            "file_organization": {
                "structure": "leagues/{league_code}/{YYYY_YYYY}.csv",
                "examples": [
                    "leagues/I1/2023_2024.csv",
                    "leagues/E0/2022_2023.csv",
                    "leagues/D1/2024_2025.csv"
                ]
            },
            "supported_leagues": {
                "total": len(LEAGUES),
                "top_tier": ["E0", "D1", "I1", "SP1", "F1", "N1", "P1"],
                "all_codes": sorted(LEAGUES.keys())
            }
        }
    }