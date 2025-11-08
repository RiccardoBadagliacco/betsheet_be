# app/api/csv_download.py
from __future__ import annotations
from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import asyncio
import httpx

from app.db.database_football import get_football_db
from app.constants.leagues import get_all_leagues
from app.services.download_service import generate_recent_seasons, download_main_league, download_other_league
from app.services.football_data_service import FootballDataService
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/download-all-leagues-centralized", tags=["CSV Download"])
async def download_all_leagues_centralized(
    n_seasons: int = Query(6),
    populate_db: bool = Query(True),
    skip_completed: bool = Query(True),
    db: Session = Depends(get_football_db),
):
    start = asyncio.get_running_loop().time()
    logger.info("🚀 Avvio processo centralizzato (n_seasons=%d, populate_db=%s)", n_seasons, populate_db)
    service = FootballDataService(db)
    recents = generate_recent_seasons(n_seasons)
    main = get_all_leagues('main')
    other = get_all_leagues('other')
    tasks = []
    
    print("Starting download of all leagues...")
    print(f"Main leagues: {list(main.keys())}")
    print(f"Other leagues: {list(other.keys())}")
    async with httpx.AsyncClient() as client:
        for code in main.keys():
            tasks.append(download_main_league(client, code, recents, overwrite_incomplete=True))
        for code in other.keys():
            tasks.append(download_other_league(client, code, n_seasons=n_seasons))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out = {"main": [], "other": []}
    idx = 0
    
    logger.info("Download completato, avvio import DB...")

    for code in main.keys():
        res = results[idx]; idx += 1
        if isinstance(res, Exception):
            out["main"].append({"league": code, "success": False, "error": str(res)})
            continue
        for season_code, path in res:
            if skip_completed and service.is_season_completed(code, season_code):
                out["main"].append({"league": code, "season": season_code, "file": path, "skipped": True, "success": True})
                continue
            db_res = service.process_csv_to_database(path, code, season_code) if populate_db else None
            out["main"].append({"league": code, "season": season_code, "file": path, "success": db_res.get("success", True) if db_res else True})
    for code in other.keys():
        res = results[idx]; idx += 1
        if isinstance(res, Exception):
            out["other"].append({"league": code, "success": False, "error": str(res)})
            continue
        for season_key, path in res:
            if populate_db:
                logger.debug("[%s %s] Import nel database...", code, season_code)
                db_res = service.process_csv_to_database(path, code, season_key)
                logger.info("[%s %s] ✅ %d partite importate", code, season_code, db_res.get("matches_processed", 0))

            out["other"].append({"league": code, "season": season_key, "file": path, "success": True})
            
    logger.info("🏁 Processo completato in %.2fs", asyncio.get_running_loop().time() - start)

    return JSONResponse(content={"success": True, "results": out})
