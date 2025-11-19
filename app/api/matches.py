from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session, joinedload
from uuid import UUID
from typing import Dict

from app.db.database_football import get_football_db
from app.db.models import Match, Season
from datetime import datetime

router = APIRouter()

import pandas as pd
from pathlib import Path

# Runtime pipeline


APP_DIR = Path(__file__).resolve().parents[1]
AFFINI_DIR = APP_DIR / "ml" / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

SLIM_PATH = DATA_DIR / "step4b_affini_index_slim_v2.parquet"
WIDE_PATH = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

@router.get("/matches/by_date")
async def get_matches_by_date(
    match_date: str = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_football_db),
):
    """
    Ritorna le partite per una certa data, raggruppate per league.
    """
    if match_date is None:
        match_date = datetime.now().date().isoformat()

    try:
        date_obj = datetime.strptime(match_date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(400, "match_date must be YYYY-MM-DD")

    q = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season).joinedload(Season.league),
        )
        .filter(Match.match_date == date_obj)
    )

    matches = q.all()
    leagues: Dict[str, Dict] = {}

    for m in matches:
        season = m.season
        league = season.league if season else None
        code = getattr(league, "code", "UNKNOWN")

        if code not in leagues:
            leagues[code] = {
                "league_name": getattr(league, "name", code),
                "fixtures": [],
            }

        leagues[code]["fixtures"].append(
            {
                "fixture_id": str(m.id),
                "home_team": m.home_team.name,
                "away_team": m.away_team.name,
                "match_time": m.match_time,
          
                "home_ft": m.home_goals_ft,
                "away_ft": m.away_goals_ft
                
            }
        )

    return {"success": True, "date": match_date, "leagues": leagues}
#


from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from app.ml.correlazioni_affini_v2.common.soft_engine_api_v2 import (
    run_soft_engine_api,
)
from app.ml.correlazioni_affini_v2.common.soft_engine_postprocess import full_postprocess
from app.ml.correlazioni_affini_v2.common.load_affini_indexes import load_affini_indexes
@router.get("/matches/{match_id}")
async def analyze_match(
    match_id: str,
    top_n: int = Query(80, ge=10, le=200),
    min_neighbors: int = Query(30, ge=5, le=100),
    db: Session = Depends(get_football_db),
):
    print("Analyzing match_id:", match_id)

    # --------------------------------------
    # 1) MATCH DAL DB
    # --------------------------------------
    try:
        match_uuid = UUID(match_id)
    except ValueError:
        raise HTTPException(400, "match_id must be a valid UUID")

    m: Match | None = (
        db.query(Match)
        .options(
            joinedload(Match.season).joinedload(Season.league),
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
        .filter(Match.id == match_uuid)
        .first()
    )

    if m is None:
        raise HTTPException(404, "Match non trovato")

    league = m.season.league if m.season else None

    # --------------------------------------
    # 2) META INFO
    # --------------------------------------
    meta = {
        "match_id": str(m.id),
        "home_team": m.home_team.name,
        "away_team": m.away_team.name,
        "league_name": league.name if league else None,
        "league_code": league.code if league else None,
        "match_date": m.match_date.isoformat() if m.match_date else None,
        "match_time": str(m.match_time) if getattr(m, "match_time", None) else None,
        "odds": {
            "avg_home_odds": m.avg_home_odds,
            "avg_draw_odds": m.avg_draw_odds,
            "avg_away_odds": m.avg_away_odds,
            "avg_over_25_odds": m.avg_over_25_odds,
            "avg_under_25_odds": m.avg_under_25_odds,
        },
        "result": {
            "home_ft": m.home_goals_ft,
            "away_ft": m.away_goals_ft
        },
    }

    # --------------------------------------
    # 3) CARICO INDICI AFFINI (SLIM + WIDE)
    # --------------------------------------
    slim_index, wide_index = load_affini_indexes()

    # --------------------------------------
    # 4) SOFT ENGINE (USANDO IL MATCH STORICO)
    # --------------------------------------
    soft_model = run_soft_engine_api(
        target_row=None,                      # per match storico non serve
        target_match_id=str(m.id),            # match da cercare in SLIM
        slim=slim_index,
        wide=wide_index,
        top_n=top_n,
        min_neighbors=min_neighbors,
    )

    if soft_model["status"] != "ok":
        return {
            "success": False,
            "meta": meta,
            "reason": soft_model.get("reason", "soft_engine_failed"),
            "debug": soft_model,
        }

    # --------------------------------------
    # 5) POSTPROCESS (GG/NG, PMF, multigoal)
    # --------------------------------------
    analytics = full_postprocess({
        "meta": meta,
        "clusters": soft_model["clusters"],
        "soft_probs": soft_model["soft_probs"],
        "affini_stats": soft_model["affini_stats"],
        "affini_list": soft_model["affini_list"],   # FONDAMENTALE
    })

    # --------------------------------------
    # 6) OUTPUT FINALE
    # --------------------------------------
    return {
        "success": True,
        "meta": meta,
        "model": {
            "status": "ok",
            "clusters": soft_model["clusters"],
            "soft_probs": soft_model["soft_probs"],
            "affini_stats": soft_model["affini_stats"],
            "config_used": soft_model["config_used"],
        },
        "analytics": analytics,
    }