#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API V2 — PROFETA ENGINE (PURE)

- Usa ESCLUSIVAMENTE step0_profeta.parquet per il modello
- Input: match_id + is_fixture
- Output:
    * intestazione match
    * fotografia statistica ProfetaEngine
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from uuid import UUID
from pathlib import Path
import pandas as pd

from app.db.database_football import get_football_db
from app.db.models import Match, Fixture, Season, League

from app.ml.profeta.step2_profeta import ProfetaEngine

# ============================================================
# PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "app" / "ml" / "profeta" / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"

# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/profeta/v2",
    tags=["profeta-engine"],
)

# ============================================================
# LOAD STEP0 UNA SOLA VOLTA
# ============================================================

try:
    STEP0_DF = pd.read_parquet(STEP0_PATH)
except Exception as e:
    raise RuntimeError(f"Impossibile caricare step0_profeta.parquet: {e}")

STEP0_BY_ID = STEP0_DF.set_index("match_id")

from app.ml.profeta.state_engine import (
    compute_control_state,
    compute_goal_state,
)


from app.ml.profeta.alert_engine import run_alert_engine
# ============================================================
# ENDPOINT
# ============================================================

@router.get("/engine/{match_id}")
def run_profeta_engine(
    match_id: str,
    is_fixture: bool = Query(False),
    db: Session = Depends(get_football_db),
):
    """
    PROFETA ENGINE PURE

    - input modellistico: step0_profeta.parquet
    - output arricchito con intestazione match
    """

    # ----------------------------------
    # VALIDAZIONE UUID
    # ----------------------------------
    try:
        match_uuid = UUID(match_id)
    except ValueError:
        raise HTTPException(400, "match_id non valido")

    # ----------------------------------
    # RECUPERO MATCH / FIXTURE DAL DB
    # (solo per intestazione)
    # ----------------------------------
    if is_fixture:
        m = (
            db.query(Fixture)
            .options(
                joinedload(Fixture.home_team),
                joinedload(Fixture.away_team),
                joinedload(Fixture.season)
                .joinedload(Season.league),
            )
            .filter(Fixture.id == match_uuid)
            .first()
        )
        if not m:
            raise HTTPException(404, "Fixture non trovata")
    else:
        m = (
            db.query(Match)
            .options(
                joinedload(Match.home_team),
                joinedload(Match.away_team),
                joinedload(Match.season)
                .joinedload(Season.league),
            )
            .filter(Match.id == match_uuid)
            .first()
        )
        if not m:
            raise HTTPException(404, "Match non trovato")

    league = m.season.league if m.season else None

    match_header = {
        "match_id": match_id,
        "is_fixture": is_fixture,
        "home_team": m.home_team.name if m.home_team else None,
        "away_team": m.away_team.name if m.away_team else None,
        "match_date": (
            m.match_date.isoformat() if getattr(m, "match_date", None) else None
        ),
        "match_time": str(m.match_time) if getattr(m, "match_time", None) else None,
        "league_code": league.code if league else None,
        "league_name": league.name if league else None,
        "result": (
            {
                "home_ft": m.home_goals_ft,
                "away_ft": m.away_goals_ft,
            }
            if not is_fixture
            else None
        ),
        "odds": {
            "home": m.avg_home_odds if getattr(m, "avg_home_odds", None) else None,
            "draw": m.avg_draw_odds if getattr(m, "avg_draw_odds", None) else None,
            "away": m.avg_away_odds if getattr(m, "avg_away_odds", None) else None,
        },
    }

    # ----------------------------------
    # RECUPERO RIGA STEP0 (INPUT MODELLO)
    # ----------------------------------
    if match_id not in STEP0_BY_ID.index:
        raise HTTPException(
            404,
            "Match non presente in step0_profeta.parquet",
        )

    row = STEP0_BY_ID.loc[match_id]

    if bool(row["is_fixture"]) != bool(is_fixture):
        raise HTTPException(
            400,
            "Mismatch is_fixture tra API e step0_profeta",
        )

    # ----------------------------------
    # RUN PROFETA ENGINE
    # ----------------------------------
    engine = ProfetaEngine(max_goals=10)

    try:
        result = engine.predict_from_row(row)

        control_state = compute_control_state(result["markets"])
        goal_state = compute_goal_state(result["markets"])

        print("DEBUG STATES:", control_state, goal_state)

        alerts = run_alert_engine(
            markets=result["markets"],
            control_state=control_state,
            goal_state=goal_state,
        )
    except Exception as e:
        raise HTTPException(500, f"Errore ProfetaEngine: {e}")

    # ----------------------------------
    # OUTPUT
    # ----------------------------------
    return {
        "success": True,
        "match": match_header,
        "engine": {
            "lambda_home": result["lambda_home"],
            "lambda_away": result["lambda_away"],
            "xg_total": result["markets"]["xg_total"],
            "markets": result["markets"],
            "goal_matrix": result["goal_matrix"].tolist(),
            "alerts": alerts,   # 👈 QUI
        },
    }