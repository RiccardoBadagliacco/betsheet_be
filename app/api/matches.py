from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload, load_only
from typing import Dict
from datetime import datetime
from pathlib import Path
import pandas as pd

from app.db.database_football import get_football_db
from app.db.models import Match, Season, Fixture, League

# ============================
# PROFETA IMPORT
# ============================

from app.ml.profeta.step2_profeta import ProfetaEngine
from app.ml.profeta.state_engine import (
    compute_control_state,
    compute_goal_state,
)
from app.ml.profeta.alert_engine import run_alert_engine

# ============================
# STEP0 LOAD (una sola volta)
# ============================

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "app" / "ml" / "profeta" / "data"
STEP0_PATH = DATA_DIR / "step0_profeta.parquet"

try:
    STEP0_DF = pd.read_parquet(STEP0_PATH)
except Exception as e:
    raise RuntimeError(f"Impossibile caricare step0_profeta.parquet: {e}")

STEP0_BY_ID = STEP0_DF.set_index("match_id")

# ============================
# ROUTER
# ============================

router = APIRouter()


# ============================
# HELPER — PROFETA ALERT
# ============================

def compute_profeta_alerts(match_id: str, is_fixture: bool):
    """
    Calcola alert Profeta per un match / fixture.
    Se il match non è presente nello step0 → ritorna [].
    """
    if match_id not in STEP0_BY_ID.index:
        return []

    row = STEP0_BY_ID.loc[match_id]

    if bool(row["is_fixture"]) != bool(is_fixture):
        return []

    engine = ProfetaEngine(max_goals=10)
    result = engine.predict_from_row(row)

    control_state = compute_control_state(result["markets"])
    goal_state = compute_goal_state(result["markets"])

    alerts = run_alert_engine(
        markets=result["markets"],
        control_state=control_state,
        goal_state=goal_state,
    )

    return alerts


# ============================
# ENDPOINT
# ============================

@router.get("/matches/by_date")
async def get_matches_by_date(
    match_date: str = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_football_db),
):
    """
    Ritorna le partite per una certa data, con ALERT PROFETA inclusi.
    """

    if match_date is None:
        match_date = datetime.now().date().isoformat()

    try:
        date_obj = datetime.strptime(match_date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(400, "match_date must be YYYY-MM-DD")

    match_date_str = date_obj.isoformat()

    # ============================
    # MATCH STORICI
    # ============================

    q = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season)
            .load_only(Season.id, Season.league_id, Season.name, Season.code)
            .joinedload(Season.league)
            .load_only(League.id, League.code, League.name, League.country_id)
            .joinedload(League.country),
        )
        .filter(
            or_(Match.match_date == match_date_str, Match.match_date == date_obj),
        )
    )

    matches = q.all()

    # ============================
    # FIXTURE
    # ============================

    fq = (
        db.query(Fixture)
        .options(
            joinedload(Fixture.home_team),
            joinedload(Fixture.away_team),
            joinedload(Fixture.season)
            .load_only(Season.id, Season.league_id, Season.name, Season.code)
            .joinedload(Season.league)
            .load_only(League.id, League.code, League.name, League.country_id)
            .joinedload(League.country),
        )
        .filter(
            or_(Fixture.match_date == match_date_str, Fixture.match_date == date_obj),
            Fixture.match_date.isnot(None),
        )
    )

    fixtures = fq.all()

    # ============================
    # OUTPUT STRUCTURE
    # ============================

    country_map: Dict[str, Dict] = {}
    matches_out = []

    def add_country_and_league(league_obj):
        if not league_obj:
            return
        country = getattr(league_obj, "country", None)
        c_code = getattr(country, "code", None) or "UNKNOWN"

        if c_code not in country_map:
            country_map[c_code] = {
                "code": c_code,
                "name": getattr(country, "name", c_code),
                "leagues": {},
            }

        l_code = getattr(league_obj, "code", "UNKNOWN")
        country_map[c_code]["leagues"][l_code] = {
            "code": l_code,
            "name": getattr(league_obj, "name", l_code),
        }

    # ============================
    # MATCH STORICI LOOP
    # ============================

    for m in matches:
        season = m.season
        league = season.league if season else None
        add_country_and_league(league)

        alerts = compute_profeta_alerts(
            match_id=str(m.id),
            is_fixture=False,
        )

        matches_out.append(
            {
                "id": str(m.id),
                "home_team": m.home_team.name if m.home_team else None,
                "away_team": m.away_team.name if m.away_team else None,
                "match_time": m.match_time,
                "home_ft": m.home_goals_ft,
                "away_ft": m.away_goals_ft,
                "is_fixture": False,
                "league_code": getattr(league, "code", None),
                "league_name": getattr(league, "name", None),
                "country_code": getattr(getattr(league, "country", None), "code", None),
                "country_name": getattr(getattr(league, "country", None), "name", None),
                "alerts": alerts,
            }
        )

    # ============================
    # FIXTURE LOOP
    # ============================

    for f in fixtures:
        season = f.season
        league = season.league if season else None
        add_country_and_league(league)

        alerts = compute_profeta_alerts(
            match_id=str(f.id),
            is_fixture=True,
        )

        matches_out.append(
            {
                "id": str(f.id),
                "home_team": f.home_team.name if f.home_team else None,
                "away_team": f.away_team.name if f.away_team else None,
                "match_time": f.match_time,
                "home_ft": f.home_goals_ft,
                "away_ft": f.away_goals_ft,
                "is_fixture": True,
                "league_code": getattr(league, "code", None) or getattr(f, "league_code", None),
                "league_name": getattr(league, "name", None) or getattr(f, "league_name", None),
                "country_code": getattr(getattr(league, "country", None), "code", None),
                "country_name": getattr(getattr(league, "country", None), "name", None),
                "alerts": alerts,
            }
        )

    # ============================
    # COUNTRIES OUTPUT
    # ============================

    countries = []
    for country in country_map.values():
        country["leagues"] = list(country["leagues"].values())
        countries.append(country)

    return {
        "success": True,
        "date": match_date,
        "countries": countries,
        "matches": matches_out,
    }