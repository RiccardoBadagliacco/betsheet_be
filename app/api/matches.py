from fastapi import APIRouter, HTTPException, Depends,Query
from sqlalchemy.orm import Session, joinedload
from uuid import UUID
from typing import Any, Dict, List, Optional

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League
from app.ml.correlazioni_affini.cluster_predictor import predict_cluster_and_profile
from datetime import datetime, date

router = APIRouter()


@router.get("/matches/{match_id}")
def predict_match_model(match_id: str, db: Session = Depends(get_football_db)):
    """
    Restituisce:
      • info match (team, lega, odds, risultato se presente)
      • cluster predetto
      • profilo statistico del cluster
    """
    # -------------------------------------------------------
    # 1) Validate UUID
    # -------------------------------------------------------
    try:
        match_uuid = UUID(match_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="match_id non valido (UUID)")

    # -------------------------------------------------------
    # 2) Query match
    # -------------------------------------------------------
    m = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season).joinedload(Season.league)
        )
        .filter(Match.id == match_uuid)
        .one_or_none()
    )

    if not m:
        raise HTTPException(status_code=404, detail="Match non trovato")

    # -------------------------------------------------------
    # 3) Ricostruzione feature per il GMM (solo quote fornite)
    #
    #  Questi nomi devono combaciare con feature_cols usate in training:
    #       bk_p1, bk_px, bk_p2, bk_pO25, bk_pU25
    #
    #  Tutte le altre feature mancanti verranno riempite automaticamente
    #  con la media di colonna dal ClusterEngine.
    # -------------------------------------------------------

    row = {
        "bk_p1": 1 / m.avg_home_odds if m.avg_home_odds else None,
        "bk_px": 1 / m.avg_draw_odds if m.avg_draw_odds else None,
        "bk_p2": 1 / m.avg_away_odds if m.avg_away_odds else None,
        "bk_pO25": 1 / m.avg_over_25_odds if m.avg_over_25_odds else None,
        "bk_pU25": 1 / m.avg_under_25_odds if m.avg_under_25_odds else None,
        # in futuro puoi aggiungere elo, poisson, forma ecc.
    }

    # -------------------------------------------------------
    # 4) Predict cluster + profilo
    # -------------------------------------------------------
    cluster_id, profile = predict_cluster_and_profile(row)

    # -------------------------------------------------------
    # 5) Preparo output info match
    # -------------------------------------------------------
    league = m.season.league if m.season else None

    match_info = {
        "id": str(m.id),
        "date": m.match_date.isoformat() if m.match_date else None,
        "time": m.match_time,
        "home_team": {
            "id": str(m.home_team.id) if m.home_team else None,
            "name": m.home_team.name if m.home_team else None,
        },
        "away_team": {
            "id": str(m.away_team.id) if m.away_team else None,
            "name": m.away_team.name if m.away_team else None,
        },
        "result_ft": {
            "home": m.home_goals_ft,
            "away": m.away_goals_ft,
        },
        "odds": {
            "1": m.avg_home_odds,
            "X": m.avg_draw_odds,
            "2": m.avg_away_odds,
            "O2.5": m.avg_over_25_odds,
            "U2.5": m.avg_under_25_odds,
        },
        "league": {
            "code": getattr(league, "code", None),
            "name": getattr(league, "name", None),
            "country": getattr(league.country, "code", None) if league and league.country else None,
        },
        "season": {
            "id": str(m.season.id) if m.season else None,
            "name": m.season.name if m.season else None,
        },
    }

    # -------------------------------------------------------
    # 6) Response finale
    # -------------------------------------------------------
    return {
        "success": True,
        "match": match_info,
        "cluster": cluster_id,
        "profile": profile,   # intero profilo cluster: 1x2, OU, GG, MG, top scores...
    }

# -------------------------------------------------------
@router.get("/matches")
async def get_matches_by_date(
    match_date: str = Query(None, description="Date in YYYY-MM-DD. If omitted returns today's date."),
    db: Session = Depends(get_football_db),
) -> Dict:
    """Return all matches for a given date grouped by league and sorted by time.

    Reads from the football DB `matches` table.
    """
    try:
        if match_date is None:
            match_date = datetime.now().date().isoformat()
        # validate date format
        try:
            date_obj = datetime.strptime(match_date, "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="match_date must be in YYYY-MM-DD format")

        # Query DB for matches on that date, eager-load teams and season->league
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
            try:
                season = m.season
                league = season.league if season is not None else None
                league_code = getattr(league, "code", "UNKNOWN") if league else "UNKNOWN"
                league_name = getattr(league, "name", None) or league_code

                if league_code not in leagues:
                    # try to extract country code from league relationship if present
                    country_code = None
                    try:
                        country_obj = getattr(league, 'country', None)
                        if country_obj is not None:
                            country_code = getattr(country_obj, 'code', None)
                    except Exception:
                        country_code = None

                    leagues[league_code] = {
                        "league_name": league_name,
                        "country": country_code,
                        "fixtures": []
                    }

                result_ft = {
                    "FT_casa": m.home_goals_ft,
                    "FT_trasferta": m.away_goals_ft,
                }

                fixture_out = {
                    "fixture_id": str(m.id),
                    "match_date": m.match_date.isoformat() if m.match_date else None,
                    "match_time": m.match_time,
                    "home_team": getattr(m.home_team, "name", None),
                    "away_team": getattr(m.away_team, "name", None),
                    "league_code": league_code,
                    "risultato": result_ft,
                }

                leagues[league_code]["fixtures"].append(fixture_out)
            except Exception:
                continue

        # sort fixtures within each league by match_time then home_team
        for lc, info in leagues.items():
            info["fixtures"] = sorted(
                info["fixtures"],
                key=lambda x: ((x.get("match_time") or "00:00"), x.get("home_team") or "")
            )

        return {"success": True, "date": match_date, "leagues": leagues}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))