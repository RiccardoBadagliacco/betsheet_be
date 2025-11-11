from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Any, Dict, List, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session, joinedload
import logging

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League
from app.analytics.get_team_stats_service import get_team_stats
from app.analytics.picchetto_tecnico import calcola_picchetto_1X2, calcola_picchetto_ou25_structured
from app.analytics.metrics import get_metrics
router = APIRouter()
logger = logging.getLogger(__name__)


from uuid import UUID


def format_stats_for_side(stats: Dict[str, Any], side: str, last_n: int = 5) -> Dict[str, Any]:
    """Normalize the stats service output into the structured blocks expected by the API."""
    data = stats.get("1_x_2", {})
    suffix = "home" if side == "home" else "away"
    label_side = suffix.upper()

    def wrap(label: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {"label": label, "stats": payload or {}}

    formatted: Dict[str, Any] = {
        "1X2": {
            "type": "1X2",
            "totali": wrap("Totale stagione", data.get("total_stats", {}) or {}),
            "recenti": wrap(f"Ultime {last_n} partite", data.get("last_n_stats", {}) or {}),
            f"totali_{suffix}": wrap(f"Totale {label_side}", data.get("total_stats_side", {}) or {}),
            f"recenti_{suffix}": wrap(
                f"Ultime {last_n} partite {label_side}", data.get("last_n_stats_side", {}) or {}
            ),
        }
    }

    ou_data = stats.get("ou_25", {}) or {}

    def build_stats(entry: Optional[Dict[str, Any]], allowed_keys: List[str]) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            return {}
        stats: Dict[str, Any] = {}
        for key in allowed_keys:
            if entry.get(key) is not None:
                stats[key] = entry.get(key)
        return stats

    if ou_data:
        formatted["OU25"] = {
            "type": "OU 2.5",
            "totali": wrap("Totale stagione", build_stats(ou_data.get("total_stats"), ["partite", "under", "over"])),
            "recenti": wrap(
                f"Ultime {last_n} partite", build_stats(ou_data.get("last_n_stats"), ["partite", "under", "over"])
            ),
            f"totali_{suffix}": wrap(
                f"Totale {label_side}", build_stats(ou_data.get("total_stats_side"), ["partite", "under", "over"])
            ),
            f"recenti_{suffix}": wrap(
                f"Ultime {last_n} partite {label_side}",
                build_stats(ou_data.get("last_n_stats_side"), ["partite", "under", "over"]),
            ),
        }

    ou15_data = stats.get("ou_15", {}) or {}
    if ou15_data:
        formatted["OU15"] = {
            "type": "OU 1.5",
            "totali": wrap("Totale stagione", build_stats(ou15_data.get("total_stats"), ["partite", "under", "over"])),
            "recenti": wrap(
                f"Ultime {last_n} partite", build_stats(ou15_data.get("last_n_stats"), ["partite", "under", "over"])
            ),
            f"totali_{suffix}": wrap(
                f"Totale {label_side}", build_stats(ou15_data.get("total_stats_side"), ["partite", "under", "over"])
            ),
            f"recenti_{suffix}": wrap(
                f"Ultime {last_n} partite {label_side}",
                build_stats(ou15_data.get("last_n_stats_side"), ["partite", "under", "over"]),
            ),
        }

    gng_data = stats.get("goal_no_goal", {}) or {}
    if gng_data:
        formatted["GNG"] = {
            "type": "Goal / No Goal",
            "totali": wrap("Totale stagione", build_stats(gng_data.get("total_stats"), ["partite", "goal", "no_goal"])),
            "recenti": wrap(
                f"Ultime {last_n} partite", build_stats(gng_data.get("last_n_stats"), ["partite", "goal", "no_goal"])
            ),
            f"totali_{suffix}": wrap(
                f"Totale {label_side}", build_stats(gng_data.get("total_stats_side"), ["partite", "goal", "no_goal"])
            ),
            f"recenti_{suffix}": wrap(
                f"Ultime {last_n} partite {label_side}",
                build_stats(gng_data.get("last_n_stats_side"), ["partite", "goal", "no_goal"]),
            ),
        }

    return formatted

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

        def build_structured_stats(team_id: str, side: str, match_obj: Match) -> Optional[Dict[str, Any]]:
            try:
                stats = get_team_stats(
                    db,
                    team_id,
                    side,
                    season_id=match_obj.season_id,
                    n=5,
                    match_date=match_obj.match_date,
                )
                return format_stats_for_side(stats, side)
            except Exception as exc:
                logger.warning("Impossibile calcolare le stats %s per match %s: %s", side, match_obj.id, exc)
                return None

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

                picchetto_1x2 = None
                picchetto_ou25 = None
                metrics_data = None
                odds_ou_payload = {
                    "U2.5": m.avg_under_25_odds,
                    "O2.5": m.avg_over_25_odds,
                }
                stats_home = build_structured_stats(m.home_team_id, "home", m)
                stats_away = build_structured_stats(m.away_team_id, "away", m)
                if stats_home and stats_away:
                    try:
                        odds_payload = {
                            "1": m.avg_home_odds,
                            "X": m.avg_draw_odds,
                            "2": m.avg_away_odds,
                        }
                        match_payload = {
                            "home_name": getattr(m.home_team, "name", "Casa"),
                            "away_name": getattr(m.away_team, "name", "Trasferta"),
                            "odds": odds_payload,
                            "odds_ou": odds_ou_payload,
                            "stats_home": stats_home,
                            "stats_away": stats_away,
                        }
                        picchetto_1x2 = calcola_picchetto_1X2(match_payload)
                        picchetto_ou25 = calcola_picchetto_ou25_structured(match_payload)
                        picchetto_map = {}
                        if picchetto_1x2:
                            picchetto_map["1X2"] = picchetto_1x2
                        if picchetto_ou25:
                            picchetto_map["OU25"] = picchetto_ou25
                        if picchetto_map:
                            metrics_data = get_metrics(picchetto_map)
                    except Exception as picchetto_exc:
                        logger.warning("Picchetto tecnico fallito per match %s: %s", m.id, picchetto_exc)

                fixture_out = {
                    "fixture_id": str(m.id),
                    "match_date": m.match_date.isoformat() if m.match_date else None,
                    "match_time": m.match_time,
                    "home_team": getattr(m.home_team, "name", None),
                    "away_team": getattr(m.away_team, "name", None),
                    "league_code": league_code,
                    "risultato": result_ft,
                    "odds_ou_25": odds_ou_payload,
                    "picchetto_tecnico_1X2": picchetto_1x2,
                    "picchetto_tecnico_OU25": picchetto_ou25,
                    "metrics": metrics_data,
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

@router.get("/matches/{match_id}")
async def get_match_detail(match_id: str, db: Session = Depends(get_football_db)) -> Dict:
    """
    Restituisce i dettagli di una partita:
      • Info principali
      • Statistiche (totali, recenti, casa/trasferta)
      • Picchetto tecnico 1X2
    """
    # 1️⃣ Validazione ID
    try:
        match_uuid = UUID(match_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="match_id deve essere un UUID valido")

    # 2️⃣ Query match
    m = (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season).joinedload(Season.league).joinedload(League.country),
        )
        .filter(Match.id == match_uuid)
        .one_or_none()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Partita non trovata")

    league = getattr(m.season, "league", None)
    country = getattr(league, "country", None)

    base_info = {
        "id": str(m.id),
        "date": m.match_date.isoformat() if m.match_date else None,
        "time": m.match_time,
        "home_team": {
            "id": str(getattr(m.home_team, "id", "")),
            "name": getattr(m.home_team, "name", None),
        },
        "away_team": {
            "id": str(getattr(m.away_team, "id", "")),
            "name": getattr(m.away_team, "name", None),
        },
        "result": {"home_ft": m.home_goals_ft, "away_ft": m.away_goals_ft},
        "odds": {"1": m.avg_home_odds, "X": m.avg_draw_odds, "2": m.avg_away_odds},
        "odds_ou_25": {"U2.5": m.avg_under_25_odds, "O2.5": m.avg_over_25_odds},
        "league": {
            "code": getattr(league, "code", None),
            "name": getattr(league, "name", None),
            "country": getattr(country, "code", None),
        },
        "season": {
            "id": str(getattr(m.season, "id", "")),
            "name": getattr(m.season, "name", None),
        },
    }

    # 3️⃣ Use the shared service to compute team stats
    def compute_team_stats(team_id: str, side: str) -> Dict[str, Any]:
        """Fetch and format stats for the requested side."""
        stats = get_team_stats(db, team_id, side, season_id=m.season_id, n=5, match_date=m.match_date)
        return format_stats_for_side(stats, side)

    stats_home = compute_team_stats(m.home_team_id, 'home')
    stats_away = compute_team_stats(m.away_team_id, 'away')
    match_payload = {
        "home_name": m.home_team.name,
        "away_name": m.away_team.name,
        "odds": {
            "1": m.avg_home_odds,
            "X": m.avg_draw_odds,
            "2": m.avg_away_odds,
        },
        "odds_ou": {
            "U2.5": m.avg_under_25_odds,
            "O2.5": m.avg_over_25_odds,
        },
        "stats_home": stats_home,
        "stats_away": stats_away,
    }

    picchetto_1x2 = calcola_picchetto_1X2(match_payload)
    picchetto_ou25 = calcola_picchetto_ou25_structured(match_payload)
    picchetti = {"1X2": picchetto_1x2}
    if picchetto_ou25:
        picchetti["OU25"] = picchetto_ou25
    metrics = get_metrics(picchetti)

 
    return {
        "success": True,
        "match": base_info,
        "stats": {"home": stats_home, "away": stats_away},
        "picchetti": picchetti,
        "metrics": metrics,
    }
