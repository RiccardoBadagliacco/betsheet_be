from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session, joinedload
import logging

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League
from app.analytics.get_team_stats_service import get_team_stats
from app.analytics.picchetto_tecnico import calcola_picchetto_structured
from app.analytics.metrics import get_metrics
router = APIRouter()
logger = logging.getLogger(__name__)


from uuid import UUID
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

        def build_structured_stats(team_id: str, side: str, match_obj: Match) -> Optional[Dict]:
            try:
                stats = get_team_stats(
                    db,
                    team_id,
                    side,
                    season_id=match_obj.season_id,
                    n=5,
                    match_date=match_obj.match_date,
                )
                data = stats.get("1_x_2", {})
                return {
                    "stats_1_x_2": {
                        "totali": {"label": "Totale stagione", "stats": data.get("total_stats", {})},
                        "recenti": {"label": "Ultime 5 partite", "stats": data.get("last_n_stats", {})},
                        "totali_side": {
                            "label": f"Totale {side.upper()}",
                            "stats": data.get("total_stats_side", {}) or {},
                        },
                        "recenti_side": {
                            "label": f"Ultime 5 partite {side.upper()}",
                            "stats": data.get("last_n_stats_side", {}) or {},
                        },
                    }
                }
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

                picchetto_data = None
                metrics_data = None
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
                            "odds": odds_payload,
                            "stats_home": stats_home,
                            "stats_away": stats_away,
                        }
                        picchetto_data = calcola_picchetto_structured(match_payload)
                        picchetto_map = {"1X2": picchetto_data}
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
                    "picchetto_tecnico_1X2": picchetto_data,
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

    def flatten_stats(s: Dict) -> Dict:
        """Map the merged service output to the flat shape expected by picchetto.

        This helper is backward-compatible: it accepts metrics either as raw
        ints (legacy) or as {value,label} objects (new format) and returns
        integer values for consumption by `calcola_picchetto`.
        """
        def val(d, k, default=0):
            if not isinstance(d, dict):
                return default
            v = d.get(k, default)
            # new format: {"value": X, "label": "..."}
            if isinstance(v, dict) and "value" in v:
                return v.get("value", default)
            # legacy format: direct int
            if isinstance(v, (int, float)):
                return v
            return default

        total = s.get("totali", {}).get("stats", {})
        recent = s.get("recenti", {}).get("stats", {})
        home_tot = s.get("totali_home_away", {}).get("stats", {}).get("home", {})
        away_tot = s.get("totali_home_away", {}).get("stats", {}).get("away", {})
        home_rec = s.get("recenti_home_away", {}).get("stats", {}).get("home", {})
        away_rec = s.get("recenti_home_away", {}).get("stats", {}).get("away", {})

        return {
            "vittorie_totali": val(total, "vittorie"),
            "sconfitte_totali": val(total, "sconfitte"),
            "partite_totali": val(total, "partite"),
            "vittorie_recenti": val(recent, "vittorie"),
            "sconfitte_recenti": val(recent, "sconfitte"),
            "partite_recenti": val(recent, "partite"),
            "vittorie_casa": val(home_tot, "vittorie"),
            "sconfitte_casa": val(home_tot, "sconfitte"),
            "partite_casa": val(home_tot, "partite"),
            "vittorie_casa_recenti": val(home_rec, "vittorie"),
            "sconfitte_casa_recenti": val(home_rec, "sconfitte"),
            "partite_casa_recenti": val(home_rec, "partite"),
            "vittorie_trasferta": val(away_tot, "vittorie"),
            "sconfitte_trasferta": val(away_tot, "sconfitte"),
            "partite_trasferta": val(away_tot, "partite"),
            "vittorie_trasferta_recenti": val(away_rec, "vittorie"),
            "sconfitte_trasferta_recenti": val(away_rec, "sconfitte"),
            "partite_trasferta_recenti": val(away_rec, "partite"),
        }
from uuid import UUID

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
        "match_id": str(m.id),
        "data": m.match_date.isoformat() if m.match_date else None,
        "ora": m.match_time,
        "casa": {"id": str(m.home_team.id), "nome": m.home_team.name},
        "trasferta": {"id": str(m.away_team.id), "nome": m.away_team.name},
        "risultato": {"FT_casa": m.home_goals_ft, "FT_trasferta": m.away_goals_ft},
        "quote": {"1": m.avg_home_odds, "X": m.avg_draw_odds, "2": m.avg_away_odds},
        "lega": {
            "codice": getattr(league, "code", None),
            "nome": getattr(league, "name", None),
            "paese": getattr(country, "code", None),
        },
        "stagione": {
            "id": str(getattr(m.season, "id", "")),
            "nome": getattr(m.season, "name", None),
        },
    }

    # 3️⃣ Use the shared service to compute team stats
    def compute_team_stats(team_id: str, side: str) -> Dict:
        """
        Build the structured stats object by calling `get_team_stats` for the
        requested side and returning the small wrapper shape expected by the
        rest of this endpoint (it contains `stats_1_x_2` with totali/recenti).
        """
        # call service for the requested side (service returns {"1_x_2": {...}})
        stats = get_team_stats(db, team_id, side, season_id=m.season_id, n=5, match_date=m.match_date)

        a = stats.get("1_x_2", {})

        combined = {
            "stats_1_x_2": {
                "totali": {"label": "Totale stagione", "stats": a.get("total_stats", {})},
                "recenti": {"label": f"Ultime {5} partite", "stats": a.get("last_n_stats", {})},
                "totali_side": {
                    "label": f"Totale {side.upper()}",
                    "stats": a.get("total_stats_side", {}) or {},
                },
                "recenti_side": {
                    "label": f"Ultime {5} partite {side.upper()}",
                    "stats": a.get("last_n_stats_side", {}) or {},
                },
            }
        }

        return combined

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
        "stats_home": stats_home,
        "stats_away": stats_away,
    }

    picchetti = {
        "1X2": calcola_picchetto_structured(match_payload)
    }
    metrics = get_metrics(picchetti)

 
    return {
        "success": True,
        "match": base_info,
        "stats": {"home": stats_home, "away": stats_away},
        "picchetti": picchetti,
        "metrics": metrics,
    }
