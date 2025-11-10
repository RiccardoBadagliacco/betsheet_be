from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List
from datetime import datetime, date
from sqlalchemy.orm import Session, joinedload

from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League
from app.services.picchetto_tecnico import calcola_picchetto

router = APIRouter()


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

                fixture_out = {
                    "fixture_id": str(m.id),
                    "match_date": m.match_date.isoformat() if m.match_date else None,
                    "match_time": m.match_time,
                    "home_team": getattr(m.home_team, "name", None),
                    "away_team": getattr(m.away_team, "name", None),
                    "league_code": league_code,
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

    # 3️⃣ Calcolo statistiche strutturate
    def compute_team_stats(team_id: str, side: str) -> Dict:
        matches = (
            db.query(Match)
            .filter(Match.season_id == m.season_id)
            .filter((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
            .order_by(Match.match_date.desc(), Match.match_time.desc())
            .all()
        )

        finished = [mm for mm in matches if mm.home_goals_ft is not None and mm.away_goals_ft is not None]
        recent = finished[:5]

        def count_results(ms):
            w = l = d = 0
            for mm in ms:
                if mm.home_goals_ft == mm.away_goals_ft:
                    d += 1
                elif (mm.home_team_id == team_id and mm.home_goals_ft > mm.away_goals_ft) or (
                    mm.away_team_id == team_id and mm.away_goals_ft > mm.home_goals_ft
                ):
                    w += 1
                else:
                    l += 1
            return {"partite": len(ms), "vittorie": w, "pareggi": d, "sconfitte": l}

        # Blocchi specifici per casa o trasferta
        if side == "home":
            tot_side_list = [mm for mm in finished if mm.home_team_id == team_id]
            rec_side_list = tot_side_list[:5]
            totali_home_away = {"home": count_results(tot_side_list)}
            recenti_home_away = {"home": count_results(rec_side_list)}
        else:
            tot_side_list = [mm for mm in finished if mm.away_team_id == team_id]
            rec_side_list = tot_side_list[:5]
            totali_home_away = {"away": count_results(tot_side_list)}
            recenti_home_away = {"away": count_results(rec_side_list)}

        # ✅ Nuovo formato con "label" e "stats"
        return {
            "totali": {
                "label": "Totale stagione",
                "stats": count_results(finished),
            },
            "recenti": {
                "label": "Ultime 5 partite",
                "stats": count_results(recent),
            },
            "totali_home_away": {
                "label": "Totale H/A",
                "stats": totali_home_away,
            },
            "recenti_home_away": {
                "label": "Ultime 5 H/A",
                "stats": recenti_home_away,
            },
        }

    stats_home = compute_team_stats(m.home_team_id, "home")
    stats_away = compute_team_stats(m.away_team_id, "away")

    # 4️⃣ Flatten per il picchetto tecnico
    def flatten_stats(s: Dict) -> Dict:
        def g(d, k, default=0):
            return d.get(k, default) if isinstance(d, dict) else default

        home_tot = s["totali_home_away"]["stats"].get("home", {})
        away_tot = s["totali_home_away"]["stats"].get("away", {})
        home_rec = s["recenti_home_away"]["stats"].get("home", {})
        away_rec = s["recenti_home_away"]["stats"].get("away", {})

        return {
            "vittorie_totali": s["totali"]["stats"]["vittorie"],
            "sconfitte_totali": s["totali"]["stats"]["sconfitte"],
            "partite_totali": s["totali"]["stats"]["partite"],
            "vittorie_recenti": s["recenti"]["stats"]["vittorie"],
            "sconfitte_recenti": s["recenti"]["stats"]["sconfitte"],
            "partite_recenti": s["recenti"]["stats"]["partite"],
            "vittorie_casa": g(home_tot, "vittorie"),
            "sconfitte_casa": g(home_tot, "sconfitte"),
            "partite_casa": g(home_tot, "partite"),
            "vittorie_casa_recenti": g(home_rec, "vittorie"),
            "sconfitte_casa_recenti": g(home_rec, "sconfitte"),
            "partite_casa_recenti": g(home_rec, "partite"),
            "vittorie_trasferta": g(away_tot, "vittorie"),
            "sconfitte_trasferta": g(away_tot, "sconfitte"),
            "partite_trasferta": g(away_tot, "partite"),
            "vittorie_trasferta_recenti": g(away_rec, "vittorie"),
            "sconfitte_trasferta_recenti": g(away_rec, "sconfitte"),
            "partite_trasferta_recenti": g(away_rec, "partite"),
        }

    flat_home = flatten_stats(stats_home)
    flat_away = flatten_stats(stats_away)

    match_payload = {
        "odds": {"1": m.avg_home_odds, "X": m.avg_draw_odds, "2": m.avg_away_odds},
        "stats_home": flat_home,
        "stats_away": flat_away,
    }

    try:
        picchetto = calcola_picchetto(match_payload)
    except Exception as e:
        picchetto = {"error": str(e)}

    return {
        "success": True,
        "match": base_info,
        "stats": {"home": stats_home, "away": stats_away},
        "picchetto_tecnico": picchetto,
    }
