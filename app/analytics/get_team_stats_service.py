from typing import Dict, Optional
from datetime import date, datetime
from sqlalchemy.orm import Session
from app.db.models_football import Match


def get_team_stats(
    db: Session,
    team_id: str,
    side: str,
    season_id: Optional[str] = None,
    n: int = 5,
    match_date: Optional[date] = None,
) -> Dict:
    """
    Calcola statistiche 1X2, Over/Under e Goal/NoGoal per una squadra.

    Gestisce automaticamente i casi in cui `match_date` è None, float o string.
    Evita confronti non validi tra tipi.
    """

    # 🧩 Conversione sicura di match_date
    safe_date = None
    if match_date:
        try:
            if isinstance(match_date, (float, int)):
                safe_date = datetime.fromtimestamp(match_date).date()
            elif isinstance(match_date, str):
                safe_date = datetime.fromisoformat(match_date).date()
            elif isinstance(match_date, datetime):
                safe_date = match_date.date()
            elif isinstance(match_date, date):
                safe_date = match_date
        except Exception as e:
            print(f"⚠️ Errore nella conversione di match_date ({match_date}): {e}")
            safe_date = None

    # 📊 Query base
    q = db.query(Match)
    if season_id:
        q = q.filter(Match.season_id == season_id)

    # ⏳ Filtro temporale sicuro
    if safe_date is not None:
        print(f"Filtering matches before date: {safe_date}")
        q = q.filter(Match.match_date != None).filter(Match.match_date < safe_date)
    else:
        q = q.filter(Match.match_date != None)

    q = q.filter((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
    q = q.order_by(Match.match_date.desc(), Match.match_time.desc())
    matches = q.all()

    # 🏁 Considera solo partite concluse e con data valida
    finished = [
        m for m in matches
        if m.match_date is not None and m.home_goals_ft is not None and m.away_goals_ft is not None
    ]

    # --- 🔹 FUNZIONI DI SUPPORTO ---
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
        return {
            "partite": {"value": len(ms), "label": "N"},
            "vittorie": {"value": w, "label": "V"},
            "pareggi": {"value": d, "label": "P"},
            "sconfitte": {"value": l, "label": "S"},
        }

    def count_over_under(ms, under_goal_max: int, under_label: str, over_label: str):
        under = over = 0
        for mm in ms:
            if mm.home_goals_ft is None or mm.away_goals_ft is None:
                continue
            total_goals = mm.home_goals_ft + mm.away_goals_ft
            if total_goals <= under_goal_max:
                under += 1
            else:
                over += 1
        return {
            "partite": {"value": len(ms), "label": "N"},
            "under": {"value": under, "label": under_label},
            "over": {"value": over, "label": over_label},
        }

    def count_goal_no_goal(ms):
        goal = no_goal = 0
        for mm in ms:
            if mm.home_goals_ft is None or mm.away_goals_ft is None:
                continue
            if mm.home_goals_ft > 0 and mm.away_goals_ft > 0:
                goal += 1
            else:
                no_goal += 1
        return {
            "partite": {"value": len(ms), "label": "N"},
            "goal": {"value": goal, "label": "Goal"},
            "no_goal": {"value": no_goal, "label": "No Goal"},
        }

    # --- 🔹 STATISTICHE ---
    total_stats = count_results(finished)
    last_n_stats = count_results(finished[:n])

    if side == "home":
        side_matches = [m for m in finished if m.home_team_id == team_id]
    else:
        side_matches = [m for m in finished if m.away_team_id == team_id]

    total_stats_side = count_results(side_matches)
    last_n_stats_side = count_results(side_matches[:n])

    ou_total = count_over_under(finished, 2, "Under 2.5", "Over 2.5")
    ou_recent = count_over_under(finished[:n], 2, "Under 2.5", "Over 2.5")
    ou_total_side = count_over_under(side_matches, 2, "Under 2.5", "Over 2.5")
    ou_recent_side = count_over_under(side_matches[:n], 2, "Under 2.5", "Over 2.5")

    ou15_total = count_over_under(finished, 1, "Under 1.5", "Over 1.5")
    ou15_recent = count_over_under(finished[:n], 1, "Under 1.5", "Over 1.5")
    ou15_total_side = count_over_under(side_matches, 1, "Under 1.5", "Over 1.5")
    ou15_recent_side = count_over_under(side_matches[:n], 1, "Under 1.5", "Over 1.5")

    gng_total = count_goal_no_goal(finished)
    gng_recent = count_goal_no_goal(finished[:n])
    gng_total_side = count_goal_no_goal(side_matches)
    gng_recent_side = count_goal_no_goal(side_matches[:n])

    return {
        "1_x_2": {
            "total_stats": total_stats,
            "last_n_stats": last_n_stats,
            "total_stats_side": total_stats_side,
            "last_n_stats_side": last_n_stats_side,
        },
        "ou_25": {
            "total_stats": ou_total,
            "last_n_stats": ou_recent,
            "total_stats_side": ou_total_side,
            "last_n_stats_side": ou_recent_side,
        },
        "ou_15": {
            "total_stats": ou15_total,
            "last_n_stats": ou15_recent,
            "total_stats_side": ou15_total_side,
            "last_n_stats_side": ou15_recent_side,
        },
        "goal_no_goal": {
            "total_stats": gng_total,
            "last_n_stats": gng_recent,
            "total_stats_side": gng_total_side,
            "last_n_stats_side": gng_recent_side,
        },
    }