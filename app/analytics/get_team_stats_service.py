from typing import Dict, Optional
from datetime import date
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
    Calcola statistiche 1-X-2 per una squadra in una season opzionale.

    Args:
      db: SQLAlchemy Session
      team_id: id della squadra (stringa/UUID)
      side: "home" o "away"
      season_id: opzionale, filtra sulla season
      n: numero di partite recenti da considerare (default 5)

    Ritorna:
      {"1_x_2": { total_stats: {...}, last_n_stats: {...}, total_stats_side: {...}, last_n_stats_side: {...} }}
    """
    q = db.query(Match)
    if season_id:
        q = q.filter(Match.season_id == season_id)

    # filter only matches before the specified match_date (if provided)
    if match_date is not None:
        print("Filtering matches before date:", match_date)
        q = q.filter(Match.match_date < match_date)

    q = q.filter((Match.home_team_id == team_id) | (Match.away_team_id == team_id))
    q = q.order_by(Match.match_date.desc(), Match.match_time.desc())
    matches = q.all()

    

    # consideriamo solo partite concluse
    finished = [m for m in matches if m.home_goals_ft is not None and m.away_goals_ft is not None]

    def count_results(ms):
        """Return stats where each metric is an object with value and label.

        Example: {"partite": {"value": 10, "label": "Partite"}, ...}
        """
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

    total_stats = count_results(finished)
    recent = finished[:n]
    last_n_stats = count_results(recent)

    if side == "home":
        tot_side_list = [m for m in finished if m.home_team_id == team_id]
    else:
        tot_side_list = [m for m in finished if m.away_team_id == team_id]

    rec_side_list = tot_side_list[:n]
    total_stats_side = count_results(tot_side_list)
    last_n_stats_side = count_results(rec_side_list)

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

    ou_total = count_over_under(finished, 2, "Under 2.5", "Over 2.5")
    ou_recent = count_over_under(recent, 2, "Under 2.5", "Over 2.5")
    ou_total_side = count_over_under(tot_side_list, 2, "Under 2.5", "Over 2.5")
    ou_recent_side = count_over_under(rec_side_list, 2, "Under 2.5", "Over 2.5")

    ou15_total = count_over_under(finished, 1, "Under 1.5", "Over 1.5")
    ou15_recent = count_over_under(recent, 1, "Under 1.5", "Over 1.5")
    ou15_total_side = count_over_under(tot_side_list, 1, "Under 1.5", "Over 1.5")
    ou15_recent_side = count_over_under(rec_side_list, 1, "Under 1.5", "Over 1.5")

    gng_total = count_goal_no_goal(finished)
    gng_recent = count_goal_no_goal(recent)
    gng_total_side = count_goal_no_goal(tot_side_list)
    gng_recent_side = count_goal_no_goal(rec_side_list)

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
