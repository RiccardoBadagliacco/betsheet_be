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

    return {
        "1_x_2": {
            "total_stats": total_stats,
            "last_n_stats": last_n_stats,
            "total_stats_side": total_stats_side,
            "last_n_stats_side": last_n_stats_side,
        }
    }
