import pandas as pd
from sqlalchemy.orm import Session
from app.db.models import Match

def load_matches_history(db: Session) -> pd.DataFrame:
    """
    Carica tutte le partite COMPLETE dal DB.
    Serve a runtime per:
      - Elo history
      - Form rolling
      - Runtime target builder fixture
    """
    q = (
        db.query(Match)
        .filter(Match.home_goals_ft.isnot(None))
        .filter(Match.away_goals_ft.isnot(None))
        .all()
    )

    rows = []
    for m in q:
        rows.append({
            "match_id": str(m.id),
            "date": m.match_date,
            "season": m.season.name if m.season else None,
            "league": m.season.league.code if m.season and m.season.league else None,
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "home_ft": m.home_goals_ft,
            "away_ft": m.away_goals_ft,
        })

    return pd.DataFrame(rows)