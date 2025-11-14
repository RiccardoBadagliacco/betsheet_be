import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from sqlalchemy.orm import Session

# ======================================================================
# FIX PYTHONPATH → aggiungiamo la ROOT DEL PROGETTO
# ======================================================================

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parents[4]   # <-- cartella betsheet_be
APP_DIR = ROOT_DIR / "app"

print(f"📌 __file__ = {THIS_FILE}")
print(f"📁 ROOT_DIR (added to PYTHONPATH): {ROOT_DIR}")

sys.path.append(str(ROOT_DIR))
sys.path.append(str(APP_DIR))

# Import DB
from app.db.database_football import get_football_db
from app.db.models_football import Match, Season, League, Team


# ======================================================================
# OUTPUT PATH
# ======================================================================

DATA_DIR = ROOT_DIR / "app" / "ml" / "correlazioni_affini" / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUT_PATH = DATA_DIR / "step0_dataset_base.parquet"

# ======================================================================
# UTILS
# ======================================================================

def safe_div(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except:
        return None
    if x <= 0:
        return None
    return 1.0 / x


def normalize_two(a, b):
    if a is None or b is None:
        return None, None
    s = a + b
    if s <= 0:
        return None, None
    return a / s, b / s


def normalize_three(a, b, c):
    if a is None or b is None or c is None:
        return None, None, None
    s = a + b + c
    if s <= 0:
        return None, None, None
    return a / s, b / s, c / s


# ======================================================================
# MAIN FUNCTION
# ======================================================================

def build_step0_dataset(db: Session) -> pd.DataFrame:
    rows: List[Dict] = []

    q = (
        db.query(Match)
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
        .join(Team, Match.home_team_id == Team.id)
    )

    matches = q.all()
    print(f"📥 Match trovati nel DB: {len(matches)}")

    for m in matches:

        if m.home_goals_ft is None or m.away_goals_ft is None:
            continue

        league = m.season.league if m.season else None
        country = getattr(league, "country", None)

        # Probabilità implicite quote
        p1_raw = safe_div(m.avg_home_odds)
        px_raw = safe_div(m.avg_draw_odds)
        p2_raw = safe_div(m.avg_away_odds)
        bk_p1, bk_px, bk_p2 = normalize_three(p1_raw, px_raw, p2_raw)

        pO_raw = safe_div(m.avg_over_25_odds)
        pU_raw = safe_div(m.avg_under_25_odds)
        bk_pO25, bk_pU25 = normalize_two(pO_raw, pU_raw)

        if bk_p1 is None:
            continue

        rows.append({
            "match_id": str(m.id),
            "date": str(m.match_date),
            "league": getattr(league, "code", None),
            "country": getattr(country, "code", None),
            "season": getattr(m.season, "name", None),
            "home_team": m.home_team.name,
            "away_team": m.away_team.name,
            "home_ft": m.home_goals_ft,
            "away_ft": m.away_goals_ft,
            "bk_p1": bk_p1,
            "bk_px": bk_px,
            "bk_p2": bk_p2,
            "bk_pO25": bk_pO25,
            "bk_pU25": bk_pU25,
            "tech_p1": None,
            "tech_px": None,
            "tech_p2": None,
            "tech_pO25": None,
            "tech_pU25": None,
            "delta_p1": None,
            "delta_px": None,
            "delta_p2": None,
        })

    df = pd.DataFrame(rows)
    print(f"✅ Dataset creato: {df.shape}")

    return df


# ======================================================================
# CLI MAIN
# ======================================================================

def main():
    print("🚀 STEP0 → Generazione dataset_base dal DB")
    print("Output:", OUT_PATH)

    db = next(get_football_db())
    df = build_step0_dataset(db)

    if df.empty:
        print("⚠️ Nessun dato generato, skip salvataggio.")
        return

    df.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato!", OUT_PATH)


if __name__ == "__main__":
    main()