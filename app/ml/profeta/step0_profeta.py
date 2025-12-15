# ============================================================
# app/ml/profeta/step0_profeta_dataset.py
# ============================================================

"""
STEP0 — Dataset base per PROFETA V0 (modello goal-based)

Obiettivo:
    Costruire un dataset minimale e pulito che contiene
    SOLO le informazioni necessarie per stimare
    i parametri ATT/DEF, league_offset, season_offset, HFA.

    Colonne finali:
        match_id
        is_fixture
        match_date
        match_year
        age_years
        weight
        league_id
        league_code
        season_id
        season_code
        home_team_id
        away_team_id
        home_team_season_id
        away_team_season_id
        home_goals
        away_goals
"""

import sys
from pathlib import Path
from typing import List
from datetime import date
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
import math

# ------------------------------------------------------------
# FIX PATH IMPORTS
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]   # adattato alla tua struttura
sys.path.append(str(ROOT_DIR))

from app.db.database_football import get_football_db
from app.db.models import Match, Fixture, Season, League


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUT_PATH = DATA_DIR / "step0_profeta.parquet"

DECAY_LAMBDA = 0.35


# ============================================================
# UTILS
# ============================================================

def compute_age_years(d: date, ref: date) -> float:
    if d is None:
        return 0.0
    delta = (ref - d).days
    if delta < 0:
        delta = 0
    return delta / 365.25


def make_team_season_id(team_id, season_id) -> str:
    return f"{team_id}::{season_id}"


# ============================================================
# MAIN LOGIC
# ============================================================

def build_profeta_step0_dataset() -> pd.DataFrame:
    session: Session = next(get_football_db())
    rows = []

    # --------------------------------------------------------
    # 1. CARICAMENTO MATCH STORICI
    # --------------------------------------------------------
    print("📥 Lettura MATCH storici…")

    q = (
        session.query(Match)
        .join(Season, Match.season_id == Season.id)
        .join(League, Season.league_id == League.id)
    )

    matches: List[Match] = q.all()
    print(f"   → {len(matches)} match storici trovati")

    # --------------------------------------------------------
    # 1A. Troviamo una data di riferimento globale
    # --------------------------------------------------------
    all_dates = [m.match_date for m in matches if m.match_date is not None]
    if not all_dates:
        reference_date = date.today()
    else:
        reference_date = max(all_dates)

    print(f"📌 Data di riferimento per age_years: {reference_date}")

    # --------------------------------------------------------
    # 1B. Per ogni match storico, assembliamo la row
    # --------------------------------------------------------
    for m in matches:
        league: League = m.season.league
        season: Season = m.season

        d = m.match_date
        age_years = compute_age_years(d, reference_date)
        weight = math.exp(-DECAY_LAMBDA * age_years)

        rows.append(
            {
                "match_id": str(m.id),
                "is_fixture": False,

                "match_date": d,
                "match_year": d.year if d else None,
                "age_years": age_years,
                "weight": weight,

                "league_id": str(league.id),
                "league_code": league.code,
                "season_id": str(season.id),
                "season_code": season.code,

                "home_team_id": str(m.home_team_id),
                "away_team_id": str(m.away_team_id),
                "home_team_season_id": make_team_season_id(m.home_team_id, season.id),
                "away_team_season_id": make_team_season_id(m.away_team_id, season.id),

                "home_goals": m.home_goals_ft,
                "away_goals": m.away_goals_ft,
            }
        )

    # --------------------------------------------------------
    # 2. CARICAMENTO FIXTURE (MODALITÀ PROFETA)
    # --------------------------------------------------------
    print("📥 Lettura FIXTURE future…")

    fq = (
        session.query(Fixture)
        .outerjoin(Season, Fixture.season_id == Season.id)
        .outerjoin(League, Season.league_id == League.id)
    )
    fixtures: List[Fixture] = fq.all()
    print(f"   → {len(fixtures)} fixture trovate")

    for f in fixtures:
        season = f.season
        league = season.league if season else None

        d = f.match_date
        # Fixture → age_years = 0, weight = 1
        age_years = 0.0
        weight = 1.0

        rows.append(
            {
                "match_id": str(f.id),
                "is_fixture": True,

                "match_date": d,
                "match_year": d.year if d else None,
                "age_years": age_years,
                "weight": weight,

                "league_id": str(league.id) if league else None,
                "league_code": league.code if league else None,

                "season_id": str(season.id) if season else None,
                "season_code": season.code if season else None,

                "home_team_id": str(f.home_team_id) if f.home_team_id else None,
                "away_team_id": str(f.away_team_id) if f.away_team_id else None,

                "home_team_season_id": make_team_season_id(f.home_team_id, season.id) if season else None,
                "away_team_season_id": make_team_season_id(f.away_team_id, season.id) if season else None,

                "home_goals": f.home_goals_ft,
                "away_goals": f.away_goals_ft,
            }
        )

    # --------------------------------------------------------
    # 3. COSTRUZIONE DATAFRAME
    # --------------------------------------------------------
    df = pd.DataFrame(rows)
    print("📊 Dataset Profeta STEP0 creato:", df.shape)

    # --------------------------------------------------------
    # 4. SALVATAGGIO
    # --------------------------------------------------------
    df.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato:", OUT_PATH)

    return df


def main():
    print("🚀 STEP0 PROFETA — Generazione dataset base")
    print(f"📦 Output: {OUT_PATH}")

    df = build_profeta_step0_dataset()

    print("✅ STEP0 completato!")
    print(df.head())


if __name__ == "__main__":
    main()