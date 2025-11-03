
# app/ml/mg_model/v1/train_calibrators_leagues.py
"""
Train isotonic calibrators (PER-LEAGUE) for MG markets.
- Reads from SQLite DB (configure DB_PATH)
- Computes raw MG probs via MGModelV1 (without calibrators)
- For each league_code with enough samples (N_MIN), fits calibrators and saves under per-league folder.
"""

import os
import sqlite3
import pickle
from pathlib import Path


import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import os, argparse
parser = argparse.ArgumentParser()
from model import MGModelV1, MARKETS

# ---- Config ----
MAX_ROWS = None
N_MIN = 1000  # minimum completed matches per league for stable isotonic


parser.add_argument("--patch", type=int, default=int(os.getenv("MG_PATCH", 0)))
args = parser.parse_args()
PATCH = args.patch
DB_PATH = os.environ.get("FOOTBALL_DB_PATH", "./data/football_dataset.db")
OUT_BASE = f"app/ml/mg/v1/calibrators/leagues"


def load_matches(limit=None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT 
        l.code AS league_code,
        m.match_date AS match_date,
        m.home_goals_ft AS home_goals_ft,
        m.away_goals_ft AS away_goals_ft,
        m.avg_home_odds  AS odd_home,
        m.avg_draw_odds  AS odd_draw,
        m.avg_away_odds  AS odd_away,
        m.avg_over_25_odds AS odd_over25,
        m.avg_under_25_odds AS odd_under25,
        s.name AS season
    FROM matches m
    JOIN seasons s ON s.id = m.season_id
    JOIN leagues l ON l.id = s.league_id
    WHERE m.home_goals_ft IS NOT NULL
      AND m.away_goals_ft IS NOT NULL
    ORDER BY m.match_date ASC
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    if limit:
        df = df.head(limit)
    return df

def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea le colonne target (y_MG_HOME_*, y_MG_AWAY_*) per i mercati MG.
    Usa i gol finali (colonne '_ft').
    """
    df = df.copy()

    # Usa i nomi corretti delle colonne dal DB
    if "home_goals_ft" not in df.columns or "away_goals_ft" not in df.columns:
        raise KeyError("Missing columns 'home_goals_ft' or 'away_goals_ft' in DataFrame")

    hg = df["home_goals_ft"].astype(int)
    ag = df["away_goals_ft"].astype(int)

    df["y_MG_HOME_1_3"] = ((hg >= 1) & (hg <= 3)).astype(int)
    df["y_MG_HOME_1_4"] = ((hg >= 1) & (hg <= 4)).astype(int)
    df["y_MG_HOME_1_5"] = ((hg >= 1) & (hg <= 5)).astype(int)

    df["y_MG_AWAY_1_3"] = ((ag >= 1) & (ag <= 3)).astype(int)
    df["y_MG_AWAY_1_4"] = ((ag >= 1) & (ag <= 4)).astype(int)
    df["y_MG_AWAY_1_5"] = ((ag >= 1) & (ag <= 5)).astype(int)

    return df

def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    df = load_matches(limit=MAX_ROWS)
    if df.empty:
        raise SystemExit("No data.")

    df = label_targets(df)
    model = MGModelV1(patch=PATCH, calibrators_global_dir=None, calibrators_leagues_dir=None)

    # Precompute probs for all
    preds = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing matches"):
        payload = {
            "avg_home_odds": float(r["odd_home"]),
            "avg_draw_odds": float(r["odd_draw"]),
            "avg_away_odds": float(r["odd_away"]),
            "avg_over_25_odds": float(r["odd_over25"]) if not pd.isna(r["odd_over25"]) else None,
            "avg_under_25_odds": float(r["odd_under25"]) if not pd.isna(r["odd_under25"]) else None,
            "league_code": r["league_code"],
            "season": r["season"],
        }
        out = model.predict(payload)
        preds.append(out.probs)
    proba_df = pd.DataFrame(preds, columns=list(MARKETS))
    data = pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

    # Train per-league
    for league_code, g in data.groupby("league_code"):
        if len(g) < N_MIN:
            print(f"[SKIP] {league_code}: only {len(g)} samples (<{N_MIN})")
            continue
        out_dir = os.path.join(OUT_BASE, league_code)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[FIT] {league_code}: {len(g)} samples")

        for mkt in MARKETS:
            p = g[mkt].astype(float).values
            y = g["y_" + mkt].astype(int).values

            # --- FILTRO PATCH 1: ignora None / NaN ---
            mask = ~np.isnan(p)
            p, y = p[mask], y[mask]

            if len(np.unique(y)) < 2 or len(p) < 100:
                print(f"  [WARN] {league_code}/{mkt} insufficient or invalid data, skip.")
                continue

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p, y)
            with open(os.path.join(out_dir, f"{mkt}.pkl"), "wb") as f:
                pickle.dump(ir, f)
            print(f"  [OK] saved {mkt}.pkl")

    print("Done.")

if __name__ == "__main__":
    main()
