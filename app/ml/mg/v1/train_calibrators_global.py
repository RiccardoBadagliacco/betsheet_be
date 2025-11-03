
# app/ml/mg_model/v1/train_calibrators_global.py
"""
Train isotonic calibrators (GLOBAL) for MG markets using historical matches.
- Reads from SQLite DB (configure DB_PATH)
- Computes raw MG probs via MGModelV1 (without calibrators)
- Fits IsotonicRegression for each MG market on completed matches
- Saves pickles to app/ml/mg/v1/calibrators/global/
"""

import os
import sqlite3
import pickle
from pathlib import Path


import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from model import MGModelV1, MARKETS

import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--patch", type=int, default=int(os.getenv("MG_PATCH", 0)))
args = parser.parse_args()
PATCH = args.patch

# ---- Config ----
DB_PATH = os.environ.get("FOOTBALL_DB_PATH", "./data/football_dataset.db")
OUT_DIR = f"app/ml/mg/v1/calibrators/global"
MAX_ROWS = None  # set to int to limit for quick tests
MIN_VALID_ODDS = 1.01

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

def valid_row(row) -> bool:
    # require 1X2 odds
    if row["odd_home"] is None or row["odd_draw"] is None or row["odd_away"] is None:
        return False
    if row["odd_home"] < MIN_VALID_ODDS or row["odd_draw"] < MIN_VALID_ODDS or row["odd_away"] < MIN_VALID_ODDS:
        return False
    # O/U optional
    return True

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
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_matches(limit=MAX_ROWS)
    df = df[df.apply(valid_row, axis=1)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No valid rows found. Check DB/odds.")

    df = label_targets(df)
    print(f"Loaded {len(df)} valid matches for training.")
    model = MGModelV1(patch=PATCH, calibrators_global_dir=None, calibrators_leagues_dir=None)

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
            # mu_total_prior fallback not used explicitly here
        }
        out = model.predict(payload)
        preds.append(out.probs)

    proba_df = pd.DataFrame(preds, columns=list(MARKETS))
    data = pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

    # Fit isotonic per market
    calibrators = {}
    for mkt in MARKETS:
        # valori e target
        p = data[mkt].astype(float).values
        y = data["y_" + mkt].astype(int).values

        # --- FILTRO PATCH 1: ignora None / NaN ---
        mask = ~np.isnan(p)
        p, y = p[mask], y[mask]

        if len(np.unique(y)) < 2 or len(p) < 100:
            print(f"[WARN] Not enough data for {mkt} after filtering, skipping.")
            continue

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        calibrators[mkt] = ir
        with open(os.path.join(OUT_DIR, f"{mkt}.pkl"), "wb") as f:
            pickle.dump(ir, f)
        print(f"[OK] Saved {mkt} calibrator.")

    print(f"Done. Saved {len(calibrators)} calibrators in {OUT_DIR}")

if __name__ == "__main__":
    main()
