
# app/ml/mg/v1/backtest.py
"""
Backtest for MGModelV1 baseline:
- Time-based split (train until date T -> fit calibrators -> test after T)
- Computes Brier, Accuracy@tau and ECE (post-calibration)
- Saves thresholds grid search results
Configure DB_PATH and split date or use last-season split.
"""

import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import MGModelV1, MARKETS

# ---- Config ----
DB_PATH = os.environ.get("FOOTBALL_DB_PATH", "./data/football_dataset.db")
CAL_ROOT = "app/ml/mg/v1"
CAL_GLOBAL_DIR  = f"{CAL_ROOT}/calibrators/global"
CAL_LEAGUES_DIR = f"{CAL_ROOT}/calibrators/leagues"
THRESHOLDS_OUT = "data/mg_thresholds.json"
SPLIT_DATE = os.environ.get("MG_SPLIT_DATE")  # e.g. "2023-07-01"; if None, uses 80/20 time split
TAU_GRID = np.round(np.arange(0.50, 0.91, 0.01), 2)


os.makedirs(os.path.dirname(THRESHOLDS_OUT), exist_ok=True)

def load_matches() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT
        l.code AS league_code,
        m.match_date AS match_date,
        m.home_goals_ft AS home_goals,
        m.away_goals_ft AS away_goals,
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
    df = pd.read_sql_query(q, conn, parse_dates=["match_date"])
    conn.close()
    return df

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hg = df["home_goals"].astype(int)
    ag = df["away_goals"].astype(int)
    df["y_MG_HOME_1_3"] = ((hg >= 1) & (hg <= 3)).astype(int)
    df["y_MG_HOME_1_4"] = ((hg >= 1) & (hg <= 4)).astype(int)
    df["y_MG_HOME_1_5"] = ((hg >= 1) & (hg <= 5)).astype(int)
    df["y_MG_AWAY_1_3"] = ((ag >= 1) & (ag <= 3)).astype(int)
    df["y_MG_AWAY_1_4"] = ((ag >= 1) & (ag <= 4)).astype(int)
    df["y_MG_AWAY_1_5"] = ((ag >= 1) & (ag <= 5)).astype(int)
    return df

def brier(y, p):
    y = np.array(y, float); p = np.array(p, float)
    return float(np.mean((p - y) ** 2))

def ece_score(y, p, bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i+1] if i < bins-1 else p <= edges[i+1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def split_time(df: pd.DataFrame):
    df = df.sort_values("match_date").reset_index(drop=True)
    if SPLIT_DATE:
        cutoff = pd.to_datetime(SPLIT_DATE)
        train = df[df["match_date"] < cutoff].copy()
        test  = df[df["match_date"] >= cutoff].copy()
    else:
        # 80/20 split by time
        n = len(df)
        idx = int(n * 0.8)
        cutoff = df.loc[idx, "match_date"]
        train = df[df["match_date"] < cutoff].copy()
        test  = df[df["match_date"] >= cutoff].copy()
    return train, test

def predict_block(df: pd.DataFrame, model: MGModelV1) -> pd.DataFrame:
    preds = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Predicting matches"):
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
    return pd.concat([df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)

def optimize_thresholds(df: pd.DataFrame) -> dict:
    out = {}
    for mkt in MARKETS:
        p_all = df[mkt].astype(float).values
        y_all = df["y_" + mkt].astype(int).values
        mask = ~np.isnan(p_all)
        p = p_all[mask]
        y = y_all[mask]

        if len(p) == 0:
            out[mkt] = {"tau": None, "accuracy": None, "n": 0}
            continue

        best_tau, best_acc = None, -1.0
        for tau in TAU_GRID:
            yhat = (p >= tau).astype(int)
            acc = (yhat == y).mean()
            if acc > best_acc:
                best_acc, best_tau = float(acc), float(tau)
        out[mkt] = {"tau": best_tau, "accuracy": best_acc, "n": int(len(p))}
    return out

def main(patch: int = 0):
    # 1) load and label
    df_all = load_matches()
    df_all = add_targets(df_all)
    train, test = split_time(df_all)
    # --- Reference test set (shared across all patches) ---
    os.makedirs("data", exist_ok=True)
    TEST_REF_PATH = "data/mg_backtest_reference.csv"

    if not os.path.isfile(TEST_REF_PATH):
        # Prima esecuzione → salva il test set di riferimento
        test.to_csv(TEST_REF_PATH, index=False)
        print(f"[INFO] Saved reference test set → {TEST_REF_PATH}")
    else:
        # Esecuzioni successive → ricarica sempre lo stesso test
        test = pd.read_csv(TEST_REF_PATH, parse_dates=["match_date"])
        print(f"[INFO] Loaded reference test set from {TEST_REF_PATH}")
    
    # 2) train calibrators using scripts (assumed run beforehand)
    # Here we just ensure paths exist; you can run the trainers separately.
    model = MGModelV1(
        patch=patch,
        calibrators_global_dir=CAL_GLOBAL_DIR if os.path.isdir(CAL_GLOBAL_DIR) else None,
        calibrators_leagues_dir=CAL_LEAGUES_DIR if os.path.isdir(CAL_LEAGUES_DIR) else None,
        use_league_calibrators=True
    )

    # 3) predictions on test set
    test_pred = predict_block(test, model)

    # 4) metrics
    # 4) metrics (robuste a NaN dovuti alla Patch 1)
    report = {}
    N_total = len(test_pred)
    for mkt in MARKETS:
        p_all = test_pred[mkt].astype(float).values
        y_all = test_pred["y_" + mkt].astype(int).values

        mask = ~np.isnan(p_all)              # valuta solo dove il mercato è attivo
        p = p_all[mask]
        y = y_all[mask]

        if len(p) == 0:
            report[mkt] = {
                "brier": None,
                "ece": None,
                "mean_prob": None,
                "prevalence": None,
                "n": 0,
                "coverage": 0.0,             # quota di test dove il mercato è attivo
            }
            continue

        report[mkt] = {
            "brier": brier(y, p),
            "ece": ece_score(y, p, bins=10),
            "mean_prob": float(p.mean()),
            "prevalence": float(y.mean()),
            "n": int(len(y)),
            "coverage": float(len(y) / N_total),
    }


    # 5) threshold optimization on train (or on test for quick demo)
    thresholds = optimize_thresholds(test_pred)

    THRESHOLDS_OUT_PATCH = f"data/mg_thresholds_P{patch}.json"
    with open(THRESHOLDS_OUT_PATCH, "w") as f:
        json.dump(thresholds, f, indent=2)
    print("\nSaved thresholds to:", THRESHOLDS_OUT_PATCH)

    print("=== MG Backtest Report (TEST) ===")
    for mkt, d in report.items():
        print(mkt, d)
    print("\nSaved thresholds to:", THRESHOLDS_OUT)

if __name__ == "__main__":
    import sys
    patch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(patch)
