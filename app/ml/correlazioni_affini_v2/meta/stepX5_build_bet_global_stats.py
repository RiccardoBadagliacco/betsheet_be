#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X5 — Build Global Betting Stats (simple version)

Genera:
    data/step5_bet_global_stats.json

Contiene statistiche globali stabili, utili per:
    - favorite profile v2
    - over/under profile v2
    - betting heuristics in stepZ_decision_engine.py

Input:
    - meta_1x2_backtest_v1.parquet
    - meta_1x2_backtest_ev_v1.parquet

Output:
    - step5_bet_global_stats.json
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

BACKTEST_META = DATA_DIR / "meta_1x2_backtest_v1.parquet"
BACKTEST_EV   = DATA_DIR / "meta_1x2_backtest_ev_v1.parquet"
OUT_PATH      = DATA_DIR / "step5_bet_global_stats.json"


# ============================================================
# HELPERS
# ============================================================
def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return None
    return float(s.mean())


def range_bucket(p, bins):
    """ Converte la probabilità favorita in range es. 0.55-0.60 """
    for low, high in bins:
        if low <= p < high:
            return f"{low:.2f}_{high:.2f}"
    return "other"


# ============================================================
# MAIN
# ============================================================
def main():
    print("===================================================")
    print("🚀 STEP X5 — BUILD BET GLOBAL STATS (simple)")
    print("===================================================")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------
    print(f"📥 BACKTEST_META: {BACKTEST_META}")
    print(f"📥 BACKTEST_EV  : {BACKTEST_EV}")

    df_meta = pd.read_parquet(BACKTEST_META)
    df_ev   = pd.read_parquet(BACKTEST_EV)

    # merge (safe)
    df = df_meta.merge(df_ev[["match_id", "ev_home","ev_draw","ev_away"]], 
                       on="match_id", how="left")

    print(f"📏 merged df shape: {df.shape}")

    # Only matches with real outcome
    df = df[df["is_home_win"].notna()].copy()

    # encode outcome 0/1/2
    def enc(r):
        if r["is_home_win"] == 1: return 2
        if r["is_draw"] == 1:     return 1
        if r["is_away_win"] == 1: return 0
        return np.nan

    df["y"] = df.apply(enc, axis=1).astype(int)

    # favorite prob from bookmaker
    df["fav_prob"] = df[["bk_p1", "bk_px", "bk_p2"]].max(axis=1)
    df["fav_side"] = df[["bk_p1", "bk_px", "bk_p2"]].idxmax(axis=1).str.replace("bk_p","")

    # bins prob favorita
    bins = [
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.80),
        (0.80, 1.00),
    ]
    df["fav_prob_bucket"] = df["fav_prob"].apply(lambda p: range_bucket(p, bins))

    # cluster exists?
    has_cluster = "cluster_1x2" in df.columns

    # ============================================================
    # GLOBAL BASELINES
    # ============================================================
    print("📊 Building global baselines...")

    global_stats = {
        "win_rate_home": float((df["y"] == 2).mean()),
        "win_rate_draw": float((df["y"] == 1).mean()),
        "win_rate_away": float((df["y"] == 0).mean()),
        "avg_goals_home": safe_mean(df["home_ft"]),
        "avg_goals_away": safe_mean(df["away_ft"]),
        "avg_total_goals": safe_mean(df["total_goals"]),
    }

    # ============================================================
    # FAVORITE PERFORMANCE BY BUCKET
    # ============================================================
    print("📊 Building favorite performance...")

    fav_stats = {}
    for bucket in sorted(df["fav_prob_bucket"].unique()):
        sub = df[df["fav_prob_bucket"] == bucket]
        if len(sub) < 50:
            continue
        fav_stats[bucket] = {
            "n": len(sub),
            "win_rate_fav": float((sub["y"] == sub["fav_side"].map({"1":2,"X":1,"2":0})).mean()),
            "avg_ev_home": safe_mean(sub["ev_home"]),
            "avg_ev_draw": safe_mean(sub["ev_draw"]),
            "avg_ev_away": safe_mean(sub["ev_away"]),
        }

    # ============================================================
    # UNDER/OVER BASELINES
    # ============================================================
    print("📊 Building OU baselines...")

    ou_stats = {
        "over25_rate": float(df["is_over25"].mean()),
        "over15_rate": float(df["is_over15"].mean()),
        "over35_rate": float(df["is_over35"].mean()),
        "under25_rate": float((df["total_goals"] < 3).mean()),
    }

    # ============================================================
    # CLUSTER STATS (if available)
    # ============================================================
    cluster_stats = {}
    if has_cluster:
        print("📊 Building cluster stats...")
        for cl in sorted(df["cluster_1x2"].dropna().unique()):
            sub = df[df["cluster_1x2"] == cl]
            if len(sub) < 40:
                continue
            cluster_stats[int(cl)] = {
                "n": len(sub),
                "home_rate": float((sub["y"]==2).mean()),
                "draw_rate": float((sub["y"]==1).mean()),
                "away_rate": float((sub["y"]==0).mean()),
                "avg_total_goals": safe_mean(sub["total_goals"]),
            }

    # ============================================================
    # FINAL OBJECT
    # ============================================================
    BET_GLOBAL_STATS = {
        "global": global_stats,
        "favorite": fav_stats,
        "ou": ou_stats,
        "cluster": cluster_stats,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        json.dump(BET_GLOBAL_STATS, f, indent=2)

    print("---------------------------------------------------")
    print(f"💾 Salvato bet_global_stats in: {OUT_PATH}")
    print("🏁 STEP X5 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()