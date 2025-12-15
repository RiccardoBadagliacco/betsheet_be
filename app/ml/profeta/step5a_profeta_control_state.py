#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5A — PROFETA CONTROL STATE (DATA-DRIVEN)

Classifica CHI controlla la partita
usando SOLO la fotografia Profeta
e soglie data-driven.

Output:
- step5a_profeta_control_state.parquet
- step5a_control_state_thresholds.json
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STEP3 = DATA_DIR / "step3_profeta_predictions.parquet"
OUT_PARQUET = DATA_DIR / "step5a_profeta_control_state.parquet"
OUT_JSON    = DATA_DIR / "step5a_control_state_thresholds.json"

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP5A — PROFETA CONTROL STATE (DATA-DRIVEN)")

    df = pd.read_parquet(STEP3)

    # solo storico
    if "is_fixture" in df.columns:
        df = df[df["is_fixture"] == False].copy()

    print(f"📦 Partite storiche: {len(df)}")

    # metriche
    df["lambda_diff"] = df["lambda_home"] - df["lambda_away"]

    # soglie data-driven
    thr = {
        "p1_dom": float(df["p1"].quantile(0.80)),
        "p2_dom": float(df["p2"].quantile(0.80)),
        "px_draw": float(df["px"].quantile(0.75)),
        "lambda_diff": float(df["lambda_diff"].abs().quantile(0.75)),
    }

    print("\n📊 SOGLIE CALCOLATE (PERCENTILI)")
    for k, v in thr.items():
        print(f"{k}: {v:.3f}")

    # classificazione
    def classify(row):
        if row["px"] >= thr["px_draw"] and abs(row["p1"] - row["p2"]) <= 0.10:
            return "DRAW_PRONE"
        if row["p1"] >= thr["p1_dom"] and row["lambda_diff"] >= thr["lambda_diff"]:
            return "HOME_DOMINANT"
        if row["p2"] >= thr["p2_dom"] and row["lambda_diff"] <= -thr["lambda_diff"]:
            return "AWAY_DOMINANT"
        return "BALANCED"

    df["control_state"] = df.apply(classify, axis=1)

    # distribuzione
    dist = df["control_state"].value_counts(normalize=True) * 100
    print("\n📊 DISTRIBUZIONE CONTROL STATE (%)")
    print(dist.round(2))

    # output parquet
    out_cols = [
        "match_id",
        "control_state",
        "p1", "px", "p2",
        "lambda_home",
        "lambda_away",
        "lambda_diff",
    ]
    df[out_cols].to_parquet(OUT_PARQUET, index=False)

    # output soglie JSON (🔥 fondamentale per runtime)
    with open(OUT_JSON, "w") as f:
        json.dump(thr, f, indent=2)

    print(f"\n💾 Salvato: {OUT_PARQUET}")
    print(f"💾 Salvato: {OUT_JSON}")
    print("✅ STEP5A COMPLETATO")

if __name__ == "__main__":
    main()