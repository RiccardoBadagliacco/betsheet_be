#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5B — PROFETA GOAL STATE (DATA-DRIVEN)

Classifica il RITMO GOL della partita
usando SOLO la fotografia Profeta.

Output:
- step5b_profeta_goal_state.parquet
- step5b_goal_state_thresholds.json
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
OUT_PARQUET = DATA_DIR / "step5b_profeta_goal_state.parquet"
OUT_JSON    = DATA_DIR / "step5b_goal_state_thresholds.json"

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP5B — PROFETA GOAL STATE (DATA-DRIVEN)")

    df = pd.read_parquet(STEP3)

    if "is_fixture" in df.columns:
        df = df[df["is_fixture"] == False].copy()

    print(f"📦 Partite storiche: {len(df)}")

    # soglie data-driven
    thr = {
        "xg_low":   float(df["xg_total"].quantile(0.25)),
        "xg_high":  float(df["xg_total"].quantile(0.75)),
        "xg_wild":  float(df["xg_total"].quantile(0.90)),
        "o25_high": float(df["p_over_2_5"].quantile(0.75)),
        "o35_wild": float(df["p_over_3_5"].quantile(0.85)),
        "u25_low":  float(df["p_under_2_5"].quantile(0.75)),
    }

    print("\n📊 SOGLIE CALCOLATE (PERCENTILI)")
    for k, v in thr.items():
        print(f"{k}: {v:.3f}")

    # classificazione
    def classify(row):
        if row["xg_total"] >= thr["xg_wild"] or row["p_over_3_5"] >= thr["o35_wild"]:
            return "WILD_GOALS"
        if row["xg_total"] <= thr["xg_low"] and row["p_under_2_5"] >= thr["u25_low"]:
            return "LOW_GOALS"
        if row["xg_total"] >= thr["xg_high"] or row["p_over_2_5"] >= thr["o25_high"]:
            return "HIGH_GOALS"
        return "MID_GOALS"

    df["goal_state"] = df.apply(classify, axis=1)

    # distribuzione
    dist = df["goal_state"].value_counts(normalize=True) * 100
    print("\n📊 DISTRIBUZIONE GOAL STATE (%)")
    print(dist.round(2))

    # output parquet
    out_cols = [
        "match_id",
        "goal_state",
        "xg_total",
        "p_over_2_5",
        "p_under_2_5",
        "p_over_3_5",
    ]
    df[out_cols].to_parquet(OUT_PARQUET, index=False)

    # output JSON soglie
    with open(OUT_JSON, "w") as f:
        json.dump(thr, f, indent=2)

    print(f"\n💾 Salvato: {OUT_PARQUET}")
    print(f"💾 Salvato: {OUT_JSON}")
    print("✅ STEP5B COMPLETATO")

if __name__ == "__main__":
    main()