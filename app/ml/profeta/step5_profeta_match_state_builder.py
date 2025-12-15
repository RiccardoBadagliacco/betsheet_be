#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — PROFETA MATCH STATE BUILDER

Obiettivo:
    Trasformare la fotografia statistica Profeta
    in uno stato di partita leggibile (1 per match)

Input:
    data/step3_profeta_predictions.parquet

Output:
    data/step5_profeta_match_states.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STEP3 = DATA_DIR / "step3_profeta_predictions.parquet"
OUT   = DATA_DIR / "step5_profeta_match_states.parquet"

# ============================================================
# STATE LOGIC
# ============================================================

def classify_match(row) -> str:
    """
    Ritorna UNO stato principale per la partita
    basato SOLO sulla fotografia Profeta.
    """

    xg = row["xg_total"]
    p1 = row["p1"]
    p2 = row["p2"]
    px = row["px"]

    lam_h = row["lambda_home"]
    lam_a = row["lambda_away"]

    # --------------------------------------------------
    # 1) LOW TEMPO / CHIUSA
    # --------------------------------------------------
    if xg <= 2.1 and row["p_under_2_5"] >= 0.65:
        return "LOW_TEMPO_CLOSED"

    # --------------------------------------------------
    # 2) DRAW TRAP
    # --------------------------------------------------
    if px >= 0.32 and abs(p1 - p2) <= 0.10:
        return "DRAW_TRAP"

    # --------------------------------------------------
    # 3) HOME CONTROL
    # --------------------------------------------------
    if (lam_h - lam_a) >= 0.60 and p1 >= 0.50 and px <= 0.28:
        return "HOME_CONTROL"

    # --------------------------------------------------
    # 4) AWAY CONTROL
    # --------------------------------------------------
    if (lam_a - lam_h) >= 0.60 and p2 >= 0.50 and px <= 0.28:
        return "AWAY_CONTROL"

    # --------------------------------------------------
    # 5) OPEN GAME
    # --------------------------------------------------
    if xg >= 2.6 and row["p_over_2_5"] >= 0.55 and px <= 0.30:
        return "OPEN_GAME"

    # --------------------------------------------------
    # 6) WILD MATCH
    # --------------------------------------------------
    if xg >= 3.2 and row["p_over_3_5"] >= 0.45:
        return "WILD_MATCH"

    # --------------------------------------------------
    # 7) BALANCED DEFAULT
    # --------------------------------------------------
    return "BALANCED_GAME"

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP5 — PROFETA MATCH STATE BUILDER")

    df = pd.read_parquet(STEP3)

    # usa SOLO storico
    if "is_fixture" in df.columns:
        df = df[df["is_fixture"] == False].copy()

    print(f"📦 Partite analizzate: {len(df)}")

    df["match_state"] = df.apply(classify_match, axis=1)

    # distribuzione stati
    dist = df["match_state"].value_counts(normalize=True) * 100
    print("\n📊 DISTRIBUZIONE STATI (%)")
    print(dist.round(2))

    out_cols = [
        "match_id",
        "match_state",
        "lambda_home",
        "lambda_away",
        "xg_total",
        "p1", "px", "p2",
        "p_over_2_5", "p_under_2_5",
        "p_mg_home_1_4",
        "p_mg_away_1_4",
        "p_mg_1_4",
    ]

    df[out_cols].to_parquet(OUT, index=False)

    print(f"\n💾 Salvato: {OUT}")
    print("✅ STEP5 COMPLETATO")

if __name__ == "__main__":
    main()