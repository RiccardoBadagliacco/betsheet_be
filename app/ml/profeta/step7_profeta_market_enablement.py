#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP7 — PROFETA MARKET ENABLEMENT (DATA-DRIVEN)

Obiettivo:
    Stabilire QUALI mercati hanno senso
    per ogni coppia:
        CONTROL_STATE × GOAL_STATE

Input:
    step6b_profeta_state_market_profile.parquet

Output:
    step7_profeta_market_enablement.parquet
"""

from pathlib import Path
import pandas as pd
import json

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INP = DATA_DIR / "step6b_profeta_state_market_profile.parquet"
OUT = DATA_DIR / "step7_profeta_market_enablement.parquet"

# ============================================================
# CONFIG
# ============================================================

MIN_SUPPORT = 300
MIN_LIFT = 1.05
MIN_EDGE = 0.05   # +5% sopra baseline

# Mercati da considerare
MARKETS = {
    "HOME_WIN": "HOME_WIN",
    "DRAW": "DRAW",
    "AWAY_WIN": "AWAY_WIN",
    "OVER25": "OVER25",
    "UNDER25": "UNDER25",

    "MG_HOME_1_4": "MG_HOME_1_4",
    "MG_AWAY_1_4": "MG_AWAY_1_4",
}

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP7 — PROFETA MARKET ENABLEMENT")

    df = pd.read_parquet(INP)
    print(f"📦 Profili caricati: {len(df)}")

    enabled_rows = []

    # --------------------------------------------------
    # Baseline globali
    # --------------------------------------------------
    baseline = {}
    for m in MARKETS:
        baseline[m] = df[m].mean()

    # --------------------------------------------------
    # Analisi per stato × stato
    # --------------------------------------------------
    for _, row in df.iterrows():
        support = row["support"]

        if support < MIN_SUPPORT:
            continue

        enabled = []

        for m in MARKETS:
            p = row[m]
            base = baseline[m]

            lift = p / base if base > 0 else 0
            edge = p - base

            if lift >= MIN_LIFT or edge >= MIN_EDGE:
                enabled.append(m)

        if enabled:
            enabled_rows.append({
                "control_state": row["control_state"],
                "goal_state": row["goal_state"],
                "support": int(support),
                "enabled_markets": json.dumps(enabled),
            })

    out_df = pd.DataFrame(enabled_rows)

    out_df.to_parquet(OUT, index=False)

    print(f"💾 Salvato: {OUT}")
    print(f"📊 Profili con mercati abilitati: {len(out_df)}")
    print("✅ STEP7 COMPLETATO")


if __name__ == "__main__":
    main()