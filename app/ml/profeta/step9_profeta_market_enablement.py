#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP9 — PROFETA MARKET ENABLEMENT

Decide quali mercati sono ABILITATI
per ogni match in base a:
- CONTROL_STATE
- GOAL_STATE
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CTRL = DATA_DIR / "step5a_profeta_control_state.parquet"
GOAL = DATA_DIR / "step5b_profeta_goal_state.parquet"
OUT  = DATA_DIR / "step9_profeta_market_enablement.parquet"

ENABLEMENT_MAP = {
    ("HOME_DOMINANT", "LOW_GOALS"): ["HOME_WIN", "MG_HOME_1_3", "UNDER_2_5"],
    ("HOME_DOMINANT", "MID_GOALS"): ["HOME_WIN", "MG_HOME_1_4"],
    ("HOME_DOMINANT", "HIGH_GOALS"): ["HOME_WIN", "MG_HOME_1_5", "OVER_2_5"],
    ("HOME_DOMINANT", "WILD_GOALS"): ["HOME_WIN", "MG_HOME_1_5", "OVER_3_5"],

    ("AWAY_DOMINANT", "LOW_GOALS"): ["AWAY_WIN", "MG_AWAY_1_3", "UNDER_2_5"],
    ("AWAY_DOMINANT", "MID_GOALS"): ["AWAY_WIN", "MG_AWAY_1_4"],
    ("AWAY_DOMINANT", "HIGH_GOALS"): ["AWAY_WIN", "MG_AWAY_1_5", "OVER_2_5"],

    ("BALANCED", "LOW_GOALS"): ["UNDER_2_5", "X"],
    ("BALANCED", "MID_GOALS"): ["MG_1_4"],
    ("BALANCED", "WILD_GOALS"): ["OVER_2_5", "MG_1_4"],

    ("DRAW_PRONE", "LOW_GOALS"): ["X", "UNDER_2_5"],
    ("DRAW_PRONE", "MID_GOALS"): ["X", "MG_1_4"],
}

def main():
    print("🚀 STEP9 — MARKET ENABLEMENT")

    df_ctrl = pd.read_parquet(CTRL)
    df_goal = pd.read_parquet(GOAL)

    df = df_ctrl.merge(df_goal, on="match_id", how="inner")

    rows = []

    for _, r in df.iterrows():
        key = (r["control_state"], r["goal_state"])
        markets = ENABLEMENT_MAP.get(key, [])

        rows.append({
            "match_id": r["match_id"],
            "control_state": r["control_state"],
            "goal_state": r["goal_state"],
            "enabled_markets": markets
        })

    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)

    print(f"💾 Salvato: {OUT}")
    print("✅ STEP9 COMPLETATO")

if __name__ == "__main__":
    main()