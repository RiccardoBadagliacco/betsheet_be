#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP6B — PROFETA STATE × STATE → MARKET PROFILE

Obiettivo:
    Profilare i mercati REALI osservati
    per ogni combinazione:
        (CONTROL_STATE, GOAL_STATE)

Nessuna decisione.
Solo statistica descrittiva.
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STEP0 = DATA_DIR / "step0_profeta.parquet"
STEP3 = DATA_DIR / "step3_profeta_predictions.parquet"
STEP5A = DATA_DIR / "step5a_profeta_control_state.parquet"
STEP5B = DATA_DIR / "step5b_profeta_goal_state.parquet"

OUT = DATA_DIR / "step6b_profeta_state_market_profile.parquet"

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP6B — STATE × STATE → MARKET PROFILE")

    df0 = pd.read_parquet(STEP0)
    df3 = pd.read_parquet(STEP3)
    df5a = pd.read_parquet(STEP5A)
    df5b = pd.read_parquet(STEP5B)

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------
    df = (
        df0
        .merge(df3, on="match_id", how="inner")
        .merge(df5a, on="match_id", how="inner")
        .merge(df5b, on="match_id", how="inner")
    )

    # FIX is_fixture
    if "is_fixture_x" in df.columns:
        df["is_fixture"] = df["is_fixture_x"]

    df = df[df["is_fixture"] == False].copy()
    print(f"📦 Partite storiche: {len(df)}")

    # --------------------------------------------------
    # EVENTI REALI
    # --------------------------------------------------
    df["HOME_WIN"] = (df["home_goals"] > df["away_goals"])
    df["DRAW"] = (df["home_goals"] == df["away_goals"])
    df["AWAY_WIN"] = (df["home_goals"] < df["away_goals"])

    df["OVER15"] = (df["home_goals"] + df["away_goals"] >= 2)
    df["UNDER15"] = ~df["OVER15"]

    df["OVER25"] = (df["home_goals"] + df["away_goals"] >= 3)
    df["UNDER25"] = ~df["OVER25"]
    

    df["MG_HOME_1_4"] = (df["home_goals"].between(1, 4))
    df["MG_AWAY_1_4"] = (df["away_goals"].between(1, 4))

    # --------------------------------------------------
    # AGGREGAZIONE
    # --------------------------------------------------
    rows = []

    for (cs, gs), g in df.groupby(["control_state", "goal_state"]):
        n = len(g)
        if n < 300:
            continue

        rows.append({
            "control_state": cs,
            "goal_state": gs,
            "support": n,
            "support_pct": round(n / len(df), 4),

            "HOME_WIN": round(g["HOME_WIN"].mean(), 4),
            "DRAW": round(g["DRAW"].mean(), 4),
            "AWAY_WIN": round(g["AWAY_WIN"].mean(), 4),

            "OVER15": round(g["OVER15"].mean(), 4),  
            "UNDER15": round(g["UNDER15"].mean(), 4),

            "OVER25": round(g["OVER25"].mean(), 4),
            "UNDER25": round(g["UNDER25"].mean(), 4),

            "MG_HOME_1_4": round(g["MG_HOME_1_4"].mean(), 4),
            "MG_AWAY_1_4": round(g["MG_AWAY_1_4"].mean(), 4),
        })

    out = pd.DataFrame(rows).sort_values(
        ["MG_HOME_1_4", "MG_AWAY_1_4"],
        ascending=False
    )

    out.to_parquet(OUT, index=False)

    print("\n📊 PROFILO STATI × MERCATI")
    print(out.head(10))

    print(f"\n💾 Salvato: {OUT}")
    print("✅ STEP6B COMPLETATO")

if __name__ == "__main__":
    main()