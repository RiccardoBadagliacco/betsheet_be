#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP6 — PROFETA STATE → MARKET BACKTEST

Valuta quanto i MATCH STATE
migliorano l'individuazione dei mercati:

- MG_HOME_1_4
- MG_AWAY_1_4
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STEP0  = DATA_DIR / "step0_profeta.parquet"
STEP5  = DATA_DIR / "step5_profeta_match_states.parquet"
OUT    = DATA_DIR / "step6_state_market_backtest.parquet"

# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 STEP6 — STATE → MARKET BACKTEST")

    df0 = pd.read_parquet(STEP0)
    df5 = pd.read_parquet(STEP5)

    df = df0.merge(df5, on="match_id", how="inner")

    # solo storico
    if "is_fixture" in df.columns:
        df = df[df["is_fixture"] == False].copy()

    print(f"📦 Partite analizzate: {len(df)}")

    # --------------------------------------------------
    # EVENTI REALI
    # --------------------------------------------------
    df["MG_HOME_1_4_REAL"] = (
        (df["home_goals"] >= 1) & (df["home_goals"] <= 4)
    )

    df["MG_AWAY_1_4_REAL"] = (
        (df["away_goals"] >= 1) & (df["away_goals"] <= 4)
    )

    # --------------------------------------------------
    # BASELINE GLOBALI
    # --------------------------------------------------
    base_home = df["MG_HOME_1_4_REAL"].mean()
    base_away = df["MG_AWAY_1_4_REAL"].mean()

    print("\n📊 BASELINE GLOBALI")
    print(f"MG HOME 1–4: {base_home:.2%}")
    print(f"MG AWAY 1–4: {base_away:.2%}")

    rows = []

    for state, g in df.groupby("match_state"):
        n = len(g)
        if n < 300:
            continue

        home_prec = g["MG_HOME_1_4_REAL"].mean()
        away_prec = g["MG_AWAY_1_4_REAL"].mean()

        rows.append({
            "match_state": state,
            "support": n,

            "home_precision": round(home_prec, 4),
            "home_lift": round(home_prec / base_home, 3),

            "away_precision": round(away_prec, 4),
            "away_lift": round(away_prec / base_away, 3),
        })

    out = pd.DataFrame(rows).sort_values(
        ["home_lift", "away_lift"],
        ascending=False
    )

    out.to_parquet(OUT, index=False)

    print("\n📊 RISULTATO PER MATCH STATE")
    print(out)

    print(f"\n💾 Salvato: {OUT}")
    print("✅ STEP6 COMPLETATO")

if __name__ == "__main__":
    main()