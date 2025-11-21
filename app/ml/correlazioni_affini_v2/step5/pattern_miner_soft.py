#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pattern_miner_soft.py

Cerca pattern del tipo:
  - soft_pO25 >= t1
  - soft_p1 >= t1 AND soft_pO15 >= t2
  - soft_pO25 >= t1 AND soft_pO15 >= t2

E valuta:
  - N match
  - % Over 1.5 reale
  - % Over 2.5 reale
  - % GG reale

Puoi estendere facilmente con nuove combinazioni.
"""

import pandas as pd
from pathlib import Path
from itertools import product

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"


def evaluate_condition(df: pd.DataFrame, mask, name: str, min_n: int = 200):
    sub = df[mask]
    n = len(sub)
    if n < min_n:
        return None

    return {
        "pattern": name,
        "n": n,
        "over15_rate": sub["is_over15_real"].mean(),
        "over25_rate": sub["is_over25_real"].mean(),
        "gg_rate": sub["is_gg_real"].mean(),
        "soft_p1_mean": sub["soft_p1"].mean(),
        "soft_pO15_mean": sub["soft_pO15"].mean(),
        "soft_pO25_mean": sub["soft_pO25"].mean(),
    }


def main():
    df = pd.read_parquet(SOFT_FILE)

    print("======================================")
    print("🧠 PATTERN MINER – SOFT ENGINE")
    print("======================================")
    print(f"Totale match in history: {len(df)}\n")

    results = []

    # -------------------------------
    # 1) Pattern su soft_pO25 alone
    # -------------------------------
    thrs_O25 = [0.55, 0.60, 0.65, 0.70, 0.75]

    for t in thrs_O25:
        mask = df["soft_pO25"] >= t
        name = f"soft_pO25 >= {t:.2f}"
        r = evaluate_condition(df, mask, name)
        if r:
            results.append(r)

    # -------------------------------
    # 2) soft_p1 & soft_pO15 → GG
    # -------------------------------
    thrs_p1  = [0.55, 0.60, 0.65, 0.70]
    thrs_O15 = [0.60, 0.65, 0.70, 0.75]

    for t1, t2 in product(thrs_p1, thrs_O15):
        mask = (df["soft_p1"] >= t1) & (df["soft_pO15"] >= t2)
        name = f"soft_p1 >= {t1:.2f} & soft_pO15 >= {t2:.2f}"
        r = evaluate_condition(df, mask, name)
        if r:
            results.append(r)

    # -------------------------------
    # 3) soft_pO25 & soft_pO15 → Over 1.5
    # -------------------------------
    for t1, t2 in product(thrs_O25, thrs_O15):
        mask = (df["soft_pO25"] >= t1) & (df["soft_pO15"] >= t2)
        name = f"soft_pO25 >= {t1:.2f} & soft_pO15 >= {t2:.2f}"
        r = evaluate_condition(df, mask, name)
        if r:
            results.append(r)

    # -------------------------------
    # 4) Output ordinato
    #    Possiamo ordinare per gg_rate, over15_rate, ecc.
    # -------------------------------
    res_df = pd.DataFrame(results)

    if res_df.empty:
        print("⚠️ Nessun pattern con N sufficiente trovato.")
        return

    print("\n🔝 Pattern ordinati per % GG:")
    print(
        res_df.sort_values("gg_rate", ascending=False)[
            ["pattern", "n", "gg_rate", "over15_rate", "over25_rate"]
        ].head(20)
    )

    print("\n🔝 Pattern ordinati per % Over 1.5:")
    print(
        res_df.sort_values("over15_rate", ascending=False)[
            ["pattern", "n", "over15_rate", "over25_rate", "gg_rate"]
        ].head(20)
    )

    print("\n🔝 Pattern ordinati per % Over 2.5:")
    print(
        res_df.sort_values("over25_rate", ascending=False)[
            ["pattern", "n", "over25_rate", "over15_rate", "gg_rate"]
        ].head(20)
    )

    # Opzionale: salva su parquet/csv
    out_path = DATA_DIR / "step5_pattern_miner_results.parquet"
    res_df.to_parquet(out_path, index=False)
    print(f"\n💾 Salvati risultati completi in: {out_path}")


if __name__ == "__main__":
    main()