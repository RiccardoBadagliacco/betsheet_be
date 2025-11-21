#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
soft_queries_basics.py

Query di base su step5_soft_history.parquet per capire:
  - calibrazione soft_pO25, soft_pO15, soft_p1
  - relazioni tipo:
      * soft_pO25 >= 0.60 → % Over 1.5 reale
      * soft_p1 >= a & soft_pO15 >= b → % GG reale
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"


def main():
    df = pd.read_parquet(SOFT_FILE)

    print("=========================================")
    print("🔍 SOFT QUERIES – DIAGNOSTICA DI BASE")
    print("=========================================")
    print(f"Totale righe in soft_history: {len(df)}\n")

    # 1) Distribuzione generale di Over 1.5 / Over 2.5 / GG
    base_over15 = df["is_over15_real"].mean()
    base_over25 = df["is_over25_real"].mean()
    base_gg     = df["is_gg_real"].mean()

    print("⚙️ Frequenze base (su tutto il dataset):")
    print(f"  - Over 1.5  : {base_over15:.3f}")
    print(f"  - Over 2.5  : {base_over25:.3f}")
    print(f"  - GG        : {base_gg:.3f}")
    print()

    # 2) Esempio 1:
    #    soft_pO25 >= 0.60 → quante volte finisce Over 1.5?
    thr = 0.60
    subset = df[df["soft_pO25"] >= thr]
    if len(subset) > 0:
        over15_rate = subset["is_over15_real"].mean()
        over25_rate = subset["is_over25_real"].mean()
        gg_rate     = subset["is_gg_real"].mean()
        print(f"🎯 Condizione: soft_pO25 >= {thr:.2f}")
        print(f"  N match           : {len(subset)}")
        print(f"  % Over 1.5 reale  : {over15_rate:.3f}")
        print(f"  % Over 2.5 reale  : {over25_rate:.3f}")
        print(f"  % GG reale        : {gg_rate:.3f}")
        print()
    else:
        print(f"🎯 Condizione: soft_pO25 >= {thr:.2f} → N=0\n")

    # 3) Esempio 2:
    #    soft_p1 >= a e soft_pO15 >= b → % GG reale
    thr1 = 0.55
    thrO15 = 0.65
    subset2 = df[(df["soft_p1"] >= thr1) & (df["soft_pO15"] >= thrO15)]
    if len(subset2) > 0:
        gg_rate = subset2["is_gg_real"].mean()
        print(f"🎯 Condizione: soft_p1 >= {thr1:.2f} AND soft_pO15 >= {thrO15:.2f}")
        print(f"  N match           : {len(subset2)}")
        print(f"  % GG reale        : {gg_rate:.3f}")
        print(f"  % Over 1.5 reale  : {subset2['is_over15_real'].mean():.3f}")
        print(f"  % Over 2.5 reale  : {subset2['is_over25_real'].mean():.3f}")
        print()
    else:
        print(f"🎯 Condizione: soft_p1 >= {thr1:.2f} AND soft_pO15 >= {thrO15:.2f} → N=0\n")

    # 4) Binning per calibrazione soft_pO25
    print("📊 Calibrazione soft_pO25 in decili:")
    df["bin_pO25"] = pd.qcut(df["soft_pO25"], 10, labels=False, duplicates="drop")
    calib = df.groupby("bin_pO25").agg(
        soft_pO25_mean=("soft_pO25", "mean"),
        over25_real=("is_over25_real", "mean"),
        n=("match_id", "count"),
    )
    print(calib)
    print()


if __name__ == "__main__":
    main()