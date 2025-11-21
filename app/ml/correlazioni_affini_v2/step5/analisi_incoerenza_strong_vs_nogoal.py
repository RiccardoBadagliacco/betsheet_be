#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisi: quando HOME_STRONG_1 o AWAY_SOLID_2 (pattern fortissimi)
falliscono nella realtà perché la favorita NON segna.

Scopo:
    • identificare condizioni storiche in cui i pattern 1/2 saltano
    • valutare se la doppia (1X o X2) è molto più sicura
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "correlazioni_affini_v2" / "data"

SOFT_FILE = DATA / "step5_soft_history.parquet"
WIDE_FILE = DATA / "step4a_affini_index_wide_v2.parquet"

def f(x):
    try: return float(x)
    except: return np.nan

def is_home_strong_1(r):
    pic_p1, pic_px, pic_p2 = f(r["pic_p1"]), f(r["pic_px"]), f(r["pic_p2"])
    bk_p1,  bk_px,  bk_p2  = f(r["bk_p1"]), f(r["bk_px"]), f(r["bk_p2"])
    return (pic_p1 > pic_px > pic_p2) and (bk_p1 > bk_px > bk_p2) and pic_p1 >= 0.65

def is_away_solid_2(r):
    pic_p1, pic_px, pic_p2 = f(r["pic_p1"]), f(r["pic_px"]), f(r["pic_p2"])
    bk_p1,  bk_px,  bk_p2  = f(r["bk_p1"]), f(r["bk_px"]), f(r["bk_p2"])
    return (pic_p2 > pic_px > pic_p1) and (bk_p2 > bk_px > bk_p1) and pic_p2 >= 0.70

def main():
    print("====================================================")
    print("🚀 ANALISI strong_1 / solid_2 → favorita NON segna")
    print("====================================================")

    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    df = pd.merge(
        wide[[
            "match_id","home_ft","away_ft",
            "bk_p1","bk_px","bk_p2",
            "pic_p1","pic_px","pic_p2",
            "tightness_index","lambda_total_form"
        ]],
        soft[["match_id"]],
        on="match_id",
        how="left"
    )

    df["HOME_STRONG_1"] = df.apply(is_home_strong_1, axis=1)
    df["AWAY_SOLID_2"]  = df.apply(is_away_solid_2, axis=1)

    # HOME_SCENARIO: forte 1 ma la favorita NON segna
    home_fail = df[df["HOME_STRONG_1"] & (df["home_ft"] == 0)]

    # AWAY_SCENARIO: forte 2 ma favorita away NON segna
    away_fail = df[df["AWAY_SOLID_2"] & (df["away_ft"] == 0)]

    # ----------------------------------------------------
    print("\n====================================================")
    print("📊 HOME_STRONG_1 che fallisce (favorita home NON segna)")
    print(f"Campione: {len(home_fail)}")

    if len(home_fail):
        print("P(X)         =", (home_fail["home_ft"]==home_fail["away_ft"]).mean())
        print("P(2)         =", (home_fail["home_ft"]<home_fail["away_ft"]).mean())
        print("P(1X)        =", (home_fail["home_ft"]>=home_fail["away_ft"]).mean())
        print("Under 2.5    =", (home_fail["home_ft"]+home_fail["away_ft"]<3).mean())
        print("Under 1.5    =", (home_fail["home_ft"]+home_fail["away_ft"]<2).mean())
        print("0–0          =", ((home_fail['home_ft']==0)&(home_fail['away_ft']==0)).mean())

    # ----------------------------------------------------
    print("\n====================================================")
    print("📊 AWAY_SOLID_2 che fallisce (favorita away NON segna)")
    print(f"Campione: {len(away_fail)}")

    if len(away_fail):
        print("P(X)         =", (away_fail["home_ft"]==away_fail["away_ft"]).mean())
        print("P(1)         =", (away_fail["home_ft"]>away_fail["away_ft"]).mean())
        print("P(X2)        =", (away_fail["away_ft"]>=away_fail["home_ft"]).mean())
        print("Under 2.5    =", (away_fail["home_ft"]+away_fail["away_ft"]<3).mean())
        print("Under 1.5    =", (away_fail["home_ft"]+away_fail["away_ft"]<2).mean())
        print("0–0          =", ((away_fail['home_ft']==0)&(away_fail['away_ft']==0)).mean())

    print("\n====================================================")
    print("🏁 ANALISI COMPLETATA")
    print("====================================================")

if __name__ == "__main__":
    main()