#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP4B — AFFINI INDEX SLIM V2 (VERSIONE FINALE)

Versione light del WIDE, ottimizzata per:
    - runtime
    - lookup affini
    - sistemi veloci
"""

import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

WIDE_FILE = DATA_DIR / "step4a_affini_index_wide_v2.parquet"
SLIM_FILE = DATA_DIR / "step4b_affini_index_slim_v2.parquet"


AFFINI_SLIM_COLS = [
    "match_id", "date", "season", "league", "home_team", "away_team",
    "is_home_win", "is_away_win", "is_draw", "is_over25", "is_over15",

    # 1X2
    "bk_p1", "bk_px", "bk_p2",
    "pic_p1", "pic_px", "pic_p2",
    "cluster_1x2",

    # OU2.5
    "bk_pO25", "bk_pU25",
    "pic_pO25", "pic_pU25",
    "cluster_ou25",

    # OU1.5
    "pic_pO15", "pic_pU15",
    "cluster_ou15",

    # driver
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "lambda_home_form", "lambda_away_form", "lambda_total_form",
    "lambda_total_market_ou25",
    "goal_supremacy_market_ou25",
    "goal_supremacy_form",

    "season_recency",
    "tightness_index",
]


def main():
    print("====================================================")
    print("🚀 STEP4B — AFFINI SLIM INDEX V2")
    print("====================================================")

    df = pd.read_parquet(WIDE_FILE)
    print(f"📏 Shape WIDE: {df.shape}")

    available = [c for c in AFFINI_SLIM_COLS if c in df.columns]
    missing   = [c for c in AFFINI_SLIM_COLS if c not in df.columns]

    if missing:
        print(f"⚠️ Mancano alcune colonne: {missing}")
        print("   Uso solo quelle disponibili.")

    df_slim = df[available].copy()

    if "date" in df_slim.columns:
        df_slim = df_slim.sort_values("date")

    df_slim.to_parquet(SLIM_FILE, index=False)
    print(f"💾 Salvato SLIM index in: {SLIM_FILE}")
    print("🏁 STEP4B COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()