#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP4B — AFFINI INDEX SLIM V2

Deriva una versione "slim" dell'indice affini a partire dal WIDE,
pensata per l'uso runtime:

    - lookup affini per match_id
    - filtri rapidi (cluster + elo + lambda)
    - serializzazione leggera

Input:
    - step4a_affini_index_wide_v2.parquet

Output:
    - step4b_affini_index_slim_v2.parquet
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


# Colonne slim scelte in modo da:
#   - coprire target
#   - cluster
#   - probabilità base
#   - driver tecnici principali (elo, lambda)
AFFINI_SLIM_COLS = [
    # meta
    "match_id",
    "date",
    "season",
    "league",
    "home_team",
    "away_team",

    # target principali
    "is_home_win",
    "is_away_win",
    "is_draw",
    "is_over25",
    "is_over15",

    # 1X2 probabilità e cluster
    "bk_p1", "bk_px", "bk_p2",
    "pic_p1", "pic_px", "pic_p2",
    "cluster_1x2",

    # OU2.5 probabilità e cluster (quando disponibili)
    "bk_pO25", "bk_pU25",
    "pic_pO25", "pic_pU25",
    "cluster_ou25",

    # OU1.5 (solo tecnico/picchetto + cluster)
    "pic_pO15", "pic_pU15",
    "cluster_ou15",

    # driver tecnici principali
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "lambda_home_form", "lambda_away_form", "lambda_total_form",
    "lambda_total_market_ou25",
    "goal_supremacy_market_ou25",
    "goal_supremacy_form",

    # posizionamento temporale
    "season_recency",
    "tightness_index"
]


def main():
    print("====================================================")
    print("🚀 STEP4B — AFFINI INDEX SLIM V2")
    print("====================================================")
    print(f"📥 Input WIDE: {WIDE_FILE}")
    print(f"💾 Output SLIM: {SLIM_FILE}")

    df = pd.read_parquet(WIDE_FILE)
    print(f"📏 Shape WIDE: {df.shape}")

    # Verifica colonne disponibili
    available = [c for c in AFFINI_SLIM_COLS if c in df.columns]
    missing   = [c for c in AFFINI_SLIM_COLS if c not in df.columns]

    if missing:
        print(f"⚠️ Alcune colonne SLIM non presenti nel WIDE: {missing}")
        print("   Procedo comunque usando solo le colonne disponibili.")

    df_slim = df[available].copy()

    # Ordino facoltativamente per data (comodo per debug e affini cronologici)
    if "date" in df_slim.columns:
        df_slim = df_slim.sort_values("date")

    print(f"📏 Shape SLIM: {df_slim.shape}")
    SLIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_slim.to_parquet(SLIM_FILE, index=False)
    print(f"💾 Salvato SLIM index: {SLIM_FILE}")
    print("🏁 STEP4B AFFINI INDEX SLIM V2 COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()