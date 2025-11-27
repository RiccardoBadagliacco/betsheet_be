#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X1 — MASTER DATASET 1X2 v1

Costruisce un unico dataset per il meta-modello 1X2:

Input:
    - data/step2b_1x2_features_v2.parquet
    - data/step1c_dataset_with_elo_form.parquet
    - data/step3a_1x2_clusters_v2.parquet

Output:
    - data/meta_1x2_master_train_v1.parquet

Contiene:
    - tutte le feature 1X2 (FEATURES_1X2_V2)
    - target reali: home_ft, away_ft, is_home_win, is_draw, is_away_win, is_over25
    - cluster_1x2
"""

from pathlib import Path
import sys
import pandas as pd

# ------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]  # .../app/ml
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

FEAT_1X2_PATH   = DATA_DIR / "step2b_1x2_features_v2.parquet"
MASTER_STEP1C   = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
CLUST_1X2_PATH  = DATA_DIR / "step3a_1x2_clusters_v2.parquet"
OUT_MASTER_PATH = DATA_DIR / "meta_1x2_master_train_v1.parquet"


def main():
    print("===================================================")
    print("🚀 STEP X1 — MASTER DATASET 1X2 v1")
    print("===================================================")
    print(f"📥 Features 1X2 : {FEAT_1X2_PATH}")
    print(f"📥 Step1C master: {MASTER_STEP1C}")
    print(f"📥 Cluster 1X2  : {CLUST_1X2_PATH}")
    print(f"💾 Output       : {OUT_MASTER_PATH}")
    print("===================================================")

    # -----------------------------
    # 1) Carica feature 1X2
    # -----------------------------
    df_feat = pd.read_parquet(FEAT_1X2_PATH)
    print(f"📏 df_feat shape : {df_feat.shape}")

    if "match_id" not in df_feat.columns:
        raise RuntimeError("df_feat non contiene 'match_id'")

    # -----------------------------
    # 2) Carica master step1c (target reali)
    # -----------------------------
    df_step1c = pd.read_parquet(MASTER_STEP1C)
    print(f"📏 df_step1c shape: {df_step1c.shape}")

    if "match_id" not in df_step1c.columns:
        raise RuntimeError("df_step1c non contiene 'match_id'")

    # Seleziono solo colonne target + qualche meta utile
    cols_target = [
        "match_id",
        "home_ft", "away_ft",
        "is_home_win", "is_draw", "is_away_win",
        "is_over05", "is_over15", "is_over25", "is_over35",
        "total_goals",
    ]
    cols_target = [c for c in cols_target if c in df_step1c.columns]
    df_target = df_step1c[cols_target].copy()

    # -----------------------------
    # 3) Carica cluster_1x2
    # -----------------------------
    df_clu = pd.read_parquet(CLUST_1X2_PATH)
    print(f"📏 df_clu shape  : {df_clu.shape}")

    if "match_id" not in df_clu.columns or "cluster_1x2" not in df_clu.columns:
        raise RuntimeError("df_clu deve avere 'match_id' e 'cluster_1x2'")

    df_clu = df_clu[["match_id", "cluster_1x2"]].copy()

    # -----------------------------
    # 4) Merge: feat + target + cluster
    # -----------------------------
    df = df_feat.merge(df_target, on="match_id", how="left")
    print(f"📏 dopo merge target: {df.shape}")

    df = df.merge(df_clu, on="match_id", how="left")
    print(f"📏 dopo merge cluster: {df.shape}")

    # Info righe con/ senza esito (per capire match vs fixture)
    n_total = len(df)
    n_with_result = df["is_home_win"].notna().sum() if "is_home_win" in df.columns else 0
    print(f"✅ Righe con esito (train): {n_with_result}")
    print(f"🚫 Righe senza esito      : {n_total - n_with_result}")

    # -----------------------------
    # 5) Salva output
    # -----------------------------
    OUT_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_MASTER_PATH, index=False)
    print(f"💾 Salvato MASTER 1X2 in: {OUT_MASTER_PATH}")
    print("🏁 STEP X1 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
