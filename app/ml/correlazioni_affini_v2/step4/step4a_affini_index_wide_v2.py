#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP4A — AFFINI INDEX WIDE V2 (VERSIONE DEFINITIVA E ROBUSTA)

- Merge di master + feature 1X2 / OU25 / OU15 + cluster
- Eliminazione sicura delle colonne duplicate
- Fix completo tightness_index
- Nessuna regressione nei merge
"""

import pandas as pd
from pathlib import Path
import sys

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

MASTER_FILE       = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
FEAT_1X2_FILE     = DATA_DIR / "step2b_1x2_features_v2.parquet"
FEAT_OU25_FILE    = DATA_DIR / "step2b_ou25_features_v2.parquet"
FEAT_OU15_FILE    = DATA_DIR / "step2b_ou15_features_v2.parquet"
CLUST_1X2_FILE    = DATA_DIR / "step3a_1x2_clusters_v2.parquet"
CLUST_OU25_FILE   = DATA_DIR / "step3b_ou25_clusters_v2.parquet"
CLUST_OU15_FILE   = DATA_DIR / "step3c_ou15_clusters_v2.parquet"

OUT_WIDE_FILE     = DATA_DIR / "step4a_affini_index_wide_v2.parquet"


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------

def clean_features(df_feat: pd.DataFrame, master_cols: set) -> pd.DataFrame:
    """
    Rimuove colonne duplicate del master MAI rimuovendo i picchetti.
    """
    ALWAYS_KEEP = [
        "pic_p1", "pic_px", "pic_p2",
        "pic_pO25", "pic_pU25",
        "pic_pO15", "pic_pU15",
        "pic_pO05", "pic_pU05"
    ]

    cols = ["match_id"]

    for c in df_feat.columns:
        if c == "match_id":
            continue
        if c in ALWAYS_KEEP:
            cols.append(c)
            continue
        if c not in master_cols:
            cols.append(c)

    return df_feat[cols]

def fix_tightness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza tightness_index dopo merge multipli.
    Mantiene solo tightness_index pulita.
    """
    cols = [c for c in df.columns if c.startswith("tightness_index")]
    if cols == ["tightness_index"]:
        return df

    df["tightness_index"] = df[cols].mean(axis=1)

    for c in cols:
        if c != "tightness_index":
            df = df.drop(columns=c)

    return df


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("====================================================")
    print("🚀 STEP4A — AFFINI INDEX WIDE (VERSIONE DEFINITIVA)")
    print("====================================================")

    # --------------------------------------------------
    # Caricamento dataset
    # --------------------------------------------------
    master = pd.read_parquet(MASTER_FILE).drop_duplicates("match_id")
    f1x2   = pd.read_parquet(FEAT_1X2_FILE)
    fou25  = pd.read_parquet(FEAT_OU25_FILE)
    fou15  = pd.read_parquet(FEAT_OU15_FILE)

    c1x2   = pd.read_parquet(CLUST_1X2_FILE)[["match_id", "cluster_1x2"]]
    cou25  = pd.read_parquet(CLUST_OU25_FILE)[["match_id", "cluster_ou25"]]
    cou15  = pd.read_parquet(CLUST_OU15_FILE)[["match_id", "cluster_ou15"]]

    print(f"📏 master: {master.shape}")
    print(f"📏 feat1x2: {f1x2.shape}")
    print(f"📏ou25: {fou25.shape}")
    print(f"📏ou15: {fou15.shape}")
    print(f"📏c1x2: {c1x2.shape}  | c25: {cou25.shape}  | c15: {cou15.shape}")

    # --------------------------------------------------
    # Pulizia feature (drop colonne già presenti nel master)
    # --------------------------------------------------
    master_cols = set(master.columns)

    f1x2_clean  = clean_features(f1x2, master_cols)
    fou25_clean = clean_features(fou25, master_cols)
    fou15_clean = clean_features(fou15, master_cols)

    # --------------------------------------------------
    # MERGE FINALE
    # --------------------------------------------------
    df = master

    df = df.merge(f1x2_clean,  on="match_id", how="left")
    df = df.merge(fou25_clean, on="match_id", how="left")
    df = df.merge(fou15_clean, on="match_id", how="left")

    df = df.merge(c1x2,  on="match_id", how="left")
    df = df.merge(cou25, on="match_id", how="left")
    df = df.merge(cou15, on="match_id", how="left")

    print(f"📏 Shape dopo merge: {df.shape}")

    # --------------------------------------------------
    # FIX tightness_index
    # --------------------------------------------------
    df = fix_tightness(df)

    # --------------------------------------------------
    # Flag mercati disponibili
    # --------------------------------------------------
    df["has_ou25"] = df["cluster_ou25"].notna().astype(int)
    df["has_ou15"] = df["cluster_ou15"].notna().astype(int)

    # --------------------------------------------------
    # Verifica colonne critiche per SLIM
    # --------------------------------------------------
    print("🔍 Check colonne chiave per SLIM:")
    key_cols = [
        "bk_p1", "bk_px", "bk_p2",
        "bk_pO25", "bk_pU25",
        "pic_pO15", "pic_pU15",
        "elo_home_pre", "elo_away_pre", "elo_diff",
        "cluster_1x2", "cluster_ou25", "cluster_ou15",
        "tightness_index"
    ]

    for c in key_cols:
        print(f"  - {c}: {'OK' if c in df.columns else 'MISSING'}")

    # --------------------------------------------------
    # Salvataggio
    # --------------------------------------------------
    OUT_WIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_WIDE_FILE, index=False)

    print(f"💾 Salvato: {OUT_WIDE_FILE}")
    print("====================================================")


if __name__ == "__main__":
    main()