#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP4A — AFFINI INDEX WIDE V2

Costruisce un indice "wide" per le correlazioni affini, pensato per:
    - analisi offline
    - diagnostica
    - sviluppo di nuove metriche

Input:
    - step1c_dataset_with_elo_form.parquet         (master: target + elo + form)
    - step2b_1x2_features_v2.parquet               (feature avanzate 1X2)
    - step2b_ou25_features_v2.parquet              (feature avanzate OU2.5)
    - step2b_ou15_features_v2.parquet              (feature avanzate OU1.5)
    - step3a_1x2_clusters_v2.parquet               (cluster_1x2)
    - step3b_ou25_clusters_v2.parquet              (cluster_ou25)
    - step3c_ou15_clusters_v2.parquet              (cluster_ou15)

Output:
    - step4a_affini_index_wide_v2.parquet
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

MASTER_FILE       = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
FEAT_1X2_FILE     = DATA_DIR / "step2b_1x2_features_v2.parquet"
FEAT_OU25_FILE    = DATA_DIR / "step2b_ou25_features_v2.parquet"
FEAT_OU15_FILE    = DATA_DIR / "step2b_ou15_features_v2.parquet"
CLUST_1X2_FILE    = DATA_DIR / "step3a_1x2_clusters_v2.parquet"
CLUST_OU25_FILE   = DATA_DIR / "step3b_ou25_clusters_v2.parquet"
CLUST_OU15_FILE   = DATA_DIR / "step3c_ou15_clusters_v2.parquet"

OUT_WIDE_FILE     = DATA_DIR / "step4a_affini_index_wide_v2.parquet"


def main():
    print("====================================================")
    print("🚀 STEP4A — AFFINI INDEX WIDE V2")
    print("====================================================")
    print(f"📥 MASTER     : {MASTER_FILE}")
    print(f"📥 FEAT 1X2   : {FEAT_1X2_FILE}")
    print(f"📥 FEAT OU2.5 : {FEAT_OU25_FILE}")
    print(f"📥 FEAT OU1.5 : {FEAT_OU15_FILE}")
    print(f"📥 CLUST 1X2  : {CLUST_1X2_FILE}")
    print(f"📥 CLUST OU2.5: {CLUST_OU25_FILE}")
    print(f"📥 CLUST OU1.5: {CLUST_OU15_FILE}")
    print(f"💾 OUT WIDE   : {OUT_WIDE_FILE}")

    # -------------------------
    # 1) Carico tutto
    # -------------------------
    master = pd.read_parquet(MASTER_FILE)
    f1x2   = pd.read_parquet(FEAT_1X2_FILE)
    fou25  = pd.read_parquet(FEAT_OU25_FILE)
    fou15  = pd.read_parquet(FEAT_OU15_FILE)

    c1x2   = pd.read_parquet(CLUST_1X2_FILE)
    cou25  = pd.read_parquet(CLUST_OU25_FILE)
    cou15  = pd.read_parquet(CLUST_OU15_FILE)

    print(f"📏 master : {master.shape}")
    print(f"📏 f1x2   : {f1x2.shape}")
    print(f"📏 fou25  : {fou25.shape}")
    print(f"📏 fou15  : {fou15.shape}")
    print(f"📏 c1x2   : {c1x2.shape}")
    print(f"📏 cou25  : {cou25.shape}")
    print(f"📏 cou15  : {cou15.shape}")

    # -------------------------
    # 2) Preparazione cluster
    #    (ci interessa solo match_id + cluster_*)
    # -------------------------
    c1x2 = c1x2[["match_id", "cluster_1x2"]].drop_duplicates("match_id")
    if "cluster_label" in c1x2.columns:
        # eventuale label auto → la teniamo come extra
        pass

    # cou25 può avere solo sottoinsieme di match (senza OU2.5)
    cou25 = cou25[["match_id", "cluster_ou25"]].drop_duplicates("match_id")
    cou15 = cou15[["match_id", "cluster_ou15"]].drop_duplicates("match_id")

    # -------------------------
    # 3) Pulizia feature per evitare duplicati meta
    # -------------------------
    meta_cols = ["match_id", "date", "season", "league", "home_team", "away_team"]

    def drop_meta(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c not in meta_cols]
        return df[["match_id"] + [c for c in cols if c != "match_id"]]

    f1x2_clean  = drop_meta(f1x2)
    fou25_clean = drop_meta(fou25)
    fou15_clean = drop_meta(fou15)

    # -------------------------
    # 4) Merge, partendo dal master
    # -------------------------
    # Ci assicuriamo che master abbia match_id univoco
    master = master.drop_duplicates("match_id")

    df = master.merge(f1x2_clean, on="match_id", how="left", suffixes=("", "_f1x2"))
    df = df.merge(fou25_clean, on="match_id", how="left", suffixes=("", "_ou25"))
    df = df.merge(fou15_clean, on="match_id", how="left", suffixes=("", "_ou15"))

    df = df.merge(c1x2,  on="match_id", how="left")
    df = df.merge(cou25, on="match_id", how="left")
    df = df.merge(cou15, on="match_id", how="left")

    print(f"📏 Shape dopo merge WIDE: {df.shape}")

    # -------------------------
    # 5) Flag utility
    # -------------------------
    # NaN → niente mercato OU2.5
    df["has_ou25"] = df["cluster_ou25"].notna().astype(int)
    df["has_ou15"] = df["cluster_ou15"].notna().astype(int)

    # -------------------------
    # 6) Check target principali (solo log, non forzo)
    # -------------------------
    target_cols = [
        "is_home_win",
        "is_away_win",
        "is_draw",
        "is_over25",
        "is_over15",
    ]
    missing_t = [c for c in target_cols if c not in df.columns]
    if missing_t:
        print(f"⚠️ WARNING: target mancanti nel WIDE index: {missing_t}")
    else:
        print("✅ Target principali presenti nel WIDE index.")

    # -------------------------
    # 7) Salvataggio
    # -------------------------
    OUT_WIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_WIDE_FILE, index=False)
    print(f"💾 Salvato WIDE index: {OUT_WIDE_FILE}")
    print("🏁 STEP4A AFFINI INDEX WIDE V2 COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()
 