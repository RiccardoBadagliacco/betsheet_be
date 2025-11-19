#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3C OU1.5 V2 — Clustering avanzato mercato OU1.5 (solo tecnico).
Basato su step2b_ou15_features_v2.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# -----------------------------------------------------------
# Import path
# -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"
MODEL_DIR = AFFINI_DIR / "models"

FEATURE_FILE = DATA_DIR / "step2b_ou15_features_v2.parquet"
MASTER_FILE  = DATA_DIR / "step1c_dataset_with_elo_form.parquet"

OUT_CLUSTER_FILE = DATA_DIR / "step3c_ou15_clusters_v2.parquet"
OUT_MODEL_FILE   = MODEL_DIR / "cluster_ou15_kmeans_v2.pkl"


# -----------------------------------------------------------
# Feature numeriche da usare nel clustering
# -----------------------------------------------------------
NUM_FEATURES = [
    # λ form
    "lambda_home_form",
    "lambda_away_form",
    "lambda_total_form",
    "lambda_total_market_ou25",
    "lambda_total_mix_ou15",

    # Tecnico Poisson
    "tech_pO05_mix",
    "tech_pO15_mix",
    "tech_pO25_mix",

    # Picchetto
    "pic_pO05",
    "pic_pO15",
    "pic_pU05",
    "pic_pU15",

    # Delta tech vs pic
    "delta_ou15_pic_vs_mix",
    "delta_ou15_pic_vs_mix_abs",

    # Tempo / fatica
    "match_density_index",
    "rest_diff_days",
    "short_rest_home",
    "short_rest_away",
    "rest_advantage_home",
    "rest_advantage_away",

    # Season recency
    "season_recency",
]


# -----------------------------------------------------------
# Scoring composito per selezione K
# -----------------------------------------------------------
def cluster_score(sil, cal, dav, stab,
                  w_s=0.45, w_c=0.25, w_d=0.15, w_st=0.15):
    if cal < 0:
        cal_adj = 0
    else:
        cal_adj = cal / (cal + 1000)

    dav_adj = 1 / (1 + dav)

    return (w_s * sil) + (w_c * cal_adj) + (w_d * dav_adj) + (w_st * stab)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    print("="*60)
    print("🚀 STEP3C OU1.5 V2 — Clustering avanzato OU1.5")
    print("="*60)

    df_feat = pd.read_parquet(FEATURE_FILE)
    df_master = pd.read_parquet(MASTER_FILE)[["match_id", "is_over15","tightness_index"]]

    print(f"📏 Feature: {df_feat.shape}")
    print(f"📏 Master : {df_master.shape}")

    df = df_feat.merge(df_master, on="match_id", how="left")
    print(f"📏 Merge feature+target: {df.shape}")


    # ---------------------------------------------------
    # FIX tightness_index (merge genera _x e _y)
    # ---------------------------------------------------
    tix = [c for c in df.columns if c.startswith("tightness_index")]

    if len(tix) == 1:
        pass
    elif len(tix) == 2 and "tightness_index_x" in tix and "tightness_index_y" in tix:
        df["tightness_index"] = df["tightness_index_x"].astype(float)
        df = df.drop(columns=["tightness_index_x", "tightness_index_y"])
    else:
        df["tightness_index"] = df[tix].mean(axis=1)
        for col in tix:
            if col != "tightness_index":
                df = df.drop(columns=col)

    print("🛠 tightness_index normalizzato:", "tightness_index" in df.columns)

    # Filtra righe valide
    df_valid = df.dropna(subset=["tech_pO15_mix", "pic_pO15"])
    print(f"📉 Righe valide per OU1.5 clustering: {df_valid.shape[0]}")

    # Prepara matrice numerica
    X = df_valid[NUM_FEATURES].copy()

    # Fix eventuali NaN (poi non dovrebbero esserne rimasti)
    X = X.fillna(X.mean())

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Recupera n_components per spiegare ~78% varianza
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.searchsorted(cum_var, 0.78) + 1
    n_comp = max(6, min(n_comp, 15))

    print(f"🎛 PCA — n_components usati: {n_comp}")
    print(f"   Varianza spiegata totale PCA: {cum_var[n_comp-1]:.3f}")

    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    # Selezione K
    results = []
    prev_labels = None

    for K in range(6, 15):
        print(f"🔍 Valutazione K = {K} ...")

        kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        cal = calinski_harabasz_score(X_pca, labels)
        dav = davies_bouldin_score(X_pca, labels)

        if prev_labels is None:
            stability = 1.0
        else:
            stability = (prev_labels == labels).mean()

        score = cluster_score(sil, cal, dav, stability)
        results.append((K, sil, cal, dav, stability, score))

        prev_labels = labels

    print("-"*60)
    print("📊 Riepilogo metriche per K:")
    print("   K | silhouette |  calinski |  davies | stability |  SCORE")
    for K, sil, cal, dav, stab, score in results:
        print(f"{K:4d} | {sil:.4f} | {cal:10.1f} | {dav:.4f} | {stab:.4f} | {score:.4f}")

    # Scegli K migliore
    best = max(results, key=lambda x: x[5])
    best_k = best[0]

    print("-"*60)
    print(f"🏆 K ottimale scelto: {best_k} (score={best[5]:.4f})")
    print("-"*60)

    # Fit finale
    kmeans_final = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    labels_final = kmeans_final.fit_predict(X_pca)

    df_valid["cluster_ou15"] = labels_final

    # Statistiche cluster
    print("📊 Distribuzione cluster OU1.5:")
    print(df_valid["cluster_ou15"].value_counts())

    agg = df_valid.groupby("cluster_ou15").agg(
        n=("match_id", "count"),
        over15_rate=("is_over15", "mean"),
        avg_lam_form=("lambda_total_form", "mean"),
        avg_lam_mix=("lambda_total_mix_ou15", "mean"),
        pic_pO15_mean=("pic_pO15", "mean"),
        tech_pO15_mean=("tech_pO15_mix", "mean"),
        delta_mean=("delta_ou15_pic_vs_mix", "mean"),
        delta_abs_mean=("delta_ou15_pic_vs_mix_abs", "mean"),
        rest_adv_home=("rest_advantage_home", "mean"),
    )
    print("-"*60)
    print(agg)

    # Salvataggio
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(kmeans_final, OUT_MODEL_FILE)
    df_valid.to_parquet(OUT_CLUSTER_FILE, index=False)

    print(f"💾 Modello clustering salvato in: {OUT_MODEL_FILE}")
    print(f"💾 Dataset clusterizzato salvato in: {OUT_CLUSTER_FILE}")
    print("🏁 STEP3C OU1.5 V2 COMPLETATO")
    print("="*60)


if __name__ == "__main__":
    main()