#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3B OU2.5 V2 — Clustering avanzato mercato OU2.5

Input feature:
    data/step2b_ou25_features_v2.parquet

Input master (target):
    data/step1c_dataset_with_elo_form.parquet

Output modello:
    models/cluster_ou25_kmeans_v2.pkl

Output dataset clusterizzato:
    data/step3b_ou25_clusters_v2.parquet
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------
# PATH / IMPORT ROOT
# -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"
MODEL_DIR = AFFINI_DIR / "models"

FEATURE_FILE = DATA_DIR / "step2b_ou25_features_v2.parquet"
MASTER_FILE = DATA_DIR / "step1c_dataset_with_elo_form.parquet"

MODEL_OUT = MODEL_DIR / "cluster_ou25_kmeans_v2.pkl"
CLUSTERS_OUT = DATA_DIR / "step3b_ou25_clusters_v2.parquet"


# -----------------------------------------------------------
# UTILITY: valutazione K (come stile 1X2)
# -----------------------------------------------------------
def evaluate_kmeans_stability(X_pca: np.ndarray, k: int, n_repeats: int = 4, base_seed: int = 42):
    """Lancia più KMeans per stimare la stabilità via ARI medio."""
    labels_list = []
    for i in range(n_repeats):
        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=base_seed + i,
        )
        lbl = km.fit_predict(X_pca)
        labels_list.append(lbl)

    # se c'è un solo run, stabilità = 1.0
    if len(labels_list) < 2:
        return 1.0

    # ARI medio su tutte le coppie
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return float(np.mean(aris))


def main():
    print("====================================================")
    print("🚀 STEP3B OU2.5 V2 — Clustering avanzato mercato OU2.5")
    print("====================================================")
    print(f"📥 Feature file: {FEATURE_FILE}")
    print(f"📥 Master file : {MASTER_FILE}")
    print(f"💾 Modello out : {MODEL_OUT}")
    print(f"💾 Clusters out: {CLUSTERS_OUT}")

    # ---------------------------------------------------
    # 1) Carico feature OU2.5 + master (target)
    # ---------------------------------------------------
    df_feat = pd.read_parquet(FEATURE_FILE)
    df_master = pd.read_parquet(MASTER_FILE)[
        ["match_id", "is_over25", "is_under25", "total_goals","tightness_index"]
    ]

    print(f"📏 Shape feature: {df_feat.shape}")
    print(f"📏 Shape master : {df_master.shape}")

    # merge per avere target per diagnostica
    df = df_feat.merge(df_master, on="match_id", how="left")
    print(f"📏 Shape merge feature+target: {df.shape}")

    # ---------------------------------------------------
    # FIX tightness_index (merge genera _x e _y)
    # ---------------------------------------------------
    # Caso tipico → due colonne identiche: tightness_index_x e tightness_index_y
    tix = [c for c in df.columns if c.startswith("tightness_index")]

    if len(tix) == 1:
        # già corretto (solo tightness_index)
        pass

    elif len(tix) == 2 and "tightness_index_x" in tix and "tightness_index_y" in tix:
        # sono identici → prendo tightness_index_x come riferimento
        df["tightness_index"] = df["tightness_index_x"].astype(float)
        df = df.drop(columns=["tightness_index_x", "tightness_index_y"])

    else:
        # fallback di sicurezza: prendi la media
        df["tightness_index"] = df[tix].mean(axis=1)
        for col in tix:
            if col != "tightness_index":
                df = df.drop(columns=col)

    print("🛠 tightness_index normalizzato:", "tightness_index" in df.columns)

    # ---------------------------------------------------
    # 2) Selezione feature numeriche per clustering
    # ---------------------------------------------------
    meta_cols = ["match_id", "date", "season", "league", "home_team", "away_team"]
    num_cols = [c for c in df_feat.columns if c not in meta_cols]

    mask_valid = df_feat["bk_pO25"].notna()
    df_feat = df_feat.loc[mask_valid].reset_index(drop=True)
    df = df.loc[mask_valid].reset_index(drop=True)

    print(f"📉 Righe valide per OU2.5 clustering: {df_feat.shape[0]}")

    # ---------------------------------------------------
    # 2ter) IMPUTAZIONE SICURA (mean) PER PCA / KMEANS
    # ---------------------------------------------------
    df_feat[num_cols] = df_feat[num_cols].astype(float)
    df_feat[num_cols] = df_feat[num_cols].fillna(df_feat[num_cols].mean())

    # ora nessun NaN
    X = df_feat[num_cols].values

    print(f"🔢 N feature numeriche usate per clustering: {X.shape[1]}")

    print(f"📉 Righe valide per OU2.5 clustering: {X.shape[0]}")
    print(f"🔢 N feature numeriche usate per clustering: {X.shape[1]}")
    # ---------------------------------------------------
    # 3) Standardizzazione + PCA
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA full per determinare n_components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    # target ~0.75 varianza spiegata (max 12 componenti)
    target_var = 0.75
    n_components = int(np.searchsorted(cumvar, target_var) + 1)
    n_components = max(2, min(n_components, 12))

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"🎛 PCA — n_components usati: {n_components}")
    print(f"   Varianza spiegata totale PCA: {cumvar[n_components - 1]:.3f}")

    # ---------------------------------------------------
    # 4) Ricerca K ottimale (6–14) con metrica composita
    # ---------------------------------------------------
    results = []
    Ks = list(range(6, 15))

    for k in Ks:
        print(f"🔍 Valutazione K = {k} ...")
        km = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=42,
        )
        labels = km.fit_predict(X_pca)

        sil = silhouette_score(X_pca, labels)
        cal = calinski_harabasz_score(X_pca, labels)
        dav = davies_bouldin_score(X_pca, labels)
        stab = evaluate_kmeans_stability(X_pca, k, n_repeats=4, base_seed=100 + k)

        results.append(
            {
                "K": k,
                "silhouette": sil,
                "calinski": cal,
                "davies": dav,
                "stability": stab,
            }
        )

    df_res = pd.DataFrame(results)

    # normalizzazione per costruire uno SCORE simile a 1X2
    sil_norm = (df_res["silhouette"] - df_res["silhouette"].min()) / (
        df_res["silhouette"].max() - df_res["silhouette"].min() + 1e-9
    )

    cal_log = np.log(df_res["calinski"] + 1.0)
    cal_norm = (cal_log - cal_log.min()) / (cal_log.max() - cal_log.min() + 1e-9)

    inv_dav = 1.0 / (df_res["davies"] + 1e-9)  # davies più basso = meglio
    inv_dav_norm = (inv_dav - inv_dav.min()) / (inv_dav.max() - inv_dav.min() + 1e-9)

    stab_norm = (df_res["stability"] - df_res["stability"].min()) / (
        df_res["stability"].max() - df_res["stability"].min() + 1e-9
    )

    # pesi: silhouette 0.4, calinski 0.3, -davies 0.2, stability 0.1
    df_res["SCORE"] = (
        0.4 * sil_norm
        + 0.3 * cal_norm
        + 0.2 * inv_dav_norm
        + 0.1 * stab_norm
    )

    print("----------------------------------------------------")
    print("📊 Riepilogo metriche per K:")
    print("   K | silhouette |  calinski |  davies | stability |  SCORE")
    for _, r in df_res.sort_values("K").iterrows():
        print(
            f"{int(r['K']):4d} | "
            f"{r['silhouette']:.4f} | "
            f"{r['calinski']:9.1f} | "
            f"{r['davies']:.4f} | "
            f"{r['stability']:.4f} | "
            f"{r['SCORE']:.4f}"
        )
    print("----------------------------------------------------")

    # scelgo K con SCORE massimo
    best_row = df_res.loc[df_res["SCORE"].idxmax()]
    best_k = int(best_row["K"])
    best_score = float(best_row["SCORE"])

    print(f"🏆 K ottimale scelto: {best_k} (score={best_score:.4f})")
    print("----------------------------------------------------")

    # ---------------------------------------------------
    # 5) Fit finale KMeans con K ottimale
    # ---------------------------------------------------
    kmeans_final = KMeans(
        n_clusters=best_k,
        n_init=30,
        random_state=123,
    )
    cluster_labels = kmeans_final.fit_predict(X_pca)

    df["cluster_ou25"] = cluster_labels

    # ---------------------------------------------------
    # 6) Statistiche per cluster
    # ---------------------------------------------------
    agg = df.groupby("cluster_ou25").agg(
        n=("match_id", "size"),
        over25_rate=("is_over25", "mean"),
        under25_rate=("is_under25", "mean"),
        avg_goals=("total_goals", "mean"),
        bk_pO25_mean=("bk_pO25", "mean"),
        pic_pO25_mean=("pic_pO25", "mean"),
        lambda_market_mean=("lambda_total_market_ou25", "mean"),
        lambda_form_mean=("lambda_total_form", "mean"),
        delta_O25_mean=("delta_O25", "mean"),
        delta_ou25_abs_sum_mean=("delta_ou25_abs_sum", "mean"),
        delta_ou25_mkt_vs_form_mean=("delta_ou25_market_vs_form", "mean"),
        entropy_bk_ou25_mean=("entropy_bk_ou25", "mean"),
        entropy_pic_ou25_mean=("entropy_pic_ou25", "mean"),
    )

    agg = agg.sort_values("over25_rate")

    print("📊 Distribuzione cluster_ou25:")
    print(df["cluster_ou25"].value_counts().sort_index())
    print("----------------------------------------------------")
    print(
        "📊 Statistiche per cluster (over25_rate, λ mercato/form, bk_pO25, pic_pO25, delta_O25):"
    )
    print(agg)

    # ---------------------------------------------------
    # 7) Salvataggio modello + dataset
    # ---------------------------------------------------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": kmeans_final,
            "scaler": scaler,
            "pca": pca,
            "num_cols": num_cols,
        },
        MODEL_OUT,
    )


    df.to_parquet(CLUSTERS_OUT, index=False)

    print(f"💾 Modello clustering salvato in: {MODEL_OUT}")
    print(f"💾 Dataset clusterizzato salvato in: {CLUSTERS_OUT}")
    print("🏁 STEP3B OU2.5 V2 COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()