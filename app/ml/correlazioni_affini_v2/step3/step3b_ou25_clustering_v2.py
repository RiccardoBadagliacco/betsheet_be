#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3B OU2.5 V2 — Clustering avanzato mercato OU2.5
(OUTCOME-AWARE + FEATURE-WEIGHTED)

Input feature:
    data/step2b_ou25_features_v2.parquet

Input master (target):
    data/step1c_dataset_with_elo_form.parquet

Output modello:
    models/cluster_ou25_kmeans_v2.pkl

Output dataset clusterizzato:
    data/step3b_ou25_clusters_v2.parquet

Logica:
- Usa solo righe con bk_pO25 non nullo
- Usa tutte le feature numeriche OU2.5 come input
- StandardScaler -> pesi feature via mutual information (con is_over25) ->
  applicazione pesi -> PCA -> KMeans(K)
- K in [6..14]
- Per ogni K:
    * silhouette
    * calinski-harabasz
    * davies-bouldin
    * stabilità (media ARI tra run multipli)
    * sep_outcome: std tra cluster di over25_rate
- Tutte le metriche normalizzate in [0,1] e combinate in:
    score = 0.30 * sil_norm
          + 0.20 * cal_norm
          + 0.15 * db_inv_norm
          + 0.15 * stab_norm
          + 0.20 * sep_norm
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
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
# UTILS
# -----------------------------------------------------------
def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizza un array in [0,1] (se costante -> 0.5)."""
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _evaluate_k_range_ou25(
    X_pca: np.ndarray,
    Ks: list[int],
    y_over25: np.ndarray,
    n_runs: int = 5,
    random_state_base: int = 42,
):
    """
    Valuta diversi K su X_pca, con più run per K per misurare stabilità
    + 'sep_outcome' (separazione sugli esiti reali is_over25).

    Parametri
    ---------
    X_pca : array (n_samples, n_components)
    Ks    : lista di K da provare
    y_over25 : array binario (0/1) allineato alle righe di X_pca

    Ritorna
    -------
    results : list[dict]
        Per ciascun K: metriche silhouette, calinski, davies, stability, sep_outcome
    """
    results = []

    for k in Ks:
        print(f"🔍 Valutazione K = {k} ...")
        labels_runs = []
        sil_scores = []
        cal_scores = []
        db_scores = []

        for r in range(n_runs):
            seed = random_state_base + r
            km = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=20,
                max_iter=300,
            )
            labels = km.fit_predict(X_pca)
            labels_runs.append(labels)

            if r == 0:
                sil_scores.append(silhouette_score(X_pca, labels))
                cal_scores.append(calinski_harabasz_score(X_pca, labels))
                db_scores.append(davies_bouldin_score(X_pca, labels))

        sil = sil_scores[0]
        cal = cal_scores[0]
        db = db_scores[0]

        # Stabilità: media ARI tra tutte le coppie di run
        ari_vals = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                ari_vals.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
        stability = float(np.mean(ari_vals)) if ari_vals else np.nan

        # -------------------------------------------------------
        # SEP_OUTCOME: quanto i cluster sono diversi su is_over25
        # -------------------------------------------------------
        labels0 = labels_runs[0]
        tmp = pd.DataFrame(
            {
                "cluster": labels0,
                "is_over25": y_over25,
            }
        )
        tmp = tmp.dropna(subset=["is_over25"])

        g = tmp.groupby("cluster")["is_over25"].mean()

        if len(g) > 1:
            sep_outcome = float(g.std(ddof=0))
        else:
            sep_outcome = 0.0

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski": cal,
                "davies": db,
                "stability": stability,
                "sep_outcome": sep_outcome,
            }
        )

    return results


def main():
    print("====================================================")
    print("🚀 STEP3B OU2.5 V2 — Clustering avanzato mercato OU2.5 (OUTCOME-AWARE + FEATURE-WEIGHTED)")
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
        ["match_id", "is_over25", "is_under25", "total_goals", "tightness_index"]
    ]

    print(f"📏 Shape feature: {df_feat.shape}")
    print(f"📏 Shape master : {df_master.shape}")

    # merge per avere target per diagnostica
    df = df_feat.merge(df_master, on="match_id", how="left")
    print(f"📏 Shape merge feature+target: {df.shape}")

    # ---------------------------------------------------
    # FIX tightness_index (merge genera _x e _y se duplicato)
    # ---------------------------------------------------
    tix = [c for c in df.columns if c.startswith("tightness_index")]
    if len(tix) == 1:
        pass
    elif len(tix) == 2 and "tightness_index_x" in tix and "tightness_index_y" in tix:
        df["tightness_index"] = df["tightness_index_x"].astype(float)
        df = df.drop(columns=["tightness_index_x", "tightness_index_y"])
    elif len(tix) > 1:
        df["tightness_index"] = df[tix].mean(axis=1)
        for col in tix:
            if col != "tightness_index":
                df = df.drop(columns=col)
    print("🛠 tightness_index normalizzato:", "tightness_index" in df.columns)

    # ---------------------------------------------------
    # 2) Filtro righe valide (bk_pO25 non nullo)
    # ---------------------------------------------------
    meta_cols = ["match_id", "date", "season", "league", "home_team", "away_team"]
    mask_valid = df_feat["bk_pO25"].notna()

    df_feat = df_feat.loc[mask_valid].reset_index(drop=True)
    df = df.loc[mask_valid].reset_index(drop=True)

    print(f"📉 Righe valide per OU2.5 clustering: {df_feat.shape[0]}")

    # ---------------------------------------------------
    # 3) Mask training: solo match chiusi (no fixture)
    # ---------------------------------------------------
    mask_train = df["is_over25"].notna()
    df_feat_train = df_feat.loc[mask_train].reset_index(drop=True)
    df_train = df.loc[mask_train].reset_index(drop=True)

    print(f"✅ Righe usate per training (match chiusi): {df_feat_train.shape[0]}")
    print(f"🚫 Righe escluse dal training (fixture o match senza esito): {df_feat.shape[0] - df_feat_train.shape[0]}")

    # ---------------------------------------------------
    # 4) Selezione feature numeriche (solo training)
    # ---------------------------------------------------
    num_cols = [
        c
        for c in df_feat_train.columns
        if c not in meta_cols
        and df_feat_train[c].dtype != "object"
        and not np.issubdtype(df_feat_train[c].dtype, np.datetime64)
    ]

    print(f"🔢 N feature numeriche usate per clustering: {len(num_cols)}")

    # ---------------------------------------------------
    # 5) Preparazione X_train / y_train (solo match chiusi)
    # ---------------------------------------------------
    X_train = df_feat_train[num_cols].astype(float).values

    # NaN → media colonna (training)
    col_means = np.nanmean(X_train, axis=0)
    idx_nan = np.where(np.isnan(X_train))
    if idx_nan[0].size > 0:
        X_train[idx_nan] = np.take(col_means, idx_nan[1])

    y_over25_train = df_train["is_over25"].astype(int).values

    print(f"🔢 N feature numeriche effettive per training: {X_train.shape[1]}")

    # ---------------------------------------------------
    # 6) Standardizzazione + pesi via mutual information
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)

    print("----------------------------------------------------")
    print("🧮 Calcolo pesi feature via mutual information (outcome-aware su is_over25)...")
    mi = mutual_info_classif(
        X_scaled_train,
        y_over25_train,
        discrete_features=False,
        random_state=42,
    )

    mi = np.nan_to_num(mi, nan=0.0)
    mi_max = mi.max()
    if mi_max <= 0:
        feature_weights = np.ones_like(mi)
    else:
        feature_weights = mi / mi_max

    # log: top 15 feature per peso
    w_series = pd.Series(feature_weights, index=num_cols).sort_values(ascending=False)
    print("🔝 Top 15 feature per peso outcome-aware (is_over25):")
    for name, w in w_series.head(15).items():
        print(f"   - {name:<35} weight={w:.3f}")

    # applicazione pesi (scaling per feature) prima di PCA (solo training)
    X_weighted_train = X_scaled_train * feature_weights

    # ---------------------------------------------------
    # 7) PCA (fit su training)
    # ---------------------------------------------------
    n_components = min(8, X_weighted_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_train = pca.fit_transform(X_weighted_train)

    explained = pca.explained_variance_ratio_.sum()
    print(f"🎛 PCA — n_components usati: {n_components}")
    print(f"   Varianza spiegata totale PCA: {explained:.3f}")

    # ---------------------------------------------------
    # 8) Ricerca K ottimale (6–14) con sep_outcome (solo training)
    # ---------------------------------------------------
    Ks = list(range(6, 15))
    results = _evaluate_k_range_ou25(
        X_pca_train,
        Ks,
        y_over25=y_over25_train,
        n_runs=5,
        random_state_base=100,
    )

    sils = np.array([r["silhouette"] for r in results], dtype=float)
    cals = np.array([r["calinski"] for r in results], dtype=float)
    dbs = np.array([r["davies"] for r in results], dtype=float)
    stabs = np.array([r["stability"] for r in results], dtype=float)
    seps = np.array([r["sep_outcome"] for r in results], dtype=float)

    sil_norm = _normalize(sils)
    cal_norm = _normalize(cals)
    db_norm_inv = _normalize(-dbs)
    stab_norm = _normalize(stabs)
    sep_norm = _normalize(seps)

    scores = (
        0.30 * sil_norm
        + 0.20 * cal_norm
        + 0.15 * db_norm_inv
        + 0.15 * stab_norm
        + 0.20 * sep_norm
    )

    for i, r in enumerate(results):
        r["sil_norm"] = float(sil_norm[i])
        r["cal_norm"] = float(cal_norm[i])
        r["db_inv_norm"] = float(db_norm_inv[i])
        r["stab_norm"] = float(stab_norm[i])
        r["sep_norm"] = float(sep_norm[i])
        r["score"] = float(scores[i])

    print("----------------------------------------------------")
    print("📊 Riepilogo metriche per K:")
    print("   K | silhouette |  calinski |  davies | stability | sep_outcome |  SCORE")
    for r in results:
        print(
            f"{r['k']:4d} | "
            f"{r['silhouette']:.4f} | "
            f"{r['calinski']:9.1f} | "
            f"{r['davies']:.4f} | "
            f"{r['stability']:.4f} | "
            f"{r['sep_outcome']:.4f} | "
            f"{r['score']:.4f}"
        )
    print("----------------------------------------------------")

    best_idx = int(np.nanargmax(scores))
    best_k = results[best_idx]["k"]
    print(f"🏆 K ottimale scelto: {best_k} (score={results[best_idx]['score']:.4f})")

    # ---------------------------------------------------
    # 9) Fit finale KMeans con K ottimale (solo training)
    # ---------------------------------------------------
    kmeans_final = KMeans(
        n_clusters=best_k,
        n_init=50,
        max_iter=500,
        random_state=123,
    )
    labels_train = kmeans_final.fit_predict(X_pca_train)

    # ---------------------------------------------------
    # 10) Applicazione modello a TUTTE le righe valide (train + fixture)
    # ---------------------------------------------------
    X_all = df_feat[num_cols].astype(float).values
    col_means_all = np.nanmean(X_all, axis=0)
    idx_nan_all = np.where(np.isnan(X_all))
    if idx_nan_all[0].size > 0:
        X_all[idx_nan_all] = np.take(col_means_all, idx_nan_all[1])

    X_all_scaled = scaler.transform(X_all)
    X_all_weighted = X_all_scaled * feature_weights
    X_all_pca = pca.transform(X_all_weighted)

    labels_all = kmeans_final.predict(X_all_pca)
    df["cluster_ou25"] = labels_all

    # ---------------------------------------------------
    # 11) Statistiche per cluster (diagnostica solo sui match chiusi)
    # ---------------------------------------------------
    df_diag = df.loc[mask_train].copy()

    agg = df_diag.groupby("cluster_ou25").agg(
        n=("match_id", "size"),
        over25_rate=("is_over25", "mean"),
        under25_rate=("is_under25", "mean"),
        avg_goals=("total_goals", "mean"),
        bk_pO25_mean=("bk_pO25", "mean"),
        bk_pU25_mean=("bk_pU25", "mean"),
        pic_pO25_mean=("pic_pO25", "mean"),
        lambda_market_mean=("lambda_total_market_ou25", "mean"),
        lambda_form_mean=("lambda_total_form", "mean"),
        delta_O25_mean=("delta_O25", "mean"),
        delta_ou25_abs_sum_mean=("delta_ou25_abs_sum", "mean"),
        delta_ou25_mkt_vs_form_mean=("delta_ou25_market_vs_form", "mean"),
        entropy_bk_ou25_mean=("entropy_bk_ou25", "mean"),
        entropy_pic_ou25_mean=("entropy_pic_ou25", "mean"),
        tight_mean=("tightness_index", "mean"),
    )

    agg = agg.sort_values("over25_rate")

    print("📊 Distribuzione cluster_ou25 (tutte le righe valide):")
    print(df["cluster_ou25"].value_counts().sort_index())
    print("----------------------------------------------------")
    print("📊 Statistiche per cluster (solo match chiusi: over25_rate, λ mercato/form, bk/pic, delta, tight):")
    print(agg)

    # ---------------------------------------------------
    # 12) Salvataggio modello + dataset clusterizzato
    # ---------------------------------------------------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kmeans": kmeans_final,
            "scaler": scaler,
            "pca": pca,
            "num_cols": num_cols,
            "feature_cols": num_cols,
            "feature_weights": feature_weights,
            "metrics": results,
            "k": best_k,
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