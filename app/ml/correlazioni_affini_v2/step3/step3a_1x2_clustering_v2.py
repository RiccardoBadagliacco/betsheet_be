#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3A 1X2 V2 — Clustering avanzato mercato 1X2

- Input feature: step2b_1x2_features_v2.parquet
- Input master:  step1c_dataset_with_elo_form.parquet (solo per diagnostica)
- Output dataset clusterizzato: step3a_1x2_clusters_v2.parquet
- Modello salvato: models/cluster_1x2_kmeans_v2.pkl

Strategia:
- Usa tutte le feature numeriche di step2b (no colonne object).
- StandardScaler -> PCA(8) -> KMeans(K)
- K in [6..14]
- Per ogni K:
    * silhouette
    * calinski-harabasz
    * davies-bouldin
    * stabilità (media ARI tra run multipli)
- Tutte le metriche vengono normalizzate in [0,1] e combinate in:
    score = 0.50 * sil_norm
          + 0.20 * cal_norm
          + 0.20 * stab_norm
          + 0.10 * db_inv_norm
- K con score massimo viene scelto come K ottimale.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"
MODEL_DIR = AFFINI_DIR / "models"

FEATURE_FILE = DATA_DIR / "step2b_1x2_features_v2.parquet"
MASTER_FILE = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
OUT_CLUSTERS_FILE = DATA_DIR / "step3a_1x2_clusters_v2.parquet"
MODEL_PATH = MODEL_DIR / "cluster_1x2_kmeans_v2.pkl"


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def _normalize(arr):
    """Normalizza un array in [0,1] (se costante -> 0.5)."""
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _evaluate_k_range(X_pca, Ks, n_runs=5, random_state_base=42):
    """
    Valuta diversi K su X_pca, con più run per K per misurare stabilità.

    Ritorna:
      - results: list di dict con metriche per ciascun K
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

            # Calcoliamo le metriche solo sul primo run,
            # ma potremmo anche mediare: qui usiamo il primo per coerenza.
            if r == 0:
                sil_scores.append(silhouette_score(X_pca, labels))
                cal_scores.append(calinski_harabasz_score(X_pca, labels))
                db_scores.append(davies_bouldin_score(X_pca, labels))

        # Silhouette / Calinski / Davies dal primo run
        sil = sil_scores[0]
        cal = cal_scores[0]
        db = db_scores[0]

        # Stabilità: media ARI tra tutte le coppie di run
        ari_vals = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                ari_vals.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
        stability = float(np.mean(ari_vals)) if ari_vals else np.nan

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski": cal,
                "davies": db,
                "stability": stability,
            }
        )

    return results


def main():
    print("====================================================")
    print("🚀 STEP3A 1X2 V2 — Clustering avanzato mercato 1X2")
    print("====================================================")
    print(f"📥 Feature file: {FEATURE_FILE}")
    print(f"📥 Master file : {MASTER_FILE}")
    print(f"💾 Modello out : {MODEL_PATH}")
    print(f"💾 Clusters out: {OUT_CLUSTERS_FILE}")

    # --------------------------------------------------
    # 1) Caricamento dati
    # --------------------------------------------------
    df_feat = pd.read_parquet(FEATURE_FILE)
    print(f"📏 Shape feature: {df_feat.shape}")

    df_master = pd.read_parquet(MASTER_FILE)[
        ["match_id", "is_home_win", "is_draw", "is_away_win", "is_over25"]
    ]
    print(f"📏 Shape master : {df_master.shape}")

    # --------------------------------------------------
    # 2) Selezione feature numeriche per clustering
    # --------------------------------------------------
    numeric_cols = [
        c for c in df_feat.columns
        if df_feat[c].dtype != "object" and not np.issubdtype(df_feat[c].dtype, np.datetime64)
    ]

    # Escludiamo eventualmente colonne troppo ID-like? (qui NO: teniamo tutto)
    # Se in futuro vuoi togliere qualcosa, modifichi questa lista.
    print(f"🔢 N feature numeriche usate per clustering: {len(numeric_cols)}")

    X = df_feat[numeric_cols].astype(float).values

    # Gestione eventuali NaN → imputiamo con media colonna
    col_means = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    X[idx_nan] = np.take(col_means, idx_nan[1])

    # --------------------------------------------------
    # 3) StandardScaler + PCA
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(8, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"🎛 PCA — n_components usati: {n_components}")
    explained = pca.explained_variance_ratio_.sum()
    print(f"   Varianza spiegata totale PCA: {explained:.3f}")

    # --------------------------------------------------
    # 4) Ricerca K ottimale (K in [6..14])
    # --------------------------------------------------
    Ks = list(range(6, 15))
    results = _evaluate_k_range(X_pca, Ks, n_runs=5, random_state_base=42)

    # Normalizzazione metriche per score combinato
    sils = np.array([r["silhouette"] for r in results], dtype=float)
    cals = np.array([r["calinski"] for r in results], dtype=float)
    dbs = np.array([r["davies"] for r in results], dtype=float)
    stabs = np.array([r["stability"] for r in results], dtype=float)

    sil_norm = _normalize(sils)
    cal_norm = _normalize(cals)
    db_norm_inv = _normalize(-dbs)  # invertito: meno è meglio → -db

    # Stabilità già [0,1] di solito, ma normalizziamo comunque
    stab_norm = _normalize(stabs)

    scores = (
        0.50 * sil_norm +
        0.20 * cal_norm +
        0.20 * stab_norm +
        0.10 * db_norm_inv
    )

    # Colleghiamo gli score ai risultati
    for i, r in enumerate(results):
        r["sil_norm"] = float(sil_norm[i])
        r["cal_norm"] = float(cal_norm[i])
        r["db_inv_norm"] = float(db_norm_inv[i])
        r["stab_norm"] = float(stab_norm[i])
        r["score"] = float(scores[i])

    # Stampa tabella riassuntiva
    print("----------------------------------------------------")
    print("📊 Riepilogo metriche per K:")
    print("   K | silhouette |  calinski |  davies | stability |  SCORE")
    for r in results:
        print(
            f"{r['k']:4d} | "
            f"{r['silhouette']:.4f} | "
            f"{r['calinski']:9.1f} | "
            f"{r['davies']:.4f} | "
            f"{r['stability']:.4f} | "
            f"{r['score']:.4f}"
        )
    print("----------------------------------------------------")

    # Scelta K migliore
    best_idx = int(np.nanargmax(scores))
    best_k = results[best_idx]["k"]
    print(f"🏆 K ottimale scelto: {best_k} (score={results[best_idx]['score']:.4f})")

    # --------------------------------------------------
    # 5) Fit finale KMeans con K ottimale
    # --------------------------------------------------
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50,
        max_iter=500,
    )
    labels = kmeans.fit_predict(X_pca)

    df_feat["cluster_1x2"] = labels

    # --------------------------------------------------
    # 6) Merge con target per diagnostica cluster
    # --------------------------------------------------
    df_merge = df_feat.merge(df_master, on="match_id", how="left")

    # Sanity: cluster distribution
    print("----------------------------------------------------")
    print("📊 Distribuzione cluster_1x2:")
    print(df_feat["cluster_1x2"].value_counts().sort_index())

    # Statistiche per cluster
    print("----------------------------------------------------")
    print("📊 Statistiche per cluster (home win rate, draw, away, bk_p1, pic_p1, delta_p1):")
    grp = df_merge.groupby("cluster_1x2").agg(
        n=("match_id", "size"),
        home_rate=("is_home_win", "mean"),
        draw_rate=("is_draw", "mean"),
        away_rate=("is_away_win", "mean"),
        over25_rate=("is_over25", "mean"),
        bk_p1_mean=("bk_p1", "mean"),
        pic_p1_mean=("pic_p1", "mean"),
        delta_p1_mean=("delta_p1", "mean"),
        delta_1x2_abs_sum_mean=("delta_1x2_abs_sum", "mean"),
    )
    print(grp)

    # --------------------------------------------------
    # 7) Salvataggio modello (scaler + pca + kmeans + feature_cols)
    # --------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_obj = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "feature_cols": numeric_cols,
        "k": best_k,
        "metrics": results,
    }
    joblib.dump(model_obj, MODEL_PATH)
    print(f"💾 Modello clustering salvato in: {MODEL_PATH}")

    # --------------------------------------------------
    # 8) Salvataggio dataset clusterizzato
    # --------------------------------------------------
    # Manteniamo meta + alcune colonne chiave + cluster
    cols_out = [
        "match_id", "date", "season", "league", "home_team", "away_team",
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_px", "pic_p2",
        "delta_p1", "delta_px", "delta_p2",
        "delta_1x2_abs_sum",
        "entropy_bk_1x2", "entropy_pic_1x2",
        "elo_home_pre", "elo_away_pre", "elo_diff",
        "team_strength_home", "team_strength_away", "team_strength_diff",
        "lambda_home_form", "lambda_away_form", "lambda_total_form",
        "lambda_total_market_ou25", "goal_supremacy_market_ou25", "goal_supremacy_form",
        "fav_prob_1x2", "market_balance_index_1x2", "fav_prob_gap_1x2",
        "second_fav_prob_1x2",
        "season_recency", "match_density_index",
        "cluster_1x2",
        "tightness_index",
    ]
    cols_out = [c for c in cols_out if c in df_feat.columns]

    df_out = df_feat[cols_out].copy()
    df_out.to_parquet(OUT_CLUSTERS_FILE, index=False)
    print(f"💾 Dataset clusterizzato salvato in: {OUT_CLUSTERS_FILE}")

    print("🏁 STEP3A 1X2 V2 COMPLETATO")
    print("====================================================")


if __name__ == "__main__":
    main()