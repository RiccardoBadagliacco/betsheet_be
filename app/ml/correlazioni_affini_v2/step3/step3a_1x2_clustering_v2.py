#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3A 1X2 V2 — Clustering avanzato mercato 1X2
(OUTCOME-AWARE + FEATURE-WEIGHTED)

- Input feature: step2b_1x2_features_v2.parquet
- Input master:  step1c_dataset_with_elo_form.parquet (per diagnostica / outcome)
- Output dataset clusterizzato: step3a_1x2_clusters_v2.parquet
- Modello salvato: models/cluster_1x2_kmeans_v2.pkl

Strategia:

1) Usa TUTTE le feature numeriche di step2b (no object / datetime).
2) Calcola un peso per ogni feature in base a quanto spiega gli esiti reali:
      - is_home_win
      - is_away_win
      - is_over25
   tramite mutual information:
      - mi_home_norm, mi_away_norm, mi_ou25_norm ∈ [0,1]
      - weight_f = 0.4*mi_home + 0.4*mi_away + 0.2*mi_ou25
3) Pipeline clustering:
      - imputazione NaN con media colonna
      - StandardScaler
      - moltiplicazione per i pesi delle feature (feature-weighting)
      - PCA (8 componenti)
      - KMeans(K) con K ∈ [6..14]
4) Per ogni K calcola:
      - silhouette
      - calinski-harabasz
      - davies-bouldin
      - stabilità (media ARI su più run)
      - sep_outcome (std tra cluster di:
            • home_win_rate
            • away_win_rate
            • over25_rate
        più alto = cluster più diversi negli esiti reali)
5) Normalizza tutte le metriche in [0,1] e combina in:
      score = 0.30 * sil_norm
            + 0.20 * cal_norm
            + 0.15 * db_inv_norm
            + 0.15 * stab_norm
            + 0.20 * sep_norm

Il K con SCORE massimo viene scelto come K ottimale.
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
from sklearn.feature_selection import mutual_info_classif

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
def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizza un array in [0,1] (se costante -> 0.5)."""
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _compute_feature_weights(
    df_feat: pd.DataFrame,
    df_y: pd.DataFrame,
    numeric_cols: list[str],
) -> np.ndarray:
    """
    Calcola un peso per ogni feature numerica usando mutual information
    rispetto agli esiti reali:
      - is_home_win
      - is_away_win
      - is_over25

    Ritorna:
      weights: np.ndarray di shape (n_features,), normalizzato in [0.1, 1.0]
    """
    print("----------------------------------------------------")
    print("🧮 Calcolo pesi feature via mutual information (outcome-aware)...")

    X = df_feat[numeric_cols].astype(float).values

    # imputazione NaN con media colonna (come nel clustering)
    col_means = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    if idx_nan[0].size > 0:
        X[idx_nan] = np.take(col_means, idx_nan[1])

    y_home = df_y["is_home_win"].values.astype(int)
    y_away = df_y["is_away_win"].values.astype(int)
    y_ou25 = df_y["is_over25"].values.astype(int)

    # Mutual information per ciascun outcome
    mi_home = mutual_info_classif(X, y_home, discrete_features=False, random_state=42)
    mi_away = mutual_info_classif(X, y_away, discrete_features=False, random_state=42)
    mi_ou25 = mutual_info_classif(X, y_ou25, discrete_features=False, random_state=42)

    # Normalizziamo ciascun vettore MI in [0,1]
    mi_home_n = _normalize(mi_home)
    mi_away_n = _normalize(mi_away)
    mi_ou25_n = _normalize(mi_ou25)

    # Combiniamo con pesi: home/away 40% ciascuno, over25 20%
    weights = 0.4 * mi_home_n + 0.4 * mi_away_n + 0.2 * mi_ou25_n

    # Evitiamo pesi nulli, e normalizziamo a [0.1, 1.0]
    if np.all(~np.isfinite(weights)):
        weights = np.ones_like(weights)
    else:
        weights = np.nan_to_num(weights, nan=0.0)
        w_min = np.min(weights)
        w_max = np.max(weights)
        if w_max == w_min:
            weights = np.ones_like(weights)
        else:
            weights = (weights - w_min) / (w_max - w_min)  # [0,1]
            weights = 0.1 + 0.9 * weights                # [0.1,1.0]

    # Log: top feature più pesate
    feat_weights = sorted(
        zip(numeric_cols, weights),
        key=lambda t: t[1],
        reverse=True,
    )
    print("🔝 Top 15 feature per peso outcome-aware:")
    for name, w in feat_weights[:15]:
        print(f"   - {name:30s}  weight={w:.3f}")

    return weights


def _evaluate_k_range(
    X_pca: np.ndarray,
    Ks: list[int],
    y_outcome: pd.DataFrame,
    n_runs: int = 5,
    random_state_base: int = 42,
):
    """
    Valuta diversi K su X_pca, con più run per K per misurare stabilità
    + 'sep_outcome' (separazione sugli esiti reali).
    """
    results = []

    y_home = y_outcome["is_home_win"].values.astype(float)
    y_away = y_outcome["is_away_win"].values.astype(float)
    y_ou25 = y_outcome["is_over25"].values.astype(float)

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

        # SEP_OUTCOME: quanto i cluster sono diversi negli esiti reali
        labels0 = labels_runs[0]
        tmp = pd.DataFrame(
            {
                "cluster": labels0,
                "is_home_win": y_home,
                "is_away_win": y_away,
                "is_over25": y_ou25,
            }
        ).dropna(subset=["is_home_win", "is_away_win", "is_over25"])

        g = tmp.groupby("cluster")[["is_home_win", "is_away_win", "is_over25"]].mean()

        if len(g) > 1:
            std_home = g["is_home_win"].std(ddof=0)
            std_away = g["is_away_win"].std(ddof=0)
            std_ou25 = g["is_over25"].std(ddof=0)
            sep_outcome = 0.4 * std_home + 0.4 * std_away + 0.2 * std_ou25
        else:
            sep_outcome = 0.0

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski": cal,
                "davies": db,
                "stability": stability,
                "sep_outcome": float(sep_outcome),
            }
        )

    return results


def main():
    print("====================================================")
    print("🚀 STEP3A 1X2 V2 — Clustering avanzato mercato 1X2 (OUTCOME-AWARE + FEATURE-WEIGHTED)")
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

    # Allineo gli outcome alle righe di df_feat (via match_id)
    df_y = df_feat[["match_id"]].merge(df_master, on="match_id", how="left")

    # 🔹 Usa SOLO match chiusi per addestrare (no fixture)
    mask_train = df_y["is_home_win"].notna() & df_y["is_away_win"].notna() & df_y["is_over25"].notna()

    df_feat_train = df_feat.loc[mask_train].reset_index(drop=True)
    df_y_train    = df_y.loc[mask_train].reset_index(drop=True)

    print(f"✅ Righe usate per training (match chiusi): {df_feat_train.shape[0]}")
    print(f"🚫 Righe escluse dal training (fixture o match senza esito): {df_feat.shape[0] - df_feat_train.shape[0]}")

    # --------------------------------------------------
    # 2) Selezione feature numeriche per clustering
    # --------------------------------------------------
    numeric_cols = [
        c
        for c in df_feat_train.columns
        if df_feat_train[c].dtype != "object"
        and not np.issubdtype(df_feat_train[c].dtype, np.datetime64)
    ]

    print(f"🔢 N feature numeriche usate per clustering: {len(numeric_cols)}")

    # --------------------------------------------------
    # 3) Calcolo pesi outcome-aware sulle feature
    # --------------------------------------------------
    feature_weights = _compute_feature_weights(df_feat_train, df_y_train, numeric_cols)
    assert feature_weights.shape[0] == len(numeric_cols)

    # --------------------------------------------------
    # 4) Preparazione X: imputazione, standardizzazione, weighting
    # --------------------------------------------------
    # TRAIN solo su match chiusi
    X_train = df_feat_train[numeric_cols].astype(float).values

    # NaN → media colonna
    col_means = np.nanmean(X_train, axis=0)
    idx_nan = np.where(np.isnan(X_train))
    if idx_nan[0].size > 0:
        X_train[idx_nan] = np.take(col_means, idx_nan[1])

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)

    X_scaled_weighted_train = X_scaled_train * feature_weights

    n_components = min(8, X_scaled_weighted_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_train = pca.fit_transform(X_scaled_weighted_train)

    print(f"🎛 PCA — n_components usati: {n_components}")
    explained = pca.explained_variance_ratio_.sum()
    print(f"   Varianza spiegata totale PCA: {explained:.3f}")

    # --------------------------------------------------
    # 6) Ricerca K ottimale (K in [6..14]) con sep_outcome
    # --------------------------------------------------
    Ks = list(range(6, 15))
    results = _evaluate_k_range(
        X_pca_train,
        Ks,
        y_outcome=df_y_train,
        n_runs=5,
        random_state_base=42,
    )

    sils = np.array([r["silhouette"] for r in results], dtype=float)
    cals = np.array([r["calinski"] for r in results], dtype=float)
    dbs = np.array([r["davies"] for r in results], dtype=float)
    stabs = np.array([r["stability"] for r in results], dtype=float)
    seps = np.array([r["sep_outcome"] for r in results], dtype=float)

    sil_norm = _normalize(sils)
    cal_norm = _normalize(cals)
    db_norm_inv = _normalize(-dbs)  # meno davies = meglio
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

    # --------------------------------------------------
    # 7) Fit finale KMeans con K ottimale
    # --------------------------------------------------
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50,
        max_iter=500,
    )
    labels_train = kmeans.fit_predict(X_pca_train)

    X_all = df_feat[numeric_cols].astype(float).values

    # imputazione NaN coerente con il training
    col_means_all = np.nanmean(X_all, axis=0)
    idx_nan_all = np.where(np.isnan(X_all))
    if idx_nan_all[0].size > 0:
        X_all[idx_nan_all] = np.take(col_means_all, idx_nan_all[1])

    X_all_scaled = scaler.transform(X_all)
    X_all_scaled_weighted = X_all_scaled * feature_weights
    X_all_pca = pca.transform(X_all_scaled_weighted)

    labels_all = kmeans.predict(X_all_pca)

    df_feat["cluster_1x2"] = labels_all

    # --------------------------------------------------
    # 8) Merge con target per diagnostica cluster
    # --------------------------------------------------
    df_merge = df_feat.loc[mask_train].merge(df_master, on="match_id", how="left")

    print("----------------------------------------------------")
    print("📊 Distribuzione cluster_1x2:")
    print(df_feat["cluster_1x2"].value_counts().sort_index())

    print("----------------------------------------------------")
    print("📊 Statistiche per cluster (home/draw/away, over25, bk_p1/pic_p1, delta):")
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
        tight_mean=("tightness_index", "mean"),
    )
    print(grp)

    # --------------------------------------------------
    # 9) Salvataggio modello
    # --------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_obj = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "feature_cols": numeric_cols,
        "feature_weights": feature_weights,
        "k": best_k,
        "metrics": results,
    }
    joblib.dump(model_obj, MODEL_PATH)
    print(f"💾 Modello clustering salvato in: {MODEL_PATH}")

    # --------------------------------------------------
    # 10) Salvataggio dataset clusterizzato
    # --------------------------------------------------
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