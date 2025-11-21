#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP3C OU1.5 V2 — Clustering avanzato mercato OU1.5
(OUTCOME-AWARE + FEATURE-WEIGHTED)

Input feature:
    data/step2b_ou15_features_v2.parquet

Input master (target):
    data/step1c_dataset_with_elo_form.parquet

Output modello:
    models/cluster_ou15_kmeans_v2.pkl

Output dataset clusterizzato:
    data/step3c_ou15_clusters_v2.parquet

Strategia:
- Usa tutte le feature numeriche di OU1.5 (no colonne object/datetime, no meta).
- Merge con master per avere il target binario is_over15.
- OUTCOME-AWARE:
    * per ogni feature calcola mutual information rispetto a is_over15
    * le MI vengono normalizzate in [0,1] → pesi di feature
- FEATURE-WEIGHTED:
    * StandardScaler → X_scaled
    * X_weighted = X_scaled * weights_feature
    * PCA su X_weighted (n_components ~ 8, varianza ~0.9+)
- Ricerca K in [6..14] con metrica combinata:
    score = 0.30 * sil_norm
          + 0.20 * cal_norm
          + 0.15 * db_inv_norm
          + 0.15 * stab_norm
          + 0.20 * sep_norm

  dove sep_outcome misura quanto i cluster sono separati su:
    - is_over15
    - lambda_total_form
    - lambda_total_mix_ou15
    - pic_pO15
    - tech_pO15_mix

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
from sklearn.feature_selection import mutual_info_classif

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"
MODEL_DIR = AFFINI_DIR / "models"

FEATURE_FILE = DATA_DIR / "step2b_ou15_features_v2.parquet"
MASTER_FILE = DATA_DIR / "step1c_dataset_with_elo_form.parquet"

OUT_CLUSTER_FILE = DATA_DIR / "step3c_ou15_clusters_v2.parquet"
OUT_MODEL_FILE = MODEL_DIR / "cluster_ou15_kmeans_v2.pkl"


# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizza un array in [0,1] (se costante → 0.5)."""
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def _evaluate_k_range(
    X_pca: np.ndarray,
    Ks: list[int],
    df_valid: pd.DataFrame,
    n_runs: int = 4,
    random_state_base: int = 100,
):
    """
    Valuta diversi K su X_pca, con più run per K per misurare stabilità + separazione su outcome OU1.5.

    df_valid deve contenere, allineato riga-per-riga a X_pca:
        - is_over15
        - lambda_total_form
        - lambda_total_mix_ou15
        - pic_pO15
        - tech_pO15_mix
    """
    results = []

    y_over15 = df_valid["is_over15"].values.astype(float)
    lam_form = df_valid.get("lambda_total_form", pd.Series(index=df_valid.index, data=np.nan)).values.astype(float)
    lam_mix = df_valid.get("lambda_total_mix_ou15", pd.Series(index=df_valid.index, data=np.nan)).values.astype(float)
    pic_pO15 = df_valid.get("pic_pO15", pd.Series(index=df_valid.index, data=np.nan)).values.astype(float)
    tech_pO15 = df_valid.get("tech_pO15_mix", pd.Series(index=df_valid.index, data=np.nan)).values.astype(float)

    for k in Ks:
        print(f"🔍 Valutazione K = {k} ...")

        labels_runs = []
        sil_scores = []
        cal_scores = []
        db_scores = []

        for r in range(n_runs):
            seed = random_state_base + k * 10 + r
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

        # Stabilità via ARI medio tra run
        ari_vals = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                ari_vals.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
        stability = float(np.mean(ari_vals)) if ari_vals else np.nan

        # ---------------- OUTCOME SEPARATION (sep_outcome) ----------------
        labels0 = labels_runs[0]
        tmp = pd.DataFrame(
            {
                "cluster": labels0,
                "is_over15": y_over15,
                "lambda_total_form": lam_form,
                "lambda_total_mix_ou15": lam_mix,
                "pic_pO15": pic_pO15,
                "tech_pO15_mix": tech_pO15,
            }
        )

        tmp = tmp.dropna(
            subset=[
                "is_over15",
                "lambda_total_form",
                "lambda_total_mix_ou15",
                "pic_pO15",
                "tech_pO15_mix",
            ]
        )

        if tmp.empty or tmp["cluster"].nunique() < 2:
            sep_outcome = 0.0
        else:
            g = tmp.groupby("cluster").agg(
                pO15=("is_over15", "mean"),
                lam_form_mean=("lambda_total_form", "mean"),
                lam_mix_mean=("lambda_total_mix_ou15", "mean"),
                pic_mean=("pic_pO15", "mean"),
                tech_mean=("tech_pO15_mix", "mean"),
            )
            std_pO15 = g["pO15"].std(ddof=0)
            std_lam_form = g["lam_form_mean"].std(ddof=0)
            std_lam_mix = g["lam_mix_mean"].std(ddof=0)
            std_pic = g["pic_mean"].std(ddof=0)
            std_tech = g["tech_mean"].std(ddof=0)

            # pesi: over15 dominante, poi lambda / quote / tecnico
            sep_outcome = (
                0.40 * std_pO15
                + 0.20 * std_lam_form
                + 0.20 * std_lam_mix
                + 0.10 * std_pic
                + 0.10 * std_tech
            )

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


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🚀 STEP3C OU1.5 V2 — Clustering avanzato OU1.5 (OUTCOME-AWARE + FEATURE-WEIGHTED)")
    print("=" * 60)
    print(f"📥 Feature file: {FEATURE_FILE}")
    print(f"📥 Master file : {MASTER_FILE}")
    print(f"💾 Modello out : {OUT_MODEL_FILE}")
    print(f"💾 Clusters out: {OUT_CLUSTER_FILE}")

    # 1) Load data
    df_feat = pd.read_parquet(FEATURE_FILE)
    df_master = pd.read_parquet(MASTER_FILE)[["match_id", "is_over15", "tightness_index"]]
    df = df_feat.merge(df_master, on="match_id", how="left")

    # Fix tightness duplicates
    tix = [c for c in df.columns if c.startswith("tightness_index")]
    if len(tix) == 2 and "tightness_index_x" in tix:
        df["tightness_index"] = df["tightness_index_x"]
        df = df.drop(columns=["tightness_index_x", "tightness_index_y"])
    elif len(tix) > 2:
        df["tightness_index"] = df[tix].mean(axis=1)
        for c in tix:
            if c != "tightness_index":
                df = df.drop(columns=c)

    # 2) Valid rows
    mask_valid = df["tech_pO15_mix"].notna() & df["pic_pO15"].notna()
    df_valid = df.loc[mask_valid].reset_index(drop=True)
    df_feat_valid = df_feat.loc[mask_valid].reset_index(drop=True)

    # 3) TRAINING ONLY — match chiusi
    mask_train = df_valid["is_over15"].notna()
    df_train = df_valid.loc[mask_train].reset_index(drop=True)
    df_feat_train = df_feat_valid.loc[mask_train].reset_index(drop=True)

    # 4) Numeric features from training only
    meta_cols = ["match_id", "date", "season", "league", "home_team", "away_team"]
    numeric_cols = [
        c for c in df_feat_train.columns
        if c not in meta_cols
        and df_feat_train[c].dtype != "object"
        and not np.issubdtype(df_feat_train[c].dtype, np.datetime64)
    ]

    # 5) Prepare X_train
    X_train = df_feat_train[numeric_cols].astype(float).values
    col_means = np.nanmean(X_train, axis=0)
    idx_nan = np.where(np.isnan(X_train))
    if idx_nan[0].size > 0:
        X_train[idx_nan] = np.take(col_means, idx_nan[1])

    y_train = df_train["is_over15"].astype(int).values

    # 6) Scaling + MI weights
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)

    mi_scores = mutual_info_classif(X_scaled_train, y_train, random_state=42)
    mi_scores = np.nan_to_num(mi_scores, nan=0.0)
    weights = mi_scores / (mi_scores.max() if mi_scores.max() > 0 else 1.0)
    weights = 0.2 + 0.8 * weights
    X_weighted_train = X_scaled_train * weights

    # 7) PCA
    n_components = min(8, X_weighted_train.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_train = pca.fit_transform(X_weighted_train)

    # 8) K evaluation
    Ks = list(range(6, 15))
    results = _evaluate_k_range(
        X_pca_train,
        Ks,
        df_train,
        n_runs=4,
        random_state_base=200,
    )

    sils = np.array([r["silhouette"] for r in results])
    cals = np.array([r["calinski"] for r in results])
    dbs = np.array([r["davies"] for r in results])
    stbs = np.array([r["stability"] for r in results])
    seps = np.array([r["sep_outcome"] for r in results])

    sil_norm = _normalize(sils)
    cal_norm = _normalize(cals)
    db_norm_inv = _normalize(-dbs)
    stab_norm = _normalize(stbs)
    sep_norm = _normalize(seps)

    scores = (
        0.30 * sil_norm +
        0.20 * cal_norm +
        0.15 * db_norm_inv +
        0.15 * stab_norm +
        0.20 * sep_norm
    )

    best_idx = int(np.nanargmax(scores))
    best_k = results[best_idx]["k"]

    # 9) Final training
    kmeans_final = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50,
        max_iter=500,
    )
    kmeans_final.fit(X_pca_train)

    # 10) Apply model to ALL valid rows
    X_all = df_feat_valid[numeric_cols].astype(float).values
    col_means_all = np.nanmean(X_all, axis=0)
    idx_nan_all = np.where(np.isnan(X_all))
    if idx_nan_all[0].size > 0:
        X_all[idx_nan_all] = np.take(col_means_all, idx_nan_all[1])

    X_all_scaled = scaler.transform(X_all)
    X_all_weighted = X_all_scaled * weights
    X_all_pca = pca.transform(X_all_weighted)

    df_valid["cluster_ou15"] = kmeans_final.predict(X_all_pca)

    # 11) Save model + dataset
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_obj = {
        "kmeans": kmeans_final,
        "scaler": scaler,
        "pca": pca,
        "feature_cols": numeric_cols,
        "mi_scores": dict(zip(numeric_cols, mi_scores)),
        "weights": dict(zip(numeric_cols, weights)),
        "k": best_k,
        "metrics": results,
    }
    joblib.dump(model_obj, OUT_MODEL_FILE)

    df_valid.to_parquet(OUT_CLUSTER_FILE, index=False)

    print(f"Modello salvato in: {OUT_MODEL_FILE}")
    print(f"Dataset clusterizzato salvato in: {OUT_CLUSTER_FILE}")
    print("🏁 STEP3C OU1.5 COMPLETATO")
    print("=" * 60)


if __name__ == "__main__":
    main()