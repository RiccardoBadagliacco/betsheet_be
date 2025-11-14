# app/utils/check/check_gmm_clusters.py

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import joblib

INPUT = Path("data/cluster_assignments.parquet")
SCALER = Path("app/models/scaler_full.pkl")
PCA_MODEL = Path("app/models/pca_full.pkl")
GMM_MODEL = Path("app/models/gmm_model_full.pkl")


def main():
    print("📥 Carico assegnazioni cluster...")
    df = pd.read_parquet(INPUT)

    scaler = joblib.load(SCALER)
    pca = joblib.load(PCA_MODEL)
    gmm = joblib.load(GMM_MODEL)

    # -------------------------------------------------
    # 🎯 FEATURE NUMERICHE DA SCALARE + PCA
    # -------------------------------------------------
    skip_cols = [
        "match_id", "league", "country", "season",
        "home_team", "away_team", "cluster"
    ]

    num_cols = [c for c in df.columns if c not in skip_cols]

    X = df[num_cols].fillna(df[num_cols].mean())
    Xs = scaler.transform(X)
    Xp = pca.transform(Xs)

    # -------------------------------------------------
    # 🧪 SILHOUETTE
    # -------------------------------------------------
    print("🧪 Calcolo silhouette score...")
    sil = silhouette_score(Xp, df["cluster"])
    print(f"📊 Silhouette score: {sil:.4f}")

    # -------------------------------------------------
    # 📦 DISTRIBUZIONE CLUSTER
    # -------------------------------------------------
    print("\n📦 Dimensione cluster:")
    print(df["cluster"].value_counts())

    # -------------------------------------------------
    # 🎯 CENTROIDI NELLO SPAZIO PCA
    # -------------------------------------------------
    print("\n🎯 Centroidi nello spazio PCA:")

    centers = gmm.means_  # shape: (k, n_components)
    pca_cols = [f"PCA_{i+1}" for i in range(centers.shape[1])]

    print(pd.DataFrame(centers, columns=pca_cols).round(6).head())

    # -------------------------------------------------
    # 🧪 ARI TRA STAGIONI (stabilità temporale)
    # -------------------------------------------------
    print("\n🧪 ARI Season-to-Season Stability Check...")

    if "season" not in df.columns:
        print("⚠️ Nessuna colonna 'season'. Salto ARI.")
    else:
        seasons = sorted(df["season"].unique())

        for i in range(len(seasons) - 1):
            s1, s2 = seasons[i], seasons[i + 1]

            df1 = df[df["season"] == s1].sort_values("match_id")
            df2 = df[df["season"] == s2].sort_values("match_id")

            # ARI richiede stesso numero di osservazioni
            if len(df1) != len(df2):
                print(f"  {s1} → {s2}: ARI = N/A (size mismatch)")
                continue

            ari = adjusted_rand_score(df1["cluster"], df2["cluster"])
            print(f"  {s1} → {s2}: ARI = {ari:.4f}")

    print("\n🟢 CHECK CLUSTER COMPLETATO!")


if __name__ == "__main__":
    main()