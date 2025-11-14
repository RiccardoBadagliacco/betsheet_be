# app/utils/check/check_cluster_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/cluster_features.parquet")


def main():
    print("📥 Carico cluster_features...\n")
    df = pd.read_parquet(INPUT)

    print(f"📏 Shape: {df.shape}\n")

    print("📄 Prime righe:")
    print(df.head(), "\n")

    # METADATA CHECK
    print("🔍 Controllo colonne metadata...")
    required_meta = ["match_id", "league", "country", "season"]
    missing_meta = [c for c in required_meta if c not in df.columns]

    if missing_meta:
        print(f"❌ Mancano metadati: {missing_meta}\n")
    else:
        print("🟢 Metadata OK\n")

    # NUMERICAL CHECK
    num_cols = [
        c for c in df.columns
        if c not in ["match_id", "league", "country", "season"]
    ]

    print("🔍 NaN per colonna:")
    print(df[num_cols].isna().sum(), "\n")

    print("🔍 Valori inf:")
    print(np.isinf(df[num_cols]).sum(), "\n")

    print("📈 Statistiche descrittive:")
    print(df[num_cols].describe().T, "\n")

    # Probability sanity check
    print("🔍 Controllo range probabilità...")
    prob_cols = [c for c in num_cols if "p1" in c or "px" in c or "p2" in c or "pO25" in c or "pU25" in c]

    out_of_range = {}
    for col in prob_cols:
        mask = (~df[col].between(0, 1, inclusive="both")) & (~df[col].isna())
        if mask.any():
            out_of_range[col] = mask.sum()

    if out_of_range:
        print(f"⚠️ Probabilità fuori range trovate: {out_of_range}\n")
    else:
        print("🟢 Tutte le probabilità sono nel range [0,1]\n")

    # PCA CHECK
    print("🧪 PCA check...")

    X = df[num_cols].fillna(df[num_cols].mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    try:
        comp = pca.fit_transform(Xs)
        print("➡ PCA OK")
        print("📊 Varianza spiegata:", pca.explained_variance_ratio_)
    except Exception as e:
        print("❌ PCA FAILED:", e)

    print("\n🟢 CHECK cluster_features COMPLETATO!")


if __name__ == "__main__":
    main()