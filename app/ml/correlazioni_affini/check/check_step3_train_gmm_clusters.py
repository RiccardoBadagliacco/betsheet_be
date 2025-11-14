# check_step3_train_gmm_clusters.py

import pandas as pd
from pathlib import Path
import joblib

# FIX: usare parents[2]
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "correlazioni_affini" / "data"
MODELS = BASE / "correlazioni_affini" / "models"

FILE = DATA / "step3_cluster_assignments.parquet"

SCALER = MODELS / "scaler_full.pkl"
PCA = MODELS / "pca_full.pkl"
GMM = MODELS / "gmm_model_full.pkl"


def main():
    print("📥 Carico cluster_assignments...")
    df = pd.read_parquet(FILE)
    print("📏 Shape:", df.shape, "\n")

    print(df.head(), "\n")

    print("🔢 Cluster unici trovati:", df["cluster"].unique())
    print("📊 Distribuzione cluster:")
    print((df["cluster"].value_counts(normalize=True) * 100).round(2), "\n")

    print("🔍 NaN per colonna cluster:")
    print(df["cluster"].isna().sum(), "\n")

    print("📦 Carico scaler / PCA / GMM...")
    try:
        scaler = joblib.load(SCALER)
        pca = joblib.load(PCA)
        gmm = joblib.load(GMM)
        print("🟢 Modelli caricati correttamente!")
    except Exception as e:
        print("❌ ERRORE nel caricamento modelli:", e)


if __name__ == "__main__":
    main()