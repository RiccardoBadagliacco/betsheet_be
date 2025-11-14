# app/ml/correlazioni_affini/step3_train_gmm_clusters.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib

# ---------------------------------------------------------
# PATHS CORRETTI / DEFINITIVI
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]  # .../app/ml/
AFF_DIR = BASE / "correlazioni_affini"

DATA_DIR = AFF_DIR / "data"
MODELS_DIR = AFF_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT = DATA_DIR / "step2b_cluster_features.parquet"

OUT_ASSIGN = DATA_DIR / "step3_cluster_assignments.parquet"
OUT_SCALER = MODELS_DIR / "step3_scaler.pkl"
OUT_PCA = MODELS_DIR / "step3_pca.pkl"
OUT_GMM = MODELS_DIR / "step3_gmm.pkl"

EXCLUDE_ALWAYS = {
    "match_id",
    "league",
    "country",
    "season",
    "home_team",
    "away_team",
    "date",
    # campi di risultato e derivati (se presenti)
    "home_ft",
    "away_ft",
    "total_goals",
    "result_1x2",
    "goal_diff",
    "abs_diff",
}


def main():

    # ---------------------------------------------------------
    # CARICAMENTO
    # ---------------------------------------------------------
    print("📥 Carico step2b_cluster_features...")
    df = pd.read_parquet(INPUT)
    print("Shape:", df.shape)

    # ---------------------------------------------------------
    # FEATURE — SOLO NUMERICHE + NO IDENTIFICATIVI
    # ---------------------------------------------------------
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_ALWAYS
        and df[c].dtype != "object"
    ]

    print(f"➡ Numero feature numeriche usate: {len(feature_cols)}")

    X = df[feature_cols].copy()

    # ---------------------------------------------------------
    # Pulizia NaN
    # ---------------------------------------------------------
    print("🧼 Pulizia NaN (media colonna)...")
    X = X.fillna(X.mean())

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------
    print("📏 Scaling...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # PCA 5 → 25 componenti
    # ---------------------------------------------------------
    print("🧪 PCA test (5 → 25 componenti)...")

    best_pca = None
    best_n = 0
    best_var = -1

    for n in range(5, 26):
        pca = PCA(n_components=n)
        Xp = pca.fit_transform(Xs)
        var = pca.explained_variance_ratio_.sum()

        print(f"   PCA {n} componenti → var = {var:.4f}")

        if var > best_var:
            best_var = var
            best_pca = pca
            best_n = n

    print(f"➡ Miglior PCA: {best_n} componenti (var={best_var:.4f})")

    # Transform finale
    Xp = best_pca.transform(Xs)

    # ---------------------------------------------------------
    # GMM 3 → 12 componenti
    # ---------------------------------------------------------
    print("🤖 Training GMM (k = 3 → 12) con BIC...")

    best_gmm = None
    best_k = None
    best_bic = np.inf

    for k in range(3, 13):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            random_state=42
        )
        gmm.fit(Xp)
        bic = gmm.bic(Xp)

        print(f"   GMM k={k}: BIC={bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm

    print(f"🎯 Miglior modello: k={best_k} (BIC={best_bic:.2f})")

    # ---------------------------------------------------------
    # Assegno cluster
    # ---------------------------------------------------------
    print("🏷 Assegno cluster alle partite...")
    df["cluster"] = best_gmm.predict(Xp)

    # ---------------------------------------------------------
    # Salvataggio
    # ---------------------------------------------------------
    print("💾 Salvataggio modelli...")
    joblib.dump(scaler, OUT_SCALER)
    joblib.dump(best_pca, OUT_PCA)
    joblib.dump(best_gmm, OUT_GMM)

    print(f"💾 Salvo assegnazioni cluster in {OUT_ASSIGN}")
    df.to_parquet(OUT_ASSIGN, index=False)

    print("✅ STEP3 COMPLETATO!")


if __name__ == "__main__":
    main()