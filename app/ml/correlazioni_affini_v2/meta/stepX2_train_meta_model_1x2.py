#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X2 — Meta-modello 1X2 v1 (Logistic Regression)

Usa il MASTER 1X2 per allenare un modello probabilistico:
    P(home win), P(draw), P(away win)

Input:
    - data/meta_1x2_master_train_v1.parquet

Output:
    - models/meta_1x2_logreg_v1.pkl
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"
MODEL_DIR  = AFFINI_DIR / "models"

MASTER_PATH = DATA_DIR / "meta_1x2_master_train_v1.parquet"
MODEL_PATH  = MODEL_DIR / "meta_1x2_logreg_v1.pkl"


def main():
    print("===================================================")
    print("🚀 STEP X2 — Meta-modello 1X2 v1 (LogisticRegression)")
    print("===================================================")
    print(f"📥 MASTER: {MASTER_PATH}")
    print(f"💾 MODEL : {MODEL_PATH}")

    df = pd.read_parquet(MASTER_PATH)
    print(f"📏 df MASTER shape: {df.shape}")

    # ----------------------------------------------------
    # 1) Filtra solo match con esito (no fixture)
    # ----------------------------------------------------
    mask_train = df["is_home_win"].notna() & df["is_away_win"].notna()
    df_train = df.loc[mask_train].copy()
    print(f"✅ Righe train (con esito): {df_train.shape[0]}")

    # target numerico: 2=home, 1=draw, 0=away (ordine arbitrario)
    def encode_outcome(r):
        if r["is_home_win"] == 1:
            return 2
        elif r["is_draw"] == 1:
            return 1
        elif r["is_away_win"] == 1:
            return 0
        else:
            return np.nan

    df_train["y_1x2"] = df_train.apply(encode_outcome, axis=1)
    df_train = df_train[df_train["y_1x2"].notna()].copy()
    df_train["y_1x2"] = df_train["y_1x2"].astype(int)

    print(f"✅ Righe con y_1x2 valida: {df_train.shape[0]}")

    # ----------------------------------------------------
    # 2) Scelta feature per v1
    # ----------------------------------------------------
    feature_cols = [
        # probabilità mercato
        "bk_p1", "bk_px", "bk_p2",
        # picchetto tecnico
        "pic_p1", "pic_px", "pic_p2",
        # fattori tecnici
        "elo_diff",
        "lambda_total_form",
        "team_strength_diff",
        # Delta mercato vs tecnico
        "delta_p1", "delta_px", "delta_p2",
        "delta_1x2_abs_sum",
        # shape mercato
        "entropy_bk_1x2", "entropy_pic_1x2",
        "market_balance_index_1x2",
        "fav_prob_1x2",
        "fav_prob_gap_1x2",
        "second_fav_prob_1x2",
        # forma
        "form_pts_diff_raw",
        "form_gf_diff_raw",
        "form_ga_diff_raw",
        # calendario / stress
        "season_recency",
        "match_density_index",
        "tightness_index",
    ]

    feature_cols = [c for c in feature_cols if c in df_train.columns]
    print(f"🔢 N feature usate: {len(feature_cols)}")

    X = df_train[feature_cols].astype(float).values
    y = df_train["y_1x2"].values

    # imputazione semplice: NaN → media colonna
    col_means = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    if idx_nan[0].size > 0:
        X[idx_nan] = np.take(col_means, idx_nan[1])

    # ----------------------------------------------------
    # 3) Train/test split
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # ----------------------------------------------------
    # 4) Fit Logistic Regression multinomiale
    # ----------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )
    clf.fit(X_train_std, y_train)

    # ----------------------------------------------------
    # 5) Valutazione
    # ----------------------------------------------------
    proba_test = clf.predict_proba(X_test_std)
    y_pred = clf.predict(X_test_std)

    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, proba_test)

    print("---------------------------------------------------")
    print(f"📊 Accuracy test : {acc:.4f}")
    print(f"📊 Log-loss test : {ll:.4f}")
    print("---------------------------------------------------")
    print("📊 Confusion matrix (0=away,1=draw,2=home):")
    print(confusion_matrix(y_test, y_pred))
    print("---------------------------------------------------")
    print("📊 Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # ----------------------------------------------------
    # 6) Salvataggio modello
    # ----------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": clf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "y_mapping": {
            "class_0": "away_win",
            "class_1": "draw",
            "class_2": "home_win",
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"💾 Modello salvato in: {MODEL_PATH}")
    print("🏁 STEP X2 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
