#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X3 v2 — Meta-modello 1X2 con CatBoost (NO LEAKAGE, class_weights)

- Input:
    data/meta_1x2_master_train_v1.parquet

- Output:
    models/meta_1x2_catboost_v2.cbm
    models/meta_1x2_catboost_features_v2.json

Il modello è leakage-free: usa solo feature disponibili PREMATCH.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

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

MASTER_PATH   = DATA_DIR / "meta_1x2_master_train_v1.parquet"
MODEL_PATH    = MODEL_DIR / "meta_1x2_catboost_v2.cbm"
FEATS_JSON    = MODEL_DIR / "meta_1x2_catboost_features_v2.json"
CALIB_INPUT   = MODEL_DIR / "meta_1x2_catboost_train_proba_v2.pkl"  # per step X4


def encode_outcome_row(r) -> int | float:
    """
    2 = home, 1 = draw, 0 = away
    """
    if r["is_home_win"] == 1:
        return 2
    elif r["is_draw"] == 1:
        return 1
    elif r["is_away_win"] == 1:
        return 0
    else:
        return np.nan


def main():
    print("===================================================")
    print("🚀 STEP X3 v2 — Meta-modello 1X2 CatBoost (NO LEAKAGE)")
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

    df_train["y_1x2"] = df_train.apply(encode_outcome_row, axis=1)
    df_train = df_train[df_train["y_1x2"].notna()].copy()
    df_train["y_1x2"] = df_train["y_1x2"].astype(int)
    print(f"✅ Righe con y_1x2 valida: {df_train.shape[0]}")

    # ----------------------------------------------------
    # 2) Selezione feature PREMATCH (NO LEAKAGE)
    # ----------------------------------------------------
    # blacklist di colonne da ESCLUDERE sicuro
    leak_cols = {
        "home_ft", "away_ft", "total_goals",
        "is_home_win", "is_draw", "is_away_win",
        "is_over05", "is_over15", "is_over25",
        "is_over35", "is_under25",
        "goal_supremacy_real", "goal_supremacy_error_market", "goal_supremacy_error_form",
    }

    # teniamo tutte le numeriche tranne quelle di leakage + target
    numeric_cols = [
        c for c in df_train.columns
        if c not in leak_cols
        and c not in {"y_1x2"}
        and df_train[c].dtype != "object"
        and not np.issubdtype(df_train[c].dtype, np.datetime64)
    ]

    # opzionale: puoi anche fare una lista manuale per pieno controllo
    print(f"🔢 N feature numeriche candidate: {len(numeric_cols)}")

    feature_cols = numeric_cols  # v2: usiamo tutte le numeriche pulite

    print("🔧 Feature usate:")
    for c in feature_cols:
        print("  -", c)

    X = df_train[feature_cols].astype(float).values
    y = df_train["y_1x2"].values

    # imputazione NaN → media colonna
    col_means = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    if idx_nan[0].size > 0:
        X[idx_nan] = np.take(col_means, idx_nan[1])

    # ----------------------------------------------------
    # 3) Split train/valid per early stopping & metriche
    # ----------------------------------------------------
    # Per CatBoost possiamo usare idx per train/valid
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)

    split = int(n * 0.75)
    train_idx = idx[:split]
    valid_idx = idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    # ----------------------------------------------------
    # 4) Class weights (draw leggermente più pesato)
    # ----------------------------------------------------
    # distribuzione grezza
    _, counts = np.unique(y_train, return_counts=True)
    n_away, n_draw, n_home = counts  # 0,1,2
    print("📊 Distribuzione y_train (0,1,2):", counts)

    # pesi: draw più pesato, away leggermente più pesato
    class_weights = [1.2, 2.5, 1.0]
    print("⚖️ class_weights:", class_weights)

    # ----------------------------------------------------
    # 5) Fit CatBoost
    # ----------------------------------------------------
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "learning_rate": 0.03,
        "depth": 7,
        "iterations": 1200,
        "random_seed": 42,
        "l2_leaf_reg": 4.0,
        "border_count": 128,
        "od_type": "Iter",
        "od_wait": 80,
        "class_weights": class_weights,
        "verbose": 200,
    }

    print("🚀 Training CatBoost v2 (no leakage, weighted classes)...")
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool)

    # ----------------------------------------------------
    # 6) Valutazione su VALID
    # ----------------------------------------------------
    proba_valid = model.predict_proba(X_valid)
    y_pred = np.argmax(proba_valid, axis=1)

    acc = accuracy_score(y_valid, y_pred)
    ll  = log_loss(y_valid, proba_valid)

    print("---------------------------------------------------")
    print(f"📊 Accuracy VALID : {acc:.4f}")
    print(f"📊 Log-loss VALID : {ll:.4f}")
    print("---------------------------------------------------")
    print("📊 Confusion matrix (0=away,1=draw,2=home):")
    print(confusion_matrix(y_valid, y_pred))
    print("---------------------------------------------------")
    print("📊 Classification report:")
    print(classification_report(y_valid, y_pred, digits=3))

    # ----------------------------------------------------
    # 7) Salvataggio modello + feature list
    # ----------------------------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"💾 Modello CatBoost salvato in: {MODEL_PATH}")

    with open(FEATS_JSON, "w") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)
    print(f"💾 Salvate feature in: {FEATS_JSON}")

    # ----------------------------------------------------
    # 8) Salva anche proba train per calibrazione (STEP X4)
    # ----------------------------------------------------
    full_pool = Pool(X, label=y)
    proba_full = model.predict_proba(full_pool)

    calib_payload = {
        "proba": proba_full,
        "y": y,
    }
    joblib.dump(calib_payload, CALIB_INPUT)
    print(f"💾 Salvato payload calibrazione in: {CALIB_INPUT}")

    print("🏁 STEP X3 v2 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
