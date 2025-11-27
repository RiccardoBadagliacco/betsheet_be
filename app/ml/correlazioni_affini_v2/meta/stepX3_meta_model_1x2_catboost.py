#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X3 FIX — Meta-modello 1X2 v2 (CatBoost LEAKAGE-FREE)
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, classification_report, confusion_matrix
)

# PATHS
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = BASE_DIR / "correlazioni_affini_v2" / "data"
MODEL_DIR  = BASE_DIR / "correlazioni_affini_v2" / "models"

MASTER_PATH = DATA_DIR / "meta_1x2_master_train_v1.parquet"
MODEL_PATH  = MODEL_DIR / "meta_1x2_catboost_fix_v1.cbm"
FEAT_PATH   = MODEL_DIR / "meta_1x2_catboost_fix_features.json"

# ----------------------------------------------------
# FEATURE LIST — GOLDEN SET (no leakage)
# ----------------------------------------------------
FEATURE_COLS = [

    # --- Book ---
    "bk_p1", "bk_px", "bk_p2",
    "bk_sum_1x2", "bk_overround_1x2", "entropy_bk_1x2",

    # --- Picchetto ---
    "pic_p1", "pic_px", "pic_p2",
    "pic_sum_1x2", "pic_overround_1x2", "entropy_pic_1x2",
    "delta_p1", "delta_px", "delta_p2",
    "delta_1x2_abs_sum", "cosine_bk_pic_1x2",
    "market_sharpness", "instability_index",

    # --- Elo ---
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "elo_diff_raw", "elo_diff_raw_z",

    # --- Form ---
    "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
    "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
    "home_form_ga_avg_lastN",  "away_form_ga_avg_lastN",
    "home_form_win_rate_lastN", "away_form_win_rate_lastN",
    "home_form_matches_lastN", "away_form_matches_lastN",
    "form_pts_diff_raw", "form_pts_diff_raw_z",
    "form_gf_diff_raw", "form_gf_diff_raw_z",
    "form_ga_diff_raw", "form_ga_diff_raw_z",

    # --- Lambda ---
    "lambda_home_form", "lambda_away_form", "lambda_total_form",

    # --- Market shape ---
    "fav_prob_1x2", "market_balance_index_1x2",
    "fav_prob_gap_1x2", "second_fav_prob_1x2",

    # --- Calendario ---
    "season_recency", "days_since_last_home", "days_since_last_away",
    "match_density_index", "rest_diff_days",
    "short_rest_home", "short_rest_away",
    "rest_advantage_home", "rest_advantage_away",
    "tightness_index",
]


def main():
    print("🚀 STEP X3 FIX — CatBoost senza leakage")
    df = pd.read_parquet(MASTER_PATH)

    # Filtra solo match giocati
    m = df["is_home_win"].notna()
    df = df[m].copy()

    # Encode target
    def enc(r):
        if r["is_home_win"] == 1: return 2
        if r["is_draw"] == 1: return 1
        return 0

    df["y"] = df.apply(enc, axis=1).astype(int)

    # Filtra colonne esistenti
    feats = [c for c in FEATURE_COLS if c in df.columns]

    X = df[feats]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    train_pool = Pool(X_train, y_train)
    test_pool  = Pool(X_test, y_test)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        depth=8,
        iterations=700,
        learning_rate=0.05,
        random_seed=42,
        class_weights=[1.0, 2.0, 1.0],
        verbose=200
    )

    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict(test_pool).flatten().astype(int)
    proba = model.predict_proba(test_pool)

    print("\n📊 Accuracy:", accuracy_score(y_test, preds))
    print("📊 Logloss :", log_loss(y_test, proba))
    print("📊 Confusion:\n", confusion_matrix(y_test, preds))
    print("📊 Report:\n", classification_report(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(FEAT_PATH, "w") as f:
        json.dump({"features": feats}, f)

    print("💾 Modello salvato in:", MODEL_PATH)


if __name__ == "__main__":
    main()
