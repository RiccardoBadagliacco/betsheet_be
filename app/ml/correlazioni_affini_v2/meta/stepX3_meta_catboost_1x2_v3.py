#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X3 v5 — Meta-modello 1X2 CatBoost (NO LEAKAGE, NO CLUSTER)

- Usa il MASTER 1X2:
    data/meta_1x2_master_train_v1.parquet

- Allena CatBoost con tutte le feature numeriche "pulite"
  (nessun cluster_1x2 per evitare mismatch a runtime).

- Output:
    models/meta_1x2_catboost_v5.cbm
    models/meta_1x2_catboost_train_proba_v5.pkl
    models/meta_1x2_catboost_features_v5.json
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

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
MODEL_PATH    = MODEL_DIR / "meta_1x2_catboost_v5.cbm"
PROBA_PATH    = MODEL_DIR / "meta_1x2_catboost_train_proba_v5.pkl"
FEATURES_PATH = MODEL_DIR / "meta_1x2_catboost_features_v5.json"


def main():
    print("===================================================")
    print("🚀 STEP X3 v5 — Meta-modello 1X2 CatBoost (NO LEAKAGE, NO CLUSTER)")
    print("===================================================")
    print(f"📥 MASTER: {MASTER_PATH}")
    print(f"💾 MODEL : {MODEL_PATH}")
    print(f"💾 PROBA : {PROBA_PATH}")

    df = pd.read_parquet(MASTER_PATH)
    print(f"📏 MASTER shape: {df.shape}")

    # ----------------------------------------------------
    # 1) Target y_1x2 = 0 (away) / 1 (draw) / 2 (home)
    # ----------------------------------------------------
    def encode_outcome(r):
        if r["is_home_win"] == 1:
            return 2
        elif r["is_draw"] == 1:
            return 1
        elif r["is_away_win"] == 1:
            return 0
        else:
            return np.nan

    df["y_1x2"] = df.apply(encode_outcome, axis=1)
    df = df[df["y_1x2"].notna()].copy()
    df["y_1x2"] = df["y_1x2"].astype(int)
    print(f"✅ Righe con y_1x2 valida: {df.shape[0]}")

    # ----------------------------------------------------
    # 2) Selezione feature NUMERICHE senza leakage
    #    (escludiamo colonne di outcome, cluster, risultato, ecc.)
    # ----------------------------------------------------
    cols_blocklist = {
        "is_home_win", "is_draw", "is_away_win",
        "is_over05", "is_over15", "is_over25", "is_over35",
        "home_ft", "away_ft", "total_goals",
        "y_1x2",
        "cluster_1x2",  # ⚠️ ESCLUSO VOLONTARIAMENTE
    }

    numeric_cols = [
        c for c in df.columns
        if (df[c].dtype != "object")
        and (not np.issubdtype(df[c].dtype, np.datetime64))
        and (c not in cols_blocklist)
    ]

    print(f"🔢 N feature numeriche candidate: {len(numeric_cols)}")

    # Ordine esplicito per stabilità (facoltativo, ma utile)
    FEATURE_COLS = [
        # Probabilità mercato e picchetto
        "bk_p1", "bk_px", "bk_p2",
        "bk_sum_1x2", "bk_overround_1x2",
        "entropy_bk_1x2",
        "pic_p1", "pic_px", "pic_p2",
        "pic_sum_1x2", "pic_overround_1x2",
        "entropy_pic_1x2",
        "delta_p1", "delta_px", "delta_p2",
        "delta_1x2_abs_sum",
        "cosine_bk_pic_1x2",
        "market_sharpness",
        "instability_index",
        # ELO / strength
        "elo_home_pre", "elo_away_pre", "elo_diff",
        "elo_diff_raw", "elo_diff_raw_z",
        "team_strength_home", "team_strength_away", "team_strength_diff",
        # Forma
        "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
        "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
        "home_form_ga_avg_lastN",  "away_form_ga_avg_lastN",
        "home_form_win_rate_lastN", "away_form_win_rate_lastN",
        "home_form_matches_lastN",  "away_form_matches_lastN",
        "form_pts_diff_raw", "form_pts_diff_raw_z",
        "form_gf_diff_raw",  "form_gf_diff_raw_z",
        "form_ga_diff_raw",  "form_ga_diff_raw_z",
        # Goal model
        "lambda_home_form", "lambda_away_form", "lambda_total_form",
        "lambda_total_market_ou25",
        "goal_supremacy_market_ou25",
        "goal_supremacy_form",
        # Info mercato extra
        "fav_prob_1x2",
        "market_balance_index_1x2",
        "fav_prob_gap_1x2",
        "second_fav_prob_1x2",
        # Calendario / stress
        "season_recency",
        "match_density_index",
        "days_since_last_home",
        "days_since_last_away",
        "rest_diff_days",
        "short_rest_home",
        "short_rest_away",
        "rest_advantage_home",
        "rest_advantage_away",
        "tightness_index",
    ]

    # Teniamo solo quelle realmente presenti nel dataset
    FEATURE_COLS = [c for c in FEATURE_COLS if c in numeric_cols]
    print("🔧 Feature usate:")
    for c in FEATURE_COLS:
        print(f"  - {c}")
    print(f"🔢 N feature usate: {len(FEATURE_COLS)}")

    X_all = df[FEATURE_COLS].astype(float).values
    y_all = df["y_1x2"].values

    # imputazione semplice: NaN → media colonna
    col_means = np.nanmean(X_all, axis=0)
    idx_nan = np.where(np.isnan(X_all))
    if idx_nan[0].size > 0:
        X_all[idx_nan] = np.take(col_means, idx_nan[1])

    # ----------------------------------------------------
    # 3) Train / Valid split
    # ----------------------------------------------------
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all,
        y_all,
        test_size=0.25,
        random_state=42,
        stratify=y_all,
    )

    print(f"📊 Train={X_train.shape[0]}  Valid={X_valid.shape[0]}")
    print(f"📊 Distribuzione y_train (0,1,2): {np.bincount(y_train)}")

    # Class weights (draw pesato)
    counts = np.bincount(y_train)
    w_away = 1.2
    w_draw = 2.2
    w_home = 1.0
    class_weights = {
        0: w_away,
        1: w_draw,
        2: w_home,
    }
    print(f"⚖️ class_weights: {class_weights}")

    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    # ----------------------------------------------------
    # 4) CatBoost training
    # ----------------------------------------------------
    print("🚀 Training CatBoost v5 (no leakage, no cluster)...")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        od_type="Iter",
        od_wait=80,
        class_weights=class_weights,
        verbose=100,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )

    # ----------------------------------------------------
    # 5) Valutazione su VALID
    # ----------------------------------------------------
    p_valid = model.predict_proba(valid_pool)
    y_pred = p_valid.argmax(axis=1)

    acc = accuracy_score(y_valid, y_pred)
    ll = log_loss(y_valid, p_valid)

    print("---------------------------------------------------")
    print("📊 VALID SET")
    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📊 LogLoss : {ll:.4f}")
    print(confusion_matrix(y_valid, y_pred))
    print(classification_report(y_valid, y_pred, digits=3))

    # ----------------------------------------------------
    # 6) Salvataggi
    # ----------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"💾 Salvato modello: {MODEL_PATH}")

    joblib.dump(
        {
            "proba": p_valid,
            "y": y_valid,
        },
        PROBA_PATH,
    )
    print(f"💾 Salvato PROBA (per calibrazione): {PROBA_PATH}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"💾 Salvato FEATURE LIST: {FEATURES_PATH}")

    print("🏁 STEP X3 v5 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
