#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X5 — Runtime predictor Meta 1X2 (v5)

Uso:
    python stepX5_meta_runtime_predictor.py --input path/to/match_step2b.parquet

Se --input NON è passato:
    - entra in AUTO-TEST MODE:
        prende una riga da step2b_1x2_features_v2.parquet
        la salva come tmp_auto_test_match.parquet
        e la usa come input.

Richiede:
    - models/meta_1x2_catboost_v5.cbm         (modello CatBoost)
    - models/meta_1x2_calibrator_iso_v5.pkl   (calibratore isotonic)
    - models/meta_1x2_catboost_features_v5.json (lista feature in ordine)
"""

from pathlib import Path
import sys
import json
import argparse

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier

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

MODEL_PATH      = MODEL_DIR / "meta_1x2_catboost_v5.cbm"
CALIBRATOR_PATH = MODEL_DIR / "meta_1x2_calibrator_iso_v5.pkl"
FEATURE_PATH    = MODEL_DIR / "meta_1x2_catboost_features_v5.json"

STEP2B_PATH     = DATA_DIR / "step2b_1x2_features_v2.parquet"
TMP_AUTO_PATH   = DATA_DIR / "tmp_auto_test_match.parquet"


# fallback hard-coded (deve coincidere con STEP X3 v5)
FEATURE_COLS_FALLBACK = [
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
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "elo_diff_raw", "elo_diff_raw_z",
    "team_strength_home", "team_strength_away", "team_strength_diff",
    "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
    "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
    "home_form_ga_avg_lastN",  "away_form_ga_avg_lastN",
    "home_form_win_rate_lastN", "away_form_win_rate_lastN",
    "home_form_matches_lastN",  "away_form_matches_lastN",
    "form_pts_diff_raw", "form_pts_diff_raw_z",
    "form_gf_diff_raw",  "form_gf_diff_raw_z",
    "form_ga_diff_raw",  "form_ga_diff_raw_z",
    "lambda_home_form", "lambda_away_form", "lambda_total_form",
    "lambda_total_market_ou25",
    "goal_supremacy_market_ou25",
    "goal_supremacy_form",
    "fav_prob_1x2",
    "market_balance_index_1x2",
    "fav_prob_gap_1x2",
    "second_fav_prob_1x2",
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


def load_feature_cols() -> list[str]:
    if FEATURE_PATH.exists():
        print(f"📥 Carico feature list da {FEATURE_PATH}")
        with open(FEATURE_PATH, "r") as f:
            cols = json.load(f)
        print(f"🔢 Feature caricate: {len(cols)}")
        return cols

    print(f"⚠️ {FEATURE_PATH} non trovato → uso FEATURE_COLS_FALLBACK e genero il JSON…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_PATH, "w") as f:
        json.dump(FEATURE_COLS_FALLBACK, f, indent=2)
    print(f"💾 Salvato {FEATURE_PATH} con {len(FEATURE_COLS_FALLBACK)} feature.")
    return FEATURE_COLS_FALLBACK


def load_models():
    print("📥 Carico modello CatBoost (.cbm)…")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    print("📥 Carico calibratore isotonic…")
    calibrator = joblib.load(CALIBRATOR_PATH)
    return model, calibrator


def auto_test_generate() -> Path:
    print("⚠️ Nessun --input passato → AUTO-TEST MODE")
    print(f"📥 Carico {STEP2B_PATH}…")
    df = pd.read_parquet(STEP2B_PATH)
    if df.empty:
        raise RuntimeError("❌ step2b_1x2_features_v2.parquet è vuoto")

    row = df.sample(1, random_state=123)
    row.to_parquet(TMP_AUTO_PATH, index=False)
    print(f"📌 AUTO-TEST: generato file → {TMP_AUTO_PATH}")
    return TMP_AUTO_PATH


def run_prediction(input_path: Path):
    print(f"📥 Carico match: {input_path}")
    df = pd.read_parquet(input_path)
    if df.empty:
        raise RuntimeError("❌ Il file di input è vuoto")

    if len(df) > 1:
        print(f"⚠️ Il file contiene {len(df)} righe, uso SOLO la prima per il test.")
    row = df.iloc[0]

    feature_cols = load_feature_cols()

    # Controllo feature mancanti
    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        raise RuntimeError(f"❌ Nel file mancano feature richieste:\n{missing}")

    model, calibrator = load_models()

    # Costruisco X (1, n_features)
    X = row[feature_cols].astype(float).values.reshape(1, -1)
    # Per sicurezza: NaN → 0.0
    X = np.nan_to_num(X, nan=0.0)

    # Probabilità grezze CatBoost
    p_raw = model.predict_proba(X)[0]  # shape (3,)
    print(f"\n📊 Probabilità RAW model: [away, draw, home] = {p_raw}")

    # Calibrazione isotonic per classe
    p_cal = []
    for cls in range(3):
        iso = calibrator.get(f"iso_{cls}")
        if iso is None:
            # fallback: nessun calibratore per questa classe → usa p_raw
            p_cal.append(float(p_raw[cls]))
        else:
            v = float(p_raw[cls])
            p_cal.append(float(iso.predict([v])[0]))

    p_cal = np.array(p_cal, dtype=float)

    # Normalizzazione per sicurezza
    s = p_cal.sum()
    if s <= 0:
        p_cal = np.full_like(p_cal, 1.0 / len(p_cal))
    else:
        p_cal = p_cal / s

    p_away, p_draw, p_home = p_cal
    print(f"📊 Probabilità CALIBRATE: [away, draw, home] = {p_cal}")

    # ----------------------------------------------------
    # Output leggibile tipo betting
    # ----------------------------------------------------
    def prob_to_odds(p):
        return np.inf if p <= 0 else 1.0 / p

    print("\n🎯 STIMA 1X2 (calibrata):")
    print(f"   P(Home) = {p_home:.3f}  → quota fair ≈ {prob_to_odds(p_home):.2f}")
    print(f"   P(Draw) = {p_draw:.3f}  → quota fair ≈ {prob_to_odds(p_draw):.2f}")
    print(f"   P(Away) = {p_away:.3f}  → quota fair ≈ {prob_to_odds(p_away):.2f}")
    print(f"   Somma   = {p_home + p_draw + p_away:.3f}")

    # per eventuale integrazione API:
    return {
        "p1": p_home,
        "px": p_draw,
        "p2": p_away,
        "raw": {
            "p_away": float(p_raw[0]),
            "p_draw": float(p_raw[1]),
            "p_home": float(p_raw[2]),
        },
        "calibrated": {
            "p_away": float(p_away),
            "p_draw": float(p_draw),
            "p_home": float(p_home),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="STEP X5 — Runtime Meta 1X2 predictor (v5)")
    ap.add_argument("--input", type=str, help="Parquet con una riga tipo step2b_1x2_features_v2")
    args = ap.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"❌ File non trovato: {input_path}")
    else:
        input_path = auto_test_generate()

    run_prediction(input_path)


if __name__ == "__main__":
    main()
