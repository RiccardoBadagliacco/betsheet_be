#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X4 v5 — Calibrazione Meta 1X2 (Isotonic)

Usa le probabilità grezze del modello CatBoost v5 e
calibra ogni classe (0=away,1=draw,2=home) con IsotonicRegression.

Input:
    models/meta_1x2_catboost_train_proba_v5.pkl
        {"proba": np.ndarray (N,3), "y": np.ndarray (N,)}

Output:
    models/meta_1x2_calibrator_iso_v5.pkl
        {"iso_0": ..., "iso_1": ..., "iso_2": ...}
"""

from pathlib import Path
import sys

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
MODEL_DIR  = AFFINI_DIR / "models"

PROBA_PATH      = MODEL_DIR / "meta_1x2_catboost_train_proba_v5.pkl"
CALIBRATOR_PATH = MODEL_DIR / "meta_1x2_calibrator_iso_v5.pkl"


def main():
    print("===================================================")
    print("🚀 STEP X4 v5 — Calibrazione Meta 1X2 (Isotonic)")
    print("===================================================")
    print(f"📥 INPUT: {PROBA_PATH}")

    raw = joblib.load(PROBA_PATH)
    proba = raw["proba"]
    y     = raw["y"]

    proba = np.asarray(proba, dtype=float)
    y     = np.asarray(y, dtype=int)

    print(f"🔢 Prob shape: {proba.shape}")
    print(f"🔢 y shape   : {y.shape}")

    calibrators = {}
    n_classes = proba.shape[1]

    for cls in range(n_classes):
        print(f"\n🎯 Classe {cls}:")
        p_raw = proba[:, cls]
        y_bin = (y == cls).astype(int)

        # IsotonicRegression richiede x ordinato: per robustezza ordiniamo p_raw
        order = np.argsort(p_raw)
        x_sorted = p_raw[order]
        y_sorted = y_bin[order]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_sorted, y_sorted)

        p_cal = iso.predict(p_raw)

        # metriche
        ll = log_loss(y_bin, np.vstack([1 - p_cal, p_cal]).T)
        bs = brier_score_loss(y_bin, p_cal)

        print(f"   Logloss: {ll:.4f}")
        print(f"   Brier : {bs:.4f}")

        calibrators[f"iso_{cls}"] = iso

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrators, CALIBRATOR_PATH)
    print(f"\n💾 Salvato calibratore: {CALIBRATOR_PATH}")
    print("🏁 STEP X4 v5 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
