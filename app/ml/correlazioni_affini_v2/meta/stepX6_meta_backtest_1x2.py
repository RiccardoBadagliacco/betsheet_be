#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X6 — BACKTEST Meta 1X2 v1
--------------------------------
Applica il modello CatBoost v5 (no leakage) + calibratore isotonic
a TUTTE le partite storiche e confronta:

- accuracy modello VS bookmaker VS picchetto tecnico
- logloss modello VS bookmaker VS picchetto
- performance per cluster
- performance per range quota favorita

Output:
  data/meta_1x2_backtest_v1.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

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

MASTER_PATH     = DATA_DIR / "meta_1x2_master_train_v1.parquet"
MODEL_PATH      = MODEL_DIR / "meta_1x2_catboost_v5.cbm"
FEATURE_PATH    = MODEL_DIR / "meta_1x2_catboost_features_v5.json"
CALIBRATOR_PATH = MODEL_DIR / "meta_1x2_calibrator_iso_v5.pkl"

OUT_PATH = DATA_DIR / "meta_1x2_backtest_v1.parquet"


# ------------------------------------------------------------
# LOAD FEATURE LIST
# ------------------------------------------------------------
def load_feature_cols():
    import json
    with open(FEATURE_PATH, "r") as f:
        data = json.load(f)

    # Caso 1: nuovo formato {"feature_cols": [...]}
    if isinstance(data, dict) and "feature_cols" in data:
        return data["feature_cols"]

    # Caso 2: fallback → file contiene direttamente la lista
    if isinstance(data, list):
        return data

    raise RuntimeError(f"Formato file feature non riconosciuto: {FEATURE_PATH}")



# ------------------------------------------------------------
# MAP OUTCOME TO 0/1/2
# ------------------------------------------------------------
def encode_outcome(df):
    def f(r):
        if r["is_home_win"] == 1:
            return 2
        if r["is_draw"] == 1:
            return 1
        if r["is_away_win"] == 1:
            return 0
        return np.nan
    return df.apply(f, axis=1)


# ------------------------------------------------------------
# CALIBRATOR HANDLING
# ------------------------------------------------------------
def extract_isotonic_calibrators(calibrator):
    """
    Supporta diversi formati di salvataggio:
      - {"iso_0": ..., "iso_1": ..., "iso_2": ...}
      - {"calibrators": {"0": ..., "1": ..., "2": ...}}
      - lista/tupla [iso0, iso1, iso2]
    Restituisce la terna (iso0, iso1, iso2) o solleva se mancano.
    """
    cal = calibrator
    if isinstance(calibrator, dict):
        cal = calibrator.get("calibrators", calibrator)

    if isinstance(cal, (list, tuple)):
        if len(cal) < 3:
            raise RuntimeError("❌ Calibratore isotonic ha meno di 3 elementi")
        return cal[0], cal[1], cal[2]

    if isinstance(cal, dict):
        iso0 = cal.get("iso_0") or cal.get("class_0") or cal.get("0") or cal.get(0)
        iso1 = cal.get("iso_1") or cal.get("class_1") or cal.get("1") or cal.get(1)
        iso2 = cal.get("iso_2") or cal.get("class_2") or cal.get("2") or cal.get(2)
    else:
        raise RuntimeError(f"❌ Formato calibratore inatteso: {type(calibrator)}")

    if iso0 is None or iso1 is None or iso2 is None:
        keys = list(cal.keys()) if isinstance(cal, dict) else "n/a"
        raise RuntimeError(f"❌ Calibratore isotonic incompleto. Chiavi disponibili: {keys}")

    return iso0, iso1, iso2


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    print("===================================================")
    print("🚀 STEP X6 — Backtest Meta 1X2 v1")
    print("===================================================")

    print(f"📥 MASTER: {MASTER_PATH}")
    df = pd.read_parquet(MASTER_PATH)
    print(f"📏 df shape: {df.shape}")

    # solo match con esito
    df = df[df["is_home_win"].notna()].copy()
    print(f"📌 Match con esito: {df.shape[0]}")

    # target numerico
    df["y"] = encode_outcome(df).astype(int)

    # load modello + calibratore + feature list
    print("📥 Carico feature list…")
    feature_cols = load_feature_cols()
    print(f"🔢 N feature: {len(feature_cols)}")

    print("📥 Carico modello CatBoost v5…")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    print("📥 Carico calibratore isotonic…")
    calibrator = joblib.load(CALIBRATOR_PATH)

    # imputazione NaN → media colonna
    X = df[feature_cols].astype(float).values
    col_means = np.nanmean(X, axis=0)
    idx_nan = np.where(np.isnan(X))
    if idx_nan[0].size > 0:
        X[idx_nan] = np.take(col_means, idx_nan[1])

    # raw prediction
    print("📊 Predict RAW probabilities…")
    proba_raw = model.predict_proba(X)

    # calibrated prediction per classe
    print("📊 Apply isotonic calibrator…")
    iso0, iso1, iso2 = extract_isotonic_calibrators(calibrator)

    # Applico trasformazione isotonic
    p0 = iso0.predict(proba_raw[:, 0])
    p1 = iso1.predict(proba_raw[:, 1])
    p2 = iso2.predict(proba_raw[:, 2])

    p_sum = np.where(p0 + p1 + p2 <= 0, 1.0, p0 + p1 + p2)
    p0, p1, p2 = p0/p_sum, p1/p_sum, p2/p_sum
    proba_cal = np.vstack([p0, p1, p2]).T

    # attach results
    df["p0_raw"], df["p1_raw"], df["p2_raw"] = proba_raw.T
    df["p0_cal"], df["p1_cal"], df["p2_cal"] = proba_cal.T

    # bookmaker (usa bk_p1 ecc.)
    df["p1_bk"] = df["bk_p1"]
    df["pX_bk"] = df["bk_px"]
    df["p2_bk"] = df["bk_p2"]

    # picchetto
    df["p1_pic"] = df["pic_p1"]
    df["pX_pic"] = df["pic_px"]
    df["p2_pic"] = df["pic_p2"]

    # ----------------------------------------------------
    # METRICHE GLOBALI
    # ----------------------------------------------------
    y = df["y"].values

    # modello meta calibrato
    y_pred_meta = np.argmax(proba_cal, axis=1)
    acc_meta = accuracy_score(y, y_pred_meta)
    ll_meta  = log_loss(y, proba_cal)

    # bookmaker
    proba_bk = df[["p0_bk","p1_bk","p2_bk"]] = df.apply(
        lambda r: [r["bk_p2"], r["bk_px"], r["bk_p1"]], axis=1, result_type="expand"
    ).values
    y_pred_bk = np.argmax(proba_bk, axis=1)
    acc_bk = accuracy_score(y, y_pred_bk)
    ll_bk  = log_loss(y, proba_bk)

    # picchetto
    proba_pic = df[["p0_pic","p1_pic","p2_pic"]] = df.apply(
        lambda r: [r["pic_p2"], r["pic_px"], r["pic_p1"]], axis=1, result_type="expand"
    ).values
    y_pred_pic = np.argmax(proba_pic, axis=1)
    acc_pic = accuracy_score(y, y_pred_pic)
    ll_pic  = log_loss(y, proba_pic)

    print("---------------------------------------------------")
    print("📊 ACCURACY")
    print(f"   Meta (calibrato): {acc_meta:.4f}")
    print(f"   Bookmaker       : {acc_bk:.4f}")
    print(f"   Picchetto       : {acc_pic:.4f}")

    print("---------------------------------------------------")
    print("📊 LOG-LOSS")
    print(f"   Meta (calibrato): {ll_meta:.4f}")
    print(f"   Bookmaker       : {ll_bk:.4f}")
    print(f"   Picchetto       : {ll_pic:.4f}")

    # ----------------------------------------------------
    # SALVA DATASET BACKTESTATO
    # ----------------------------------------------------
    df.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato backtest in:", OUT_PATH)
    print("🏁 STEP X6 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
