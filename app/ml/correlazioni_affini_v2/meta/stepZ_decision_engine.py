#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP Z — DECISION ENGINE COMPLETO (runtime da match_id)
-------------------------------------------------------

Input:
    --match_id <fixture_id_o_match_storico>

Il motore carica:
    • step1c_dataset_with_elo_form.parquet   (match + fixture)
    • modello meta 1X2 CatBoost v5
    • calibratore isotonic
    • indici affini slim + wide

E restituisce:
    • probabilità meta 1X2 calibrate
    • probabilità affini (1X2 + O/U + PMF goals)
    • fusione meta+affini
    • quote fair
    • Expected Value
    • Top scores / multigoal / GG-NG
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import joblib
import argparse

# Ensure repository root is on sys.path so `app` imports always work
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# PATHS
# ============================================================

BASE_DIR   = Path(__file__).resolve().parents[2]
AFF_DIR    = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFF_DIR / "data"
MODEL_DIR  = AFF_DIR / "models"

STEP1C_PATH       = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
FEATURES_JSON     = MODEL_DIR / "meta_1x2_catboost_features_v5.json"
META_MODEL_PATH   = MODEL_DIR / "meta_1x2_catboost_v5.cbm"
CALIBRATOR_PATH   = MODEL_DIR / "meta_1x2_calibrator_iso_v5.pkl"
STEP2B_1X2_PATH   = DATA_DIR / "step2b_1x2_features_v2.parquet"

# funzioni affini
from app.ml.correlazioni_affini_v2.common.load_affini_indexes import load_affini_indexes
from app.ml.correlazioni_affini_v2.common.soft_engine_api_v2 import run_soft_engine_api
from app.ml.correlazioni_affini_v2.common.soft_engine_postprocess import full_postprocess
from app.ml.correlazioni_affini_v2.meta.stepZ_formatter import build_final_forecast

# funzione generatore riga feature runtime

# ============================================================
# META MODELLO (CALIBRATO)
# ============================================================

def load_feature_cols():
    """Legge la lista delle feature del modello meta."""
    with open(FEATURES_JSON, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "feature_cols" in data:
        return data["feature_cols"]
    if isinstance(data, list):
        return data

    raise RuntimeError(f"Formato file feature non riconosciuto: {FEATURES_JSON}")


def normalize_calibrators(raw_calibrator) -> dict:
    """
    Accetta diversi formati:
      - {"calibrators": {...}}
      - {"iso_0": ..., "iso_1": ..., "iso_2": ...}
      - lista/tupla [iso0, iso1, iso2]
    Restituisce sempre un dizionario interrogabile con get().
    """
    cal = raw_calibrator

    if isinstance(cal, dict) and "calibrators" in cal:
        cal = cal["calibrators"]

    if isinstance(cal, dict):
        return cal

    if isinstance(cal, (list, tuple)):
        if len(cal) < 3:
            raise RuntimeError("❌ Calibratore isotonic ha meno di 3 elementi")
        return {str(i): cal[i] for i in range(len(cal))}

    raise RuntimeError(f"❌ Formato calibratore inatteso: {type(raw_calibrator)}")


def meta_predict(row, feature_cols):
    """Ritorna proba calibrate {p_home, p_draw, p_away}."""

    # modello
    model = CatBoostClassifier()
    model.load_model(str(META_MODEL_PATH))

    # calibratore
    calib = joblib.load(CALIBRATOR_PATH)
    calibrators = normalize_calibrators(calib)
    get_calib = lambda k: (
        calibrators.get(f"class_{k}")
        or calibrators.get(f"iso_{k}")
        or calibrators.get(k)
        or calibrators.get(str(k))
    )

    X = row[feature_cols].astype(float).values.reshape(1, -1)
    raw = model.predict_proba(X)[0]      # ordine: [away, draw, home]

    # isotonic per classe
    c0 = get_calib(0)
    c1 = get_calib(1)
    c2 = get_calib(2)

    if c0 is None or c1 is None or c2 is None:
        raise RuntimeError("Calibratori isotonic mancanti o con chiave inattesa")

    p0 = c0.predict([raw[0]])[0]
    p1 = c1.predict([raw[1]])[0]
    p2 = c2.predict([raw[2]])[0]

    p = np.array([p0, p1, p2])
    s = p.sum()
    if s <= 0:
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / s

    return {
        "p_away": float(p[0]),
        "p_draw": float(p[1]),
        "p_home": float(p[2]),
    }


# ============================================================
# FUSIONE META + AFFINI
# ============================================================

def fuse(meta, aff, w_meta=0.6, w_aff=0.4):
    p_home = w_meta*meta["p_home"] + w_aff*aff["p1"]
    p_draw = w_meta*meta["p_draw"] + w_aff*aff["px"]
    p_away = w_meta*meta["p_away"] + w_aff*aff["p2"]
    s = p_home + p_draw + p_away
    return {
        "p_home": p_home/s,
        "p_draw": p_draw/s,
        "p_away": p_away/s
    }


# ============================================================
# EV
# ============================================================

def compute_ev(p, q_book):
    if q_book is None or q_book <= 1e-9:
        return None
    return q_book*p - 1


# ============================================================
# DECISION ENGINE PRINCIPALE
# ============================================================

def run_decision(match_id):
    feature_cols = load_feature_cols()

    # ----------------------------------------------------------
    # 1) Carica STEP1C per validare il match_id
    # ----------------------------------------------------------
    df = pd.read_parquet(STEP1C_PATH)
    row_fx = df.loc[df["match_id"] == match_id]

    if row_fx.empty:
        raise RuntimeError(f"match_id {match_id} NON trovato in step1c!")
    
    match_date = row_fx.iloc[0]["date"]

    # Filtro dataset generale
    df = df[df["date"] < match_date]

    # ----------------------------------------------------------
    # 2) Carica indici affini (riusati sia per target row sia per soft engine)
    # ----------------------------------------------------------
    slim, wide = load_affini_indexes()

    # ----------------------------------------------------------
    # FILTRO TEMPORALE ANCHE PER GLI INDICI AFFINI
    # ----------------------------------------------------------
    if "date" in slim.columns:
        slim = slim[slim["date"] < match_date]

    if "date" in wide.columns:
        wide = wide[wide["date"] < match_date]

    slim_all, wide_all = load_affini_indexes()
    target = wide_all.loc[wide_all["match_id"] == match_id]
    if target.empty:
        raise RuntimeError(f"match_id {match_id} NON trovato in affini WIDE.")

    row = target.iloc[0].copy()

    # Integra eventuali feature mancanti dal dataset step2b (meta features complete)
    missing_cols = [c for c in feature_cols if c not in row.index]
    if missing_cols:
        df2b = pd.read_parquet(STEP2B_1X2_PATH)
        row2b = df2b.loc[df2b["match_id"] == match_id]
        if row2b.empty:
            raise RuntimeError(
                f"match_id {match_id} non trovato in {STEP2B_1X2_PATH} per colmatura feature"
            )
        r2b = row2b.iloc[0]
        for col in missing_cols:
            row[col] = r2b.get(col, np.nan)

    # ----------------------------------------------------------
    # 4) Predizione META MODELLO
    # ----------------------------------------------------------
    meta = meta_predict(row, feature_cols)

    # fair odds
    fair_home = 1/meta["p_home"]
    fair_draw = 1/meta["p_draw"]
    fair_away = 1/meta["p_away"]

    # ----------------------------------------------------------
    # 5) AFFINI SOFT
    # ----------------------------------------------------------
    s = run_soft_engine_api(
        target_row=row,
        slim=slim,
        wide=wide,
        top_n=80,
        min_neighbors=30
    )

    if s["status"] != "ok":
        soft = None
        post = None
    else:
        soft = s["soft_probs"]
        post = full_postprocess({
            "meta": {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
            },
            "clusters": s["clusters"],
            "soft_probs": soft,
            "affini_stats": s["affini_stats"],
            "affini_list": s["affini_list"],
        })

    # ----------------------------------------------------------
    # 6) Fusione META + AFFINI
    # ----------------------------------------------------------
    if soft:
        fused = fuse(meta, soft)
    else:
        fused = meta

    # ----------------------------------------------------------
    # 7) EV
    # ----------------------------------------------------------
    bk1 = row.get("bk_p1")
    bkx = row.get("bk_px")
    bk2 = row.get("bk_p2")

    q1 = 1/bk1 if bk1 else None
    qx = 1/bkx if bkx else None
    q2 = 1/bk2 if bk2 else None

    ev_home = compute_ev(fused["p_home"], q1)
    ev_draw = compute_ev(fused["p_draw"], qx)
    ev_away = compute_ev(fused["p_away"], q2)

    # ----------------------------------------------------------
    # 8) OUTPUT JSON COMPLETO
    # ----------------------------------------------------------
    return {
        "match_id": match_id,
        "teams": {
            "home": row["home_team"],
            "away": row["away_team"],
        },
        "meta_1x2": meta,
        "meta_fair": {
            "home": fair_home,
            "draw": fair_draw,
            "away": fair_away
        },
        "affini_1x2": soft,
        "affini_post": post,
        "fused_1x2": fused,
        "fused_fair": {
            "home": 1/fused["p_home"],
            "draw": 1/fused["p_draw"],
            "away": 1/fused["p_away"],
        },
        "ev": {
            "home": ev_home,
            "draw": ev_draw,
            "away": ev_away
        }
    }


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_id", required=True)
    args = ap.parse_args()

    out = run_decision(args.match_id)
    final = build_final_forecast(out)
    print(json.dumps(final, indent=2, default=str))


if __name__ == "__main__":
    main()
