#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP X7 — Backtest Meta 1X2 v1 con EV e strategie betting

- Usa il MASTER 1X2 + modello CatBoost v5 + calibratore isotonic v2
- Calcola:
    • Probabilità calibrate P(away, draw, home)
    • Quote implicite del book (da bk_p*)
    • EV per 1 / X / 2
    • Strategia: punta sull'esito con EV max se EV ≥ soglia τ
- Stampa una tabella riassuntiva per diverse soglie τ
- Salva un parquet dettagliato con, per ogni match:
    meta_probs, book_probs, quote, EV, esito reale, puntata, profitto

Output:
    data/meta_1x2_backtest_ev_v1.parquet
"""

from pathlib import Path
import sys
import json

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

MASTER_PATH      = DATA_DIR / "meta_1x2_master_train_v1.parquet"
MODEL_PATH       = MODEL_DIR / "meta_1x2_catboost_v5.cbm"
FEATURES_PATH    = MODEL_DIR / "meta_1x2_catboost_features_v5.json"
CALIBRATOR_PATH  = MODEL_DIR / "meta_1x2_calibrator_iso_v5.pkl"
BACKTEST_EV_PATH = DATA_DIR / "meta_1x2_backtest_ev_v1.parquet"


# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def encode_outcome(row) -> int:
    """
    Encoding consistente con gli altri step:
      0 = away win
      1 = draw
      2 = home win
    """
    if row["is_home_win"] == 1:
        return 2
    elif row["is_draw"] == 1:
        return 1
    elif row["is_away_win"] == 1:
        return 0
    return -1


def load_feature_cols() -> list[str]:
    with open(FEATURES_PATH, "r") as f:
        data = json.load(f)
    # il file v5 è una semplice lista di nomi
    if isinstance(data, dict) and "feature_cols" in data:
        return data["feature_cols"]
    return data


def normalize_calibrators(raw_calibrator) -> dict:
    """
    Gestisce diversi formati di salvataggio del calibratore isotonic:
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


def load_model_and_calibrator():
    print(f"📥 Carico modello CatBoost da: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    print(f"📥 Carico calibratore isotonic da: {CALIBRATOR_PATH}")
    calib = joblib.load(CALIBRATOR_PATH)

    calibrators = normalize_calibrators(calib)  # dict: "0","1","2" -> IsotonicRegression
    return model, calibrators


def apply_isotonic_calibration(proba_raw: np.ndarray, calibrators: dict) -> np.ndarray:
    """
    Applica il calibratore isotonic classe per classe.
    """
    # supporta entrambe le versioni: "0"/"1"/"2" e "class_0"/"class_1"/"class_2"
    iso0 = (
        calibrators.get("class_0")
        or calibrators.get("iso_0")
        or calibrators.get("0")
        or calibrators.get(0)
    )
    iso1 = (
        calibrators.get("class_1")
        or calibrators.get("iso_1")
        or calibrators.get("1")
        or calibrators.get(1)
    )
    iso2 = (
        calibrators.get("class_2")
        or calibrators.get("iso_2")
        or calibrators.get("2")
        or calibrators.get(2)
    )

    if iso0 is None or iso1 is None or iso2 is None:
        raise RuntimeError(
            f"❌ Calibratore isotonic non contiene class_0/class_1/class_2. "
            f"Chiavi presenti: {list(calibrators.keys())}"
        )

    # isotonic regression usa predict() su array 1D
    p0 = iso0.predict(proba_raw[:, 0])
    p1 = iso1.predict(proba_raw[:, 1])
    p2 = iso2.predict(proba_raw[:, 2])

    P = np.vstack([p0, p1, p2]).T
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return P / row_sum


def compute_ev_for_row(row) -> dict:
    """
    Calcola:
      - quote del book (da bk_p*)
      - EV per 1/X/2
    Usa:
      bk_p1, bk_px, bk_p2
      meta_p_away, meta_p_draw, meta_p_home
    """
    # Probabilità meta calibrate
    pA = float(row["meta_p_away"])
    pX = float(row["meta_p_draw"])
    pH = float(row["meta_p_home"])

    # Prob dalla view del bookmaker
    bk_p1 = float(row["bk_p1"])
    bk_px = float(row["bk_px"])
    bk_p2 = float(row["bk_p2"])

    # Quote implicite (assumiamo bk_p* ≈ 1/quote con overround)
    # → quote_book = 1 / bk_p*
    def safe_quote(p):
        if p is None or p <= 0 or not np.isfinite(p):
            return np.nan
        return 1.0 / p

    qH = safe_quote(bk_p1)  # home
    qX = safe_quote(bk_px)  # draw
    qA = safe_quote(bk_p2)  # away

    # EV = p_meta * quota - 1
    evH = pH * qH - 1 if np.isfinite(qH) else np.nan
    evX = pX * qX - 1 if np.isfinite(qX) else np.nan
    evA = pA * qA - 1 if np.isfinite(qA) else np.nan

    return {
        "quote_home": qH,
        "quote_draw": qX,
        "quote_away": qA,
        "ev_home": evH,
        "ev_draw": evX,
        "ev_away": evA,
    }


def simulate_strategy(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """
    Simula una strategia semplice:
      - per ogni match:
          • trova outcome j con EV massimo
          • se EV_max >= τ → bet=1 unità su j
      - calcola ROI, hit-rate, numero scommesse per ciascuna soglia τ
    """
    results = []

    for tau in thresholds:
        mask_bet = df["ev_max"] >= tau
        sub = df.loc[mask_bet].copy()

        n_bets = len(sub)
        if n_bets == 0:
            results.append(
                {
                    "threshold_ev": tau,
                    "n_bets": 0,
                    "hit_rate": np.nan,
                    "roi": np.nan,
                    "profit_total": 0.0,
                }
            )
            continue

        # profit se puntiamo 1 unità su esito chosen_idx
        # chosen_idx: 0=away,1=draw,2=home
        profit = []
        hits = 0

        for _, r in sub.iterrows():
            outcome = int(r["y_1x2"])
            choice = int(r["bet_choice"])

            if choice == 2:
                odd = r["quote_home"]
            elif choice == 1:
                odd = r["quote_draw"]
            else:
                odd = r["quote_away"]

            if not np.isfinite(odd) or odd <= 1.0:
                # non bettiamo, ma in teoria qui non dovremmo arrivare se le quote sono sane
                profit.append(0.0)
                continue

            if outcome == choice:
                profit.append(odd - 1.0)
                hits += 1
            else:
                profit.append(-1.0)

        profit = np.array(profit, dtype=float)
        profit_total = float(profit.sum())
        roi = profit_total / n_bets
        hit_rate = hits / n_bets

        results.append(
            {
                "threshold_ev": tau,
                "n_bets": n_bets,
                "hit_rate": hit_rate,
                "roi": roi,
                "profit_total": profit_total,
            }
        )

    return pd.DataFrame(results)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("===================================================")
    print("🚀 STEP X7 — Backtest Meta 1X2 v1 con EV")
    print("===================================================")
    print(f"📥 MASTER: {MASTER_PATH}")
    print(f"💾 OUT   : {BACKTEST_EV_PATH}")

    df = pd.read_parquet(MASTER_PATH)
    print(f"📏 df shape: {df.shape}")

    # solo match con esito
    mask_target = (
        df["is_home_win"].notna()
        & df["is_draw"].notna()
        & df["is_away_win"].notna()
    )
    df = df.loc[mask_target].copy()
    print(f"📌 Match con esito: {df.shape[0]}")

    # outcome numerico
    df["y_1x2"] = df.apply(encode_outcome, axis=1)
    df = df[df["y_1x2"] >= 0].copy()

    # feature
    feature_cols = load_feature_cols()
    print("📥 Feature usate per il meta-model:")
    for c in feature_cols:
        print(f"   - {c}")
    print(f"🔢 N feature: {len(feature_cols)}")

    # check colonne
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"❌ Mancano colonne nel MASTER per il meta model:\n{missing}")

    # prepara X
    X = df[feature_cols].astype(float).values

    # carica modello + calibratore
    model, calibrators = load_model_and_calibrator()

    # pred proba raw
    print("📊 Predizione probabilità RAW (meta 1X2)…")
    proba_raw = model.predict_proba(X)  # shape (N,3) [away, draw, home]

    # calibrazione
    print("📊 Applicazione calibratore isotonic…")
    proba_cal = apply_isotonic_calibration(proba_raw, calibrators)

    # append meta proba
    df["meta_p_away"] = proba_cal[:, 0]
    df["meta_p_draw"] = proba_cal[:, 1]
    df["meta_p_home"] = proba_cal[:, 2]

    # EV per ogni riga
    print("📊 Calcolo EV per ogni match…")
    ev_records = df.apply(compute_ev_for_row, axis=1, result_type="expand")
    df = pd.concat([df, ev_records], axis=1)

    # EV max e scelta esito migliore
    ev_array = np.vstack(
        [
            df["ev_away"].values,
            df["ev_draw"].values,
            df["ev_home"].values,
        ]
    ).T  # shape (N,3)

    df["ev_max"] = ev_array.max(axis=1)
    df["bet_choice"] = ev_array.argmax(axis=1)  # 0=away,1=draw,2=home

    # --- piccola diagnosi media EV & edge medio book vs meta ---
    df["bk_prob_away"] = df["bk_p2"]
    df["bk_prob_draw"] = df["bk_px"]
    df["bk_prob_home"] = df["bk_p1"]

    df["edge_home"] = df["meta_p_home"] - df["bk_prob_home"]
    df["edge_draw"] = df["meta_p_draw"] - df["bk_prob_draw"]
    df["edge_away"] = df["meta_p_away"] - df["bk_prob_away"]

    print("---------------------------------------------------")
    print("📊 Edge medio meta vs book (tutto il dataset):")
    print(
        f"   Home: {df['edge_home'].mean():+.4f}, "
        f"Draw: {df['edge_draw'].mean():+.4f}, "
        f"Away: {df['edge_away'].mean():+.4f}"
    )

    # Strategie con diverse soglie di EV
    thresholds = [0.00, 0.02, 0.05, 0.08, 0.10]
    print("---------------------------------------------------")
    print(f"📊 Backtest strategie EV (soglie: {thresholds})")

    strat_df = simulate_strategy(df, thresholds)

    print("\n   soglia_EV | n_bets | hit_rate |    ROI  | profit_total")
    for _, r in strat_df.iterrows():
        tau = r["threshold_ev"]
        n_bets = int(r["n_bets"])
        hit = r["hit_rate"]
        roi = r["roi"]
        prof = r["profit_total"]
        if np.isnan(roi):
            roi_str = "  nan "
        else:
            roi_str = f"{roi:+.4f}"
        if np.isnan(hit):
            hit_str = "  nan "
        else:
            hit_str = f"{hit:.3f}"
        print(
            f"   {tau:9.2%} | {n_bets:6d} | {hit_str} | {roi_str} | {prof:+.1f}"
        )

    # Salva parquet dettagliato
    cols_keep = [
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "bk_p1", "bk_px", "bk_p2",
        "quote_home", "quote_draw", "quote_away",
        "meta_p_home", "meta_p_draw", "meta_p_away",
        "ev_home", "ev_draw", "ev_away", "ev_max",
        "bet_choice", "y_1x2",
        "edge_home", "edge_draw", "edge_away",
        "is_home_win", "is_draw", "is_away_win",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]

    df_out = df[cols_keep].copy()
    BACKTEST_EV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(BACKTEST_EV_PATH, index=False)
    print("---------------------------------------------------")
    print(f"💾 Backtest EV dettagliato salvato in: {BACKTEST_EV_PATH}")
    print("🏁 STEP X7 COMPLETATO")
    print("===================================================")


if __name__ == "__main__":
    main()
