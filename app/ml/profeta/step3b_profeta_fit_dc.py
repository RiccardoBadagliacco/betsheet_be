# ============================================================
# app/ml/profeta/step1b_profeta_fit_dc.py
# ============================================================

"""
STEP1B — Stima del parametro Dixon–Coles (rho) per PROFETA

Usa:
  - step0_profeta.parquet      (risultati reali)
  - step3_profeta_predictions.parquet (lambda_home, lambda_away)

Output:
  - step1b_profeta_dc_params.json con il valore ottimo di rho
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import math

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

FILE_DIR = Path(__file__).resolve().parent
DATA_DIR = FILE_DIR / "data"

STEP0_PATH = DATA_DIR / "step0_profeta.parquet"
STEP3_PATH = DATA_DIR / "step3_profeta_predictions.parquet"
OUT_JSON   = DATA_DIR / "step1b_profeta_dc_params.json"


def dixon_coles_correction_factor(x, y, lam_h, lam_a, rho):
    """
    Fattore di correzione Dixon–Coles C(x,y, lambda_home, lambda_away, rho)

    Formula classica:
      - se (x,y) non è in {(0,0),(0,1),(1,0),(1,1)} → C = 1
      - altrimenti:
            C = 1 + rho * f(x,y, lambda_home, lambda_away)
    """
    # Solo i 4 casi speciali hanno correzione
    if x == 0 and y == 0:
        return 1.0 + rho * (-lam_h * lam_a)
    elif x == 0 and y == 1:
        return 1.0 + rho * lam_h
    elif x == 1 and y == 0:
        return 1.0 + rho * lam_a
    elif x == 1 and y == 1:
        return 1.0 + rho * (1 - lam_h - lam_a)
    else:
        return 1.0


def dc_loglik_for_rho(df, rho):
    """
    Calcola la log-likelihood totale Dixon–Coles per un dato rho.
    """
    lam_h = df["lambda_home"].values
    lam_a = df["lambda_away"].values
    gh    = df["home_goals"].values
    ga    = df["away_goals"].values

    ll = 0.0
    for lh, la, x, y in zip(lam_h, lam_a, gh, ga):
        # Poisson indipendente
        # log P(X=x) = -lam + x*log(lam) - log(x!)
        # log P(Y=y) analogo
        # log-lik base
        if lh <= 0 or la <= 0:
            continue  # skip casi patologici, non dovrebbero esserci

        # Poisson log-likelihood
        base = (
            -lh + x * np.log(lh) - math.lgamma(x + 1) +
            -la + y * np.log(la) - math.lgamma(y + 1)
        )

        # Fattore correttivo DC
        C = dixon_coles_correction_factor(x, y, lh, la, rho)
        if C <= 0:
            # likelihood nulla o negativa → penalizza molto
            return -np.inf

        ll += base + np.log(C)

    return ll


def main():
    print("📥 Carico STEP0:", STEP0_PATH)
    df0 = pd.read_parquet(STEP0_PATH)

    print("📥 Carico STEP3:", STEP3_PATH)
    df3 = pd.read_parquet(STEP3_PATH)

    print("🔗 Merge su match_id…")
    df = df0.merge(df3, on="match_id", suffixes=("_step0", "_step3"))

    # Storici con gol not null
    df = df[(df["is_fixture_step0"] == False)]
    df = df.dropna(subset=["home_goals", "away_goals", "lambda_home", "lambda_away"])

    print("🔢 Partite usate per stima rho:", len(df))

    # Griglia di ricerca per rho
    rhos = np.linspace(-0.2, 0.2, 81)  # passo 0.005
    best_rho = None
    best_ll = -np.inf

    print("🧮 Ricerca del rho ottimale (Dixon–Coles)…")

    for r in rhos:
        ll = dc_loglik_for_rho(df, r)
        if ll > best_ll:
            best_ll = ll
            best_rho = r

    print("\n✅ Miglior rho trovato:")
    print(f"   rho = {best_rho:.4f}")
    print(f"   log-likelihood = {best_ll:.2f}")

    out = {
        "rho_dc": float(best_rho),
        "loglik": float(best_ll),
        "grid_min": float(rhos.min()),
        "grid_max": float(rhos.max()),
        "grid_step": float(rhos[1] - rhos[0]),
        "n_matches": int(len(df)),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("💾 Salvato:", OUT_JSON)


if __name__ == "__main__":
    main()
