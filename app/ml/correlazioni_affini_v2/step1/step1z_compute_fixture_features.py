# ============================================================
# step1/step1z_compute_fixture_features.py
# ============================================================
"""
STEP1Z — Feature per FIXTURE (Elo, Form, Picchetto)
---------------------------------------------------
Questo step integra Elo, Form e alcune feature tecniche per le
FIXTURE future, usando ESCLUSIVAMENTE lo storico precedente.

Perché serve:
  - step1a e step1b non calcolano valori per fixture (home_ft=None)
  - step1c richiede Elo/Form/picchetto completi per tutte le righe
  - la pipeline FULL deve assegnare le feature corrette alle fixture
"""

import sys
from pathlib import Path


# ------------------------------------------------------------
# PATH FIX
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from app.ml.correlazioni_affini_v2.common.elo import compute_fixture_elo
from app.ml.correlazioni_affini_v2.common.form import compute_fixture_form

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

# INPUT
STEP0_PATH = DATA_DIR / "step0_dataset_base.parquet"
ELO_PATH   = DATA_DIR / "step1a_elo.parquet"
FORM_PATH  = DATA_DIR / "step1b_form_recent.parquet"

# OUTPUT
OUT_PATH   = DATA_DIR / "step1z_fixture_features.parquet"


# ============================================================
# UTILITIES
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


def load_data():
    df0 = pd.read_parquet(STEP0_PATH)
    df_elo = pd.read_parquet(ELO_PATH)
    df_form = pd.read_parquet(FORM_PATH)
    return df0, df_elo, df_form


# ============================================================
# MAIN LOGIC
# ============================================================

def build_fixture_features(df0, df_elo, df_form):
    # 1) Identifica fixture (senza risultato reale)
    df0["home_ft"] = pd.to_numeric(df0["home_ft"], errors="coerce")
    df0["away_ft"] = pd.to_numeric(df0["away_ft"], errors="coerce")

    df_fixt = df0[df0["home_ft"].isna() & df0["away_ft"].isna()].copy()
    if df_fixt.empty:
        print("⚠️ Nessuna fixture trovata.")
        return pd.DataFrame()

    print(f"🔍 Fixture trovate: {len(df_fixt)}")

    df_fixt["date"] = pd.to_datetime(df_fixt["date"], errors="coerce")
    df_elo["date"] = pd.to_datetime(df_elo["date"], errors="coerce")
    df_form = df_form.copy()
    df_form["date"] = pd.to_datetime(df_form["date"], errors="coerce")

    rows = []

    for _, fx in df_fixt.iterrows():
        mid = fx["match_id"]
        ht = fx["home_team"]
        at = fx["away_team"]
        dfx = fx["date"]

        elo = compute_fixture_elo(df_elo, ht, at, dfx)
        form = compute_fixture_form(df_form, ht, at, dfx)

        rows.append({
            "match_id": mid,
            **elo,
            **form,
            "lambda_home_form": form["home_form_gf_avg_lastN"],
            "lambda_away_form": form["away_form_gf_avg_lastN"],
            "lambda_total_form": form["home_form_gf_avg_lastN"] + form["away_form_gf_avg_lastN"],
        })

    df_fx = pd.DataFrame(rows)
    print(f"📊 Fixture feature generate: {df_fx.shape}")
    return df_fx

# ============================================================
# SAVE OUTPUT
# ============================================================

def main():
    print("================================================")
    print("🚀 STEP1Z — Calcolo feature FIXTURE (Elo/Form)")
    print("================================================")

    df0, df_elo, df_form = load_data()

    df_fx = build_fixture_features(df0, df_elo, df_form)

    if df_fx.empty:
        print("⚠️ Nessuna fixture → Nessun file generato.")
        return

    df_fx.to_parquet(OUT_PATH, index=False)
    print(f"💾 Salvato: {OUT_PATH}")
    print("================================================")


if __name__ == "__main__":
    main()
