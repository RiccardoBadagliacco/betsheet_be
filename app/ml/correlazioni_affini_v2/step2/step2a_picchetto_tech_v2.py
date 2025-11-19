# app/ml/correlazioni_affini_v2/step2/step2a_picchetto_tech_fix.py

import pandas as pd
from pathlib import Path
import sys
import joblib

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.picchetto import (
    apply_picchetto_tech_fix,
    DIFF_COLS_FIX,
    fit_picchetto_stats,
)

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"
MODEL_DIR = AFFINI_DIR / "models"

IN_PATH = DATA_DIR / "step1c_dataset_with_elo_form.parquet"
OUT_PATH = DATA_DIR / "step2a_features_with_picchetto_fix.parquet"
STATS_PATH = MODEL_DIR / "picchetto_stats_fix.pkl"


def build_picchetto_raw_features_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ELO POST ---
    df["elo_diff_raw"] = df["elo_home_post"] - df["elo_away_post"]

    # --- FORMA SMOOTHATA SE DISPONIBILE ---
    def choose(raw_col, smooth_col):
        return df[smooth_col] if smooth_col in df.columns else df[raw_col]

    home_pts = choose("home_form_pts_avg_lastN", "home_form_pts_smooth")
    away_pts = choose("away_form_pts_avg_lastN", "away_form_pts_smooth")

    home_gf  = choose("home_form_gf_avg_lastN", "home_form_gf_smooth")
    away_gf  = choose("away_form_gf_avg_lastN", "away_form_gf_smooth")

    home_ga  = choose("home_form_ga_avg_lastN", "home_form_ga_smooth")
    away_ga  = choose("away_form_ga_avg_lastN", "away_form_ga_smooth")

    df["form_pts_diff_raw"] = home_pts - away_pts
    df["form_gf_diff_raw"]  = home_gf - away_gf
    df["form_ga_diff_raw"]  = home_ga - away_ga  # segno corretto

    return df


def main():
    print("============================================")
    print("🚀 STEP2A FIX — Picchetto Tecnico Corretto")
    print("============================================")

    df = pd.read_parquet(IN_PATH)
    print("📏 input:", df.shape)

    df_raw = build_picchetto_raw_features_fix(df)

    stats = fit_picchetto_stats(df_raw)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(stats, STATS_PATH)
    print("💾 Stats salvate:", STATS_PATH)

    df_out = apply_picchetto_tech_fix(
        df_raw,
        alpha=0.5,          # 1X2
        stats=stats,
        alpha_ou=0.3,
        beta_ou_lambda=0.3,
    )

    print("📏 output:", df_out.shape)
    df_out.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato:", OUT_PATH)
    print("✅ STEP2A FIX COMPLETATO!")


if __name__ == "__main__":
    main()