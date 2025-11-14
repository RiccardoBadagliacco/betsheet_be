# app/ml/correlazioni_affini/step2/step2a_picchetto_tech_pro.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ---------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

PATH_STEP1D = DATA_DIR / "step1d_dataset_matches_features_with_elo.parquet"
PATH_POISSON = DATA_DIR / "step1b_poisson_expected_goals.parquet"
PATH_FORM = DATA_DIR / "step1c_form_regression.parquet"

OUTPUT = DATA_DIR / "step2a_features_with_picchetto.parquet"


# ---------------------------------------------------------
# PICCHETTO TECNICO PRO
# ---------------------------------------------------------
def compute_picchetto(df):
    """
    Picchetto tecnico basato su:
      - Poisson expected goals
      - Form Regression GF/GA/GD
      - Elo difference
    """

    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="⚙️ Picchetto PRO"):

        # --- Poisson ---
        lam_home = r.get("exp_goals_home", 1.2)
        lam_away = r.get("exp_goals_away", 1.0)

        # --- Form Regression (guardrail: può essere NaN) ---
        home_momentum = r.get("home_trend_gd", 0)
        away_momentum = r.get("away_trend_gd", 0)

        # --- Elo ---
        elo_diff = r.get("elo_diff", 0) / 100  # normalizzazione

        # --- Base Poisson win prob ---
        p_home = np.exp(-lam_away)
        p_away = np.exp(-lam_home)
        p_draw = max(0.05, 1 - (p_home + p_away))

        # --- applico momentum ---
        p_home *= (1 + home_momentum * 0.1 + elo_diff * 0.1)
        p_away *= (1 + away_momentum * 0.1 - elo_diff * 0.1)

        # --- normalizzo ---
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        rows.append({
            "match_id": r.match_id,
            "pic_p1": p_home,
            "pic_px": p_draw,
            "pic_p2": p_away
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():

    print("📥 Carico BASE dataset (STEP1D)...")
    df_base = pd.read_parquet(PATH_STEP1D)
    print("  →", df_base.shape)

    print("📥 Carico POISSON (STEP1B)...")
    df_poisson = pd.read_parquet(PATH_POISSON)
    print("  →", df_poisson.shape)

    print("📥 Carico FORM REGRESSION (STEP1C)...")
    df_form = pd.read_parquet(PATH_FORM)
    print("  →", df_form.shape)

    # -----------------------------------------------------
    # MERGE SENZA PERDERE METADATA
    # -----------------------------------------------------
    print("🔗 Merge dei 3 dataset...")

    df = df_base.merge(df_poisson, on="match_id", how="left", suffixes=("", "_pois"))
    df = df.merge(df_form, on="match_id", how="left", suffixes=("", "_form"))

    print("📏 Final shape post-merge:", df.shape)

    # -----------------------------------------------------
    # FIX metadata (league, country, season)
    # -----------------------------------------------------
    rename_map = {
        "league": "league",
        "league_x": "league",
        "country": "country",
        "country_x": "country",
        "season": "season",
        "season_x": "season",
    }

    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]

    # rimuovi tutte le colonne _y e _form duplicate
    df = df[[c for c in df.columns if not c.endswith("_y")]]
    df = df[[c for c in df.columns if not c.endswith("_form")]]

    print("📌 Colonne dopo fix metadata:", len(df.columns))

    # -----------------------------------------------------
    # PICCHETTO TECNICO PRO
    # -----------------------------------------------------
    print("🔥 Calcolo Picchetto Tecnico PRO...")
    df_pic = compute_picchetto(df)

    # merge finale
    df_final = df.merge(df_pic, on="match_id", how="left")

    print("📏 Output shape:", df_final.shape)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------
    print("💾 Salvo →", OUTPUT)
    df_final.to_parquet(OUTPUT, index=False)

    print("✅ STEP2A COMPLETATO!")


if __name__ == "__main__":
    main()