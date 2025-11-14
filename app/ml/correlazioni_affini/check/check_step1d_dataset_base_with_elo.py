# app/ml/correlazioni_affini/check/check_step1d_dataset_base_with_elo.py

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]   # .../app/ml/
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

FILE = DATA_DIR / "step1d_dataset_matches_features_with_elo.parquet"


# ---------------------------------------------------------
# MAIN CHECK
# ---------------------------------------------------------

def main():
    print("📥 Carico dataset STEP1D...")
    df = pd.read_parquet(FILE)

    print("\n📏 Shape:", df.shape)

    print("\n📄 Prime 5 righe:")
    print(df.head())

    print("\n📊 Colonne presenti:")
    print(df.columns.tolist())

    # -----------------------------------------------------
    # CHECK NaN
    # -----------------------------------------------------
    print("\n🔍 NaN per colonna:")
    print(df.isna().sum())

    # -----------------------------------------------------
    # RANGE DATE
    # -----------------------------------------------------
    print("\n⏳ Range date:")
    print(df["date"].min(), "→", df["date"].max())

    # -----------------------------------------------------
    # ELO sanity check
    # -----------------------------------------------------
    print("\n📈 Statistiche ELO:")
    print(df[["elo_home_pre", "elo_away_pre", "elo_diff"]].describe())

    # anomalie elo
    weird_elo = df[(df["elo_home_pre"] < 1000) | (df["elo_home_pre"] > 2400)]
    print("\n⚠ Squadre con Elo anomalo:")
    print(weird_elo[["home_team", "elo_home_pre"]].head())

    # -----------------------------------------------------
    # BOOKMAKER sanity check
    # -----------------------------------------------------
    print("\n📈 Bookmaker stats:")
    print(df[["bk_p1", "bk_px", "bk_p2"]].describe())

    # -----------------------------------------------------
    # TECH BASE sanity check
    # -----------------------------------------------------
    print("\n📈 Tech base stats:")
    print(df[["tech_p1", "tech_px", "tech_p2"]].describe())

    # -----------------------------------------------------
    # CONTROLLA CHE NON CI SIANO MATCH DUPLICATI
    # -----------------------------------------------------
    n_dupes = df["match_id"].duplicated().sum()
    print(f"\n🔁 Duplicati match_id: {n_dupes}")

    # -----------------------------------------------------
    # RISULTATI
    # -----------------------------------------------------
    print("\n📈 Statistiche FT goals:")
    print(df[["home_ft", "away_ft"]].describe())

    # missing results
    missing_ft = df[df["home_ft"].isna() | df["away_ft"].isna()]
    print("\n⚠ Match senza risultati (NON dovrebbero esserci):", len(missing_ft))

    print("\n🟢 CHECK STEP1D COMPLETATO!")


if __name__ == "__main__":
    main()