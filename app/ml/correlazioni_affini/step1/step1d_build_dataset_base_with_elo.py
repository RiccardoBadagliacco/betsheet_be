import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # .../app/ml/
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

# Input CORRETTI
PATH_BASE_FEATURES = DATA_DIR / "step0_dataset_base.parquet"
PATH_ELO = DATA_DIR / "step1a_elo.parquet"

# Output
PATH_OUT = DATA_DIR / "step1d_dataset_matches_features_with_elo.parquet"


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("📥 Carico dataset base (STEP0)...")
    df = pd.read_parquet(PATH_BASE_FEATURES)
    print(f"  → Shape base: {df.shape}")

    print("📥 Carico Elo Step1A...")
    elo = pd.read_parquet(PATH_ELO)
    print(f"  → Shape Elo: {elo.shape}")

    # Tenere solo colonne necessarie
    elo = elo[["match_id", "elo_home_pre", "elo_away_pre"]]

    print("🔗 Merge su match_id...")
    df2 = df.merge(elo, on="match_id", how="left")

    print("📊 Creo elo_diff...")
    df2["elo_diff"] = df2["elo_home_pre"] - df2["elo_away_pre"]

    print("\n🔍 Controllo NaN Elo:")
    print(df2[["elo_home_pre", "elo_away_pre", "elo_diff"]].isna().sum())

    # Output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Salvo dataset completo → {PATH_OUT}")
    df2.to_parquet(PATH_OUT, index=False)

    print("✅ STEP1D COMPLETATO!")
    print(f"   → File salvato: {PATH_OUT}")


if __name__ == "__main__":
    main()