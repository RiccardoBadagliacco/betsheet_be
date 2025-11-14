import pandas as pd
from pathlib import Path

FILE = Path("data/form_features.parquet")

def main():
    print("📥 Carico Form...")
    df = pd.read_parquet(FILE)

    print("\n📏 Shape:", df.shape)

    print("\n📄 Prime righe:")
    print(df.head())

    # NaN check
    print("\n🔍 NaN per colonna:")
    print(df.isna().sum())

    # Range check
    print("\n📈 Statistiche form_home_points_5 / form_away_points_5:")
    cols = ["form_home_points_5", "form_away_points_5"]
    print(df[cols].describe())

    # Momentum sanity
    print("\n📈 Statistiche momentum:")
    print(df[["form_home_momentum", "form_away_momentum"]].describe())

    print("\n🟢 CHECK COMPLETATO")

if __name__ == "__main__":
    main()