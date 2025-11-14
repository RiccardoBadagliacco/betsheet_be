import pandas as pd
from pathlib import Path

FILE = Path("data/dataset_matches_features_picchetto.parquet")

def main():
    print("📥 Carico dataset picchetto...")
    df = pd.read_parquet(FILE)

    print("\n📏 Shape:", df.shape)

    print("\n📄 Prime righe:")
    print(df[[
        "match_id","tech_p1","tech_px","tech_p2",
        "tech_pO25","tech_pU25",
        "tech_pGG","tech_pNG"
    ]].head())

    print("\n🔍 Controllo NaN:")
    print(df[["tech_p1","tech_px","tech_p2","tech_pO25","tech_pU25","tech_pGG","tech_pNG"]].isna().sum())

    print("\n📈 Distribuzione tech_p1:")
    print(df["tech_p1"].describe())

    print("\n⚽ Somma 1X2 (deve essere 1):")
    print((df["tech_p1"] + df["tech_px"] + df["tech_p2"]).describe())

    print("\n🟢 CHECK COMPLETATO.")


if __name__ == "__main__":
    main()