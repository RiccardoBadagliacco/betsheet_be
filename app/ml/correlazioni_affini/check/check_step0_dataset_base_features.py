import pandas as pd
from pathlib import Path

PATH = Path("app/ml/correlazioni_affini/data/step0_dataset_base.parquet")

def main():
    print("📥 Carico STEP0 dataset base...\n")
    df = pd.read_parquet(PATH)

    print(f"📏 Shape: {df.shape}\n")

    print("📄 Prime 5 righe:")
    print(df.head(), "\n")

    print("📊 Colonne presenti:")
    print(list(df.columns), "\n")

    print("🔍 NaN per colonna:")
    print(df.isna().sum(), "\n")

    print("⏳ Range date:")
    print(df["date"].min(), "→", df["date"].max(), "\n")

    print("🔁 Duplicati match_id:", df["match_id"].duplicated().sum(), "\n")

    print("📈 Bookmaker stats:")
    print(df[["bk_p1", "bk_px", "bk_p2", "bk_pO25", "bk_pU25"]].describe(include="all"), "\n")

    print("📈 Tech base stats:")
    print(df[["tech_p1", "tech_px", "tech_p2", "tech_pO25", "tech_pU25"]].describe(include="all"), "\n")

    print("📈 Statistiche FT goals:")
    print(df[["home_ft", "away_ft"]].describe(), "\n")

    print("🟢 STEP0 check completato!")


if __name__ == "__main__":
    main()