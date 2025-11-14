import pandas as pd

PATH = "app/ml/correlazioni_affini/data/step1a_elo.parquet"

def main():
    print("📥 Carico Elo...")
    df = pd.read_parquet(PATH)

    print("\n📄 Prime righe:")
    print(df.head())

    print("\n📊 Colonne presenti:")
    print(df.columns.tolist())

    print("\n📏 Shape:", df.shape)

    # --- Calcolo elo_diff se assente ---
    if "elo_diff" not in df.columns:
        df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    print("\n🔍 Statistiche descrittive Elo:")
    print(
        df[
            [
                'elo_home_pre',
                'elo_away_pre',
                'elo_home_post',
                'elo_away_post',
                'elo_diff'
            ]
        ].describe()
    )

    print("\n🧪 Controllo NaN:")
    print(df.isna().sum())

    print("\n🔁 Controllo duplicati match_id:")
    print(df['match_id'].duplicated().sum(), "duplicati")

    print("\n🎯 Statistiche elo_diff:")
    print(df['elo_diff'].describe())

    print("\n⏳ Range date:")
    print(df['date'].min(), "→", df['date'].max())

    print("\n🔎 Squadre con Elo anomalo (<1000 o >2400):")
    outliers = df[(df['elo_home_pre'] < 1000) | (df['elo_home_pre'] > 2400)]
    print(outliers[['home_team', 'elo_home_pre']].head())

if __name__ == "__main__":
    main()