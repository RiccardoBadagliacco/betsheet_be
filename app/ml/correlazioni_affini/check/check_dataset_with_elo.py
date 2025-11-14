import pandas as pd

PATH = "data/dataset_matches_features_with_elo.parquet"


def main():
    print("📥 Carico dataset con Elo...")
    df = pd.read_parquet(PATH)

    print("\n📏 Shape:", df.shape)

    print("\n📄 Prime 5 righe:")
    print(df.head())

    print("\n📊 Colonne:")
    print(df.columns.tolist())

    # --- Controllo NaN ---
    print("\n🔍 NaN per colonna:")
    print(df.isna().sum())

    # --- Statistiche Elo ---
    print("\n📈 Statistiche descrittive Elo:")
    print(df[["elo_home_pre", "elo_away_pre", "elo_diff"]].describe())

    # --- Analisi elo_diff rispetto ai risultati ---
    print("\n🎯 Media elo_diff per risultato (1 / X / 2):")
    if "result" in df.columns:
        df["esito"] = df["result"].map({"1": 1, "X": 0, "2": -1})
    else:
        # oppure deriviamo dagli FT
        df["esito"] = df.apply(
            lambda r: 1 if r["home_ft"] > r["away_ft"]
            else -1 if r["home_ft"] < r["away_ft"]
            else 0,
            axis=1
        )

    print(df.groupby("esito")["elo_diff"].mean())

    # --- Distribuzione Elo per campionato ---
    print("\n📚 Media elo per campionato:")
    try:
        print(df.groupby("league")["elo_home_pre"].mean().sort_values(ascending=False))
    except:
        pass

    # --- Correlazione Elo con goal ---
    print("\n⚽ Correlazione Elo_diff → differenza gol:")
    df["goal_diff"] = df["home_ft"] - df["away_ft"]
    print(df[["elo_diff", "goal_diff"]].corr())

    # --- Controllo outlier ---
    print("\n🧨 Outlier Elo_diff (< -400 o > 400):")
    out = df[(df["elo_diff"].abs() > 400)]
    print(out[["match_id", "home_team", "away_team", "elo_diff"]].head())

    print("\n🟢 ANALISI COMPLETATA.\n"
          "Se tutto è OK, possiamo procedere allo STEP B (Poisson).")


if __name__ == "__main__":
    main()