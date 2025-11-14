# app/utils/check/check_form_regression_pro.py

import pandas as pd
from pathlib import Path

FILE = Path("app/ml/correlazioni_affini/data/step1c_form_regression.parquet")

def main():
    print("📥 Carico Form Regression PRO...")
    df = pd.read_parquet(FILE)

    print("\n📏 Shape:", df.shape)

    print("\n📄 Prime righe:")
    print(df.head())

    print("\n🔍 NaN per colonna:")
    print(df.isna().sum())

    print("\n📈 Statistiche punti:")
    print(df[["home_pts", "away_pts"]].describe())

    print("\n📈 Statistiche trend:")
    print(df[["home_trend_gf", "home_trend_ga", "home_trend_gd"]].describe())
    print(df[["away_trend_gf", "away_trend_ga", "away_trend_gd"]].describe())

    print("\n📈 Statistiche varianze:")
    print(df[["home_var_gf", "home_var_ga", "home_var_gd"]].describe())
    print(df[["away_var_gf", "away_var_ga", "away_var_gd"]].describe())

    print("\n🧨 Controllo valori negativi improbabili:")
    print(df[df["home_pts"] < 0].head())

    print("\n🟢 CHECK COMPLETATO")


if __name__ == "__main__":
    main()