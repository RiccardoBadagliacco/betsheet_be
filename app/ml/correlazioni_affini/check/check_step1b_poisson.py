import pandas as pd
from pathlib import Path

FILE = Path("app/ml/correlazioni_affini/data/step1b_poisson_expected_goals.parquet")

def main():
    print("📥 Carico Poisson...")
    df = pd.read_parquet(FILE)

    print("\n📏 Shape:", df.shape)
    print("\n📄 Prime righe:")
    print(df.head())

    print("\n🔍 Controllo NaN:")
    print(df.isna().sum())

    print("\n🔍 Valori inf:")
    print(df.replace([float('inf'), float('-inf')], None).isna().sum())

    print("\n📈 Statistiche exp_goals_home / away:")
    print(df[['exp_goals_home', 'exp_goals_away']].describe())

    print("\n⚠ Match con exp_goals_home > 5:")
    print(df[df.exp_goals_home > 5].head())

    print("\n⚠ Match con exp_goals_away > 5:")
    print(df[df.exp_goals_away > 5].head())

    print("\n⚠ Match con valori negativi (NON devono esistere):")
    print(df[(df.exp_goals_home < 0) | (df.exp_goals_away < 0)].head())

    print("\n🔍 Range probabilità:")
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    print(df[prob_cols].describe())

    print("\n🧨 Conta valori NaN dopo replace inf:")
    print(df[prob_cols].replace([float('inf'), float('-inf')], None).isna().sum())

    print("\n🟢 CHECK COMPLETATO")

if __name__ == "__main__":
    main()