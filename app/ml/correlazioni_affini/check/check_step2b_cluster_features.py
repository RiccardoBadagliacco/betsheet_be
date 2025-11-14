# app/ml/correlazioni_affini/check/check_step2b_cluster_features.py

import pandas as pd
from pathlib import Path

FILE = Path("app/ml/correlazioni_affini/data/step2b_cluster_features.parquet")


def main():
    print("📥 Carico STEP2B cluster_features...")
    df = pd.read_parquet(FILE)

    print(f"\n📏 Shape: {df.shape}\n")

    print("📄 Prime 5 righe:")
    print(df.head(), "\n")

    # Duplicati
    print("🔁 Duplicati match_id:", df["match_id"].duplicated().sum(), "\n")

    # Controllo che non ci siano suffissi _x/_y
    bad_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if bad_cols:
        print("⚠️ Colonne con suffisso _x/_y (NON dovrebbero esserci):")
        print(bad_cols, "\n")
    else:
        print("✅ Nessuna colonna con suffissi _x / _y\n")

    # Alcune colonne chiave da controllare
    key_cols = [
        "p1_book", "px_book", "p2_book",
        "p1_tech_full", "px_tech_full", "p2_tech_full",
        "elo_diff", "exp_goals_home", "exp_goals_away",
        "home_last5_pts", "away_last5_pts", "form_diff_pts5",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    print("🔍 NaN per colonne chiave:")
    print(df[key_cols].isna().sum(), "\n")

    num_cols = [c for c in key_cols if df[c].dtype != "O"]
    if num_cols:
        print("📈 Statistiche descrittive (colonne chiave):")
        print(df[num_cols].describe(), "\n")

    print("🟢 CHECK STEP2B COMPLETATO!")


if __name__ == "__main__":
    main()