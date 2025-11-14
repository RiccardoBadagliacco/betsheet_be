# app/utils/check/check_picchetto_pro.py

import pandas as pd
from pathlib import Path

FILE = Path("app/ml/correlazioni_affini/data/step2a_features_with_picchetto.parquet")


def main():
    print("📥 Carico Picchetto Tecnico PRO (STEP2A)...")
    df = pd.read_parquet(FILE)

    print(f"\n📏 Shape: {df.shape}\n")

    # --------------------------------------------------
    # Prime righe
    # --------------------------------------------------
    print("📄 Prime righe:")
    print(df[[
        "match_id",
        "p1_picchetto",
        "px_picchetto",
        "p2_picchetto"
    ]].head(), "\n")

    # --------------------------------------------------
    # Controllo NaN
    # --------------------------------------------------
    print("🔍 NaN:")
    print(df[["p1_picchetto", "px_picchetto", "p2_picchetto"]].isna().sum(), "\n")

    # --------------------------------------------------
    # Controllo probabilità fuori range
    # --------------------------------------------------
    bad_prob = df[
        (df["p1_picchetto"] < 0) | (df["p1_picchetto"] > 1) |
        (df["px_picchetto"] < 0) | (df["px_picchetto"] > 1) |
        (df["p2_picchetto"] < 0) | (df["p2_picchetto"] > 1)
    ]

    print(f"⚠️ Probabilità fuori range: {len(bad_prob)}\n")

    # --------------------------------------------------
    # Controllo somme = 1
    # --------------------------------------------------
    df["sum_prob"] = (
        df["p1_picchetto"] +
        df["px_picchetto"] +
        df["p2_picchetto"]
    )

    bad_sum = df[(df["sum_prob"] < 0.999) | (df["sum_prob"] > 1.001)]

    print(f"⚠️ Somme probabilità ≠ 1: {len(bad_sum)}\n")

    # --------------------------------------------------
    # Statistiche descrittive
    # --------------------------------------------------
    print("📈 Statistiche Probabilità:")
    print(df[["p1_picchetto", "px_picchetto", "p2_picchetto"]].describe(), "\n")

    # --------------------------------------------------
    # Media globale
    # --------------------------------------------------
    print("📊 Media probabilità (globale):")
    print(df[["p1_picchetto", "px_picchetto", "p2_picchetto"]].mean(), "\n")

    print("🟢 CHECK COMPLETATO")


if __name__ == "__main__":
    main()