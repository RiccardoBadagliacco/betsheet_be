# app/ml/correlazioni_affini/step2/step2b_build_cluster_features.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini"
DATA_DIR = AFFINI_DIR / "data"

PATH_STEP2A = DATA_DIR / "step2a_features_with_picchetto.parquet"
OUTPUT = DATA_DIR / "step2b_cluster_features.parquet"


def main():

    print("📥 Carico STEP2A (features + Picchetto)...")
    df = pd.read_parquet(PATH_STEP2A)
    print("  → shape:", df.shape)

    # ------------------------------------------------------------------
    # Controllo colonne picchetto
    # ------------------------------------------------------------------
    expected_pic_cols = ["pic_p1", "pic_px", "pic_p2"]
    missing = [c for c in expected_pic_cols if c not in df.columns]

    if missing:
        print("❌ ERRORE: mancano colonne picchetto:", missing)
        print("   (STEP2A deve creare pic_p1, pic_px, pic_p2)")
        return

    # ------------------------------------------------------------------
    # Costruzione probabilità bookmaker normalizzate
    # ------------------------------------------------------------------
    print("🎯 Costruisco probabilità bookmaker normalizzate...")

    df["bk_sum"] = df["bk_p1"] + df["bk_px"] + df["bk_p2"]
    df["bk_p1_norm"] = df["bk_p1"] / df["bk_sum"]
    df["bk_px_norm"] = df["bk_px"] / df["bk_sum"]
    df["bk_p2_norm"] = df["bk_p2"] / df["bk_sum"]

    # ------------------------------------------------------------------
    # Seleziono tutte le caratteristiche numeriche
    # (match_id sempre incluso)
    # ------------------------------------------------------------------
    exclude_cols = [
        "league", "country", "season",
        "home_team", "away_team",
        "date"
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    out = df[feature_cols].copy()

    print("📏 Output finale shape:", out.shape)

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------
    print("💾 Salvo →", OUTPUT)
    out.to_parquet(OUTPUT, index=False)

    print("✅ STEP2B COMPLETATO!")


if __name__ == "__main__":
    main()