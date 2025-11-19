"""
step1a_elo_v2.py – Versione ROBUSTA
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.elo import compute_full_elo_history

BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

IN_PATH = DATA_DIR / "step0_dataset_base.parquet"
OUT_PATH = DATA_DIR / "step1a_elo.parquet"


def main():
    print("============================================")
    print("🚀 STEP1A v2 — Calcolo Elo ROBUSTO")
    print("============================================")
    print(f"📥 Input  : {IN_PATH}")
    print(f"💾 Output : {OUT_PATH}")
    print("============================================")

    df0 = pd.read_parquet(IN_PATH)
    print(f"📏 Shape step0: {df0.shape}")

    # Rimozione di eventuali date invalide prima del passaggio
    df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
    df0 = df0.dropna(subset=["date"])

    df_elo = compute_full_elo_history(df0)
    print(f"📏 Shape Elo output: {df_elo.shape}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_elo.to_parquet(OUT_PATH, index=False)

    print(f"💾 Salvato: {OUT_PATH}")
    print("✅ STEP1A v2 COMPLETATO (versione robusta)")
    print("============================================")


if __name__ == "__main__":
    main()