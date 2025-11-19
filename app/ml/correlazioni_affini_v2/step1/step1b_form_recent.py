"""
step1b_form_recent.py — VERSIONE ROBUSTA v3
Usa compute_form_lastN rivisto (no season, pura data reale).
"""

from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.form import (
    compute_form_lastN,
    DEFAULT_LAST_N
)

# Path
BASE_DIR = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR = AFFINI_DIR / "data"

IN_PATH = DATA_DIR / "step0_dataset_base.parquet"
OUT_PATH = DATA_DIR / "step1b_form_recent.parquet"


def main():
    print("============================================")
    print("🚀 STEP1B v3 — Form recente ROBUSTA (no season)")
    print("============================================")
    print(f"📥 Input  : {IN_PATH}")
    print(f"💾 Output : {OUT_PATH}")
    print(f"⚙️  N_LAST = {DEFAULT_LAST_N}")
    print("============================================")

    df0 = pd.read_parquet(IN_PATH)
    print(f"📏 Shape step0: {df0.shape}")

    # Compute form
    df_form = compute_form_lastN(df0, last_n=DEFAULT_LAST_N)
    print(f"📏 Shape form: {df_form.shape}")

    # Merge 1:1
    df_out = df0.merge(df_form, on="match_id", how="left")
    print(f"📏 Shape output: {df_out.shape}")

    df_out.to_parquet(OUT_PATH, index=False)
    print("💾 Salvato:", OUT_PATH)
    print("✅ STEP1B v3 COMPLETATO")


if __name__ == "__main__":
    main()