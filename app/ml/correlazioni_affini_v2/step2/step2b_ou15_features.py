# app/ml/correlazioni_affini_v2/step2/step2b_ou15_features_v2.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP2B OU1.5 V2 — Feature avanzate per OU 1.5 (derivato da λ mix)

Input:
    data/step2a_features_with_picchetto_fix.parquet

Output:
    data/step2b_ou15_features_v2.parquet
"""

from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.features_ou15 import (
    build_features_ou15_v2,
    FEATURES_OU15_V2,
)

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

INPUT_FILE  = DATA_DIR / "step2a_features_with_picchetto_fix.parquet"
OUTPUT_FILE = DATA_DIR / "step2b_ou15_features_v2.parquet"


def main():
    print("============================================")
    print("🚀 STEP2B OU1.5 V2 — Feature avanzate OU1.5")
    print("============================================")
    print(f"📥 Input:  {INPUT_FILE}")
    print(f"💾 Output: {OUTPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    print(f"📏 Shape input: {df.shape}")

    rows = [build_features_ou15_v2(r) for _, r in df.iterrows()]
    df_out = pd.DataFrame(rows)

    missing = [c for c in FEATURES_OU15_V2 if c not in df_out.columns]
    if missing:
        raise RuntimeError(f"❌ STEP2B OU1.5 V2: mancano colonne obbligatorie: {missing}")

    df_out = df_out[FEATURES_OU15_V2].copy()

    df_out.to_parquet(OUTPUT_FILE, index=False)
    print(f"📏 Shape output: {df_out.shape}")
    print(f"💾 Salvato: {OUTPUT_FILE}")
    print("✅ STEP2B OU1.5 V2 COMPLETATO!")


if __name__ == "__main__":
    main()
    