# ============================================================
# app/ml/profeta/check/step1_profeta_sanity_check.py
# ============================================================

"""
Sanity check per PROFETA STEP1 (parametri allenati)
"""

import sys
from pathlib import Path
import json
import torch
import pandas as pd
from pprint import pprint
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PARAMS_PATH = DATA_DIR / "step1_profeta_params.pth"
META_PATH   = DATA_DIR / "step1_profeta_metadata.json"


def main():

    print("🔍 SANITY CHECK PROFETA STEP1")
    print("📦 Params file:", PARAMS_PATH)
    print("📦 Meta file:  ", META_PATH)

    if not PARAMS_PATH.exists():
        print("❌ ERRORE: File parametri non trovato.")
        return

    if not META_PATH.exists():
        print("❌ ERRORE: File metadata non trovato.")
        return

    # Carica metadata
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    league_to_idx = meta["league_to_idx"]
    season_to_idx = meta["season_to_idx"]
    ts_to_idx     = meta["teamseason_to_idx"]

    print(f"🏟  Leagues:       {len(league_to_idx)}")
    print(f"📅 Seasons:       {len(season_to_idx)}")
    print(f"👕 Team-Seasons:  {len(ts_to_idx)}")

    # Carica parametri
    params = torch.load(PARAMS_PATH, map_location=torch.device("cpu"))

    # Funzione helper per print stats
    def print_stats(name, t):
        print(f"\n{name}:")
        print(f"  shape:  {tuple(t.shape)}")
        print(f"  mean:   {t.mean().item():.4f}")
        print(f"  std:    {t.std().item():.4f}")
        print(f"  min:    {t.min().item():.4f}")
        print(f"  max:    {t.max().item():.4f}")

    # Stampa statistiche dei parametri
    for key, tensor in params.items():
        print_stats(key, tensor)

    print("\n✅ Sanity check completato!")
    print("Se tutti i range sono ragionevoli, puoi procedere allo STEP2.")


if __name__ == "__main__":
    main()
