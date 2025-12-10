#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggiunge al parquet:
- real_score
- hit_score1 .. hit_score4
- hit_any_score_top4
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
PARQUET = REPO_ROOT / "app" / "ml" / "backtests" / "data" / "matches_fav_le_195.parquet"

print(f"📥 Carico parquet: {PARQUET}")
df = pd.read_parquet(PARQUET)

print("🔧 Aggiungo colonne score hit...")

# Risultato reale come stringa
df["real_score"] = df["home_ft"].astype(str) + "-" + df["away_ft"].astype(str)

# Inizializza hit col default = 0
for i in range(1,5):
    df[f"hit_score{i}"] = 0

# Calcola hit
for i in range(1,5):
    col = f"score{i}"
    if col in df.columns:
        df[f"hit_score{i}"] = (df["real_score"] == df[col]).astype(int)

# Hit entro i primi 4
df["hit_any_score_top4"] = (
    df["hit_score1"] | df["hit_score2"] |
    df["hit_score3"] | df["hit_score4"]
).astype(int)

print("💾 Salvo parquet aggiornato...")
df.to_parquet(PARQUET, index=False)

print("✅ Patch completata.")