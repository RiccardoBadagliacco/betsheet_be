import pandas as pd
from pathlib import Path

base = Path("app/ml/correlazioni_affini_v2/data")

# 1) Dataset step2 (quello da cui costruisci i cluster/indici affini)
for name in [
    "step2a_1x2_dataset.parquet",
    "step2b_ou15_dataset.parquet",
    "step2c_cluster_dataset.parquet",
]:
    p = base / name
    if p.exists():
        df = pd.read_parquet(p)
        print(name, "→ tightness_index in columns?", "tightness_index" in df.columns)

# 2) Indici affini (SLIM/WIDE)
for name in [
    "affini_index_slim_v2.parquet",
    "affini_index_wide_v2.parquet",
]:
    p = base / name
    if p.exists():
        df = pd.read_parquet(p)
        print(name, "→ tightness_index in columns?", "tightness_index" in df.columns)