# diagnostica_2_distance_stats.py
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.correlazioni_affini_v2.common.soft_engine_api_v2 import compute_block_weighted_distances
SLIM = "app/ml/correlazioni_affini_v2/data/step4b_affini_index_slim_v2.parquet"

df = pd.read_parquet(SLIM)

cols = ["elo_diff", "lambda_total_form", "tightness_index"]

dist_avgs = []
dist_median = []
dist_min = []
dist_max = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Computing distances",
    unit="match",
):
    # prendiamo candidati solo su cluster_1x2
    cands = df[(df.cluster_1x2 == row.cluster_1x2) & (df.match_id != row.match_id)]
    d = compute_block_weighted_distances(cands, row)
    dist_avgs.append(np.mean(d))
    dist_median.append(np.median(d))
    dist_min.append(np.min(d))
    dist_max.append(np.max(d))

print("\n==============================")
print("🔍 DIAGNOSTICA DISTANZE SOFT")
print("==============================")
print(f"Mean distance       : {np.mean(dist_avgs):.4f}")
print(f"Median distance     : {np.median(dist_avgs):.4f}")
print(f"Min distance (avg)  : {np.mean(dist_min):.4f}")
print(f"Max distance (avg)  : {np.mean(dist_max):.4f}")
