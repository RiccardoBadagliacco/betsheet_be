# diagnostica_1_filters.py
import pandas as pd
from tqdm import tqdm

SLIM = "app/ml/correlazioni_affini_v2/data/step4b_affini_index_slim_v2.parquet"

df = pd.read_parquet(SLIM)

total = len(df)
fails = 0
single_cluster_only = 0

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Scanning filters",
    unit="match",
):
    c1 = row.cluster_1x2
    c25 = row.cluster_ou25
    c15 = row.cluster_ou15

    # filtro più stretto
    mask = (
        (df.cluster_1x2 == c1) &
        (df.cluster_ou25 == c25) &
        (df.cluster_ou15 == c15)
    )

    # rimuovi se stesso
    mask &= (df.match_id != row.match_id)

    if mask.sum() == 0:
        fails += 1

    # fallimento cluster multipli → almeno cluster_1x2 funziona?
    if (df.cluster_1x2 == c1).sum() <= 1:
        single_cluster_only += 1

print("\n==============================")
print("🔍 DIAGNOSTICA FILTRI HARD")
print("==============================")
print(f"Match totali           : {total}")
print(f"Match senza candidati  : {fails}  ({fails/total:.2%})")
print(f"Cluster1x2 isolati     : {single_cluster_only} ({single_cluster_only/total:.2%})")
