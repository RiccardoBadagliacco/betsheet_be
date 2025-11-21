#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VALIDAZIONE COERENZA STEP3 → STEP4

Confronta i cluster generati nei file:
 - step3a_1x2_clusters_v2.parquet
 - step3b_ou25_clusters_v2.parquet
 - step3c_ou15_clusters_v2.parquet

con i cluster presenti in:
 - step4a_affini_index_wide_v2.parquet
 - step4b_affini_index_slim_v2.parquet
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "correlazioni_affini_v2" / "data"

# Step3
C1X2  = BASE / "step3a_1x2_clusters_v2.parquet"
COU25 = BASE / "step3b_ou25_clusters_v2.parquet"
COU15 = BASE / "step3c_ou15_clusters_v2.parquet"

# Step4
WIDE = BASE / "step4a_affini_index_wide_v2.parquet"
SLIM = BASE / "step4b_affini_index_slim_v2.parquet"


def validate(cluster_name, df3, df4):
    print(f"\n==============================")
    print(f"🔍 VALIDAZIONE {cluster_name}")
    print("==============================")

    df3 = df3[["match_id", cluster_name]].drop_duplicates("match_id").copy()
    df4 = df4[["match_id", cluster_name]].copy()

    merged = df3.merge(df4, on="match_id", how="outer", suffixes=("_3", "_4"))

    # confronti
    merged["same"] = merged[f"{cluster_name}_3"] == merged[f"{cluster_name}_4"]

    total = len(merged)
    ok = merged["same"].sum()
    ko = (~merged["same"]).sum()

    missing_3 = merged[f"{cluster_name}_3"].isna().sum()
    missing_4 = merged[f"{cluster_name}_4"].isna().sum()

    print(f"📊 Totale match: {total}")
    print(f"✔️ Coerenti    : {ok}")
    print(f"❌ Non coerenti: {ko}")
    print(f"⚠️ Presenti in STEP4 ma non in STEP3 : {missing_3}")
    print(f"⚠️ Presenti in STEP3 ma non in STEP4 : {missing_4}")

    accuracy = ok / total * 100
    print(f"🎯 ACCURATEZZA FINALE: {accuracy:.2f}%")


def main():
    print("==============================================")
    print("🚀 VALIDAZIONE INTERA FILIERA STEP3 → STEP4")
    print("==============================================\n")

    df1x2  = pd.read_parquet(C1X2)
    df25   = pd.read_parquet(COU25)
    df15   = pd.read_parquet(COU15)

    df_wide = pd.read_parquet(WIDE)
    df_slim = pd.read_parquet(SLIM)

    # ------------------------------------------
    # VALIDAZIONE SU WIDE
    # ------------------------------------------
    print("\n\n===== 🧪 VALIDAZIONE WIDE =====")
    validate("cluster_1x2", df1x2, df_wide)
    validate("cluster_ou25", df25, df_wide)
    validate("cluster_ou15", df15, df_wide)

    # ------------------------------------------
    # VALIDAZIONE SU SLIM
    # ------------------------------------------
    print("\n\n===== 🧪 VALIDAZIONE SLIM =====")
    validate("cluster_1x2", df1x2, df_slim)
    validate("cluster_ou25", df25, df_slim)
    validate("cluster_ou15", df15, df_slim)


if __name__ == "__main__":
    main()