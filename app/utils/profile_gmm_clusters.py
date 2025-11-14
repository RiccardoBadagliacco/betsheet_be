# app/utils/profile_gmm_clusters.py

import pandas as pd
import numpy as np
from pathlib import Path

CLUSTERS = Path("data/cluster_assignments.parquet")
BASE = Path("data/dataset_matches_features_with_elo.parquet")  # contiene home_ft, away_ft, ecc.


def main():

    print("📥 Carico cluster_assignments...")
    df_cluster = pd.read_parquet(CLUSTERS)

    print("📥 Carico dataset base con risultati...")
    df_base = pd.read_parquet(BASE)[[
        "match_id", "home_ft", "away_ft", "total_goals", "result_1x2"
    ]]

    print("🔗 Merge cluster + risultati...")
    df = df_cluster.merge(df_base, on="match_id", how="left")

    # ============ CALCOLO STATISTICHE REALI ============

    print("\n📊 Profilazione cluster...\n")

    clusters = sorted(df["cluster"].unique())

    for c in clusters:
        sub = df[df["cluster"] == c]

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔵 CLUSTER {c} — {len(sub)} partite")

        # 1X2 reali
        dist_1x2 = sub["result_1x2"].value_counts(normalize=True).reindex(
            ["1", "X", "2"], fill_value=0
        )

        # Over/Under 2.5
        ou25 = (sub["total_goals"] > 2.5).value_counts(normalize=True)

        # GG/NG
        ggng = ((sub["home_ft"] > 0) & (sub["away_ft"] > 0)).value_counts(normalize=True)

        print("\n📌 Risultati reali:")
        print(f"   1: {dist_1x2['1']:.2%}")
        print(f"   X: {dist_1x2['X']:.2%}")
        print(f"   2: {dist_1x2['2']:.2%}")

        print("\n📌 Over/Under 2.5:")
        print(f"   Over 2.5: {ou25.get(True, 0):.2%}")
        print(f"   Under 2.5: {ou25.get(False, 0):.2%}")

        print("\n📌 GG/NG:")
        print(f"   GG: {ggng.get(True, 0):.2%}")
        print(f"   NG: {ggng.get(False, 0):.2%}")

        # Statistiche numeriche dei modelli
        num_cols = [
            "p1_book", "px_book", "p2_book",
            "tech_p1", "tech_px", "tech_p2",
            "p1_tech_full", "px_tech_full", "p2_tech_full",
            "elo_diff", "exp_goals_home", "exp_goals_away"
        ]

        stats = sub[num_cols].mean().round(3)

        print("\n📈 Statistiche modello / bookmaker:")
        for col in num_cols:
            print(f"   {col}: {stats[col]}")

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    print("🟢 Profilazione completata!")


if __name__ == "__main__":
    main()