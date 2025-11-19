#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALYZE_CLUSTERS_V2 — Diagnostica dei modelli di clustering 1X2 / OU2.5 / OU1.5

Obiettivi:
  - Capire cosa rappresenta ogni cluster (feature medie)
  - Valutare quanto i cluster separano gli esiti reali (win/over)
  - Studiare tightness_index per cluster
  - Dare indizi su dove migliorare il clustering

Input attesi:
  - data/step3a_1x2_clusters_v2.parquet
  - data/step3b_ou25_clusters_v2.parquet
  - data/step3c_ou15_clusters_v2.parquet
  - data/step4a_affini_index_wide_v2.parquet   (per outcomes completi)

Uso:
  python app/ml/correlazioni_affini_v2/analysis/analyze_clusters_v2.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------------------------------------
# PATH BASE (stile tuoi script)
# -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"


# -----------------------------------------------------------
# UTILITY
# -----------------------------------------------------------
def entropy_from_probs(p: np.ndarray) -> float:
    """Entropia di Shannon (base 2) date probabilità p."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def safe_rate(x):
    """Helper per rate: mean ignorando NaN."""
    return float(np.nanmean(x)) if len(x) > 0 else np.nan


# -----------------------------------------------------------
# ANALISI CLUSTER 1X2
# -----------------------------------------------------------
def analyze_1x2(step3_path: Path, wide_path: Path):
    print("\n" + "="*80)
    print("🔍 ANALISI CLUSTER 1X2")
    print("="*80)

    if not step3_path.exists():
        print(f"❌ File step3a non trovato: {step3_path}")
        return
    if not wide_path.exists():
        print(f"⚠️ WIDE index non trovato: {wide_path} — analisi esiti limitata.")
    
    df = pd.read_parquet(step3_path)
    print(f"📏 step3a shape: {df.shape}")

    # cluster col
    cluster_col = "cluster_1x2"
    if cluster_col not in df.columns:
        cluster_col = [c for c in df.columns if c.startswith("cluster_")][0]
    print(f"📌 Colonna cluster: {cluster_col}")

    # distribuzione dimensioni cluster
    vc = df[cluster_col].value_counts().sort_index()
    print("\n📊 Distribuzione cluster (conteggio & %):")
    total = len(df)
    for k, n in vc.items():
        print(f"  - cluster {k}: {n:6d} ({n/total*100:5.1f}%)")

    # feature medie chiave (solo pre-match)
    cols_key = [
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_px", "pic_p2",
        "delta_p1", "delta_px", "delta_p2",
        "delta_1x2_abs_sum",
        "entropy_bk_1x2", "entropy_pic_1x2",
        "elo_diff",
        "team_strength_diff",
        "lambda_total_form",
        "lambda_total_market_ou25",
        "fav_prob_1x2",
        "market_balance_index_1x2",
        "fav_prob_gap_1x2",
        "second_fav_prob_1x2",
        "tightness_index",
    ]
    cols_key = [c for c in cols_key if c in df.columns]

    print("\n📈 Medie feature chiave per cluster (step3a):")
    print(df.groupby(cluster_col)[cols_key].mean().round(4))

    # Se possibile, unisco con WIDE per analisi esiti reali
    if wide_path.exists():
        wide = pd.read_parquet(wide_path)[[
            "match_id",
            "is_home_win", "is_draw", "is_away_win",
            "is_over25", "is_over15",
            "total_goals",
        ]]
        dfj = df.merge(wide, on="match_id", how="left")
        print("\n📈 Esiti reali per cluster (dal WIDE index):")
        agg = dfj.groupby(cluster_col).agg(
            n=("match_id", "size"),
            home_win_rate=("is_home_win", "mean"),
            draw_rate=("is_draw", "mean"),
            away_win_rate=("is_away_win", "mean"),
            over25_rate=("is_over25", "mean"),
            over15_rate=("is_over15", "mean"),
            avg_goals=("total_goals", "mean"),
            tight_mean=("tightness_index", "mean"),
        ).round(4)
        print(agg)

        # Entropia distribuzione 1X2 per cluster
        print("\n📉 Entropia (p1/px/p2) per cluster:")
        rows = []
        for c, sub in dfj.groupby(cluster_col):
            p1 = safe_rate(sub["is_home_win"])
            px = safe_rate(sub["is_draw"])
            p2 = safe_rate(sub["is_away_win"])
            H = entropy_from_probs(np.array([p1, px, p2]))
            rows.append({"cluster": c, "p1": p1, "px": px, "p2": p2, "entropy_1x2": H})
        ent_df = pd.DataFrame(rows).set_index("cluster").round(4)
        print(ent_df)

        # Dispersione dei tassi tra cluster (quanto sono separati)
        disp_home = ent_df["p1"].std()
        disp_away = ent_df["p2"].std()
        disp_over25 = agg["over25_rate"].std()
        print("\n📌 DISPERSIONE tra cluster (std tra medie di cluster):")
        print(f"   std(p_home_win) tra cluster   = {disp_home:.4f}")
        print(f"   std(p_away_win) tra cluster   = {disp_away:.4f}")
        print(f"   std(p_over25) tra cluster     = {disp_over25:.4f}")

        print("\n💡 Hint miglioramento 1X2:")
        print("   - Cluster con entropia 1X2 alta ~ 'misti', poco predittivi.")
        print("   - Cluster con home_win_rate / away_win_rate molto simili fra loro ⇒ poco separati.")
        print("   - Per migliorare: supervisionare la scelta di K usando std(p1/p2) tra cluster come parte dello SCORE.\n")


# -----------------------------------------------------------
# ANALISI CLUSTER OU2.5
# -----------------------------------------------------------
def analyze_ou25(step3_path: Path):
    print("\n" + "="*80)
    print("🔍 ANALISI CLUSTER OU2.5")
    print("="*80)

    if not step3_path.exists():
        print(f"❌ File step3b non trovato: {step3_path}")
        return

    df = pd.read_parquet(step3_path)
    print(f"📏 step3b shape: {df.shape}")

    cluster_col = "cluster_ou25"
    if cluster_col not in df.columns:
        cluster_col = [c for c in df.columns if c.startswith("cluster_")][0]
    print(f"📌 Colonna cluster: {cluster_col}")

    vc = df[cluster_col].value_counts().sort_index()
    print("\n📊 Distribuzione cluster (conteggio & %):")
    total = len(df)
    for k, n in vc.items():
        print(f"  - cluster {k}: {n:6d} ({n/total*100:5.1f}%)")

    # Esiti e feature chiave
    cols_key = [
        "bk_pO25", "bk_pU25",
        "pic_pO25", "pic_pU25",
        "delta_O25", "delta_U25", "delta_ou25_abs_sum",
        "delta_ou25_market_vs_form",
        "lambda_total_market_ou25",
        "lambda_total_form",
        "goal_supremacy_market_ou25",
        "goal_supremacy_form",
        "entropy_bk_ou25",
        "entropy_pic_ou25",
        "tightness_index",
    ]
    cols_key = [c for c in cols_key if c in df.columns]

    print("\n📈 Medie feature chiave + esiti per cluster OU2.5:")
    agg = df.groupby(cluster_col).agg(
        n=("match_id", "size"),
        over25_rate=("is_over25", "mean"),
        under25_rate=("is_under25", "mean"),
        avg_goals=("total_goals", "mean"),
        **{c: (c, "mean") for c in cols_key},
    ).round(4)
    print(agg)

    # Entropia over/under per cluster
    print("\n📉 Entropia (over/under 2.5) per cluster:")
    rows = []
    for c, sub in df.groupby(cluster_col):
        pov = safe_rate(sub["is_over25"])
        pun = safe_rate(sub["is_under25"])
        H = entropy_from_probs(np.array([pov, pun]))
        rows.append({"cluster": c, "pO25": pov, "pU25": pun, "entropy_ou25": H})
    ent_df = pd.DataFrame(rows).set_index("cluster").round(4)
    print(ent_df)

    disp_over = ent_df["pO25"].std()
    print("\n📌 DISPERSIONE tra cluster (std p_over25 tra medie di cluster): "
          f"{disp_over:.4f}")

    print("\n💡 Hint miglioramento OU2.5:")
    print("   - Cluster con pO25 molto simili ⇒ non aggiungono informazione.")
    print("   - Guardare quelli con over25_rate ~0.3 vs ~0.7: sono i più 'puri'.")
    print("   - Per migliorare: includere dispersione di over25_rate nello SCORE di scelta K.")
    print("   - Possibile aggiunta feature: normalizzare lambda e delta per lega/campionato.\n")


# -----------------------------------------------------------
# ANALISI CLUSTER OU1.5
# -----------------------------------------------------------
def analyze_ou15(step3_path: Path):
    print("\n" + "="*80)
    print("🔍 ANALISI CLUSTER OU1.5")
    print("="*80)

    if not step3_path.exists():
        print(f"❌ File step3c non trovato: {step3_path}")
        return

    df = pd.read_parquet(step3_path)
    print(f"📏 step3c shape: {df.shape}")

    cluster_col = "cluster_ou15"
    if cluster_col not in df.columns:
        cluster_col = [c for c in df.columns if c.startswith("cluster_")][0]
    print(f"📌 Colonna cluster: {cluster_col}")

    vc = df[cluster_col].value_counts().sort_index()
    print("\n📊 Distribuzione cluster (conteggio & %):")
    total = len(df)
    for k, n in vc.items():
        print(f"  - cluster {k}: {n:6d} ({n/total*100:5.1f}%)")

    cols_key = [
        "lambda_home_form", "lambda_away_form", "lambda_total_form",
        "lambda_total_market_ou25", "lambda_total_mix_ou15",
        "tech_pO05_mix", "tech_pO15_mix", "tech_pO25_mix",
        "pic_pO15", "pic_pU15",
        "pic_pO05", "pic_pU05",
        "delta_ou15_pic_vs_mix", "delta_ou15_pic_vs_mix_abs",
        "season_recency", "match_density_index",
        "rest_diff_days",
        "short_rest_home", "short_rest_away",
        "rest_advantage_home", "rest_advantage_away",
        "tightness_index",
    ]
    cols_key = [c for c in cols_key if c in df.columns]

    print("\n📈 Medie feature chiave + esiti per cluster OU1.5:")
    agg = df.groupby(cluster_col).agg(
        n=("match_id", "size"),
        over15_rate=("is_over15", "mean"),
        **{c: (c, "mean") for c in cols_key},
    ).round(4)
    print(agg)

    # Entropia over/under 1.5 per cluster
    print("\n📉 Entropia (over/under 1.5) per cluster:")
    rows = []
    for c, sub in df.groupby(cluster_col):
        pov = safe_rate(sub["is_over15"])
        pun = 1.0 - pov if not np.isnan(pov) else np.nan
        H = entropy_from_probs(np.array([pov, pun]))
        rows.append({"cluster": c, "pO15": pov, "pU15": pun, "entropy_ou15": H})
    ent_df = pd.DataFrame(rows).set_index("cluster").round(4)
    print(ent_df)

    disp_over = ent_df["pO15"].std()
    print("\n📌 DISPERSIONE tra cluster (std p_over15 tra medie di cluster): "
          f"{disp_over:.4f}")

    print("\n💡 Hint miglioramento OU1.5:")
    print("   - Cluster con over15_rate molto vicini → meno utili per betting.")
    print("   - Molto interessante quando alcuni cluster hanno over15_rate >0.9 e altri ~0.6.")
    print("   - Si può usare dispersione over15_rate come componente dello SCORE di scelta K.")
    print("   - Eventualmente includere info da cluster_1x2 (joinando anche quel cluster qui).\n")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    step3_1x2 = DATA_DIR / "step3a_1x2_clusters_v2.parquet"
    step3_ou25 = DATA_DIR / "step3b_ou25_clusters_v2.parquet"
    step3_ou15 = DATA_DIR / "step3c_ou15_clusters_v2.parquet"
    wide_file  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

    print("============================================================")
    print("🚀 ANALYZE_CLUSTERS_V2 — Diagnostica cluster 1X2 / OU2.5 / OU1.5")
    print("============================================================")
    print(f"📥 step3a 1X2  : {step3_1x2}")
    print(f"📥 step3b OU25 : {step3_ou25}")
    print(f"📥 step3c OU15 : {step3_ou15}")
    print(f"📥 WIDE index  : {wide_file}")
    print("============================================================")

    analyze_1x2(step3_1x2, wide_file)
    analyze_ou25(step3_ou25)
    analyze_ou15(step3_ou15)

    print("\n🏁 ANALYZE_CLUSTERS_V2 COMPLETATO")
    print("============================================================")


if __name__ == "__main__":
    main()