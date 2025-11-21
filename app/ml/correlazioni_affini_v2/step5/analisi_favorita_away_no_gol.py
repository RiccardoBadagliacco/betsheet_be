#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI FAVORITA AWAY CHE NON SEGNA (AWAY_FT=0)

Utilizza:
    - step5_soft_history.parquet
    - step4a_affini_index_wide_v2.parquet

Output:
    - diagnostica completa su favorita away che non segna
    - grid-search pattern favorita fragile (away)
    - salvataggio regole e dataset annotato
"""

import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

OUT_DATA   = DATA_DIR / "step5_favorita_away_nogoal_dataset.parquet"
OUT_RULES  = DATA_DIR / "step5_favorita_away_nogoal_rules.parquet"


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def print_section(title: str):
    print("\n" + "="*50)
    print(title)
    print("="*50)

def is_away_fav(row):
    try:
        return row["bk_p2"] > row["bk_p1"] and row["bk_p2"] > row["bk_px"]
    except:
        return False

def safe_prop(x: pd.Series):
    return float(x.mean()) if len(x) else float("nan")


# -------------------------------------------------------
# GRID SEARCH FAVORITA AWAY NON SEGNA
# -------------------------------------------------------

def grid_search_fav_away_nogoal(
        df: pd.DataFrame,
        min_acc: float = 0.22,
        max_acc: float = 0.40,
        min_cov: float = 0.25):

    needed = ["is_fav_away", "is_no_goal",
              "pic_p1", "pic_p2",
              "tightness_index", "lambda_total_form",
              "soft_p2"]

    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")

    df = df[df["is_fav_away"] == 1].dropna(subset=needed)
    if df.empty:
        print("⚠️ Nessuna riga valida.")
        return pd.DataFrame()

    y = df["is_no_goal"].astype(int)
    n = len(df)

    print(f"Righe disponibili per grid-search: {n}")
    print(f"Baseline favorito-away-noGoal: {y.mean():.3f}")

    # Range soglie ricavati dalle analisi base
    pic2_vals  = np.linspace(0.35, 0.80, 10)
    pic1_vals  = np.linspace(0.05, 0.40, 10)
    tight_vals = np.linspace(0.40, 0.85, 10)
    lam_vals   = np.linspace(1.0, 3.5, 10)

    results = []
    total = len(pic1_vals)*len(pic2_vals)*len(tight_vals)*len(lam_vals)
    print(f"Combinazioni testate: {total}")

    for p2_max in pic2_vals:
        for p1_min in pic1_vals:
            for t_min in tight_vals:
                for lam_max in lam_vals:

                    mask = (
                        (df["pic_p2"] <= p2_max) &
                        (df["pic_p1"] >= p1_min) &
                        (df["tightness_index"] >= t_min) &
                        (df["lambda_total_form"] <= lam_max)
                    )

                    n_sel = mask.sum()
                    if n_sel == 0:
                        continue

                    cov = n_sel / n
                    if cov < min_cov:
                        continue

                    acc = safe_prop(y[mask])
                    if not(min_acc <= acc <= max_acc):
                        continue

                    score = acc * cov
                    results.append({
                        "rule": f"pic_p2 <= {p2_max:.3f} AND pic_p1 >= {p1_min:.3f} "
                                f"AND tight >= {t_min:.3f} AND lambda <= {lam_max:.3f}",
                        "coverage": cov,
                        "accuracy": acc,
                        "score": score,
                        "n": n_sel
                    })

    if not results:
        print("❌ Nessuna regola trovata.")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)



# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("====================================================")
    print("🚀 STEP5 — ANALISI FAVORITA AWAY CHE NON SEGNA")
    print("====================================================")

    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    # ---------------------------------------------------
    # MERGE COME NELLA VERSIONE HOME
    # ---------------------------------------------------
    df = pd.merge(
        wide[[
            "match_id","bk_p1","bk_px","bk_p2","away_ft",
            "pic_p1","pic_p2","cluster_1x2","cluster_ou25",
            "tightness_index","lambda_total_form"
        ]],
        soft[["match_id","soft_p2"]],
        on="match_id",
        how="left"
    )

    print_section("📌 Dataset unito")
    print(df.head())

    df["is_fav_away"] = df.apply(is_away_fav, axis=1).astype(int)
    df["is_no_goal"]  = (df["away_ft"]==0).astype(int)

    fav = df[df["is_fav_away"] == 1]

    # ---------------------------------------------------
    # STATISTICHE BASE
    # ---------------------------------------------------

    print_section("📊 Statistiche favorite AWAY")
    print(f"Totale favorite away: {len(fav)}")
    print(f"Favorite away che NON segnano: {fav['is_no_goal'].sum()}  ({fav['is_no_goal'].mean():.3f})")

    # ---------------------------------------------------
    # CLUSTER DISTRIBUTION
    # ---------------------------------------------------

    print_section("📊 No goal per cluster 1X2")
    print(fav.groupby("cluster_1x2")["is_no_goal"].mean())

    print_section("📊 No goal per cluster OU25")
    print(fav.groupby("cluster_ou25")["is_no_goal"].mean())


    # ---------------------------------------------------
    # PICCHETTO ANALYSIS
    # ---------------------------------------------------

    print_section("📊 Picchetto")
    print(pd.qcut(fav["pic_p2"], 10, duplicates="drop").astype(str).value_counts().sort_index())
    print("\nno-goal rate by pic_p2 decile:")
    print(fav.groupby(pd.qcut(fav["pic_p2"], 10, duplicates='drop'))["is_no_goal"].mean())

    # ---------------------------------------------------
    # Tightness & Lambda
    # ---------------------------------------------------

    print_section("📊 Tightness & Lambda")
    print("no goal rate by tightness:")
    print(fav.groupby(pd.qcut(fav["tightness_index"], 10, duplicates='drop'))["is_no_goal"].mean())

    print("\nno goal rate by lambda_total_form:")
    print(fav.groupby(pd.qcut(fav["lambda_total_form"], 10, duplicates='drop'))["is_no_goal"].mean())

    # ---------------------------------------------------
    # GRID SEARCH
    # ---------------------------------------------------

    print_section("🔍 GRID SEARCH PATTERN FAVORITA AWAY NON SEGNA")
    rules = grid_search_fav_away_nogoal(fav)

    if not rules.empty:
        print_section("🏆 TOP 25 REGOLE")
        print(rules.head(25))
        rules.to_parquet(OUT_RULES, index=False)
        print(f"\n💾 Salvate regole in: {OUT_RULES}")

    df.to_parquet(OUT_DATA, index=False)
    print(f"💾 Salvato dataset annotato in: {OUT_DATA}")

    print("\n====================================================")
    print("🏁 ANALISI COMPLETATA")
    print("====================================================")


if __name__ == "__main__":
    main()