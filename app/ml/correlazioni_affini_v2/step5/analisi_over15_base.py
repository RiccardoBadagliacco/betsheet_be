#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI OVER 1.5 COMPLETA

Utilizza:
    - step5_soft_history.parquet
    - step4a_affini_index_wide_v2.parquet

Output:
    - diagnostica sul comportamento dell'Over 1.5
    - analisi per cluster, lambda, tightness, soft probabilities
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


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



# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def print_section(title: str):
    print("\n" + "="*60)
    print(title)
    print("="*60)



# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print_section("🚀 ANALISI COMPLETA OVER 1.5")

    # ----------------------------------------
    # LOAD DATA
    # ----------------------------------------
    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    df = pd.merge(
        soft,
        wide[[
            "match_id","cluster_1x2","cluster_ou25","cluster_ou15",
            "tightness_index","lambda_total_form"
        ]],
        on="match_id",
        how="left"
    )

    # Assicuriamoci che esista la colonna outcome
    df["is_over15_real"] = (df["total_goals"] >= 2).astype(int)

    print_section("📌 INFO DATASET")
    print(df.head())
    print(df.describe())

    print("\nTotale partite:", len(df))
    print("Over 1.5 globali:", df["is_over15_real"].mean().round(3))

    # ----------------------------------------
    # DISTRIBUZIONE OVER1.5 PER CLUSTER
    # ----------------------------------------

    print_section("📊 OVER 1.5 per CLUSTER OU15")
    print(df.groupby("cluster_ou15_x")["is_over15_real"].mean())

    print_section("📊 OVER 1.5 per CLUSTER OU25")
    print(df.groupby("cluster_ou25_x")["is_over15_real"].mean())


    # ----------------------------------------
    # LAMBDA TOTAL FORM (expected goals combined)
    # ----------------------------------------

    print_section("📊 OVER 1.5 per decili di λ_total_form")
    df["lambda_decile"] = pd.qcut(df["lambda_total_form"], 10, duplicates="drop")
    print(df.groupby("lambda_decile")["is_over15_real"].mean())


    # ----------------------------------------
    # TIGHTNESS INDEX — misura quanto la partita è “bloccata”
    # ----------------------------------------

    print_section("📊 OVER 1.5 per decili di tightness_index")
    df["tight_decile"] = pd.qcut(df["tightness_index"], 10, duplicates="drop")
    print(df.groupby("tight_decile")["is_over15_real"].mean())


    # ----------------------------------------
    # SOFT PREDICTIONS (affini): pO15
    # ----------------------------------------

    print_section("📊 OVER 1.5 per decili soft_pO15")
    df["soft_o15_decile"] = pd.qcut(df["soft_pO15"], 10, duplicates="drop")
    print(df.groupby("soft_o15_decile")["is_over15_real"].mean())


    # ----------------------------------------
    # INCROCI CLUSTER OU15 × LAMBDA
    # ----------------------------------------

    print_section("🔍 MATRICE: OVER 1.5 per (cluster_ou15 × decile λ)")
    pivot = df.pivot_table(
        values="is_over15_real",
        index="cluster_ou15_x",
        columns="lambda_decile",
        aggfunc="mean"
    )
    print(pivot.round(3))


    # ----------------------------------------
    # INCROCI TIGHTNESS × LAMBDA
    # ----------------------------------------

    print_section("🔍 MATRICE: OVER 1.5 per (tightness_decile × λ_decile)")
    pivot2 = df.pivot_table(
        values="is_over15_real",
        index="tight_decile",
        columns="lambda_decile",
        aggfunc="mean"
    )
    print(pivot2.round(3))


    # ----------------------------------------
    # INCROCIO FINALE soft_pO15 × lambda
    # ----------------------------------------

    print_section("🔍 MATRICE: soft_pO15_decile × λ_decile")
    pivot3 = df.pivot_table(
        values="is_over15_real",
        index="soft_o15_decile",
        columns="lambda_decile",
        aggfunc="mean"
    )
    print(pivot3.round(3))

    print_section("🏁 ANALISI COMPLETATA")


if __name__ == "__main__":
    main()