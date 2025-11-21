#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI FAVORITA MULTIGOAL (1–3, 1–4, 1–5)

Studiamo i casi dove la FAVORITA segna:
    • 1–3 gol
    • 1–4 gol
    • 1–5 gol

Usiamo:
    - step5_soft_history.parquet  (risultati + modelli soft)
    - step4a_affini_index_wide_v2.parquet (quote bk + pic + features)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# PATH
# ----------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

OUT_DATA  = DATA_DIR / "step5_favorita_multigol_dataset.parquet"
OUT_RULES = DATA_DIR / "step5_favorita_multigol_rules.parquet"


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_mean(x: pd.Series) -> float:
    return float(x.mean()) if len(x) else float("nan")


# ----------------------------------------------------------
# MAIN ANALYSIS
# ----------------------------------------------------------

def main():
    print("============================================================")
    print("🚀 STEP5 — ANALISI FAVORITA MULTIGOAL (1–3, 1–4, 1–5)")
    print("============================================================")

    # -------------------------------------------
    # 1) Caricamento file
    # -------------------------------------------
    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    print(f"Soft shape: {soft.shape}")
    print(f"Wide shape: {wide.shape}")

    # Selezioniamo solo le colonne che ci servono per evitare _x / _y
    soft_cols = [
        "match_id",
        "status", "date", "league", "home_team", "away_team",
        "cluster_1x2", "cluster_ou25", "cluster_ou15",
        "soft_p1", "soft_px", "soft_p2",
        "soft_pO15", "soft_pU15", "soft_pO25", "soft_pU25",
        "home_ft", "away_ft", "total_goals",
        "is_over15_real", "is_over25_real", "is_gg_real",
    ]

    wide_cols = [
        "match_id",
        "bk_p1", "bk_px", "bk_p2",
        "pic_p1", "pic_px", "pic_p2",
        "tightness_index", "lambda_total_form",
    ]

    soft_use = soft[soft_cols].copy()
    wide_use = wide[wide_cols].copy()

    # -------------------------------------------
    # 2) Merge unico
    # -------------------------------------------
    df = pd.merge(
        soft_use,
        wide_use,
        on="match_id",
        how="inner"
    )

    print(f"Dataset unito: {df.shape}")

    # -------------------------------------------
    # 3) Definizione favorita (lato bookmaker)
    # -------------------------------------------
    # bk_p* sono già probabilità normalizzate → favorita = prob. massima
    df["is_home_fav"] = (df["bk_p1"] >= df["bk_px"]) & (df["bk_p1"] >= df["bk_p2"])
    df["is_away_fav"] = (df["bk_p2"] >= df["bk_px"]) & (df["bk_p2"] >= df["bk_p1"])

    df["fav_side"] = np.where(
        df["is_home_fav"] & ~df["is_away_fav"], "home",
        np.where(df["is_away_fav"] & ~df["is_home_fav"], "away", "none")
    )

    # Teniamo solo match con favorita chiara
    df = df[df["fav_side"] != "none"].copy()

    # Gol reali della favorita
    df["fav_goals"] = np.where(df["fav_side"] == "home", df["home_ft"], df["away_ft"])

    # -------------------------------------------
    # 4) Etichette multigol favorita
    # -------------------------------------------
    df["fav_1_3"] = df["fav_goals"].between(1, 3).astype(int)
    df["fav_1_4"] = df["fav_goals"].between(1, 4).astype(int)
    df["fav_1_5"] = df["fav_goals"].between(1, 5).astype(int)

    # -------------------------------------------
    # 5) Statistiche globali
    # -------------------------------------------
    print_section("📊 STATISTICHE GLOBALI")

    tot = len(df)
    print(f"Totale partite con favorita chiara: {tot}")
    print(f"P(favorita 1–3 gol) = {df['fav_1_3'].mean():.3f}")
    print(f"P(favorita 1–4 gol) = {df['fav_1_4'].mean():.3f}")
    print(f"P(favorita 1–5 gol) = {df['fav_1_5'].mean():.3f}")

    # -------------------------------------------
    # 6) Breakdown per cluster
    # -------------------------------------------
    print_section("📊 CLUSTER 1X2 → FAVORITA MULTIGOAL")
    print(df.groupby("cluster_1x2")[["fav_1_3", "fav_1_4", "fav_1_5"]].mean())

    print_section("📊 CLUSTER OU25 → FAVORITA MULTIGOAL")
    if "cluster_ou25" in df.columns:
        print(df.groupby("cluster_ou25")[["fav_1_3", "fav_1_4", "fav_1_5"]].mean())
    else:
        print("cluster_ou25 non disponibile.")

    print_section("📊 CLUSTER OU15 → FAVORITA MULTIGOAL")
    if "cluster_ou15" in df.columns:
        print(df.groupby("cluster_ou15")[["fav_1_3", "fav_1_4", "fav_1_5"]].mean())
    else:
        print("cluster_ou15 non disponibile.")

    # -------------------------------------------
    # 7) Breakdown per tightness + lambda
    # -------------------------------------------
    df = df.dropna(subset=["tightness_index", "lambda_total_form"]).copy()

    df["tight_decile"] = pd.qcut(df["tightness_index"], 10, duplicates="drop")
    df["lambda_decile"] = pd.qcut(df["lambda_total_form"], 10, duplicates="drop")

    print_section("📊 MULTIGOAL per decili tightness_index")
    print(df.groupby("tight_decile")[["fav_1_3", "fav_1_4", "fav_1_5"]].mean())

    print_section("📊 MULTIGOAL per decili λ_total_form")
    print(df.groupby("lambda_decile")[["fav_1_3", "fav_1_4", "fav_1_5"]].mean())

    # -------------------------------------------
    # 8) GRID SEARCH (pattern favorita multigol)
    # -------------------------------------------
    print_section("🔍 GRID SEARCH FAVORITA MULTIGOAL (target = fav_1_3)")

    grid_rules = []

    tight_vals = np.linspace(
        df["tightness_index"].quantile(0.1),
        df["tightness_index"].quantile(0.9),
        12
    )
    lam_vals = np.linspace(
        df["lambda_total_form"].quantile(0.1),
        df["lambda_total_form"].quantile(0.9),
        12
    )
    pic_vals = np.linspace(0.25, 0.90, 12)

    target = "fav_1_3"  # se vuoi puoi cambiare in fav_1_4 / fav_1_5

    n_total = len(df)

    for t_pic in pic_vals:
        for t_ti in tight_vals:
            for t_la in lam_vals:

                mask = (
                    ((df["fav_side"] == "home") & (df["pic_p1"] >= t_pic)) |
                    ((df["fav_side"] == "away") & (df["pic_p2"] >= t_pic))
                ) & (df["tightness_index"] <= t_ti) & (df["lambda_total_form"] >= t_la)

                n = int(mask.sum())
                if n < 400:  # minimo campione
                    continue

                acc = float(df.loc[mask, target].mean())
                coverage = n / n_total
                score = acc * coverage

                grid_rules.append({
                    "rule": f"pic≥{t_pic:.2f} & tight≤{t_ti:.2f} & lambda≥{t_la:.2f}",
                    "n": n,
                    "coverage": coverage,
                    "accuracy": acc,
                    "score": score,
                })

    if grid_rules:
        res = pd.DataFrame(grid_rules).sort_values("score", ascending=False)
        print_section("🏆 TOP 20 REGOLE MULTIGOAL FAVORITA")
        print(res.head(20))

        OUT_RULES.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(OUT_RULES, index=False)
        print(f"💾 Regole salvate in: {OUT_RULES}")
    else:
        print("⚠️ Nessuna regola multigol trovata con i vincoli scelti.")

    # -------------------------------------------
    # 9) Salvataggio dataset annotato
    # -------------------------------------------
    # pyarrow non gestisce bene le colonne con Interval/Categorical → le convertiamo a stringa
    for col in ["tight_decile", "lambda_decile"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DATA, index=False)
    print(f"💾 Dataset annotato salvato in: {OUT_DATA}")

    print("\n============================================================")
    print("🏁 ANALISI FAVORITA MULTIGOAL COMPLETATA")
    print("============================================================")


if __name__ == "__main__":
    main()