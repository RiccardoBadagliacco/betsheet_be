#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI 0–0 (ZEROZERO)

Obiettivi:
- Misurare in modo strutturato:
    * frequenza globale dello 0–0
    * 0–0 per cluster (ou15, ou25, 1x2)
    * 0–0 per decili di:
        - lambda_total_form
        - tightness_index
        - soft_pU15 / soft_pU25
        - bk_px (probabilità implicita pareggio)
    * matrici 2D (lambda x tightness, ecc.)
    * comportamento dello 0–0 quando la favorita NON segna

- Lanciare una grid-search per individuare pattern/regole del tipo:
    "lambda bassa + tightness alta + cluster underish + pU15 alta"

Output:
- Report a console
- Dataset annotato:
    data/step5_zerozero_dataset.parquet
- Regole candidate:
    data/step5_zerozero_rules.parquet

Uso:
    python analisi_zero_zero.py
    python analisi_zero_zero.py --n 20000
"""

import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------
# PATH & IMPORT
# -----------------------

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR   = Path(__file__).resolve().parents[2]
AFFINI_DIR = BASE_DIR / "correlazioni_affini_v2"
DATA_DIR   = AFFINI_DIR / "data"

SOFT_FILE  = DATA_DIR / "step5_soft_history.parquet"
WIDE_FILE  = DATA_DIR / "step4a_affini_index_wide_v2.parquet"

OUT_DATA   = DATA_DIR / "step5_over15_trap_dataset.parquet"
OUT_RULES  = DATA_DIR / "step5_over15_trap_rules.parquet"


# --------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def safe_int(x, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


# --------------------------------------------------------------------
# Caricamento e merge
# --------------------------------------------------------------------
def load_dataset(n_matches: Optional[int] = None) -> pd.DataFrame:
    print("====================================================")
    print("🚀 STEP5 — ANALISI 0–0 (ZEROZERO)")
    print("====================================================\n")

    print(f"📥 Carico soft_history da: {SOFT_FILE}")
    df_soft = pd.read_parquet(SOFT_FILE)

    print(f"📥 Carico wide da:        {WIDE_FILE}")
    wide = pd.read_parquet(WIDE_FILE)

    # Seleziono dal wide solo le colonne che servono
    wide_cols_needed = [
    "match_id",
    "date",
    "league",
    "home_team",
    "away_team",
    "home_ft",
    "away_ft",
    "bk_p1", "bk_px", "bk_p2",
    "pic_p1", "pic_p2",
    "tightness_index",
    "lambda_total_form",
    "cluster_1x2",
    "cluster_ou15",
    "cluster_ou25",
]
    missing = [c for c in wide_cols_needed if c not in wide.columns]
    if missing:
        raise RuntimeError(f"Mancano colonne nel wide: {missing}")

    wide_sel = wide[wide_cols_needed].copy()

    # Merge soft_history + wide su match_id
    df = df_soft.merge(wide_sel, on="match_id", how="inner", suffixes=("", "_w"))

    # Filtra solo status ok
    df = df[df["status"] == "ok"].copy()

    if n_matches is not None and n_matches < len(df):
        df = df.sample(n_matches, random_state=42).copy()
        print(f"✂️  Uso un sottoinsieme di {len(df)} partite per l'analisi.")
    else:
        print(f"➡️  Analisi su tutte le partite disponibili: {len(df)}")

    # Target 0–0
    df["home_ft"] = df["home_ft"].astype(int)
    df["away_ft"] = df["away_ft"].astype(int)
    df["is_zerozero"] = (df["home_ft"] == 0) & (df["away_ft"] == 0)

    return df


# --------------------------------------------------------------------
# Analisi descrittiva
# --------------------------------------------------------------------
def analisi_descrittiva(df: pd.DataFrame) -> None:
    print("\n====================================================")
    print("📌 INFO DATASET")
    print("====================================================")
    print(df[["match_id", "status", "date", "league", "home_team", "away_team",
              "home_ft", "away_ft"]].head())
    print()
    print(df[["home_ft", "away_ft", "lambda_total_form",
              "tightness_index", "cluster_ou15", "cluster_ou25"]].describe())
    print()

    n = len(df)
    base_rate = df["is_zerozero"].mean()
    print(f"Totale partite: {n}")
    print(f"Frequenza globale 0–0: {base_rate:.3f}\n")

    # Per league
    print("====================================================")
    print("📊 0–0 per LEGA (top 20 per numero di match)")
    print("====================================================")
    g_league = (
        df.groupby("league")
        .agg(
            n=("is_zerozero", "size"),
            zerozero_rate=("is_zerozero", "mean"),
        )
        .sort_values("n", ascending=False)
    )
    print(g_league.head(20).to_string())
    print()

    # Per cluster OU15, OU25, 1x2
    print("====================================================")
    print("📊 0–0 per CLUSTER OU15")
    print("====================================================")
    print(df.groupby("cluster_ou15")["is_zerozero"].mean().to_string())
    print()

    print("====================================================")
    print("📊 0–0 per CLUSTER OU25")
    print("====================================================")
    print(df.groupby("cluster_ou25")["is_zerozero"].mean().to_string())
    print()

    print("====================================================")
    print("📊 0–0 per CLUSTER 1X2")
    print("====================================================")
    print(df.groupby("cluster_1x2")["is_zerozero"].mean().to_string())
    print()

    # Decili lambda
    print("====================================================")
    print("📊 0–0 per decili λ_total_form")
    print("====================================================")
    df = df.copy()
    df["lambda_decile"] = pd.qcut(
        df["lambda_total_form"],
        10,
        duplicates="drop"
    )
    print(df.groupby("lambda_decile")["is_zerozero"].mean().to_string())
    print()

    # Decili tightness
    print("====================================================")
    print("📊 0–0 per decili tightness_index")
    print("====================================================")
    df["tight_decile"] = pd.qcut(
        df["tightness_index"],
        10,
        duplicates="drop"
    )
    print(df.groupby("tight_decile")["is_zerozero"].mean().to_string())
    print()

    # Decili soft_pU15 / soft_pU25 (se presenti)
    if "soft_pU15" in df.columns:
        print("====================================================")
        print("📊 0–0 per decili soft_pU15")
        print("====================================================")
        df["soft_u15_decile"] = pd.qcut(
            df["soft_pU15"],
            10,
            duplicates="drop"
        )
        print(df.groupby("soft_u15_decile")["is_zerozero"].mean().to_string())
        print()

    if "soft_pU25" in df.columns:
        print("====================================================")
        print("📊 0–0 per decili soft_pU25")
        print("====================================================")
        df["soft_u25_decile"] = pd.qcut(
            df["soft_pU25"],
            10,
            duplicates="drop"
        )
        print(df.groupby("soft_u25_decile")["is_zerozero"].mean().to_string())
        print()

    # Decili bk_px (probabilità implicita pareggio)
    print("====================================================")
    print("📊 0–0 per decili bk_px (probabilità implicita X)")
    print("====================================================")
    df["bk_px_decile"] = pd.qcut(
        df["bk_px"],
        10,
        duplicates="drop"
    )
    print(df.groupby("bk_px_decile")["is_zerozero"].mean().to_string())
    print()

    # Matrice lambda x tight
    print("====================================================")
    print("🔍 MATRICE: 0–0 per (lambda_decile × tight_decile)")
    print("====================================================")
    pivot = df.pivot_table(
        index="tight_decile",
        columns="lambda_decile",
        values="is_zerozero",
        aggfunc="mean",
        observed=False,
    )
    print(pivot.to_string())
    print()

    # Matrice cluster_ou25 x lambda_decile
    print("====================================================")
    print("🔍 MATRICE: 0–0 per (cluster_ou25 × lambda_decile)")
    print("====================================================")
    pivot2 = df.pivot_table(
        index="cluster_ou25",
        columns="lambda_decile",
        values="is_zerozero",
        aggfunc="mean",
        observed=False,
    )
    print(pivot2.to_string())
    print()

    # Matrice soft_pU15_decile x tight_decile (se disponibile)
    if "soft_u15_decile" in df.columns:
        print("====================================================")
        print("🔍 MATRICE: 0–0 per (soft_u15_decile × tight_decile)")
        print("====================================================")
        pivot3 = df.pivot_table(
            index="soft_u15_decile",
            columns="tight_decile",
            values="is_zerozero",
            aggfunc="mean",
            observed=False,
        )
        print(pivot3.to_string())
        print()


# --------------------------------------------------------------------
# Analisi favorita non segna
# --------------------------------------------------------------------
def analisi_favorita_nosegna(df: pd.DataFrame) -> None:
    print("====================================================")
    print("📊 0–0 E FAVORITA NON SEGNA")
    print("====================================================")

    df = df.copy()

    # Individua favorita tramite bk_p1/bk_p2/bk_px (probabilità implicite)
    bk_p1 = df["bk_p1"].astype(float)
    bk_p2 = df["bk_p2"].astype(float)
    bk_px = df["bk_px"].astype(float)

    is_home_fav = (bk_p1 > bk_p2) & (bk_p1 > bk_px)
    is_away_fav = (bk_p2 > bk_p1) & (bk_p2 > bk_px)

    df["fav_side"] = np.where(is_home_fav, "home",
                              np.where(is_away_fav, "away", "none"))

    df["fav_scores"] = np.where(
        df["fav_side"] == "home",
        df["home_ft"],
        np.where(df["fav_side"] == "away", df["away_ft"], np.nan),
    )
    df["fav_nogoal"] = (df["fav_side"] != "none") & (df["fav_scores"] == 0)

    # Statistiche base
    mask_fav = df["fav_side"].isin(["home", "away"])
    fav_df = df[mask_fav].copy()
    if fav_df.empty:
        print("Nessuna favorita trovata, impossibile analizzare.")
        return

    base_zero_all = df["is_zerozero"].mean()
    base_zero_fav = fav_df["is_zerozero"].mean()
    base_zero_fav_nogoal = fav_df.loc[fav_df["fav_nogoal"], "is_zerozero"].mean()
    base_zero_fav_goal = fav_df.loc[~fav_df["fav_nogoal"], "is_zerozero"].mean()

    print(f"Frequenza globale 0–0:                    {base_zero_all:.3f}")
    print(f"Frequenza 0–0 nei match con favorita:      {base_zero_fav:.3f}")
    print(f"Frequenza 0–0 quando favorita NON segna:   {base_zero_fav_nogoal:.3f}")
    print(f"Frequenza 0–0 quando favorita segna:       {base_zero_fav_goal:.3f}")
    print()

    # Split home/away
    fav_home = fav_df[fav_df["fav_side"] == "home"]
    fav_away = fav_df[fav_df["fav_side"] == "away"]

    def _rate_zero(s: pd.DataFrame) -> float:
        if s.empty:
            return math.nan
        return float(s["is_zerozero"].mean())

    print("0–0 quando favorita HOME non segna:")
    print(f"Campione: {len(fav_home[fav_home['fav_nogoal']])}")
    print(f"Rate 0–0: {_rate_zero(fav_home[fav_home['fav_nogoal']]):.3f}\n")

    print("0–0 quando favorita AWAY non segna:")
    print(f"Campione: {len(fav_away[fav_away['fav_nogoal']])}")
    print(f"Rate 0–0: {_rate_zero(fav_away[fav_away['fav_nogoal']]):.3f}\n")

    # Cluster_ou25 quando fav_nogoal
    print("Distribuzione 0–0 per cluster_ou25 nei match con favorita non segna:")
    print(
        fav_df[fav_df["fav_nogoal"]]
        .groupby("cluster_ou25")["is_zerozero"]
        .mean()
        .to_string()
    )
    print()

    # tightness / lambda per fav_nogoal
    fav_df["lambda_decile"] = pd.qcut(
        fav_df["lambda_total_form"],
        10,
        duplicates="drop",
    )
    fav_df["tight_decile"] = pd.qcut(
        fav_df["tightness_index"],
        10,
        duplicates="drop",
    )

    print("Matrice 0–0 per (tight_decile × lambda_decile) nei match con favorita non segna:")
    pivot = fav_df[fav_df["fav_nogoal"]].pivot_table(
        index="tight_decile",
        columns="lambda_decile",
        values="is_zerozero",
        aggfunc="mean",
        observed=False,
    )
    print(pivot.to_string())
    print()


# --------------------------------------------------------------------
# Grid search regole 0–0
# --------------------------------------------------------------------
def grid_search_zerozero(df: pd.DataFrame, max_rules: int = 25) -> pd.DataFrame:
    print("====================================================")
    print("🔍 GRID SEARCH PATTERN 0–0")
    print("====================================================")

    n_total = len(df)
    base_zero = df["is_zerozero"].mean()
    print(f"Righe disponibili per grid-search: {n_total}")
    print(f"Baseline 0–0: {base_zero:.3f}")

    lam = df["lambda_total_form"].astype(float)
    tight = df["tightness_index"].astype(float)
    c15 = df["cluster_ou15"].apply(safe_int)
    c25 = df["cluster_ou25"].apply(safe_int)
    pU15 = df.get("soft_pU15", pd.Series(index=df.index, data=math.nan))
    pU25 = df.get("soft_pU25", pd.Series(index=df.index, data=math.nan))
    bk_px = df["bk_px"].astype(float)

    # cluster "underish"
    ou15_underish = {0, 5}
    ou25_underish = {1, 4}

    lam_th = [1.6, 1.8, 2.0, 2.2, 2.4]
    tight_th = [0.68, 0.70, 0.72, 0.74]
    pU15_th = [0.55, 0.60, 0.65, 0.70]
    pU25_th = [0.45, 0.50, 0.55, 0.60]
    bkpx_th = [0.27, 0.29, 0.31, 0.33]

    rules: List[Dict[str, Any]] = []

    min_n = max(150, int(0.01 * n_total))  # almeno 1% del campione

    comb_count = 0

    for lam_max in lam_th:
        for tight_min in tight_th:
            for pu15_min in pU15_th + [None]:
                for pu25_min in pU25_th + [None]:
                    for bkpx_min in bkpx_th + [None]:
                        for use_c15 in [False, True]:
                            for use_c25 in [False, True]:
                                comb_count += 1
                                # Costruisco la condizione
                                cond = pd.Series(True, index=df.index)

                                cond &= lam <= lam_max
                                cond &= tight >= tight_min

                                if pu15_min is not None:
                                    cond &= pU15 >= pu15_min
                                if pu25_min is not None:
                                    cond &= pU25 >= pu25_min
                                if bkpx_min is not None:
                                    cond &= bk_px >= bkpx_min

                                if use_c15:
                                    cond &= c15.isin(ou15_underish)
                                if use_c25:
                                    cond &= c25.isin(ou25_underish)

                                n = int(cond.sum())
                                if n < min_n:
                                    continue

                                sub = df[cond]
                                acc = sub["is_zerozero"].mean()

                                # score tipo (accuracy - baseline) * coverage
                                coverage = n / n_total
                                score = (acc - base_zero) * coverage

                                # costruzione descrizione regola
                                parts = [
                                    f"lambda <= {lam_max:.2f}",
                                    f"tight >= {tight_min:.2f}",
                                ]
                                if pu15_min is not None:
                                    parts.append(f"pU15 >= {pu15_min:.2f}")
                                if pu25_min is not None:
                                    parts.append(f"pU25 >= {pu25_min:.2f}")
                                if bkpx_min is not None:
                                    parts.append(f"bk_px >= {bkpx_min:.2f}")
                                if use_c15:
                                    parts.append("cluster_ou15 in {0,5}")
                                if use_c25:
                                    parts.append("cluster_ou25 in {1,4}")

                                rule_str = " AND ".join(parts)

                                rules.append(
                                    {
                                        "rule": rule_str,
                                        "coverage": coverage,
                                        "accuracy": acc,
                                        "score": score,
                                        "n": n,
                                    }
                                )

    print(f"Combinazioni testate: {comb_count}")

    if not rules:
        print("❌ Nessuna regola 0–0 trovata con i vincoli impostati.")
        return pd.DataFrame()

    df_rules = pd.DataFrame(rules).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    print("\n====================================================")
    print("🏆 TOP REGOLE 0–0")
    print("====================================================")
    print(df_rules.head(max_rules).to_string(index=True))
    print()

    # Salva su parquet
    out_rules = DATA_DIR / "step5_zerozero_rules.parquet"
    df_rules.to_parquet(out_rules, index=False)
    print(f"💾 Regole 0–0 salvate in: {out_rules}")

    return df_rules


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def run_analysis(n_matches: Optional[int] = None) -> None:
    df = load_dataset(n_matches=n_matches)

    # Analisi descrittiva pura
    analisi_descrittiva(df)

    # Analisi relazione 0–0 vs favorita non segna
    analisi_favorita_nosegna(df)

    # Grid search per pattern 0–0
    df_rules = grid_search_zerozero(df, max_rules=25)

    # Salvataggio dataset annotato
    out_ds = DATA_DIR / "step5_zerozero_dataset.parquet"
    cols_save = list(df.columns)
    # Già contiene is_zerozero; aggiungiamo eventuali colonne decili se create
    df.to_parquet(out_ds, index=False)
    print(f"💾 Dataset annotato salvato in: {out_ds}")

    print("\n====================================================")
    print("🏁 ANALISI 0–0 COMPLETATA")
    print("====================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Numero massimo di partite da usare (sample). Se omesso usa tutte.",
    )
    args = parser.parse_args()

    run_analysis(n_matches=args.n)


if __name__ == "__main__":
    main()