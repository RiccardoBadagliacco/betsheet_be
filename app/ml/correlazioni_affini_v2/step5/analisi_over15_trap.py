#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP5 — ANALISI OVER 1.5 TRAP (OVER atteso ma spesso NON esce)

Utilizza:
    - step5_soft_history.parquet
    - step4a_affini_index_wide_v2.parquet

Output:
    - diagnostica base sugli OVER 1.5
    - grid-search per pattern di OVER "TRAPPOLA"
    - salvataggio:
        * step5_over15_trap_rules.parquet
        * step5_over15_trap_dataset.parquet
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

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

OUT_DATA   = DATA_DIR / "step5_over15_trap_dataset.parquet"
OUT_RULES  = DATA_DIR / "step5_over15_trap_rules.parquet"


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_prop(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean()) if len(x) else float("nan")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assicura le colonne chiave per l'analisi:
      - is_over15_real
      - soft_pO15
      - cluster_ou15_x, cluster_ou25_x
      - tightness_index, lambda_total_form
    """

    # --- is_over15_real -----------------------------------
    if "is_over15_real" not in df.columns:
        # fallback: prova a costruirla da eventuali colonne score
        candidate_home = [c for c in df.columns
                          if c.lower() in ("home_ft", "home_ft_x", "home_ft_y",
                                           "goals_home_ft", "home_score")]
        candidate_away = [c for c in df.columns
                          if c.lower() in ("away_ft", "away_ft_x", "away_ft_y",
                                           "goals_away_ft", "away_score")]

        if candidate_home and candidate_away:
            h = df[candidate_home[0]].astype(float)
            a = df[candidate_away[0]].astype(float)
            df["is_over15_real"] = (h + a >= 2).astype(int)
        else:
            raise RuntimeError(
                "Manca 'is_over15_real' e non trovo colonne di risultato (home_ft/away_ft). "
                "Controlla i parquet step5_soft_history/step4a_affini_index_wide_v2."
            )
    else:
        df["is_over15_real"] = df["is_over15_real"].astype(int)

    # --- soft_pO15 ----------------------------------------
    if "soft_pO15" not in df.columns:
        # a volte può chiamarsi soft_po15, soft_o15 ecc: tentativo robusto
        alt = [c for c in df.columns if c.lower() in ("soft_po15", "soft_pover15", "soft_o15")]
        if not alt:
            raise RuntimeError("Colonna 'soft_pO15' non trovata nel dataframe.")
        df["soft_pO15"] = df[alt[0]]

    # --- cluster, tightness, lambda -----------------------
    # NB: dopo il merge avremo cluster_ou15_x / cluster_ou25_x dal SOFT
    #     e cluster_ou15_y / cluster_ou25_y dal WIDE (se presenti)
    for needed in ["cluster_ou15_x", "cluster_ou25_x"]:
        if needed not in df.columns:
            raise RuntimeError(f"Colonna '{needed}' mancante nel dataframe unito.")

    if "tightness_index" not in df.columns or "lambda_total_form" not in df.columns:
        raise RuntimeError("Mancano 'tightness_index' o 'lambda_total_form' dal WIDE parquet.")

    # Decili lambda e tightness per analisi descrittiva
    df["lambda_decile"] = pd.qcut(df["lambda_total_form"], 10, duplicates="drop")
    df["tight_decile"]  = pd.qcut(df["tightness_index"], 10, duplicates="drop")

    # Decili soft_pO15
    df["soft_o15_decile"] = pd.qcut(df["soft_pO15"], 10, duplicates="drop")

    return df


def grid_search_over15_trap(
    df: pd.DataFrame,
    baseline_over: float,
    min_soft: float = 0.75,
    max_over_rate: float = 0.72,
    min_cov: float = 0.02
) -> pd.DataFrame:
    """
    Cerca pattern tipo OVER-TRAPPOLA:
      - soft_pO15 alto (>= min_soft)
      - tightness alta
      - lambda bassa
      - cluster OU15 / OU25 "freddi"
      - frequenza reale Over1.5 <= max_over_rate (molto sotto baseline)

    Score = (baseline_over - over_rate) * coverage
    (più grande = drop forte + buon campione)
    """

    needed = [
        "is_over15_real", "soft_pO15",
        "cluster_ou15_x", "cluster_ou25_x",
        "tightness_index", "lambda_total_form"
    ]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Manca colonna necessaria per grid-search: {c}")

    # filtro base: solo match dove il modello vede OVER "alto"
    base = df[df["soft_pO15"] >= min_soft].copy()
    if base.empty:
        print("⚠️ Nessuna partita con soft_pO15 >= ", min_soft)
        return pd.DataFrame()

    n0 = len(base)
    print(f"\nPartite per grid-search (soft_pO15 ≥ {min_soft:.2f}): {n0}")
    print(f"Over1.5 in questo sottoinsieme: {safe_prop(base['is_over15_real']):.3f}")

    # range soglie
    tight_vals = np.linspace(0.55, 0.70, 7)   # da abbastanza alto a molto alto
    lam_vals   = np.linspace(1.6, 3.0, 8)     # lambda relativamente basse
    soft_vals  = np.linspace(max(min_soft, 0.75), 0.90, 7)

    # cluster "freddi" da analisi base:
    #   OU25: 1,4 tra quelli con over15 più bassi
    #   OU15: 3,4 mediamente più "blocchi"
    ou25_cold = [1.0, 4.0]
    ou15_cold = [3.0, 4.0]

    results: List[Dict[str, Any]] = []

    for s_min in soft_vals:
        for t_min in tight_vals:
            for l_max in lam_vals:

                mask = (
                    (base["soft_pO15"] >= s_min) &
                    (base["tightness_index"] >= t_min) &
                    (base["lambda_total_form"] <= l_max) &
                    (base["cluster_ou25_x"].isin(ou25_cold)) &
                    (base["cluster_ou15_x"].isin(ou15_cold))
                )

                n_sel = int(mask.sum())
                if n_sel == 0:
                    continue

                cov = n_sel / n0
                if cov < min_cov:
                    continue

                over_rate = safe_prop(base.loc[mask, "is_over15_real"])

                if over_rate > max_over_rate:
                    # non è abbastanza "trappola", troppi Over
                    continue

                drop = baseline_over - over_rate
                score = drop * cov

                rule_str = (
                    f"soft_pO15 ≥ {s_min:.3f} AND tight ≥ {t_min:.3f} "
                    f"AND lambda ≤ {l_max:.3f} AND cluster_ou25_x∈{ou25_cold} "
                    f"AND cluster_ou15_x∈{ou15_cold}"
                )

                results.append({
                    "rule": rule_str,
                    "soft_min": s_min,
                    "tight_min": t_min,
                    "lambda_max": l_max,
                    "coverage": cov,
                    "n": n_sel,
                    "over_rate": over_rate,
                    "drop_vs_global": drop,
                    "score": score
                })

    if not results:
        print("❌ Nessun pattern di OVER-TRAP trovato con i vincoli impostati.")
        return pd.DataFrame()

    rules = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return rules


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("=" * 60)
    print("🚀 ANALISI OVER 1.5 — TRAP FINDER")
    print("=" * 60)

    soft = pd.read_parquet(SOFT_FILE)
    wide = pd.read_parquet(WIDE_FILE)

    # Merge minimo necessario:
    # prendiamo da SOFT: status, date, league, teams, cluster_ou15_x, cluster_ou25_x, soft_pO15, is_over15_real
    # da WIDE: tightness_index, lambda_total_form
    df = pd.merge(
        soft[
            [
                "match_id", "status", "date", "league",
                "home_team", "away_team",
                "cluster_1x2", "cluster_ou25", "cluster_ou15",
                "soft_pO15", "is_over15_real"
            ]
        ].rename(
            columns={
                "cluster_1x2": "cluster_1x2_x",
                "cluster_ou25": "cluster_ou25_x",
                "cluster_ou15": "cluster_ou15_x",
            }
        ),
        wide[
            [
                "match_id",
                "cluster_1x2",
                "cluster_ou25",
                "cluster_ou15",
                "tightness_index",
                "lambda_total_form",
            ]
        ].rename(
            columns={
                "cluster_1x2": "cluster_1x2_y",
                "cluster_ou25": "cluster_ou25_y",
                "cluster_ou15": "cluster_ou15_y",
            }
        ),
        on="match_id",
        how="left",
    )

    print_section("📌 INFO DATASET")
    print(df.head())
    print(df.describe(include="all"))

    # Feature engineering
    df = add_features(df)

    # Filtra solo match con status ok
    df = df[df["status"] == "ok"].copy()

    # Baseline Over 1.5
    total_matches = len(df)
    baseline_over = safe_prop(df["is_over15_real"])
    print(f"\nTotale partite: {total_matches}")
    print(f"Over 1.5 globali: {baseline_over:.3f}")

    # ----------------------------------------------------
    # ANALISI DESCRITTIVA RAPIDA
    # ----------------------------------------------------
    print_section("📊 OVER 1.5 per CLUSTER OU15_x")
    print(df.groupby("cluster_ou15_x")["is_over15_real"].mean())

    print_section("📊 OVER 1.5 per CLUSTER OU25_x")
    print(df.groupby("cluster_ou25_x")["is_over15_real"].mean())

    print_section("📊 OVER 1.5 per decili λ_total_form")
    print(df.groupby("lambda_decile")["is_over15_real"].mean())

    print_section("📊 OVER 1.5 per decili tightness_index")
    print(df.groupby("tight_decile")["is_over15_real"].mean())

    print_section("📊 OVER 1.5 per decili soft_pO15")
    print(df.groupby("soft_o15_decile")["is_over15_real"].mean())

    # ----------------------------------------------------
    # GRID SEARCH — TRAP OVER 1.5
    # ----------------------------------------------------
    print_section("🔍 GRID SEARCH OVER 1.5 TRAP")
    rules = grid_search_over15_trap(
        df,
        baseline_over=baseline_over,
        min_soft=0.75,
        max_over_rate=0.72,   # se vuoi ancora più "trappola", abbassa a 0.70
        min_cov=0.02          # almeno 2% del sottoinsieme high-soft
    )

    if not rules.empty:
        print_section("🏆 TOP 25 PATTERN TRAPPOLA OVER 1.5")
        print(rules.head(25))
        rules.to_parquet(OUT_RULES, index=False)
        print(f"\n💾 Salvate regole in: {OUT_RULES}")

    # Salvataggio dataset annotato (per verifiche future / join con altre analisi)
    # Salvataggio dataset annotato (per verifiche future / join con altre analisi)
    df_save = df.copy()

    # le colonne di decile sono Interval/Categorical: le convertiamo in stringa
    for col in ["lambda_decile", "tight_decile", "soft_o15_decile"]:
        if col in df_save.columns:
            df_save[col] = df_save[col].astype(str)

    df_save.to_parquet(OUT_DATA, index=False)
    print(f"💾 Salvato dataset annotato in: {OUT_DATA}")

    print("\n" + "=" * 60)
    print("🏁 ANALISI OVER 1.5 TRAP COMPLETATA")
    print("=" * 60)


if __name__ == "__main__":
    main()