#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep study sulle relazioni tra:
- quota favorita
- MG Favorita 1-3 / 1-4 / 1-5 (mg_fav_1_3 / 1_4 / 1_5)
- risultati reali (gol favorita)
- scoreline top4

Funzioni chiave:
- run_deep_parquet_study(parquet_path, min_samples=200)

Ritorna un dict di DataFrame con i risultati principali, e stampa un report
abbastanza approfondito in console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# FEATURE ENGINEERING DI BASE
# ============================================================

def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
    - fav_goals
    - is_mg13_real, is_mg14_real, is_mg15_real
    - real_score
    """
    df = df.copy()

    # Gol reali della favorita
    df["fav_goals"] = np.where(df["fav_side"] == "home", df["home_ft"], df["away_ft"])

    # Esiti reali MG 1-3, 1-4, 1-5
    df["is_mg13_real"] = df["fav_goals"].between(1, 3).astype(int)
    df["is_mg14_real"] = df["fav_goals"].between(1, 4).astype(int)
    df["is_mg15_real"] = df["fav_goals"].between(1, 5).astype(int)

    # Risultato reale
    df["real_score"] = df["home_ft"].astype(str) + "-" + df["away_ft"].astype(str)

    return df


def _add_score_hits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
    - hit_score1..4
    - hit_any_score_top4
    """
    df = df.copy()

    for i in range(1, 5):
        col = f"score{i}"
        hit_col = f"hit_score{i}"
        if col in df.columns:
            df[hit_col] = (df["real_score"] == df[col]).astype(int)
        else:
            df[hit_col] = 0

    df["hit_any_score_top4"] = (
        df["hit_score1"] |
        df["hit_score2"] |
        df["hit_score3"] |
        df["hit_score4"]
    ).astype(int)

    return df


def _add_score_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
    - score_top4_fav_0  : quanti scoreline top4 con favorita a 0 gol
    - score_top4_fav_1_4: quanti scoreline top4 con favorita 1-4 gol
    - score_top4_fav_5p : quanti scoreline top4 con favorita >=5 gol
    """
    df = df.copy()

    def _count_patterns(row):
        fav_side = row["fav_side"]
        c0 = c14 = c5 = 0

        for i in range(1, 5):
            s = row.get(f"score{i}")
            if not isinstance(s, str) or "-" not in s:
                continue
            try:
                gh_str, ga_str = s.split("-")
                gh = int(gh_str)
                ga = int(ga_str)
            except Exception:
                continue

            fav_g = gh if fav_side == "home" else ga

            if fav_g == 0:
                c0 += 1
            elif 1 <= fav_g <= 4:
                c14 += 1
            elif fav_g >= 5:
                c5 += 1

        return pd.Series({
            "score_top4_fav_0": c0,
            "score_top4_fav_1_4": c14,
            "score_top4_fav_5p": c5,
        })

    df[["score_top4_fav_0", "score_top4_fav_1_4", "score_top4_fav_5p"]] = df.apply(
        _count_patterns, axis=1
    )

    return df


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge binning per:
    - fav_odds
    - mg_fav_1_3 / 1_4 / 1_5
    - differenze mg_15 - mg_14
    - numero di scoreline top4 con favorita 1-4 gol
    """
    df = df.copy()

    # Bins quota favorita (puoi cambiare i cut se vuoi più finezza)
    df["fav_odds_bin"] = pd.cut(
        df["fav_odds"],
        bins=[1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.20],
        include_lowest=True
    )

    # Bins MG 1-3 / 1-4 / 1-5
    df["mg13_bin"] = pd.cut(
        df["mg_fav_1_3"],
        bins=[0.0, 0.60, 0.70, 0.80, 0.90, 1.01],
        include_lowest=True
    )
    df["mg14_bin"] = pd.cut(
        df["mg_fav_1_4"],
        bins=[0.0, 0.60, 0.70, 0.80, 0.90, 1.01],
        include_lowest=True
    )
    df["mg15_bin"] = pd.cut(
        df["mg_fav_1_5"],
        bins=[0.0, 0.60, 0.70, 0.80, 0.90, 1.01],
        include_lowest=True
    )

    # Differenza MG 1-5 - 1-4 (misura rischio goleada)
    df["mg_15_minus_14"] = df["mg_fav_1_5"] - df["mg_fav_1_4"]
    df["mg_15_minus_13"] = df["mg_fav_1_5"] - df["mg_fav_1_3"]

    df["mg_diff14_bin"] = pd.cut(
        df["mg_15_minus_14"],
        bins=[-1.0, 0.02, 0.05, 0.10, 1.0],
        include_lowest=True
    )
    df["mg_diff13_bin"] = pd.cut(
        df["mg_15_minus_13"],
        bins=[-1.0, 0.02, 0.05, 0.10, 1.0],
        include_lowest=True
    )

    # Binning su quanti scoreline top4 hanno la favorita in 1-4 gol
    df["score1_4_bin"] = pd.cut(
        df["score_top4_fav_1_4"],
        bins=[-0.1, 1, 2, 3, 4],
        labels=["<=1", "2", "3", "4"]
    )

    return df


# ============================================================
# ANALISI GLOBALI
# ============================================================

def _global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche globali su:
    - P(MG 1-3, 1-4, 1-5 reali)
    - media gol favorita
    """
    stats = {
        "n_matches": len(df),
        "p_mg13_real": df["is_mg13_real"].mean(),
        "p_mg14_real": df["is_mg14_real"].mean(),
        "p_mg15_real": df["is_mg15_real"].mean(),
        "fav_goals_mean": df["fav_goals"].mean(),
        "fav_zero_gol": (df["fav_goals"] == 0).mean(),
    }

    print("\n=== STATISTICHE GLOBALI ===")
    print(f"Totale match       : {stats['n_matches']}")
    print(f"P(MG 1-3 reale)    : {stats['p_mg13_real']*100:.2f}%")
    print(f"P(MG 1-4 reale)    : {stats['p_mg14_real']*100:.2f}%")
    print(f"P(MG 1-5 reale)    : {stats['p_mg15_real']*100:.2f}%")
    print(f"Media gol favorita : {stats['fav_goals_mean']:.3f}")
    print(f"P(favorita 0 gol)  : {stats['fav_zero_gol']*100:.2f}%")

    return pd.DataFrame([stats])


def _correlation_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice di correlazione su un set di variabili chiave.
    """
    cols = [
        "fav_odds",
        "mg_fav_1_3",
        "mg_fav_1_4",
        "mg_fav_1_5",
        "fav_goals",
        "is_mg13_real",
        "is_mg14_real",
        "is_mg15_real",
        "hit_score1",
        "hit_score2",
        "hit_score3",
        "hit_score4",
        "hit_any_score_top4",
        "score_top4_fav_0",
        "score_top4_fav_1_4",
        "score_top4_fav_5p",
    ]

    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()
    print("\n=== CORRELAZIONI PRINCIPALI ===")
    print(corr.round(3))

    return corr


def _stats_by_fav_odds_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche aggregate per bande di quota favorita.
    """
    gb = df.groupby("fav_odds_bin").agg(
        n_matches=("is_mg14_real", "count"),
        p_mg13_real=("is_mg13_real", "mean"),
        p_mg14_real=("is_mg14_real", "mean"),
        p_mg15_real=("is_mg15_real", "mean"),
        fav_zero_gol=("fav_goals", lambda x: (x == 0).mean()),
        fav_goals_mean=("fav_goals", "mean"),
        hit_score1=("hit_score1", "mean"),
        hit_top4=("hit_any_score_top4", "mean"),
    ).reset_index()

    print("\n=== PER FASCIA DI QUOTA FAVORITA ===")
    print(gb.assign(
        p_mg13_real=lambda d: (d["p_mg13_real"] * 100).round(2),
        p_mg14_real=lambda d: (d["p_mg14_real"] * 100).round(2),
        p_mg15_real=lambda d: (d["p_mg15_real"] * 100).round(2),
        fav_zero_gol=lambda d: (d["fav_zero_gol"] * 100).round(2),
        hit_score1=lambda d: (d["hit_score1"] * 100).round(2),
        hit_top4=lambda d: (d["hit_top4"] * 100).round(2),
    ))

    return gb


# ============================================================
# RICERCA SEGMENTI "FORTI" PER MG 1-3, 1-4, 1-5
# ============================================================

def _search_best_segments(
    df: pd.DataFrame,
    target_col: str,
    min_samples: int = 200,
    min_prob: float = 0.86,
) -> pd.DataFrame:
    """
    Cerca segmenti per il target (is_mg13_real / is_mg14_real / is_mg15_real)
    combinando:
      - fav_odds_bin
      - mg13_bin / mg14_bin / mg15_bin
      - mg_diff13_bin / mg_diff14_bin
      - score1_4_bin

    Restituisce i segmenti con:
      - almeno min_samples partite
      - probabilità target >= min_prob
    """
    if target_col not in ("is_mg13_real", "is_mg14_real", "is_mg15_real"):
        raise ValueError("target_col deve essere uno tra is_mg13_real, is_mg14_real, is_mg15_real")

    if target_col == "is_mg13_real":
        mg_bin_col = "mg13_bin"
        diff_bin_col = "mg_diff13_bin"
    elif target_col == "is_mg14_real":
        mg_bin_col = "mg14_bin"
        diff_bin_col = "mg_diff14_bin"
    else:
        mg_bin_col = "mg15_bin"
        diff_bin_col = "mg_diff14_bin"  # per MG 1-5 ha senso guardare 5-4

    group_cols = ["fav_odds_bin", mg_bin_col, diff_bin_col, "score1_4_bin"]

    g = (
        df
        .groupby(group_cols, observed=False)[target_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_matches", "mean": "p_target"})
    )

    mask = (g["n_matches"] >= min_samples) & (g["p_target"] >= min_prob)
    best = g[mask].sort_values(by="p_target", ascending=False)

    print(f"\n=== SEGMENTI FORTI PER {target_col} (p >= {min_prob:.2f}, N >= {min_samples}) ===")
    if best.empty:
        print("Nessun segmento soddisfa i criteri. Prova ad abbassare min_prob o min_samples.")
    else:
        print(best.to_string(index=False))

    return best


# ============================================================
# ENTRYPOINT PRINCIPALE
# ============================================================

def run_deep_parquet_study(
    parquet_path: str | Path,
    min_samples: int = 200,
    mg13_min_prob: float = 0.80,
    mg14_min_prob: float = 0.86,
    mg15_min_prob: float = 0.90,
) -> Dict[str, Any]:
    """
    Esegue uno studio approfondito sul parquet:

    - Legge il file
    - Aggiunge tutte le feature derivate
    - Stampa statistiche globali, correlazioni, analisi per quota
    - Cerca segmenti forti per MG 1-3, 1-4, 1-5

    Ritorna un dict con:
      - "df"               : il DataFrame arricchito
      - "global_stats"     : DataFrame 1xN con stat globali
      - "corr"             : matrice di correlazione
      - "by_fav_odds_bin"  : aggregazioni per fascia di quota
      - "best_mg13"        : segmenti forti per MG 1-3
      - "best_mg14"        : segmenti forti per MG 1-4
      - "best_mg15"        : segmenti forti per MG 1-5
    """
    parquet_path = Path(parquet_path)
    print(f"📥 Carico parquet da: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Righe lette: {len(df)}")

    # Feature engineering
    df = _add_base_features(df)
    df = _add_score_hits(df)
    df = _add_score_pattern_features(df)
    df = _add_bins(df)

    # Analisi principali
    global_stats = _global_stats(df)
    corr = _correlation_block(df)
    by_fav_bin = _stats_by_fav_odds_bin(df)

    # Segmenti forti per MG 1-3 / 1-4 / 1-5
    best_mg13 = _search_best_segments(
        df, target_col="is_mg13_real", min_samples=min_samples, min_prob=mg13_min_prob
    )
    best_mg14 = _search_best_segments(
        df, target_col="is_mg14_real", min_samples=min_samples, min_prob=mg14_min_prob
    )
    best_mg15 = _search_best_segments(
        df, target_col="is_mg15_real", min_samples=min_samples, min_prob=mg15_min_prob
    )

    return {
        "df": df,
        "global_stats": global_stats,
        "corr": corr,
        "by_fav_odds_bin": by_fav_bin,
        "best_mg13": best_mg13,
        "best_mg14": best_mg14,
        "best_mg15": best_mg15,
    }


if __name__ == "__main__":
    # esempio d'uso “standalone”
    REPO_ROOT = Path(__file__).resolve().parents[4]
    PARQUET = REPO_ROOT / "app" / "ml" / "backtests" / "data" / "matches_fav_le_195.parquet"

    results = run_deep_parquet_study(PARQUET)