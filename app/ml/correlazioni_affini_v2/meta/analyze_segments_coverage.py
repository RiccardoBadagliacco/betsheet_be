#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep study sulle relazioni tra:
- quota favorita
- MG Favorita 1-3 / 1-4 / 1-5 (mg_fav_1_3 / mg_fav_1_4 / mg_fav_1_5)
- risultati reali (gol favorita)
- pattern degli scoreline top4

Obiettivi principali:
- Capire come quota, mg_fav_* e pattern score interagiscono
- Trovare segmenti "forti" per MG 1-4 e MG 1-5
- Derivare soglie di probabilità (mg_fav_1_4 / mg_fav_1_5) ottimali
- Estrarre pseudo-regole in forma leggibile, utili per il motore di alert

Novità di questa versione:
- Binning della quota favorita con step di 0.05 su fav_odds_bin
- Per ogni segmento forte trovato, viene riportata anche:
    - copertura rispetto all'intero dataset
    - copertura rispetto alla sola fascia di quota (fav_odds_bin)

Uso principale:
    from analyze import run_deep_parquet_study
    results = run_deep_parquet_study("matches_fav_le_195.parquet")

Run standalone:
    python analyze.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# FEATURE ENGINEERING DI BASE
# ============================================================

def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
    - fav_goals: gol reali della favorita (home o away in base a fav_side)
    - is_mg13_real, is_mg14_real, is_mg15_real
    - real_score: stringa "home_ft-away_ft"
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
    - hit_score1..4: 1 se il relativo score_i è stato centrato
    - hit_any_score_top4: 1 se almeno uno dei primi 4 score è corretto
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
    - score_top4_fav_0  : quanti scoreline top4 hanno favorita a 0 gol
    - score_top4_fav_1_4: quanti scoreline top4 hanno favorita a 1-4 gol
    - score_top4_fav_5p : quanti scoreline top4 hanno favorita a >=5 gol
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
    - fav_odds (step 0.05)
    - mg_fav_1_3 / 1_4 / 1_5
    - differenze mg_15 - mg_14 e mg_15 - mg_13
    - numero di scoreline top4 con favorita 1-4 gol
    """
    df = df.copy()

    # Bins quota favorita con step 0.05 (es. 1.00-1.05, 1.05-1.10, ..., 2.15-2.20)
    # Adattato al range tipico dello studio (fav_odds <= 1.95), ma leggermente più largo.
    min_odds = 1.00
    max_odds = 2.20
    fav_bins = np.arange(min_odds, max_odds + 0.001, 0.05)

    df["fav_odds_bin"] = pd.cut(
        df["fav_odds"],
        bins=fav_bins,
        include_lowest=True
    )

    # Bins MG 1-3 / 1-4 / 1-5
    bins_mg = [0.0, 0.60, 0.70, 0.80, 0.90, 1.01]
    df["mg13_bin"] = pd.cut(df["mg_fav_1_3"], bins=bins_mg, include_lowest=True)
    df["mg14_bin"] = pd.cut(df["mg_fav_1_4"], bins=bins_mg, include_lowest=True)
    df["mg15_bin"] = pd.cut(df["mg_fav_1_5"], bins=bins_mg, include_lowest=True)

    # Differenze MG 1-5 - 1-4 / 1-5 - 1-3
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


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature "interpretative" aggiuntive:
    - goleada_risk: mg_15_minus_14
    - is_perfect_pattern: (score_top4_fav_0 == 0 & score_top4_fav_1_4 == 4)
    - fav_strong_bin: indica se la quota è in zona "premium" (1.2-1.6)
    """
    df = df.copy()

    df["goleada_risk"] = df["mg_15_minus_14"]

    df["is_perfect_pattern"] = (
        (df["score_top4_fav_0"] == 0) &
        (df["score_top4_fav_1_4"] == 4)
    ).astype(int)

    # Bandiera di "quota forte" nel range tipico delle regole premium
    df["is_fav_odds_12_16"] = df["fav_odds"].between(1.20, 1.60).astype(int)

    return df


# ============================================================
# ANALISI GLOBALI
# ============================================================

def _global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche globali su:
    - P(MG 1-3, 1-4, 1-5 reali)
    - media gol favorita
    - P(favorita 0 gol)
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
        "mg_15_minus_14",
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
        "is_perfect_pattern",
        "goleada_risk",
    ]

    cols = [c for c in cols if c in df.columns]

    corr = df[cols].corr()
    print("\n=== CORRELAZIONI PRINCIPALI ===")
    print(corr.round(3))

    return corr


def _stats_by_fav_odds_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche aggregate per bande di quota favorita (fav_odds_bin a step 0.05).
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

    print("\n=== PER FASCIA DI QUOTA FAVORITA (step 0.05) ===")
    print(
        gb.assign(
            p_mg13_real=lambda d: (d["p_mg13_real"] * 100).round(2),
            p_mg14_real=lambda d: (d["p_mg14_real"] * 100).round(2),
            p_mg15_real=lambda d: (d["p_mg15_real"] * 100).round(2),
            fav_zero_gol=lambda d: (d["fav_zero_gol"] * 100).round(2),
            hit_score1=lambda d: (d["hit_score1"] * 100).round(2),
            hit_top4=lambda d: (d["hit_top4"] * 100).round(2),
        )
    )

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
      - fav_odds_bin (step 0.05)
      - mg13_bin / mg14_bin / mg15_bin
      - mg_diff13_bin / mg_diff14_bin
      - score1_4_bin

    Restituisce i segmenti con:
      - almeno min_samples partite
      - probabilità target >= min_prob

    In più aggiunge:
      - n_bin          : numero totale di match nella fascia fav_odds_bin
      - coverage_total : n_matches / len(df)
      - coverage_in_bin: n_matches / n_bin
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
        # per MG 1-5 è sensato usare la diff 5-4
        diff_bin_col = "mg_diff14_bin"

    group_cols = ["fav_odds_bin", mg_bin_col, diff_bin_col, "score1_4_bin"]

    # Aggregazione principale
    g = (
        df
        .groupby(group_cols, observed=False)[target_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_matches", "mean": "p_target"})
    )

    # Conteggi per fascia di quota
    bin_counts = (
        df
        .groupby("fav_odds_bin", observed=False)
        .size()
        .reset_index(name="n_bin")
    )

    # Merge per avere n_bin
    g = g.merge(bin_counts, on="fav_odds_bin", how="left")

    total_n = len(df)

    # Filtra segmenti forti
    mask = (g["n_matches"] >= min_samples) & (g["p_target"] >= min_prob)
    best = g[mask].copy()

    if not best.empty:
        best["coverage_total"] = best["n_matches"] / total_n
        best["coverage_in_bin"] = best["n_matches"] / best["n_bin"]

    best = best.sort_values(by="p_target", ascending=False)

    print(f"\n=== SEGMENTI FORTI PER {target_col} (p >= {min_prob:.2f}, N >= {min_samples}) ===")
    if best.empty:
        print("Nessun segmento soddisfa i criteri. Prova ad abbassare min_prob o min_samples.")
    else:
        # Stampa tabellare compatta (senza coperture, che vengono dettagliate dopo)
        print(
            best[
                [
                    "fav_odds_bin",
                    mg_bin_col,
                    diff_bin_col,
                    "score1_4_bin",
                    "n_matches",
                    "n_bin",
                    "p_target",
                    "coverage_total",
                    "coverage_in_bin",
                ]
            ].to_string(index=False)
        )

    return best


def _confidence_interval(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Intervallo di confidenza approssimato (normal approx) per una proporzione.
    """
    if n == 0:
        return (np.nan, np.nan)
    se = np.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0.0
    return (p - z * se, p + z * se)


def _print_segment_rules(best: pd.DataFrame, label: str) -> None:
    """
    Genera una versione "regola leggibile" dei segmenti trovati,
    includendo la copertura:
      - rispetto all'intero dataset
      - rispetto alla sola fascia di quota (fav_odds_bin)
    """
    if best.empty:
        return

    print(f"\n=== PSEUDO-REGOLE DERIVATE PER {label} ===")
    for idx, row in best.iterrows():
        fav_odds_bin = row["fav_odds_bin"]
        # trova la colonna mg_bin e diff_bin dinamicamente
        mg_bin_col = [c for c in row.index if "mg" in str(c) and "bin" in str(c) and "diff" not in str(c)]
        diff_bin_col = [c for c in row.index if "diff" in str(c) and "bin" in str(c)]

        mg_bin = row[mg_bin_col[0]] if mg_bin_col else None
        diff_bin = row[diff_bin_col[0]] if diff_bin_col else None

        score_bin = row["score1_4_bin"]
        n = int(row["n_matches"])
        p = float(row["p_target"])
        n_bin = int(row.get("n_bin", np.nan))
        cov_tot = float(row.get("coverage_total", np.nan))
        cov_bin = float(row.get("coverage_in_bin", np.nan))
        lo, hi = _confidence_interval(p, n)

        print("\n---------------------------------------------")
        print(f"SE fav_odds in {fav_odds_bin}")
        print(f" E prob_MG bin {mg_bin}")
        print(f" E diff_MG bin {diff_bin}")
        print(f" E score_top4_fav_1_4_bin = {score_bin}")
        print(
            f"ALLORA P(target) ≈ {p*100:.2f}% "
            f"(N={n}, CI95% ≈ [{lo*100:.1f}%, {hi*100:.1f}%])"
        )
        if not np.isnan(cov_tot):
            print(f"   Copertura su TUTTO il dataset : {cov_tot*100:.2f}%")
        if not (np.isnan(cov_bin) or np.isnan(n_bin)):
            print(
                f"   Copertura nella FASCIA QUOTA  : {cov_bin*100:.2f}% "
                f"(N_fascia={n_bin})"
            )


# ============================================================
# SOGLIE OTTIMALI SU MG_FAV_* (tipo ROC semplificata)
# ============================================================

def _find_best_threshold(
    df: pd.DataFrame,
    prob_col: str,
    target_col: str,
    min_support: int = 300,
) -> Optional[Dict[str, Any]]:
    """
    Cerca una soglia sul prob_col (es. mg_fav_1_4) che separi il meglio possibile
    successi/insuccessi del target (es. is_mg14_real), usando una logica tipo
    massimizzazione di Youden's J (TPR - FPR).

    Ritorna:
        {
          "threshold": float,
          "tpr": float,
          "fpr": float,
          "support": int
        }
    oppure None se dati insufficienti.
    """
    df = df[[prob_col, target_col]].dropna().copy()
    if df.empty:
        return None

    df = df.sort_values(prob_col)
    y = df[target_col].values
    p = df[prob_col].values

    # Possibili soglie: i valori unici della prob stimata
    uniq = np.unique(p)
    if len(uniq) < 5:
        return None

    best = None
    best_j = -1.0

    positives = y.sum()
    negatives = len(y) - positives

    if positives == 0 or negatives == 0:
        return None

    # TPR/FPR al crescere della soglia
    for thr in uniq:
        mask = p >= thr
        support = int(mask.sum())
        if support < min_support:
            continue

        y_hat = mask.astype(int)
        tp = int(((y_hat == 1) & (y == 1)).sum())
        fp = int(((y_hat == 1) & (y == 0)).sum())
        # fn e tn non servono esplicitamente per J

        tpr = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        j = tpr - fpr

        if j > best_j:
            best_j = j
            best = {
                "threshold": float(thr),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "support": support,
            }

    return best


def _print_best_thresholds(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcola e stampa le migliori soglie per:
    - mg_fav_1_4 → is_mg14_real
    - mg_fav_1_5 → is_mg15_real
    """
    results: Dict[str, Any] = {}

    print("\n=== SOGLIE OTTIMALI SU MG_FAV_* (tipo ROC semplificata) ===")

    for prob_col, target_col, label in [
        ("mg_fav_1_4", "is_mg14_real", "MG 1–4"),
        ("mg_fav_1_5", "is_mg15_real", "MG 1–5"),
    ]:
        if prob_col not in df.columns or target_col not in df.columns:
            continue

        best = _find_best_threshold(df, prob_col, target_col)
        results[label] = best

        if best is None:
            print(f"- Nessuna soglia robusta trovata per {label} ({prob_col} vs {target_col}).")
        else:
            thr = best["threshold"]
            tpr = best["tpr"]
            fpr = best["fpr"]
            support = best["support"]
            print(
                f"- {label}: soglia ottimale ≈ {thr:.3f} "
                f"(TPR={tpr*100:.1f}%, FPR={fpr*100:.1f}%, supporto N={support})"
            )

    return results


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
    - Trova soglie ottimali su mg_fav_1_4 / 1_5 rispetto agli esiti reali
    - Cerca segmenti forti per MG 1-3, 1-4, 1-5
    - Stampa pseudo-regole leggibili per MG 1-4 e 1-5, con coperture

    Ritorna un dict con:
      - "df"               : il DataFrame arricchito
      - "global_stats"     : DataFrame 1xN con stat globali
      - "corr"             : matrice di correlazione
      - "by_fav_odds_bin"  : aggregazioni per fascia di quota (step 0.05)
      - "best_mg13"        : segmenti forti per MG 1-3 (con coperture)
      - "best_mg14"        : segmenti forti per MG 1-4 (con coperture)
      - "best_mg15"        : segmenti forti per MG 1-5 (con coperture)
      - "best_thresholds"  : soglie ottimali su mg_fav_1_4 / 1_5
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
    df = _add_extra_features(df)

    # Analisi principali
    global_stats = _global_stats(df)
    corr = _correlation_block(df)
    by_fav_bin = _stats_by_fav_odds_bin(df)

    # Soglie ottimali su mg_fav_1_4 / 1_5
    best_thresholds = _print_best_thresholds(df)

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

    # Pseudo-regole leggibili per MG 1-4 e MG 1-5, con coperture
    _print_segment_rules(best_mg14, "MG 1–4")
    _print_segment_rules(best_mg15, "MG 1–5")

    return {
        "df": df,
        "global_stats": global_stats,
        "corr": corr,
        "by_fav_odds_bin": by_fav_bin,
        "best_mg13": best_mg13,
        "best_mg14": best_mg14,
        "best_mg15": best_mg15,
        "best_thresholds": best_thresholds,
    }


if __name__ == "__main__":
    # esempio d'uso “standalone”
    REPO_ROOT = Path(__file__).resolve().parents[4]
    PARQUET = REPO_ROOT / "app" / "ml" / "backtests" / "data" / "matches_fav_le_195.parquet"

    results = run_deep_parquet_study(PARQUET)