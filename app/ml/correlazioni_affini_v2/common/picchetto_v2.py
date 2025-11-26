# app/ml/correlazioni_affini_v2/common/picchetto_v2.py

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# UTILITY
# ============================================================

def softmax3(u1: float, uX: float, u2: float) -> Tuple[float, float, float]:
    """
    Softmax a 3 classi, numericamente stabile.
    """
    m = max(u1, uX, u2)
    e1 = math.exp(u1 - m)
    eX = math.exp(uX - m)
    e2 = math.exp(u2 - m)
    s = e1 + eX + e2
    return e1 / s, eX / s, e2 / s


def poisson_pmf(lmbd: float, k: int) -> float:
    """
    Poisson PMF: P(K=k) con parametro lambda.
    """
    if lmbd <= 0:
        return 0.0
    return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)


def poisson_1x2_prob(lambda_home: float, lambda_away: float, max_goals: int = 10) -> Tuple[float, float, float]:
    """
    Calcola P(1), P(X), P(2) assumendo:
    - goal_home ~ Poisson(lambda_home)
    - goal_away ~ Poisson(lambda_away)
    indipendenti (modello Dixon–Coles semplificato).

    max_goals: taglio massimo per la somma (10 è già sufficiente).
    """
    if lambda_home <= 0 or lambda_away <= 0:
        return np.nan, np.nan, np.nan

    p1 = 0.0
    px = 0.0
    p2 = 0.0

    # Precompute pmf per efficienza
    pmf_home = [poisson_pmf(lambda_home, i) for i in range(max_goals + 1)]
    pmf_away = [poisson_pmf(lambda_away, j) for j in range(max_goals + 1)]

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_home[i] * pmf_away[j]
            if i > j:
                p1 += p
            elif i == j:
                px += p
            else:
                p2 += p

    # Normalizza per assorbire la massa tagliata fuori
    s = p1 + px + p2
    if s <= 0:
        return np.nan, np.nan, np.nan

    return p1 / s, px / s, p2 / s


# ============================================================
# COLONNE DI DIFF PER IL MARGIN ELO/FORM
# ============================================================

DIFF_COLS_V2 = [
    "elo_diff_raw",
    "form_pts_diff_raw",
    "form_gf_diff_raw",
    "form_ga_diff_raw",
]


def fit_picchetto_v2_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calcola mean/std per le colonne DIFF_COLS_V2.
    Puoi chiamarlo su tutto il dataset storico e passare `stats` a apply_picchetto_tech_v2
    per avere una normalizzazione stabile anche in runtime.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in DIFF_COLS_V2:
        if col not in df.columns:
            raise RuntimeError(f"fit_picchetto_v2_stats: manca colonna {col}")
        m = float(df[col].mean(skipna=True))
        s = float(df[col].std(skipna=True))
        if not np.isfinite(s) or s == 0:
            s = 1.0
        stats[col] = {"mean": m, "std": s}
    return stats


# ============================================================
# PICCHETTO TECNICO V2 (1X2)
# ============================================================

def _ensure_raw_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea le colonne *_raw se mancano, usando i campi di step1c:
    - elo_diff_raw
    - form_pts_diff_raw
    - form_gf_diff_raw
    - form_ga_diff_raw
    """
    df = df.copy()

    if "elo_diff_raw" not in df.columns:
        if "elo_home_pre" in df.columns and "elo_away_pre" in df.columns:
            df["elo_diff_raw"] = df["elo_home_pre"] - df["elo_away_pre"]
        elif "elo_diff" in df.columns:
            df["elo_diff_raw"] = df["elo_diff"]
        else:
            raise RuntimeError("picchetto_v2: mancano colonne per calcolare elo_diff_raw")

    if "form_pts_diff_raw" not in df.columns:
        df["form_pts_diff_raw"] = df["home_form_pts_avg_lastN"] - df["away_form_pts_avg_lastN"]

    if "form_gf_diff_raw" not in df.columns:
        df["form_gf_diff_raw"] = df["home_form_gf_avg_lastN"] - df["away_form_gf_avg_lastN"]

    if "form_ga_diff_raw" not in df.columns:
        df["form_ga_diff_raw"] = df["away_form_ga_avg_lastN"] - df["home_form_ga_avg_lastN"]

    return df


def _compute_margin_z(df: pd.DataFrame,
                      stats: Optional[Dict[str, Dict[str, float]]] = None,
                      w_elo: float = 0.5,
                      w_pts: float = 0.3,
                      w_gf: float = 0.1,
                      w_ga: float = 0.1) -> pd.DataFrame:
    """
    Normalizza le diff (z-score) e costruisce un margin combinato.
    """
    df = df.copy()

    # Normalizzazione z-score
    for col in DIFF_COLS_V2:
        if stats is not None:
            m = stats[col]["mean"]
            s = stats[col]["std"]
        else:
            m = float(df[col].mean(skipna=True))
            s = float(df[col].std(skipna=True))
            if not np.isfinite(s) or s == 0:
                s = 1.0
        df[col + "_z"] = (df[col] - m) / s

    df["margin_raw_v2"] = (
        w_elo * df["elo_diff_raw_z"] +
        w_pts * df["form_pts_diff_raw_z"] +
        w_gf * df["form_gf_diff_raw_z"] +
        w_ga * df["form_ga_diff_raw_z"]
    )

    df["margin_clipped_v2"] = df["margin_raw_v2"].clip(-3, 3)
    return df


def _compute_lambdas_row(row: pd.Series,
                         beta_form: float = 0.4) -> Tuple[float, float]:
    """
    Costruisce lambda_home e lambda_away combinando:
    - lambda_home_form / lambda_away_form
    - lambda_home_market_ou25 / lambda_away_market_ou25
    - fallback su lambda_total_* se mancano i dettagli per squadra
    """
    # --- Form ---
    lam_h_form = row.get("lambda_home_form", np.nan)
    lam_a_form = row.get("lambda_away_form", np.nan)
    lam_tot_form = row.get("lambda_total_form", np.nan)

    if not np.isfinite(lam_h_form) or lam_h_form <= 0:
        # fallback: split del totale usando elo_prob_home
        lam_tot = lam_tot_form if np.isfinite(lam_tot_form) and lam_tot_form > 0 else 2.6  # media mondiale
        elo_ph = row.get("elo_prob_home", 0.5)
        elo_ph = float(min(max(elo_ph, 0.2), 0.8))
        lam_h_form = lam_tot * elo_ph
        lam_a_form = lam_tot * (1 - elo_ph)

    if not np.isfinite(lam_a_form) or lam_a_form <= 0:
        # se manca solo away, usa simmetria
        lam_a_form = max(0.1, lam_h_form * 0.7)

    # --- Market ---
    lam_h_mkt = row.get("lambda_home_market_ou25", np.nan)
    lam_a_mkt = row.get("lambda_away_market_ou25", np.nan)
    lam_tot_mkt = row.get("lambda_total_market_ou25", np.nan)

    if (not np.isfinite(lam_h_mkt) or lam_h_mkt <= 0) or (not np.isfinite(lam_a_mkt) or lam_a_mkt <= 0):
        if np.isfinite(lam_tot_mkt) and lam_tot_mkt > 0:
            # split del market total usando le probabilità 1X2 di mercato
            bk_p1 = row.get("bk_p1", np.nan)
            bk_p2 = row.get("bk_p2", np.nan)
            if np.isfinite(bk_p1) and np.isfinite(bk_p2):
                # approssima share offensivo in base a p1/p2
                w_h = float(min(max(bk_p1 / (bk_p1 + bk_p2 + 1e-9), 0.2), 0.8))
            else:
                w_h = 0.5
            lam_h_mkt = lam_tot_mkt * w_h
            lam_a_mkt = lam_tot_mkt * (1 - w_h)
        else:
            # fallback totale: usa form
            lam_h_mkt = lam_h_form
            lam_a_mkt = lam_a_form

    # --- Blend finale ---
    # beta_form: peso su λ_form, (1 - beta_form) su λ_market
    lam_h = beta_form * lam_h_form + (1.0 - beta_form) * lam_h_mkt
    lam_a = beta_form * lam_a_form + (1.0 - beta_form) * lam_a_mkt

    # safety clip
    lam_h = float(min(max(lam_h, 0.1), 5.0))
    lam_a = float(min(max(lam_a, 0.1), 5.0))

    return lam_h, lam_a


def apply_picchetto_tech_v2(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    # pesi di blending fra le 3 sorgenti: Elo, Poisson, Book
    w_elo_source: float = 0.4,
    w_poisson_source: float = 0.3,
    w_book_source: float = 0.3,
    # pesi interni per il margin Elo/Form
    w_elo_diff: float = 0.5,
    w_pts_diff: float = 0.3,
    w_gf_diff: float = 0.1,
    w_ga_diff: float = 0.1,
    # blending fra λ form e λ market
    beta_form_lambda: float = 0.4,
    max_poisson_goals: int = 10,
) -> pd.DataFrame:
    """
    Picchetto tecnico v2 per il mercato 1X2.

    Combina tre fonti di probabilità:
    - Elo/Form (margin + softmax)           -> sorgente "Elo"
    - Poisson λ_home/λ_away (forma/market)  -> sorgente "Poisson"
    - Probabilità di mercato (bk_p1/px/p2)  -> sorgente "Book"

    Parametri principali:
    - w_elo_source, w_poisson_source, w_book_source: pesi di blending tra le tre sorgenti
    - w_elo_diff, w_pts_diff, w_gf_diff, w_ga_diff: pesi per costruire il margin da diff_z
    - beta_form_lambda: blend tra λ di forma e λ di mercato (0=form only, 1=market only)
    """
    df = df.copy()

    # ----------------------------------------------------
    # 1) Pre-condizioni minime
    # ----------------------------------------------------
    required_cols = [
        "bk_p1", "bk_px", "bk_p2",
        "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
        "home_form_gf_avg_lastN", "away_form_gf_avg_lastN",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"apply_picchetto_tech_v2: manca colonna {c}")

    # Creiamo le diff raw se mancano
    df = _ensure_raw_diffs(df)

    # Costruiamo il margin z-based
    df = _compute_margin_z(
        df,
        stats=stats,
        w_elo=w_elo_diff,
        w_pts=w_pts_diff,
        w_gf=w_gf_diff,
        w_ga=w_ga_diff,
    )

    # ----------------------------------------------------
    # 2) Sorgente Elo/Form: softmax sul margin
    # ----------------------------------------------------
    df["elo_p1"] = np.nan
    df["elo_px"] = np.nan
    df["elo_p2"] = np.nan

    for i in df.index:
        m = float(df.at[i, "margin_clipped_v2"])
        u1 = m
        u2 = -m
        uX = -0.5 * abs(m)
        p1_e, pX_e, p2_e = softmax3(u1, uX, u2)
        df.at[i, "elo_p1"] = p1_e
        df.at[i, "elo_px"] = pX_e
        df.at[i, "elo_p2"] = p2_e

    # ----------------------------------------------------
    # 3) Sorgente Poisson: λ_home, λ_away -> P(1,X,2)
    # ----------------------------------------------------
    df["pois_p1"] = np.nan
    df["pois_px"] = np.nan
    df["pois_p2"] = np.nan

    for i in df.index:
        row = df.loc[i]
        lam_h, lam_a = _compute_lambdas_row(row, beta_form=beta_form_lambda)
        p1_p, pX_p, p2_p = poisson_1x2_prob(lam_h, lam_a, max_goals=max_poisson_goals)
        df.at[i, "pois_p1"] = p1_p
        df.at[i, "pois_px"] = pX_p
        df.at[i, "pois_p2"] = p2_p

    # ----------------------------------------------------
    # 4) Sorgente Book: bk_p1, bk_px, bk_p2 già overround corrected
    # ----------------------------------------------------
    # (assumo che bk_p* siano già le probabilità "fair" / normalizzate – come da step0/1)
    # Se non lo fossero, puoi sempre rinormalizzarle qui:
    s_bk = df["bk_p1"] + df["bk_px"] + df["bk_p2"]
    df["bk_p1_norm"] = df["bk_p1"] / s_bk
    df["bk_px_norm"] = df["bk_px"] / s_bk
    df["bk_p2_norm"] = df["bk_p2"] / s_bk

    # ----------------------------------------------------
    # 5) Blending delle 3 sorgenti per ogni match
    # ----------------------------------------------------
    df["pic_v2_p1"] = np.nan
    df["pic_v2_px"] = np.nan
    df["pic_v2_p2"] = np.nan

    for i in df.index:
        row = df.loc[i]

        sources = []

        # Elo source
        if np.isfinite(row["elo_p1"]) and np.isfinite(row["elo_px"]) and np.isfinite(row["elo_p2"]):
            sources.append(("elo", w_elo_source, row["elo_p1"], row["elo_px"], row["elo_p2"]))

        # Poisson source
        if np.isfinite(row["pois_p1"]) and np.isfinite(row["pois_px"]) and np.isfinite(row["pois_p2"]):
            sources.append(("pois", w_poisson_source, row["pois_p1"], row["pois_px"], row["pois_p2"]))

        # Book source
        if np.isfinite(row["bk_p1_norm"]) and np.isfinite(row["bk_px_norm"]) and np.isfinite(row["bk_p2_norm"]):
            sources.append(("book", w_book_source, row["bk_p1_norm"], row["bk_px_norm"], row["bk_p2_norm"]))

        if not sources:
            # fallback: se per qualche motivo non ho nessuna sorgente, salta
            continue

        w_sum = sum(w for _, w, _, _, _ in sources)
        if w_sum <= 0:
            continue

        p1 = 0.0
        pX = 0.0
        p2 = 0.0
        for _, w, s1, sX, s2 in sources:
            alpha = w / w_sum
            p1 += alpha * s1
            pX += alpha * sX
            p2 += alpha * s2

        # norma finale per assurdità numeriche
        s = p1 + pX + p2
        if s > 0:
            p1 /= s
            pX /= s
            p2 /= s

        df.at[i, "pic_v2_p1"] = p1
        df.at[i, "pic_v2_px"] = pX
        df.at[i, "pic_v2_p2"] = p2

    # ----------------------------------------------------
    # 6) Quote corrispondenti
    # ----------------------------------------------------
    df["pic_v2_odd1"] = 1 / df["pic_v2_p1"]
    df["pic_v2_oddX"] = 1 / df["pic_v2_px"]
    df["pic_v2_odd2"] = 1 / df["pic_v2_p2"]

    return df