# app/ml/correlazioni_affini_v2/common/picchetto_fix.py

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# ============================================================
# UTILITY
# ============================================================

def softmax3(u1: float, uX: float, u2: float):
    m = max(u1, uX, u2)
    e1 = math.exp(u1 - m)
    eX = math.exp(uX - m)
    e2 = math.exp(u2 - m)
    s = e1 + eX + e2
    return e1 / s, eX / s, e2 / s


def poisson_over_under_half_goalline(lmbd: float, line: float):
    if lmbd is None or not np.isfinite(lmbd) or lmbd <= 0:
        return None, None
    threshold = int(math.floor(line))
    p0 = math.exp(-lmbd)
    p_under = p0
    pk = p0
    for k in range(1, threshold + 1):
        pk = pk * lmbd / k
        p_under += pk
    return 1 - p_under, p_under


# ============================================================
# DIFF COLONNE
# ============================================================

DIFF_COLS_FIX = [
    "elo_diff_raw",
    "form_pts_diff_raw",
    "form_gf_diff_raw",
    "form_ga_diff_raw",
]


# ============================================================
# FIT MEAN/STD
# ============================================================

def fit_picchetto_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in DIFF_COLS_FIX:
        if col not in df.columns:
            raise RuntimeError(f"fit_picchetto_stats: manca colonna {col}")
        m = float(df[col].mean(skipna=True))
        s = float(df[col].std(skipna=True))
        if s == 0 or not np.isfinite(s):
            s = 1.0
        stats[col] = {"mean": m, "std": s}
    return stats


# ============================================================
# PICCHETTO TECNICO FIXATO
# ============================================================

def apply_picchetto_tech_fix(
    df: pd.DataFrame,
    alpha: float = 0.5,          # per 1X2
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    alpha_ou: float = 0.3,       # NUOVO: blending OU vs book
    beta_ou_lambda: float = 0.3  # NUOVO: blending λ_form vs λ_market
) -> pd.DataFrame:

    df = df.copy()

    # ---------------------------
    # Check colonne minime
    # ---------------------------
    required = [
        "match_id",
        "bk_p1", "bk_px", "bk_p2",
        "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
        "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
    ]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"apply_picchetto_tech_fix: manca colonna {c}")

    # ======================================================
    # RAW DIFF — AUTOCALCOLATI SE MANCANTI (RUNTIME SAFE)
    # ======================================================
    if "elo_diff_raw" not in df.columns:
        df["elo_diff_raw"] = df["elo_home_pre"] - df["elo_away_pre"]

    if "form_pts_diff_raw" not in df.columns:
        df["form_pts_diff_raw"] = (
            df["home_form_pts_avg_lastN"] - df["away_form_pts_avg_lastN"]
        )

    if "form_gf_diff_raw" not in df.columns:
        df["form_gf_diff_raw"] = (
            df["home_form_gf_avg_lastN"] - df["away_form_gf_avg_lastN"]
        )

    if "form_ga_diff_raw" not in df.columns:
        df["form_ga_diff_raw"] = (
            df["away_form_ga_avg_lastN"] - df["home_form_ga_avg_lastN"]
        )

    # Ora abbiamo la certezza matematica che tutti i raw diff esistono
    # ======================================================
    # NORMALIZZAZIONE (Z-SCORE)
    # ======================================================
    for col in DIFF_COLS_FIX:
        if stats is not None:
            m = stats[col]["mean"]
            s = stats[col]["std"]
        else:
            m = float(df[col].mean(skipna=True))
            s = float(df[col].std(skipna=True))
            if s == 0 or not np.isfinite(s):
                s = 1.0
        df[col + "_z"] = (df[col] - m) / s

    # ======================================================
    # MARGIN
    # ======================================================
    w_elo = 0.5
    w_pts = 0.3
    w_gf  = 0.1
    w_ga  = 0.1

    df["margin_raw"] = (
        w_elo * df["elo_diff_raw_z"] +
        w_pts * df["form_pts_diff_raw_z"] +
        w_gf  * df["form_gf_diff_raw_z"] +
        w_ga  * df["form_ga_diff_raw_z"]
    )

    df["margin_clipped"] = df["margin_raw"].clip(-3, 3)

    # ======================================================
    # FALLBACK BOOKMAKER
    # ======================================================
    df["pic_p1"] = df["bk_p1"]
    df["pic_px"] = df["bk_px"]
    df["pic_p2"] = df["bk_p2"]

    df["pic_pO25"] = df.get("bk_pO25", np.nan)
    df["pic_pU25"] = df.get("bk_pU25", np.nan)
    df["pic_pO15"] = np.nan
    df["pic_pU15"] = np.nan
    df["pic_pO05"] = np.nan
    df["pic_pU05"] = np.nan

    # ======================================================
    # PICCHETTO TECNICO
    # ======================================================
    valid_mask = (
        df["home_form_pts_avg_lastN"].fillna(0) > 0
    ) & (
        df["away_form_pts_avg_lastN"].fillna(0) > 0
    )

    for i in df[valid_mask].index:
        row = df.loc[i]

        # --- 1X2 tecnico ---
        m = float(row["margin_clipped"])
        u1 = m
        u2 = -m
        uX = -0.5 * abs(m)
        p1_f, pX_f, p2_f = softmax3(u1, uX, u2)

        if pd.isna(row["bk_p1"]) or pd.isna(row["bk_px"]) or pd.isna(row["bk_p2"]):
            p1, pX, p2 = p1_f, pX_f, p2_f
        else:
            p1 = alpha * p1_f + (1 - alpha) * float(row["bk_p1"])
            pX = alpha * pX_f + (1 - alpha) * float(row["bk_px"])
            p2 = alpha * p2_f + (1 - alpha) * float(row["bk_p2"])

        s = p1 + pX + p2
        p1, pX, p2 = p1 / s, pX / s, p2 / s

        df.at[i, "pic_p1"] = p1
        df.at[i, "pic_px"] = pX
        df.at[i, "pic_p2"] = p2

       # --- OU tecnico ---
        # λ da forma (fallback)
        lam_form = float(row["home_form_gf_avg_lastN"] + row["away_form_gf_avg_lastN"])

        # λ tecnico avanzato se disponibile
        lam_form_adv = row.get("lambda_total_form", np.nan)
        if pd.notna(lam_form_adv) and lam_form_adv > 0:
            lam_form = float(lam_form_adv)

        # λ di mercato se disponibile
        lam_mkt = row.get("lambda_total_market_ou25", np.nan)
        if pd.isna(lam_mkt) or lam_mkt <= 0:
            lam_mkt = lam_form

        # blending λ tecnico/mercato
        lam = beta_ou_lambda * lam_form + (1.0 - beta_ou_lambda) * float(lam_mkt)
        if lam <= 0:
            lam = max(lam_form, 0.1)

        # OU 2.5 con λ mixato
        pO25_f, pU25_f = poisson_over_under_half_goalline(lam, 2.5)
        if pO25_f is not None:
            bk_pO25 = row.get("bk_pO25", np.nan)
            bk_pU25 = row.get("bk_pU25", np.nan)

            if pd.notna(bk_pO25) and pd.notna(bk_pU25):
                pO25 = alpha_ou * pO25_f + (1.0 - alpha_ou) * float(bk_pO25)
                pU25 = alpha_ou * pU25_f + (1.0 - alpha_ou) * float(bk_pU25)
            else:
                # fallback: solo tecnico
                pO25, pU25 = pO25_f, pU25_f

            # normalizza per sicurezza
            s_ou = pO25 + pU25
            if s_ou > 0:
                pO25 /= s_ou
                pU25 /= s_ou

            df.at[i, "pic_pO25"] = pO25
            df.at[i, "pic_pU25"] = pU25

        # OU 1.5
        # ---------------------------------------------
        # NUOVO PICCHETTO OU15 — INDIPENDENTE
        # ---------------------------------------------
        # λ attacco/difesa
        gf_h = row["home_form_gf_avg_lastN"]
        gf_a = row["away_form_gf_avg_lastN"]
        ga_h = row["home_form_ga_avg_lastN"]
        ga_a = row["away_form_ga_avg_lastN"]

        # safety
        if pd.notna(gf_h) and pd.notna(gf_a):
            lambda_att = gf_h + gf_a
            lambda_def = (ga_h + ga_a) if (pd.notna(ga_h) and pd.notna(ga_a)) else lambda_att

            # leggero effetto ELO: ±10%
            elo = row.get("elo_diff", 0)
            elo_factor = 1.0 + (elo / 400)
            elo_factor = max(0.7, min(1.3, elo_factor))

            lam15 = (0.6 * lambda_att + 0.4 * lambda_def) * elo_factor
            lam15 = max(0.2, min(6.0, lam15))

            # calcolo Poisson over 1.5
            p0 = math.exp(-lam15)
            p1 = p0 * lam15
            pO15 = 1 - (p0 + p1)
            pU15 = 1 - pO15

            df.at[i, "pic_pO15"] = pO15
            df.at[i, "pic_pU15"] = pU15
        else:
            df.at[i, "pic_pO15"] = np.nan
            df.at[i, "pic_pU15"] = np.nan

        # OU 0.5 (tecnico puro)
        pO05_f, pU05_f = poisson_over_under_half_goalline(lam, 0.5)
        if pO05_f is not None:
            df.at[i, "pic_pO05"] = pO05_f
            df.at[i, "pic_pU05"] = pU05_f

    return df