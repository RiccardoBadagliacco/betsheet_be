# app/ml/correlazioni_affini_v2/common/features_1x2_v2.py

import math
from typing import Dict, Any

import numpy as np
import pandas as pd

# ============================================================
# FEATURE LIST UFFICIALE V2 PER IL MODELLO / CLUSTER 1X2
# ============================================================

FEATURES_1X2_V2 = [
    # --- META ---
    "match_id", "date", "season", "league",
    "home_team", "away_team",

    # --- BOOKMAKER 1X2 ---
    "bk_p1", "bk_px", "bk_p2",
    "bk_sum_1x2", "bk_overround_1x2",
    "entropy_bk_1x2",

    # --- PICCHETTO TECNICO 1X2 ---
    "pic_p1", "pic_px", "pic_p2",
    "pic_sum_1x2", "pic_overround_1x2",
    "entropy_pic_1x2",

    # --- DELTA PICCHETTO vs BOOK ---
    "delta_p1", "delta_px", "delta_p2",
    "delta_1x2_abs_sum",
    "cosine_bk_pic_1x2",
    "market_sharpness",
    "instability_index",

    # --- ELO / STRENGTH ---
    "elo_home_pre", "elo_away_pre", "elo_diff",
    "elo_diff_raw", "elo_diff_raw_z",
    "team_strength_home", "team_strength_away", "team_strength_diff",

    # --- FORMA (MEDIA ULTIMI N) ---
    "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
    "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
    "home_form_ga_avg_lastN",  "away_form_ga_avg_lastN",
    "home_form_win_rate_lastN", "away_form_win_rate_lastN",
    "home_form_matches_lastN",  "away_form_matches_lastN",

    # --- DIFF FORM Z-SCORE ---
    "form_pts_diff_raw", "form_pts_diff_raw_z",
    "form_gf_diff_raw",  "form_gf_diff_raw_z",
    "form_ga_diff_raw",  "form_ga_diff_raw_z",

    # --- GOAL MODEL / LAMBDA ---
    "lambda_home_form", "lambda_away_form", "lambda_total_form",
    "lambda_total_market_ou25",
    "goal_supremacy_market_ou25",
    "goal_supremacy_form",

    # --- INFO MERCATO AGGIUNTIVE ---
    "fav_prob_1x2",
    "market_balance_index_1x2",
    "fav_prob_gap_1x2",
    "second_fav_prob_1x2",

    # --- CALENDARIO / STRESS FISICO ---
    "season_recency",
    "match_density_index",
    "days_since_last_home",
    "days_since_last_away",
    "rest_diff_days",
    "short_rest_home",
    "short_rest_away",
    "rest_advantage_home",
    "rest_advantage_away",

    "tightness_index"
]


# ============================================================
# UTILS
# ============================================================

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity robusta (se vettori nulli → 0, se NaN → NaN)."""
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# CORE BUILDER
# ============================================================

def build_features_1x2_v2(row: pd.Series) -> Dict[str, Any]:
    """
    Costruisce le feature 1X2 V2 per UNA sola riga del dataframe
    step2a_features_with_picchetto_fix.

    Assume che la riga contenga già:
      - bk_p1, bk_px, bk_p2
      - pic_p1, pic_px, pic_p2
      - elo_*, form_*, lambda_*, goal_supremacy_*, season_recency, ecc.
    """

    out: Dict[str, Any] = {}

    # --- META ---
    out["match_id"]  = row.get("match_id")
    out["date"]      = row.get("date")
    out["season"]    = row.get("season")
    out["league"]    = row.get("league")
    out["home_team"] = row.get("home_team")
    out["away_team"] = row.get("away_team")

    # ==========================
    # 1) PROBABILITÀ MERCATO
    # ==========================
    bk_p1 = float(row.get("bk_p1", np.nan))
    bk_px = float(row.get("bk_px", np.nan))
    bk_p2 = float(row.get("bk_p2", np.nan))

    out["bk_p1"] = bk_p1
    out["bk_px"] = bk_px
    out["bk_p2"] = bk_p2

    s_bk = bk_p1 + bk_px + bk_p2
    out["bk_sum_1x2"] = s_bk
    out["bk_overround_1x2"] = s_bk - 1.0 if np.isfinite(s_bk) else np.nan

    p_bk = np.array([bk_p1, bk_px, bk_p2], dtype=float)
    out["entropy_bk_1x2"] = _entropy(p_bk)

    # ==========================
    # 2) PROBABILITÀ PICCHETTO
    # ==========================
    pic_p1 = float(row.get("pic_p1", np.nan))
    pic_px = float(row.get("pic_px", np.nan))
    pic_p2 = float(row.get("pic_p2", np.nan))

    out["pic_p1"] = pic_p1
    out["pic_px"] = pic_px
    out["pic_p2"] = pic_p2

    s_pic = pic_p1 + pic_px + pic_p2
    out["pic_sum_1x2"] = s_pic
    out["pic_overround_1x2"] = s_pic - 1.0 if np.isfinite(s_pic) else np.nan

    p_pic = np.array([pic_p1, pic_px, pic_p2], dtype=float)
    out["entropy_pic_1x2"] = _entropy(p_pic)

    # ==========================
    # 3) DELTA / COERENZA
    # ==========================
    out["delta_p1"] = pic_p1 - bk_p1 if np.isfinite(pic_p1) and np.isfinite(bk_p1) else np.nan
    out["delta_px"] = pic_px - bk_px if np.isfinite(pic_px) and np.isfinite(bk_px) else np.nan
    out["delta_p2"] = pic_p2 - bk_p2 if np.isfinite(pic_p2) and np.isfinite(bk_p2) else np.nan

    deltas = [out["delta_p1"], out["delta_px"], out["delta_p2"]]
    if all(np.isfinite(d) for d in deltas):
        out["delta_1x2_abs_sum"] = float(sum(abs(d) for d in deltas))
    else:
        out["delta_1x2_abs_sum"] = np.nan

    out["cosine_bk_pic_1x2"] = _safe_cosine(p_bk, p_pic)

    if all(np.isfinite([bk_p1, bk_px, bk_p2, pic_p1, pic_px, pic_p2])):
        out["market_sharpness"] = float(
            np.linalg.norm([bk_p1 - pic_p1, bk_px - pic_px, bk_p2 - pic_p2])
        )
    else:
        out["market_sharpness"] = np.nan

    if out["market_sharpness"] and np.isfinite(out["market_sharpness"]) and out["market_sharpness"] != 0:
        out["instability_index"] = out["delta_1x2_abs_sum"] / out["market_sharpness"]
    else:
        out["instability_index"] = 0.0 if np.isfinite(out["delta_1x2_abs_sum"]) else np.nan

    # ==========================
    # 4) ELO / STRENGTH
    # ==========================
    out["elo_home_pre"] = float(row.get("elo_home_pre", np.nan))
    out["elo_away_pre"] = float(row.get("elo_away_pre", np.nan))
    out["elo_diff"]     = float(row.get("elo_diff", np.nan))

    out["elo_diff_raw"]   = float(row.get("elo_diff_raw", out["elo_home_pre"] - out["elo_away_pre"]))
    out["elo_diff_raw_z"] = float(row.get("elo_diff_raw_z", np.nan))

    out["team_strength_home"] = float(row.get("team_strength_home", np.nan))
    out["team_strength_away"] = float(row.get("team_strength_away", np.nan))
    out["team_strength_diff"] = float(row.get("team_strength_diff",
                                              out["team_strength_home"] - out["team_strength_away"]
                                              if np.isfinite(out["team_strength_home"]) and np.isfinite(out["team_strength_away"])
                                              else np.nan))

    # ==========================
    # 5) FORMA
    # ==========================
    for col in [
        "home_form_pts_avg_lastN", "away_form_pts_avg_lastN",
        "home_form_gf_avg_lastN",  "away_form_gf_avg_lastN",
        "home_form_ga_avg_lastN",  "away_form_ga_avg_lastN",
        "home_form_win_rate_lastN", "away_form_win_rate_lastN",
        "home_form_matches_lastN",  "away_form_matches_lastN",
        "form_pts_diff_raw", "form_pts_diff_raw_z",
        "form_gf_diff_raw",  "form_gf_diff_raw_z",
        "form_ga_diff_raw",  "form_ga_diff_raw_z",
    ]:
        out[col] = float(row.get(col, np.nan))

    # ==========================
    # 6) GOAL MODEL / LAMBDA
    # ==========================
    out["lambda_home_form"]   = float(row.get("lambda_home_form", np.nan))
    out["lambda_away_form"]   = float(row.get("lambda_away_form", np.nan))
    out["lambda_total_form"]  = float(row.get("lambda_total_form", np.nan))

    out["lambda_total_market_ou25"] = float(row.get("lambda_total_market_ou25", np.nan))

    out["goal_supremacy_market_ou25"] = float(row.get("goal_supremacy_market_ou25", np.nan))
    out["goal_supremacy_form"]        = float(row.get("goal_supremacy_form", np.nan))

    # ==========================
    # 7) INFO MERCATO EXTRA
    # ==========================
    out["fav_prob_1x2"]           = float(row.get("fav_prob_1x2", np.nan))
    out["market_balance_index_1x2"] = float(row.get("market_balance_index_1x2", np.nan))
    out["fav_prob_gap_1x2"]       = float(row.get("fav_prob_gap_1x2", np.nan))
    out["second_fav_prob_1x2"]    = float(row.get("second_fav_prob_1x2", np.nan))

    # ==========================
    # 8) CALENDARIO / STRESS
    # ==========================
    out["season_recency"]      = float(row.get("season_recency", 0.0))
    out["match_density_index"] = float(row.get("match_density_index", np.nan))

    out["days_since_last_home"] = float(row.get("days_since_last_home", np.nan))
    out["days_since_last_away"] = float(row.get("days_since_last_away", np.nan))

    out["rest_diff_days"]        = float(row.get("rest_diff_days", np.nan))
    out["short_rest_home"]       = float(row.get("short_rest_home", 0.0))
    out["short_rest_away"]       = float(row.get("short_rest_away", 0.0))
    out["rest_advantage_home"]   = float(row.get("rest_advantage_home", 0.0))
    out["rest_advantage_away"]   = float(row.get("rest_advantage_away", 0.0))

    out["tightness_index"]      = float(row.get("tightness_index", np.nan))

    return out