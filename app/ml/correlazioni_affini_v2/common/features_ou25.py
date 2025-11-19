# app/ml/correlazioni_affini_v2/common/features_ou25_v2.py

import math
from typing import Dict, Any

import numpy as np
import pandas as pd

FEATURES_OU25_V2 = [
    # META
    "match_id", "date", "season", "league",
    "home_team", "away_team",

    # BOOKMAKER OU2.5
    "bk_pO25", "bk_pU25",
    "bk_sum_ou25_raw", "bk_overround_ou25",
    "entropy_bk_ou25",

    # PICCHETTO OU2.5
    "pic_pO25", "pic_pU25",
    "pic_sum_ou25", "pic_overround_ou25",
    "entropy_pic_ou25",

    # DELTA PICCHETTO vs BOOK
    "delta_O25", "delta_U25",
    "delta_ou25_abs_sum",
    "delta_ou25_market_vs_form",

    # GOAL MODEL / LAMBDA
    "lambda_total_market_ou25",
    "lambda_total_form",
    "goal_supremacy_market_ou25",
    "goal_supremacy_form",

    # ELO / FORM DIFFERENCES
    "elo_diff_raw", "elo_diff_raw_z",
    "form_gf_diff_raw", "form_gf_diff_raw_z",
    "form_ga_diff_raw", "form_ga_diff_raw_z",

    # CALENDARIO / STRESS
    "season_recency",
    "match_density_index",
    "days_since_last_home",
    "days_since_last_away",
    "rest_diff_days",
    "short_rest_home",
    "short_rest_away",
    "rest_advantage_home",
    "rest_advantage_away",
    "tightness_index",
]


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def build_features_ou25_v2(row: pd.Series) -> Dict[str, Any]:
    """
    Feature OU2.5 per clustering / modelling.
    Usa picchetto + λ mercato e λ form.
    """
    out: Dict[str, Any] = {}

    # META
    out["match_id"]  = row.get("match_id")
    out["date"]      = row.get("date")
    out["season"]    = row.get("season")
    out["league"]    = row.get("league")
    out["home_team"] = row.get("home_team")
    out["away_team"] = row.get("away_team")

    # BOOKMAKER
    bk_pO25 = float(row.get("bk_pO25", np.nan))
    bk_pU25 = float(row.get("bk_pU25", np.nan))
    out["bk_pO25"] = bk_pO25
    out["bk_pU25"] = bk_pU25

    s_bk = bk_pO25 + bk_pU25
    out["bk_sum_ou25_raw"]   = float(row.get("bk_sum_ou25_raw", s_bk))
    out["bk_overround_ou25"] = float(row.get("bk_overround_ou25", out["bk_sum_ou25_raw"] - 1.0))

    out["entropy_bk_ou25"] = _entropy(np.array([bk_pO25, bk_pU25], dtype=float))

    # PICCHETTO
    pic_pO25 = float(row.get("pic_pO25", np.nan))
    pic_pU25 = float(row.get("pic_pU25", np.nan))
    out["pic_pO25"] = pic_pO25
    out["pic_pU25"] = pic_pU25

    s_pic = pic_pO25 + pic_pU25
    out["pic_sum_ou25"]   = s_pic
    out["pic_overround_ou25"] = s_pic - 1.0 if np.isfinite(s_pic) else np.nan
    out["entropy_pic_ou25"] = _entropy(np.array([pic_pO25, pic_pU25], dtype=float))

    # DELTA
    out["delta_O25"] = pic_pO25 - bk_pO25 if np.isfinite(pic_pO25) and np.isfinite(bk_pO25) else np.nan
    out["delta_U25"] = pic_pU25 - bk_pU25 if np.isfinite(pic_pU25) and np.isfinite(bk_pU25) else np.nan

    if np.isfinite(out["delta_O25"]) and np.isfinite(out["delta_U25"]):
        out["delta_ou25_abs_sum"] = abs(out["delta_O25"]) + abs(out["delta_U25"])
    else:
        out["delta_ou25_abs_sum"] = np.nan

    out["delta_ou25_market_vs_form"] = float(row.get("delta_ou25_market_vs_form", np.nan))

    # GOAL MODEL
    out["lambda_total_market_ou25"] = float(row.get("lambda_total_market_ou25", np.nan))
    out["lambda_total_form"]        = float(row.get("lambda_total_form", np.nan))
    out["goal_supremacy_market_ou25"] = float(row.get("goal_supremacy_market_ou25", np.nan))
    out["goal_supremacy_form"]        = float(row.get("goal_supremacy_form", np.nan))

    # ELO / FORM
    out["elo_diff_raw"]   = float(row.get("elo_diff_raw", np.nan))
    out["elo_diff_raw_z"] = float(row.get("elo_diff_raw_z", np.nan))

    out["form_gf_diff_raw"]   = float(row.get("form_gf_diff_raw", np.nan))
    out["form_gf_diff_raw_z"] = float(row.get("form_gf_diff_raw_z", np.nan))

    out["form_ga_diff_raw"]   = float(row.get("form_ga_diff_raw", np.nan))
    out["form_ga_diff_raw_z"] = float(row.get("form_ga_diff_raw_z", np.nan))

    # CALENDARIO / STRESS
    out["season_recency"]      = float(row.get("season_recency", 0.0))
    out["match_density_index"] = float(row.get("match_density_index", np.nan))
    out["days_since_last_home"] = float(row.get("days_since_last_home", np.nan))
    out["days_since_last_away"] = float(row.get("days_since_last_away", np.nan))

    out["rest_diff_days"]      = float(row.get("rest_diff_days", np.nan))
    out["short_rest_home"]     = float(row.get("short_rest_home", 0.0))
    out["short_rest_away"]     = float(row.get("short_rest_away", 0.0))
    out["rest_advantage_home"] = float(row.get("rest_advantage_home", 0.0))
    out["rest_advantage_away"] = float(row.get("rest_advantage_away", 0.0))
    out["tightness_index"]      = float(row.get("tightness_index", np.nan))

    return out