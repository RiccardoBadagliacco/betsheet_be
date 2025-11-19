# app/ml/correlazioni_affini_v2/common/features_ou15_v2.py

import math
from typing import Dict, Any

import numpy as np
import pandas as pd

FEATURES_OU15_V2 = [
    # META
    "match_id", "date", "season", "league",
    "home_team", "away_team",

    # LAMBDA FORM & MERCATO
    "lambda_home_form", "lambda_away_form", "lambda_total_form",
    "lambda_total_market_ou25",
    "lambda_total_mix_ou15",

    # POISSON TECH (MIX)
    "tech_pO05_mix", "tech_pO15_mix", "tech_pO25_mix",

    # PICCHETTO OU1.5
    "pic_pO15", "pic_pU15",
    "pic_pO05", "pic_pU05",

    # MISALIGNMENT
    "delta_ou15_pic_vs_mix",
    "delta_ou15_pic_vs_mix_abs",

    # CALENDARIO / STRESS
    "season_recency",
    "match_density_index",
    "rest_diff_days",
    "short_rest_home",
    "short_rest_away",
    "rest_advantage_home",
    "rest_advantage_away",
    "tightness_index",
]


def _poisson_over_probs(lam: float) -> Dict[str, float]:
    """Probabilità Poisson Over 0.5 / 1.5 / 2.5."""
    if lam is None or not np.isfinite(lam) or lam <= 0:
        return {"pO05": np.nan, "pO15": np.nan, "pO25": np.nan}

    e = math.exp(-lam)
    p0 = e
    p1 = e * lam
    p2 = e * lam * lam / 2.0

    pO05 = 1 - p0
    pO15 = 1 - (p0 + p1)
    pO25 = 1 - (p0 + p1 + p2)

    return {
        "pO05": max(0.0, min(1.0, pO05)),
        "pO15": max(0.0, min(1.0, pO15)),
        "pO25": max(0.0, min(1.0, pO25)),
    }


def build_features_ou15_v2(row: pd.Series) -> Dict[str, Any]:
    """
    Versione V2 OU1.5:
      - miscela λ mercato OU2.5 e λ form
      - calcola Poisson tech (0.5 / 1.5 / 2.5)
      - confronta con picchetto OU1.5 / OU0.5
    """
    out: Dict[str, Any] = {}

    # META
    out["match_id"]  = row.get("match_id")
    out["date"]      = row.get("date")
    out["season"]    = row.get("season")
    out["league"]    = row.get("league")
    out["home_team"] = row.get("home_team")
    out["away_team"] = row.get("away_team")

    # LAMBDA FORM + MERCATO
    lam_form  = float(row.get("lambda_total_form", np.nan))
    lam_mkt   = float(row.get("lambda_total_market_ou25", np.nan))

    out["lambda_home_form"]  = float(row.get("lambda_home_form", np.nan))
    out["lambda_away_form"]  = float(row.get("lambda_away_form", np.nan))
    out["lambda_total_form"] = lam_form
    out["lambda_total_market_ou25"] = lam_mkt

    if np.isfinite(lam_form) and np.isfinite(lam_mkt):
        lam_mix = 0.5 * lam_form + 0.5 * lam_mkt
    elif np.isfinite(lam_mkt):
        lam_mix = lam_mkt
    else:
        lam_mix = lam_form

    out["lambda_total_mix_ou15"] = lam_mix

    # POISSON TECH (MIX)
    if np.isfinite(lam_mix):
        p = _poisson_over_probs(lam_mix)
        out["tech_pO05_mix"] = p["pO05"]
        out["tech_pO15_mix"] = p["pO15"]
        out["tech_pO25_mix"] = p["pO25"]
    else:
        out["tech_pO05_mix"] = np.nan
        out["tech_pO15_mix"] = np.nan
        out["tech_pO25_mix"] = np.nan

    # PICCHETTO
    out["pic_pO15"] = float(row.get("pic_pO15", np.nan))
    out["pic_pU15"] = float(row.get("pic_pU15", np.nan))
    out["pic_pO05"] = float(row.get("pic_pO05", np.nan))
    out["pic_pU05"] = float(row.get("pic_pU05", np.nan))

    # MISALIGNMENT
    if np.isfinite(out["pic_pO15"]) and np.isfinite(out["tech_pO15_mix"]):
        delta = out["pic_pO15"] - out["tech_pO15_mix"]
        out["delta_ou15_pic_vs_mix"] = delta
        out["delta_ou15_pic_vs_mix_abs"] = abs(delta)
    else:
        out["delta_ou15_pic_vs_mix"] = np.nan
        out["delta_ou15_pic_vs_mix_abs"] = np.nan

    # CALENDARIO / STRESS
    out["season_recency"]      = float(row.get("season_recency", 0.0))
    out["match_density_index"] = float(row.get("match_density_index", np.nan))
    out["rest_diff_days"]      = float(row.get("rest_diff_days", np.nan))
    out["short_rest_home"]     = float(row.get("short_rest_home", 0.0))
    out["short_rest_away"]     = float(row.get("short_rest_away", 0.0))
    out["rest_advantage_home"] = float(row.get("rest_advantage_home", 0.0))
    out["rest_advantage_away"] = float(row.get("rest_advantage_away", 0.0))

    out["tightness_index"]      = float(row.get("tightness_index", np.nan))

    return out