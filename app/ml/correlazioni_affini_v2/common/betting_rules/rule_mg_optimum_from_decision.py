#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regole OPTIMUM + STRONG per Multigol Favorita (1–4 e 1–5)

Derivate dai segmenti forti trovati nello studio su 17.377 match con
quota favorita <= 1.95.

============================================================
REGOLE FINALI INSERITE (4 scenari):

1) MG 1–5 OPTIMUM (≈92–93%)
2) MG 1–4 OPTIMUM (≈87–88%)
3) MG 1–5 STRONG  (≈88–90%)
4) MG 1–4 STRONG  (≈84–86%)

============================================================

MG 1–5 OPTIMUM  
----------------------------
- fav_odds in (1.2, 1.4]
- mg_fav_1_5 >= 0.88
- diff14 in (0.02, 0.10]
- pattern top4 perfetto (0,4,0)

MG 1–4 OPTIMUM  
----------------------------
CASO A:
    - fav_odds in (1.2, 1.4]
    - mg14 in (0.8, 0.9]
    - diff14 in (-1.001, 0.10]
CASO B:
    - fav_odds in (1.4, 1.6]
    - mg14 in (0.9, 1.01]
    - diff14 in (-1.001, 0.02]
sempre con pattern (0,4,0)

MG 1–5 STRONG  
----------------------------
- 1.15 < fav_odds ≤ 1.40
- 0.88 < mg15 ≤ 1.03
- 0.02 < diff14 ≤ 0.12
- pattern perfetto

MG 1–4 STRONG  
----------------------------
- 1.25 < fav_odds ≤ 1.45
- 0.78 < mg14 ≤ 0.92
- 0.02 < diff14 ≤ 0.12
- pattern perfetto
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import math
import pandas as pd

from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions


# ============================================================
# HELPER
# ============================================================

def _f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def _match(value: float, low: float, high: float) -> bool:
    """Intervallo (low, high] come pandas."""
    return (value > low) and (value <= high)


def _compute_score_pattern(decision: Dict[str, Any], fav_side: str):
    scores = decision.get("score_prediction") or []
    score0 = score14 = score5p = 0

    for s in scores[:4]:
        score_str = s.get("score")
        if not isinstance(score_str, str) or "-" not in score_str:
            continue

        try:
            gh_str, ga_str = score_str.split("-")
            gh, ga = int(gh_str), int(ga_str)
        except Exception:
            continue

        fav_g = gh if fav_side == "home" else ga

        if fav_g == 0:
            score0 += 1
        elif 1 <= fav_g <= 4:
            score14 += 1
        elif fav_g >= 5:
            score5p += 1

    return score0, score14, score5p


def _build_alert(
    *,
    scenario: str,
    fav_side: str,
    fav_odds: float,
    mg13: float,
    mg14: float,
    mg15: float,
    diff14: float,
    score0: int,
    score14: int,
    score5p: int,
    mg_label: str,
    descr: str,
    bet_tags: List[str]
) -> BettingAlert:
    print('ciao')
    message = (
        f"Contesto per {mg_label}: "
        f"quota favorita={fav_odds:.2f}, "
        f"MG 1–3={mg13:.3f}, 1–4={mg14:.3f}, 1–5={mg15:.3f} "
        f"(diff14={diff14:.3f}). "
        f"Top4 scoreline → 0gol={score0}, 1–4gol={score14}, 5+={score5p}. "
        + descr
    )

    bet_suggestions = build_bet_suggestions(
        bet_tags=bet_tags,
        severity="high",
        side=fav_side,
    )

    return BettingAlert(
        code=scenario,
        severity="high",
        message=message,
        tags=["MULTIGOL", scenario] + bet_tags,
        bets=bet_suggestions.get("suggestions", []),
        meta={
            "scenario": scenario,
            "fav_side": fav_side,
            "fav_odds": fav_odds,
            "mg_fav_1_3": mg13,
            "mg_fav_1_4": mg14,
            "mg_fav_1_5": mg15,
            "diff14": diff14,
            "score_top4_fav_0": score0,
            "score_top4_fav_1_4": score14,
            "score_top4_fav_5p": score5p,
        },
    )


# ============================================================
# ENTRYPOINT
# ============================================================

def rule_mg_optimum_signal(t0: pd.Series, ctx: Dict[str, Any]):
    try:
        decision = ctx.get("decision") or ctx.get("ml_decision") or {}
        meta = ctx.get("meta") or ctx.get("match_meta") or {}
        if not decision:
            return []

        alert = build_mg_fav_optimum_alert_from_decision(meta, decision)
        return [alert] if alert else []

    except Exception as e:
        print("MG Optimum ERROR:", e)
        return []


# ============================================================
# CORE REGOLE
# ============================================================

def build_mg_fav_optimum_alert_from_decision(
    meta: Dict[str, Any],
    decision: Dict[str, Any]
) -> Optional[BettingAlert]:
    print('ciao')
    # ---- 1) Identifica favorita ----
    odds = meta.get("odds") or {}
    oh = _f(odds.get("avg_home_odds"))
    oa = _f(odds.get("avg_away_odds"))

    if math.isnan(oh) or math.isnan(oa):
        return None

    if oh < oa:
        fav_side, fav_odds = "home", oh
    elif oa < oh:
        fav_side, fav_odds = "away", oa
    else:
        return None

    # ---- 2) Probabilità MG ----
    multigol = (decision.get("multigol") or {}).get("fav") or {}
    mg13 = _f(multigol.get("1-3"))
    mg14 = _f(multigol.get("1-4"))
    mg15 = _f(multigol.get("1-5"))

    if any(math.isnan(x) for x in (mg13, mg14, mg15)):
        return None

    diff14 = mg15 - mg14

    # ---- 3) Pattern TOP4 ----
    score0, score14, score5p = _compute_score_pattern(decision, fav_side)

    # Pattern perfetto richiesto da TUTTE le regole
    if not(score0 == 0 and score14 == 4):
        return None

    # ============================================================
    # --------------------- MG 1–5 OPTIMUM ------------------------
    # ============================================================
    if (
        _match(fav_odds, 1.2, 1.4)
        and mg15 >= 0.88
        and _match(diff14, 0.02, 0.10)
    ):
        return _build_alert(
            scenario="MG1_5_OPTIMUM",
            fav_side=fav_side,
            fav_odds=fav_odds,
            mg13=mg13, mg14=mg14, mg15=mg15,
            diff14=diff14,
            score0=score0, score14=score14, score5p=score5p,
            mg_label="Multigol Favorita 1–5",
            descr="Segmento OPTIMUM (~92–93%).",
            bet_tags=["BET_MG15_FAV_OPT", "BET_MG15_FAV"]
        )

    # ============================================================
    # --------------------- MG 1–4 OPTIMUM ------------------------
    # ============================================================
    cond_A = (
        _match(fav_odds, 1.2, 1.4)
        and _match(mg14, 0.8, 0.9)
        and _match(diff14, -1.001, 0.10)
    )

    cond_B = (
        _match(fav_odds, 1.4, 1.6)
        and _match(mg14, 0.9, 1.01)
        and _match(diff14, -1.001, 0.02)
    )

    if cond_A or cond_B:
        return _build_alert(
            scenario="MG1_4_OPTIMUM",
            fav_side=fav_side,
            fav_odds=fav_odds,
            mg13=mg13, mg14=mg14, mg15=mg15,
            diff14=diff14,
            score0=score0, score14=score14, score5p=score5p,
            mg_label="Multigol Favorita 1–4",
            descr="Segmento OPTIMUM (~87–88%).",
            bet_tags=["BET_MG14_FAV_OPT", "BET_MG14_FAV"]
        )

    # ============================================================
    # ---------------------- MG 1–5 STRONG ------------------------
    # ============================================================
    if (
        _match(fav_odds, 1.15, 1.40)
        and _match(mg15, 0.88, 1.03)
        and _match(diff14, 0.02, 0.12)
    ):
        return _build_alert(
            scenario="MG1_5_STRONG",
            fav_side=fav_side,
            fav_odds=fav_odds,
            mg13=mg13, mg14=mg14, mg15=mg15,
            diff14=diff14,
            score0=score0, score14=score14, score5p=score5p,
            mg_label="Multigol Favorita 1–5",
            descr="Segmento STRONG (~88–90%), copertura elevata.",
            bet_tags=["BET_MG15_FAV", "BET_MG15_STRONG"]
        )

    # ============================================================
    # ---------------------- MG 1–4 STRONG ------------------------
    # ============================================================
    if (
        _match(fav_odds, 1.25, 1.45)
        and _match(mg14, 0.78, 0.92)
        and _match(diff14, 0.02, 0.12)
    ):
        return _build_alert(
            scenario="MG1_4_STRONG",
            fav_side=fav_side,
            fav_odds=fav_odds,
            mg13=mg13, mg14=mg14, mg15=mg15,
            diff14=diff14,
            score0=score0, score14=score14, score5p=score5p,
            mg_label="Multigol Favorita 1–4",
            descr="Segmento STRONG (~84–86%), ottimo volume match.",
            bet_tags=["BET_MG14_FAV", "BET_MG14_STRONG"]
        )

    return None