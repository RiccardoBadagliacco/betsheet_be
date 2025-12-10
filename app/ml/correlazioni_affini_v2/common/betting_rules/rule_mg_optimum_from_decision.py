#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Optional, List
import math

from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions


def _f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def build_mg_fav_optimum_alert_from_decision(
    meta: Dict[str, Any],
    decision: Dict[str, Any],
) -> Optional[BettingAlert]:
    """
    Regola OPTIMUM MG (Favorita 1–4 / 1–5) basata su:
      - decision['multigol']['fav']   (1-3, 1-4, 1-5, ...)
      - meta['odds']['avg_home_odds'/'avg_away_odds']
      - decision['score_prediction']  (top score)

    Ritorna un BettingAlert oppure None se non scatta nulla.
    """

    # -----------------------------------------------------
    # 1) Identifico favorita dalle quote REALI
    # -----------------------------------------------------
    odds = meta.get("odds") or {}
    oh = _f(odds.get("avg_home_odds"))
    oa = _f(odds.get("avg_away_odds"))

    if math.isnan(oh) or math.isnan(oa):
        return None

    if oh < oa:
        fav_side = "home"
        fav_odds = oh
    elif oa < oh:
        fav_side = "away"
        fav_odds = oa
    else:
        return None

    # -----------------------------------------------------
    # 2) Prendo le probabilità multigol *già pronte*
    #    decision["multigol"]["fav"]
    # -----------------------------------------------------
    multigol = (decision.get("multigol") or {}).get("fav") or {}
    mg13 = _f(multigol.get("1-3"))
    mg14 = _f(multigol.get("1-4"))
    mg15 = _f(multigol.get("1-5"))

    if any(math.isnan(x) for x in (fav_odds, mg14, mg15)):
        return None

    diff14 = mg15 - mg14

    # -----------------------------------------------------
    # 3) Top 4 scoreline: pattern 0 gol vs 1–4 gol favorita
    # -----------------------------------------------------
    scores = decision.get("score_prediction") or []
    score0 = 0
    score14 = 0

    for s in scores[:4]:  # solo top4
        score_str = s.get("score")
        if not isinstance(score_str, str) or "-" not in score_str:
            continue
        try:
            gh_str, ga_str = score_str.split("-")
            gh = int(gh_str)
            ga = int(ga_str)
        except Exception:
            continue

        fav_g = gh if fav_side == "home" else ga
        if fav_g == 0:
            score0 += 1
        elif 1 <= fav_g <= 4:
            score14 += 1
        # se >=5 non ci serve ora

    # -----------------------------------------------------
    # 4) Classificazione segmenti premium
    # -----------------------------------------------------
    scenario = None
    descr = ""
    mg_label = ""
    bet_tags: List[str] = []

    # 1) MG 1–5 STRONG (92–93%)
    if (
        1.20 < fav_odds <= 1.40
        and mg15 >= 0.90
        and diff14 <= 0.05
        and score0 == 0
        and score14 == 4
    ):
        scenario = "MG1_5_STRONG"
        mg_label = "Multigol Favorita 1–5"
        descr = "Segmento premium: MG Favorita 1–5 con frequenza storica ~92–93% nel backtest."
        bet_tags = ["BET_MG15_FAV_OPT", "BET_MG15_FAV"]

    # 2) MG 1–4 STRONG (87–88%)
    elif (
        1.20 < fav_odds <= 1.60
        and mg14 >= 0.80
        and diff14 <= 0.10
        and score0 == 0
        and score14 == 4
    ):
        scenario = "MG1_4_STRONG"
        mg_label = "Multigol Favorita 1–4"
        descr = "Segmento premium: MG Favorita 1–4 con frequenza storica ~87–88% nel backtest."
        bet_tags = ["BET_MG14_FAV_OPT", "BET_MG14_FAV"]

    if scenario is None:
        return None

    # -----------------------------------------------------
    # 5) Messaggio + BettingAlert
    # -----------------------------------------------------
    parts: List[str] = []
    parts.append(f"Contesto OTTIMALE per {mg_label}: ")
    parts.append(f"quota favorita ≈ {fav_odds:.2f}, ")
    parts.append(
        f"MG favorita: 1–3={mg13:.2f}, 1–4={mg14:.2f}, 1–5={mg15:.2f}. "
    )
    parts.append(
        "Nei top4 scoreline la favorita compare SEMPRE con 1–4 gol "
        "(0x scenari a 0 gol, 4x scenari 1–4 gol). "
    )
    parts.append(descr)

    message = "".join(parts)

    severity = "high"
    bet_suggestions = build_bet_suggestions(
        bet_tags=bet_tags,
        severity=severity,
        side=fav_side,
    )
    bets = bet_suggestions.get("suggestions", [])

    return BettingAlert(
        code="MG_FAV_OPTIMUM_SIGNAL",
        severity=severity,
        message=message,
        tags=["MULTIGOL", "MG_FAVORITA_OPTIMUM", scenario] + bet_tags,
        bets=bets,
        meta={
            "scenario": scenario,
            "fav_side": fav_side,
            "fav_odds": fav_odds,
            "mg_fav_1_3": mg13,
            "mg_fav_1_4": mg14,
            "mg_fav_1_5": mg15,
            "score_top4_fav_0": score0,
            "score_top4_fav_1_4": score14,
            "bet_tags_raw": bet_tags,
            "bet_suggestions": bet_suggestions,
        },
    )