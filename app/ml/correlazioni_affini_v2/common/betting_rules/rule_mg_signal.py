import math
from typing import List, Dict, Any

import pandas as pd

from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions



def _f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def safe_int(x, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


# ============================================================================
# PROFILO STATISTICO MULTIGOL FAVORITA (MG 1–4)
# ============================================================================

def _determine_fav_side_and_prob(t0: pd.Series) -> Dict[str, Any]:
    """
    Determina favorita (home/away) e probabilità implicita
    a partire da bk_p1 / bk_p2.

    Ritorna:
        {
          "has_fav": bool,
          "fav_side": "home" | "away" | None,
          "fav_prob": float | nan
        }
    """
    p1 = _f(t0.get("bk_p1"))
    p2 = _f(t0.get("bk_p2"))

    if math.isnan(p1) or math.isnan(p2):
        return {"has_fav": False, "fav_side": None, "fav_prob": math.nan}

    if p1 > p2:
        return {"has_fav": True, "fav_side": "home", "fav_prob": p1}
    if p2 > p1:
        return {"has_fav": True, "fav_side": "away", "fav_prob": p2}

    return {"has_fav": False, "fav_side": None, "fav_prob": math.nan}


def build_mg14_stat_profile(t0: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Costruisce un profilo statistico per il mercato
    MULTIGOL FAVORITA 1–4 (MG 1–4).

    Output:
        {
          "has_profile": bool,
          "in_context": bool,
          "checklist_pass": bool,
          "fav_side": "home"/"away"/None,
          "fav_prob": float,
          "lambda_fav": float,
          "fav_gf_avg": float,
          "opp_ga_avg": float,
          "lambda_total": float,
          "fav_matches": int,
          "opp_matches": int,
          "fav_enough": bool,
          "opp_enough": bool,
          "fav_att_ok": bool,
          "opp_def_permeable": bool,
          "total_ok": bool,
        }
    """
    try:
        fav_info = _determine_fav_side_and_prob(t0)
        if not fav_info["has_fav"]:
            return {
                "has_profile": False,
                "in_context": False,
                "checklist_pass": False,
            }

        fav_side = fav_info["fav_side"]
        fav_prob = fav_info["fav_prob"]

        # Range di probabilità equivalente a quota 1.15–1.90
        FAV_PROB_MIN = 1.0 / 1.90
        FAV_PROB_MAX = 1.0 / 1.15

        in_context = (
            not math.isnan(fav_prob)
            and (FAV_PROB_MIN <= fav_prob <= FAV_PROB_MAX)
        )
        if not in_context:
            return {
                "has_profile": True,
                "in_context": False,
                "checklist_pass": False,
                "fav_side": fav_side,
                "fav_prob": fav_prob,
            }

        # -----------------------------------------------------
        # Estrazione feature di forma a seconda del lato
        # -----------------------------------------------------
        if fav_side == "home":
            fav_matches = safe_int(t0.get("home_form_matches_lastN"), 0)
            fav_gf_avg = _f(t0.get("home_form_gf_avg_lastN"))
            fav_lambda = _f(t0.get("lambda_home_form"))

            opp_matches = safe_int(t0.get("away_form_matches_lastN"), 0)
            opp_ga_avg = _f(t0.get("away_form_ga_avg_lastN"))
        else:
            fav_matches = safe_int(t0.get("away_form_matches_lastN"), 0)
            fav_gf_avg = _f(t0.get("away_form_gf_avg_lastN"))
            fav_lambda = _f(t0.get("lambda_away_form"))

            opp_matches = safe_int(t0.get("home_form_matches_lastN"), 0)
            opp_ga_avg = _f(t0.get("home_form_ga_avg_lastN"))

        lambda_total = _f(t0.get("lambda_total_form"))

        # -----------------------------------------------------
        # Regole “statistiche” estratte dal backtest
        # -----------------------------------------------------

        # Storico minimo
        fav_enough = fav_matches >= 5
        opp_enough = opp_matches >= 5

        # Attacco favorita: devono esserci gol con una certa continuità
        fav_att_ok = (
            (not math.isnan(fav_lambda) and fav_lambda >= 1.2)
            or (not math.isnan(fav_gf_avg) and fav_gf_avg >= 1.2)
        )

        # Difesa avversaria: non ermetica
        opp_def_permeable = (
            (not math.isnan(opp_ga_avg))
            and opp_ga_avg >= 1.0
        )

        # Volume totale gol “ragionevole” per MG 1–4
        total_ok = (
            not math.isnan(lambda_total)
            and 2.0 <= lambda_total <= 4.0
        )

        checklist_pass = (
            fav_enough
            and opp_enough
            and fav_att_ok
            and opp_def_permeable
            and total_ok
        )

        return {
            "has_profile": True,
            "in_context": True,
            "checklist_pass": checklist_pass,
            "fav_side": fav_side,
            "fav_prob": fav_prob,
            "lambda_fav": fav_lambda,
            "fav_gf_avg": fav_gf_avg,
            "opp_ga_avg": opp_ga_avg,
            "lambda_total": lambda_total,
            "fav_matches": fav_matches,
            "opp_matches": opp_matches,
            # flag per debug
            "fav_enough": fav_enough,
            "opp_enough": opp_enough,
            "fav_att_ok": fav_att_ok,
            "opp_def_permeable": opp_def_permeable,
            "total_ok": total_ok,
        }

    except Exception:
        return {
            "has_profile": False,
            "in_context": False,
            "checklist_pass": False,
        }
# ============================================================================
# REGOLA PRINCIPALE: MULTIGOL FAVORITA 1–4 (CHECKLIST STATISTICA)
# ============================================================================

def rule_mg_fav_signal(t0: pd.Series, ctx: Dict[str, Any]) -> List[BettingAlert]:
    """
    MG_FAVORITA_SIGNAL (v1.0 - checklist statistica)

    Regola che individua i match in cui la favorita (home/away),
    con quota nel range 1.15–1.90, ha un profilo storico coerente
    con il multigol 1–4 (MG 1–4) e genera un alert dedicato.

    Usa solo feature “prematch” già presenti nel wide:
      - bk_p1 / bk_p2        → favorita + range quota
      - lambda_home_form / lambda_away_form
      - lambda_total_form
      - home_form_* / away_form_* (gf_avg, ga_avg, matches_lastN)

    Dal backtest:
      - baseline MG 1–4 su tutte le favorite: ~82.7%
      - checklist statistica (copertura ~22%): ~83.2%
        → filtro di stabilità, non edge gigantesco ma solido per scremare le partite.
    """
    alerts: List[BettingAlert] = []

    try:
        print(f"    - t0: {t0.to_dict()}")
        print("    - Costruzione profilo MG favorita (checklist statistica)...")

        profile = build_mg14_stat_profile(t0, ctx)
        if not profile.get("has_profile", False):
            print('Check 1')
            return alerts

        if not profile.get("in_context", False):
            # favorita fuori range quota → nessun alert MG
            print('Check 2')
            return alerts

        if not profile.get("checklist_pass", False):

            fav_side = profile.get("fav_side")
            fav_prob = profile.get("fav_prob")
            fav_matches = profile.get("fav_matches", 0)
            opp_matches = profile.get("opp_matches", 0)
            lambda_fav = profile.get("lambda_fav", math.nan)
            fav_gf_avg = profile.get("fav_gf_avg", math.nan)
            opp_ga_avg = profile.get("lambda_opp_ga", math.nan)
            lambda_total = profile.get("lambda_total", math.nan)

            fav_enough = profile.get("fav_enough", False)
            opp_enough = profile.get("opp_enough", False)
            fav_att_ok = profile.get("fav_att_ok", False)
            opp_def_permeable = profile.get("opp_def_permeable", False)
            total_ok = profile.get("total_ok", False)

            # Safe formatting
            def fmt(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) and not math.isnan(x) else "nan"

            try:
                eq_odds = 1.0 / fav_prob if fav_prob and fav_prob > 0 else math.nan
            except Exception:
                eq_odds = math.nan

            print("    - [MG] CHECKLIST FALLITA (Check3):")
            print(f"        fav_side={fav_side}, fav_prob={fmt(fav_prob)}, eq_odds≈{fmt(eq_odds)}")
            print(
                "        "
                f"fav_matches={fav_matches}, "
                f"opp_matches={opp_matches}, "
                f"lambda_fav={fmt(lambda_fav)}, "
                f"fav_gf_avg={fmt(fav_gf_avg)}, "
                f"opp_ga_avg={fmt(opp_ga_avg)}, "
                f"lambda_total={fmt(lambda_total)}"
            )
            print(
                "        FLAGS: "
                f"fav_enough={fav_enough}, "
                f"opp_enough={opp_enough}, "
                f"fav_att_ok={fav_att_ok}, "
                f"opp_def_permeable={opp_def_permeable}, "
                f"total_ok={total_ok}"
            )

            return alerts

        fav_side = profile.get("fav_side", "home")
        fav_prob = profile.get("fav_prob", math.nan)
        lambda_fav = profile.get("lambda_fav", math.nan)
        lambda_opp_ga = profile.get("lambda_opp_ga", math.nan)
        lambda_total = profile.get("lambda_total", math.nan)
        fav_matches = profile.get("fav_matches", 0)
        opp_matches = profile.get("opp_matches", 0)

        # lato per i bet tag: per coerenza con altri modelli usiamo "home"/"away"
        side = fav_side

        # ---------------------------
        # Messaggio testuale
        # ---------------------------
        parts: List[str] = []
        parts.append("Contesto favorevole al Multigol 1–4 sulla favorita: ")

        # quota favorita (in termini di probabilità)
        if not math.isnan(fav_prob):
            eq_odds = 1.0 / fav_prob if fav_prob > 0 else math.nan
            if not math.isnan(eq_odds):
                parts.append(
                    f"la squadra favorita ha una quota implicita intorno a {eq_odds:.2f}, "
                )
            else:
                parts.append("la squadra favorita ha una quota nel range 1.15–1.90, ")
        else:
            parts.append("la squadra favorita è nel range di quota medio-bassa, ")

        # storico favorita
        parts.append(
            f"con almeno {fav_matches} partite recenti sullo stesso lato "
            f"e una produzione offensiva stabile"
        )
        if not math.isnan(lambda_fav):
            parts.append(f" (λ attacco ≈ {lambda_fav:.2f})")

        parts.append(
            f". L'avversaria ha uno storico di almeno {opp_matches} match "
            "in cui concede gol con una certa frequenza"
        )
        if not math.isnan(lambda_opp_ga):
            parts.append(f" (gol subiti medi ≈ {lambda_opp_ga:.2f})")

        if not math.isnan(lambda_total):
            parts.append(
                f". Il volume reti complessivo atteso è moderato/alto "
                f"(λ totale ≈ {lambda_total:.2f}), "
                "coerente con un esito in cui la favorita segna tra 1 e 4 gol."
            )
        else:
            parts.append(
                ". Il volume reti complessivo stimato è comunque compatibile "
                "con scenari in cui la favorita si ferma in un range 1–4 gol."
            )

        parts.append(
            " Nel backtest questa configurazione ha mostrato una frequenza "
            "di Multigol 1–4 della favorita intorno all'83%, su circa un quinto "
            "dei match complessivi nel range quota considerato."
        )

        message = "".join(parts)

        # ---------------------------
        # Suggerimenti di bet
        # ---------------------------
        bet_tags = ["BET_MG14_FAV"]  # multigol favorita 1–4

        bet_suggestions = build_bet_suggestions(
            bet_tags=bet_tags,
            severity="medium",
            side=side,
        )
        bets = bet_suggestions.get("suggestions", [])

        alert = BettingAlert(
            code="MG_FAVORITA_SIGNAL",
            severity="medium",
            message=message,
            tags=["MULTIGOL", "MG_FAVORITA_1_4"] + bet_tags,
            bets=bets,
            meta={
                "scenario": "MG_FAV_1_4_STAT_CHECK",
                "fav_side": fav_side,
                "fav_prob": fav_prob,
                "lambda_fav": lambda_fav,
                "lambda_opp_ga": lambda_opp_ga,
                "lambda_total": lambda_total,
                "fav_matches": fav_matches,
                "opp_matches": opp_matches,
                "bet_tags_raw": bet_tags,
                "bet_suggestions": bet_suggestions,
            },
        )
        alerts.append(alert)
        return alerts

    except Exception as e:
        print("    - ERRORE (MG_FAVORITA_SIGNAL):", e)
        raise e
