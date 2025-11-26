import math
from typing import List, Dict, Any
import pandas as pd

from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions


# ---------------------------------------------------------------------------
# HELPERS DI BASE (già presenti, li metto solo per contesto)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HELPERS POISSON
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    if lam < 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0


def _poisson_p_range(lam: float, k_min: int, k_max: int) -> float:
    """
    P(k_min <= X <= k_max) per X ~ Poisson(lam)
    """
    if math.isnan(lam) or lam < 0:
        return math.nan
    s = 0.0
    for k in range(k_min, k_max + 1):
        s += _poisson_pmf(k, lam)
    return s


def _poisson_p_ge(lam: float, k_min: int) -> float:
    """
    P(X >= k_min) = 1 - P(X <= k_min-1)
    """
    if math.isnan(lam) or lam < 0:
        return math.nan
    s = 0.0
    for k in range(0, k_min):
        s += _poisson_pmf(k, lam)
    return max(0.0, 1.0 - s)


# ---------------------------------------------------------------------------
# PROFILO POISSON MG 1–4 V2D
# ---------------------------------------------------------------------------

def build_mg14_poisson_profile_v2d(t0: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Profilo Poisson per MULTIGOL FAVORITA 1–4 (versione V2D).

    Usa SOLO info prematch:

      - bk_p1 / bk_p2                 → favorita + range quota
      - lambda_home_form / lambda_away_form
      - home_form_ga_avg_lastN / away_form_ga_avg_lastN

    Step:

      1) Determina favorita (home/away) e prob implicita
      2) Verifica contesto quota: fav_prob ∈ [1/1.90, 1/1.15]
      3) Stima:
            λ_att_fav = lambda_home_form / lambda_away_form
            λ_def_opp = form_ga_avg_lastN dell’avversaria
         e combina:
            λ_fav = 0.6 * λ_att_fav + 0.4 * λ_def_opp
      4) Calcola via Poisson:
            P_MG14 = P(1 ≤ X ≤ 4)
            P_MG15 = P(X ≥ 1)
            P_Ge5  = P(X ≥ 5)
      5) Applica soglie V2D (calibrate dal backtest grid):

         - storico minimo:
             * fav_matches >= 5
             * opp_matches >= 5
         - λ_fav “ragionevole”:
             * 1.4 <= λ_fav <= 3.3
         - Probabilità multigol:
             * P_MG14 >= 0.80
             * P_MG15 >= 0.89
             * P_Ge5  <= 0.25   (filtra contesti troppo esplosivi)

    Output:
        {
          "has_profile": bool,
          "in_context": bool,
          "checklist_pass": bool,

          "fav_side": "home"/"away"/None,
          "fav_prob": float,
          "eq_odds": float | nan,

          "lambda_att": float,
          "lambda_def_opp": float,
          "lambda_fav": float,

          "fav_matches": int,
          "opp_matches": int,

          "P_MG14": float,
          "P_MG15": float,
          "P_Ge5": float,
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

        # Range prob equivalente a quota 1.15–1.90
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
                "eq_odds": (1.0 / fav_prob) if (fav_prob and fav_prob > 0) else math.nan,
            }

        # -------------------------
        # Estrazione feature di forma
        # -------------------------
        if fav_side == "home":
            fav_matches = safe_int(t0.get("home_form_matches_lastN"), 0)
            lambda_att = _f(t0.get("lambda_home_form"))
            fav_gf_avg = _f(t0.get("home_form_gf_avg_lastN"))

            opp_matches = safe_int(t0.get("away_form_matches_lastN"), 0)
            lambda_def_opp = _f(t0.get("away_form_ga_avg_lastN"))
        else:
            fav_matches = safe_int(t0.get("away_form_matches_lastN"), 0)
            lambda_att = _f(t0.get("lambda_away_form"))
            fav_gf_avg = _f(t0.get("away_form_gf_avg_lastN"))

            opp_matches = safe_int(t0.get("home_form_matches_lastN"), 0)
            lambda_def_opp = _f(t0.get("home_form_ga_avg_lastN"))

        # fallback se manca la lambda attacco: usa gf_avg
        if math.isnan(lambda_att) and not math.isnan(fav_gf_avg):
            lambda_att = fav_gf_avg

        # se senza numeri, niente profilo
        if math.isnan(lambda_att) or math.isnan(lambda_def_opp):
            return {
                "has_profile": False,
                "in_context": True,
                "checklist_pass": False,
                "fav_side": fav_side,
                "fav_prob": fav_prob,
                "eq_odds": (1.0 / fav_prob) if (fav_prob and fav_prob > 0) else math.nan,
            }

        # Combinazione Poisson “attacco + difesa opp”
        lambda_fav = 0.6 * lambda_att + 0.4 * lambda_def_opp

        # -------------------------
        # Poisson: P_MG14, P_MG15, P_Ge5
        # -------------------------
        P_MG14 = _poisson_p_range(lambda_fav, 1, 4)   # 1–4 gol favorita
        P_MG15 = _poisson_p_ge(lambda_fav, 1)         # almeno 1 gol
        P_Ge5  = _poisson_p_ge(lambda_fav, 5)         # 5+ gol

        # -------------------------
        # Regole V2D
        # -------------------------

        fav_enough = fav_matches >= 5
        opp_enough = opp_matches >= 5

        lambda_ok = (
            not math.isnan(lambda_fav)
            and 1.4 <= lambda_fav <= 3.3
        )

        prob_ok = (
            (not math.isnan(P_MG14)) and P_MG14 >= 0.80 and
            (not math.isnan(P_MG15)) and P_MG15 >= 0.89 and
            (not math.isnan(P_Ge5))  and P_Ge5  <= 0.25
        )

        checklist_pass = (
            fav_enough
            and opp_enough
            and lambda_ok
            and prob_ok
        )

        eq_odds = (1.0 / fav_prob) if (fav_prob and fav_prob > 0) else math.nan

        return {
            "has_profile": True,
            "in_context": True,
            "checklist_pass": checklist_pass,
            "fav_side": fav_side,
            "fav_prob": fav_prob,
            "eq_odds": eq_odds,
            "lambda_att": lambda_att,
            "lambda_def_opp": lambda_def_opp,
            "lambda_fav": lambda_fav,
            "fav_matches": fav_matches,
            "opp_matches": opp_matches,
            "P_MG14": P_MG14,
            "P_MG15": P_MG15,
            "P_Ge5": P_Ge5,
            "fav_enough": fav_enough,
            "opp_enough": opp_enough,
            "lambda_ok": lambda_ok,
            "prob_ok": prob_ok,
        }

    except Exception:
        return {
            "has_profile": False,
            "in_context": False,
            "checklist_pass": False,
        }
    
# ============================================================================
# REGOLA PRINCIPALE: MULTIGOL FAVORITA 1–4 (POISSON V2D)
# ============================================================================

def rule_mg_fav_signal_v2d(t0: pd.Series, ctx: Dict[str, Any]) -> List[BettingAlert]:
    """
    MG_FAVORITA_SIGNAL_V2D (Poisson-based, grid v2D)

    Regola che individua i match in cui la favorita (home/away),
    con quota nel range 1.15–1.90, ha un profilo Poisson favorevole
    al Multigol 1–4 (e 1+ gol) sulla base di:

      - λ_att (lambda_home_form / lambda_away_form)
      - λ_def_opp (gol subiti medi recenti avversaria)
      - λ_fav = 0.6*λ_att + 0.4*λ_def_opp
      - P_MG14 = P(1–4 gol favorita)
      - P_MG15 = P(1+ gol favorita)
      - P_Ge5  = P(5+ gol favorita)

    Soglie V2D (dalla grid search):

      - fav_matches >= 5
      - opp_matches >= 5
      - 1.4 <= λ_fav <= 3.3
      - P_MG14 >= 0.80
      - P_MG15 >= 0.89
      - P_Ge5  <= 0.25

    Nel backtest:

      - copertura ≈ 44%
      - MG 1–4 ≈ 83.3%
      - MG 1–5 ≈ 90.2%

    → filtro “robusto” e ad ampia copertura.
    """
    alerts: List[BettingAlert] = []

    try:
        print("    - Costruzione profilo MG favorita (Poisson v2D)...")

        prof = build_mg14_poisson_profile_v2d(t0, ctx)

        if not prof.get("has_profile", False):
            print("    - [MG v2D] Nessun profilo valido (has_profile=False).")
            return alerts

        if not prof.get("in_context", False):
            print("    - [MG v2D] Match fuori contesto quota 1.15–1.90.")
            return alerts

        if not prof.get("checklist_pass", False):
            # log di debug compatto
            print("    - [MG v2D] CHECKLIST FALLITA:")
            print(
                f"        fav_side={prof.get('fav_side')}, "
                f"eq_odds≈{prof.get('eq_odds', math.nan):.2f}, "
                f"λ_att={prof.get('lambda_att', math.nan):.3f}, "
                f"λ_def_opp={prof.get('lambda_def_opp', math.nan):.3f}, "
                f"λ_fav={prof.get('lambda_fav', math.nan):.3f}, "
                f"P_MG14={prof.get('P_MG14', math.nan):.3f}, "
                f"P_MG15={prof.get('P_MG15', math.nan):.3f}, "
                f"P_Ge5={prof.get('P_Ge5', math.nan):.3f}, "
                f"fav_matches={prof.get('fav_matches')}, "
                f"opp_matches={prof.get('opp_matches')}"
            )
            return alerts

        # Se siamo qui, checklist v2D PASSATA
        fav_side = prof.get("fav_side", "home")
        fav_prob = prof.get("fav_prob", math.nan)
        eq_odds = prof.get("eq_odds", math.nan)
        lambda_att = prof.get("lambda_att", math.nan)
        lambda_def_opp = prof.get("lambda_def_opp", math.nan)
        lambda_fav = prof.get("lambda_fav", math.nan)
        fav_matches = prof.get("fav_matches", 0)
        opp_matches = prof.get("opp_matches", 0)
        P_MG14 = prof.get("P_MG14", math.nan)
        P_MG15 = prof.get("P_MG15", math.nan)
        P_Ge5 = prof.get("P_Ge5", math.nan)

        # lato per i bet tag
        side = fav_side

        # ---------------------------
        # Messaggio testuale
        # ---------------------------
        parts: List[str] = []
        parts.append("Profilo Poisson favorevole al Multigol 1–4 sulla favorita: ")

        if not math.isnan(eq_odds):
            parts.append(
                f"la squadra favorita è stimata intorno a quota {eq_odds:.2f}, "
            )
        else:
            parts.append(
                "la squadra favorita è nel range di quota 1.15–1.90, "
            )

        parts.append(
            f"con almeno {fav_matches} partite recenti sullo stesso lato "
            f"e una produzione offensiva che porta a una λ attesa≈{lambda_att:.2f}. "
        )
        parts.append(
            f"L'avversaria ha almeno {opp_matches} partite storiche, "
            f"con una difesa che concede in media≈{lambda_def_opp:.2f} gol. "
        )
        parts.append(
            f"Combinando questi dati, la λ attesa della favorita è≈{lambda_fav:.2f}, "
            f"da cui il modello Poisson stima P(MG 1–4)≈{P_MG14:.3f} "
            f"e P(1+ gol favorita)≈{P_MG15:.3f}, "
            f"con probabilità di goleada (5+ gol) contenuta a≈{P_Ge5:.3f}. "
        )
        parts.append(
            "Nel backtest questa configurazione (versione v2D) ha mostrato una frequenza "
            "di Multigol 1–4 intorno all'83% e di almeno un gol favorita oltre il 90%, "
            "su circa il 40–45% dei match nel range quota considerato."
        )

        message = "".join(parts)

        # ---------------------------
        # Suggerimenti di bet
        # ---------------------------
        bet_tags = ["BET_MG14_FAV"]

        # opzionale: se P_MG15 molto alta, puoi aggiungere anche un tag per 1+ gol
        if not math.isnan(P_MG15) and P_MG15 >= 0.93:
            bet_tags.append("BET_MG15_FAV")

        bet_suggestions = build_bet_suggestions(
            bet_tags=bet_tags,
            severity="medium",
            side=side,
        )
        bets = bet_suggestions.get("suggestions", [])

        alert = BettingAlert(
            code="MG_FAVORITA_SIGNAL_V2D",
            severity="medium",
            message=message,
            tags=["MULTIGOL", "MG_FAVORITA_1_4", "POISSON_V2D"] + bet_tags,
            bets=bets,
            meta={
                "scenario": "MG_FAV_1_4_POISSON_V2D",
                "fav_side": fav_side,
                "fav_prob": fav_prob,
                "eq_odds": eq_odds,
                "lambda_att": lambda_att,
                "lambda_def_opp": lambda_def_opp,
                "lambda_fav": lambda_fav,
                "fav_matches": fav_matches,
                "opp_matches": opp_matches,
                "P_MG14": P_MG14,
                "P_MG15": P_MG15,
                "P_Ge5": P_Ge5,
                "bet_tags_raw": bet_tags,
                "bet_suggestions": bet_suggestions,
            },
        )
        alerts.append(alert)
        return alerts

    except Exception as e:
        print("    - ERRORE (MG_FAVORITA_SIGNAL_V2D):", e)
        raise e