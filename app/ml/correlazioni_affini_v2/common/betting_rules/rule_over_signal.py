import math
from typing import List, Dict, Any

import pandas as pd

from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions


# ============================================================================
# HELPERS DI BASE
# ============================================================================

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
# COSTRUZIONE PROFILO OVER/UNDER (versione "intelligente" calibrata)
# ============================================================================

def build_over_profile(t0: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Costruisce un profilo sintetico per il mercato Over/Under,
    basato sui pattern emersi in STEP5:

    - OVER 1.5 globale ≈ 0.75
    - OVER 1.5 aumenta con:
        * cluster_ou15 in {1,2,3}
        * cluster_ou25 in {0,2,3,5}
        * lambda_total_form alta (>= 3.0, top decili)
        * tightness_index bassa (<= 0.56, primi decili)
        * soft_pO15 alta (>= 0.80)
    - UNDER forte quando:
        * lambda bassa (<= 2.0)
        * tightness alta (>= 0.70)
        * cluster_ou25 in {1,4}
        * cluster_ou15 in {0,5}
    """
    try:
        lam = float(_f(t0.get("lambda_total_form")))
        tight = float(_f(t0.get("tightness_index")))
        c15 = safe_int(t0.get("cluster_ou15"))
        c25 = safe_int(t0.get("cluster_ou25"))

        soft = ctx.get("soft_probs", {}) or {}
        pO15 = float(_f(soft.get("pO15")))
        pO25 = float(_f(soft.get("pO25")))

        # Se manca praticamente tutto, non ha senso proseguire
        if all(
            math.isnan(x)
            for x in [lam, tight, pO15, pO25]
        ):
            return {"has_profile": False}

        # cluster over/under "tipici" dai dati step5
        overish_ou15 = {1, 2, 3}   # 3 è il più over (~0.86), 1-2 molto buoni
        underish_ou15 = {0, 5}     # 0 e 5 sono le code più deboli

        overish_ou25 = {0, 2, 3, 5}  # 3 e 5 altissimi, 0-2 buoni
        underish_ou25 = {1, 4}       # cluster chiaramente più "sotto"

        c15_overish = c15 in overish_ou15
        c15_underish = c15 in underish_ou15
        c25_overish = c25 in overish_ou25
        c25_underish = c25 in underish_ou25

        # Livelli di λ dai decili:
        #   (0.099, 1.667]       → ~0.71 over 1.5
        #   (3.5, 3.833], (3.833,11] → ~0.82 over 1.5
        lam_low = (not math.isnan(lam)) and lam <= 2.0
        lam_mid = (not math.isnan(lam)) and 2.0 < lam < 3.0
        lam_high = (not math.isnan(lam)) and lam >= 3.0
        lam_very_high = (not math.isnan(lam)) and lam >= 3.5

        # tightness: più bassa = match più "aperti" (over)
        #   primi decili (~0.19–0.50) over 1.5 ~0.86
        #   ultimi decili (~0.71–0.79) over 1.5 ~0.65
        tight_loose = (not math.isnan(tight)) and tight <= 0.56
        tight_mid = (not math.isnan(tight)) and 0.56 < tight < 0.68
        tight_high = (not math.isnan(tight)) and tight >= 0.68

        # soft_pO15: top decili (>= 0.80) hanno over 1.5 ~0.83
        o15_high = (not math.isnan(pO15)) and pO15 >= 0.80
        o15_mid = (not math.isnan(pO15)) and 0.72 <= pO15 < 0.80
        o15_low = (not math.isnan(pO15)) and pO15 < 0.72

        # per pO25 usiamo soglie ragionevoli (globalmente intorno al 50–55%)
        o25_high = (not math.isnan(pO25)) and pO25 >= 0.62
        o25_mid = (not math.isnan(pO25)) and 0.52 <= pO25 < 0.62
        o25_low = (not math.isnan(pO25)) and pO25 < 0.52

        # -----------------------------
        #  OVER 1.5 molto forte
        # -----------------------------
        over15_strong = (
            o15_high
            and lam_high            # λ >= 3.0
            and not tight_high      # escludiamo i match più “stretti”
            and c15_overish
            and c25_overish
        )

        # OVER 1.5 buono/standard
        over15_good = (
            (o15_high or o15_mid)
            and (lam_mid or lam_high)
            and not (tight_high and c25_underish)
        )

        # contesto che spinge verso under sul 1.5
        over15_risk_under = (
            (o15_low and lam_low)
            or (tight_high and (c15_underish or c25_underish))
        )

        # -----------------------------
        #  OVER 2.5 forte / under
        # -----------------------------
        over25_strong = (
            (o25_high and lam_high and not tight_high and c25_overish)
            or (lam_very_high and tight_loose)
        )

        over25_good = (
            (o25_high or o25_mid)
            and (lam_mid or lam_high)
            and not (tight_high and c25_underish)
        )

        over25_risk_under = (
            o25_low
            or (lam_low and (c25_underish or tight_high))
        )

        under25_strong = (
            over25_risk_under
            and lam_low
            and (tight_high or c25_underish)
        )

        under15_strong = (
            over15_risk_under
            and (lam_low or tight_high)
            and (c15_underish or c25_underish)
        )

        # -----------------------------
        #  SHAPE sintetico (macro-classe)
        # -----------------------------
        if over15_strong and (over25_strong or over25_good):
            shape = "VERY_OVER"
        elif over15_good and (over25_good or o25_mid):
            shape = "OVER"
        elif under25_strong and under15_strong:
            shape = "VERY_UNDER"
        elif under25_strong or over25_risk_under or over15_risk_under:
            shape = "UNDER_EDGE"
        else:
            shape = "MIXED"

        return {
            "has_profile": True,
            "lambda": lam,
            "tight": tight,
            "cluster_ou15": c15,
            "cluster_ou25": c25,
            "pO15": pO15,
            "pO25": pO25,
            "lam_low": lam_low,
            "lam_mid": lam_mid,
            "lam_high": lam_high,
            "lam_very_high": lam_very_high,
            "tight_loose": tight_loose,
            "tight_mid": tight_mid,
            "tight_high": tight_high,
            "c15_overish": c15_overish,
            "c15_underish": c15_underish,
            "c25_overish": c25_overish,
            "c25_underish": c25_underish,
            "over15_strong": over15_strong,
            "over15_good": over15_good,
            "over15_risk_under": over15_risk_under,
            "over25_strong": over25_strong,
            "over25_good": over25_good,
            "over25_risk_under": over25_risk_under,
            "under25_strong": under25_strong,
            "under15_strong": under15_strong,
            "shape": shape,
        }

    except Exception:
        return {"has_profile": False}


# ============================================================================
# HELPERS TESTUALI
# ============================================================================

def _describe_lambda(lam: float) -> str:
    if math.isnan(lam):
        return "senza indicazioni chiare dal volume gol"
    if lam >= 3.5:
        return "con un volume reti storicamente molto alto (λ elevata)"
    if lam >= 3.0:
        return "con un volume reti mediamente alto"
    if lam >= 2.3:
        return "con un volume reti nella media o leggermente sopra"
    return "con un volume reti tendenzialmente contenuto"


def _describe_tight(tight: float) -> str:
    if math.isnan(tight):
        return ""
    if tight <= 0.56:
        return " e partite spesso aperte (tightness bassa)"
    if tight >= 0.70:
        return " ma con partite spesso bloccate (tightness alta)"
    return ""


def _build_over_positive_message(profile: Dict[str, Any]) -> str:
    lam = profile.get("lambda", math.nan)
    tight = profile.get("tight", math.nan)
    shape = profile.get("shape", "")

    parts: List[str] = []
    parts.append("Profilo molto favorevole ai gol: ")

    if shape == "VERY_OVER":
        parts.append(
            "lo storico delle partite affini mostra una frequenza molto elevata "
            "di Over 1.5 e Over 2.5. "
        )
    else:
        parts.append(
            "lo storico delle partite affini è spesso orientato verso l'Over, "
            "soprattutto sulla linea 1.5. "
        )

    parts.append("Il match appare " + _describe_lambda(lam))
    parts.append(_describe_tight(tight))
    parts.append(
        ". In ottica betting ha senso lavorare sui mercati Over, "
        "eventualmente combinando Over 1.5 e Over 2.5 a seconda delle quote. "
    )
    return "".join(parts)


def _build_under_positive_message(profile: Dict[str, Any]) -> str:
    lam = profile.get("lambda", math.nan)
    tight = profile.get("tight", math.nan)

    parts: List[str] = []
    parts.append("Profilo tendenzialmente chiuso: ")

    parts.append(
        "nei match affini le reti sono spesso poche, con una buona frequenza "
        "di Under 2.5 e in molti casi anche Under 1.5. "
    )

    if not math.isnan(lam):
        if lam <= 2.0:
            parts.append("La pericolosità offensiva complessiva risulta bassa (λ contenuta)")
        else:
            parts.append("La pericolosità offensiva resta comunque moderata")
    else:
        parts.append("La pericolosità offensiva stimata non è particolarmente elevata")

    if not math.isnan(tight):
        if tight >= 0.70:
            parts.append(
                " e le partite simili sono spesso bloccate e tattiche "
                "(tightness alta). "
            )
        else:
            parts.append(". ")

    parts.append(
        "In ottica betting questo contesto sostiene maggiormente i mercati Under, "
        "con priorità alla linea 2.5."
    )
    return "".join(parts)


def _build_balanced_message(profile: Dict[str, Any]) -> str:
    lam = profile.get("lambda", math.nan)
    tight = profile.get("tight", math.nan)

    parts: List[str] = []
    parts.append("Profilo equilibrato sul fronte gol: ")

    parts.append(
        "lo storico degli affini non spinge in modo estremo né verso Over né verso Under, "
        "ma evidenzia comunque una buona probabilità che si superi la linea 1.5. "
    )

    parts.append("Il match appare " + _describe_lambda(lam))
    parts.append(_describe_tight(tight))
    parts.append(
        ". In ottica betting l'Over 1.5 può essere la prima scelta, mentre sulla "
        "linea 2.5 è preferibile valutare anche il prezzo offerto dal bookmaker."
    )
    return "".join(parts)


def _build_under_edge_message(profile: Dict[str, Any]) -> str:
    lam = profile.get("lambda", math.nan)
    tight = profile.get("tight", math.nan)

    parts: List[str] = []
    parts.append("Leggera prevalenza di contesti chiusi: ")

    parts.append(
        "nei match storicamente simili si nota una certa tendenza a rimanere "
        "sotto la linea 2.5, pur senza un dominio assoluto dell'Under. "
    )

    if not math.isnan(lam):
        if lam <= 2.3:
            parts.append("Il volume reti atteso è piuttosto moderato")
        else:
            parts.append("Il volume reti atteso non è particolarmente esplosivo")

    parts.append(_describe_tight(tight))
    parts.append(
        ". In ottica betting l'Under 2.5 può essere una scelta interessante, "
        "ma va gestito con stake prudente."
    )
    return "".join(parts)


# ============================================================================
# REGOLA PRINCIPALE: OVER / UNDER SIGNAL (versione B)
# ============================================================================

def rule_over_signal(t0: pd.Series, ctx: Dict[str, Any]) -> List[BettingAlert]:
    """
    OVER_UNDER_SIGNAL (v2.0, versione “intelligente” calibrata)
    Regola che sintetizza il profilo Over/Under della partita, basandosi su:

      - soft_pO15 / soft_pO25 (affini)
      - lambda_total_form (volume gol atteso)
      - tightness_index (apertura/chiusura del match)
      - cluster_ou15 / cluster_ou25 (tipologia di partita)

    e produce alert con suggerimenti su:
      - BET_OVER15
      - BET_OVER25
      - BET_UNDER25
      - BET_UNDER15

    Ordine gerarchico scenari (B):
      STRONG_OVER → OVER_EDGE → STRONG_UNDER → UNDER_EDGE → nessun alert
    """
    try:
        alerts: List[BettingAlert] = []

        print("    - Costruzione profilo Over/Under (v2.0)...")

        profile = build_over_profile(t0, ctx)
        if not profile.get("has_profile"):
            return alerts

        shape = profile["shape"]
        over15_strong = profile["over15_strong"]
        over15_good = profile["over15_good"]
        over15_risk_under = profile["over15_risk_under"]
        over25_strong = profile["over25_strong"]
        over25_good = profile["over25_good"]
        over25_risk_under = profile["over25_risk_under"]
        under25_strong = profile["under25_strong"]
        under15_strong = profile["under15_strong"]

        pO15 = profile["pO15"]
        pO25 = profile["pO25"]
        lam = profile["lambda"]
        tight = profile["tight"]
        c15 = profile["cluster_ou15"]
        c25 = profile["cluster_ou25"]

        # Per i mercati Over/Under il lato non conta:
        side = "home"

        print("    - Valutazione scenario Over/Under (v2.0)...")

        # ================================
        # 1) STRONG_OVER (scenario più aggressivo)
        # ================================
        if shape == "VERY_OVER" and over15_strong:
            bet_tags = ["BET_OVER15"]

            # Over 2.5 solo dove il contesto è davvero robusto
            if over25_strong or (over25_good and not over25_risk_under):
                bet_tags.append("BET_OVER25")

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="high",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_over_positive_message(profile)

            alert = BettingAlert(
                code="OVER_UNDER_SIGNAL",
                severity="high",
                message=message,
                tags=["OVER_UNDER", "STRONG_OVER"] + bet_tags,
                bets=bets,
                meta={
                    "scenario": "STRONG_OVER",
                    "shape": shape,
                    "pO15": pO15,
                    "pO25": pO25,
                    "lambda": lam,
                    "tightness_index": tight,
                    "cluster_ou15": c15,
                    "cluster_ou25": c25,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 2) OVER_EDGE (Over buono ma non estremo)
        # ================================
        if (shape in {"VERY_OVER", "OVER", "MIXED"}) and (over15_good and not over15_risk_under):
            bet_tags = ["BET_OVER15"]

            # Over 2.5 solo se non ci sono segnali di rischio evidente
            if over25_good and not over25_risk_under and pO25 >= 0.55:
                bet_tags.append("BET_OVER25")

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="medium",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_balanced_message(profile)

            alert = BettingAlert(
                code="OVER_UNDER_SIGNAL",
                severity="medium",
                message=message,
                tags=["OVER_UNDER", "OVER_EDGE"] + bet_tags,
                bets=bets,
                meta={
                    "scenario": "OVER_EDGE",
                    "shape": shape,
                    "pO15": pO15,
                    "pO25": pO25,
                    "lambda": lam,
                    "tightness_index": tight,
                    "cluster_ou15": c15,
                    "cluster_ou25": c25,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 3) STRONG_UNDER
        # ================================
        if shape == "VERY_UNDER" and (under25_strong or over25_risk_under):
            bet_tags = ["BET_UNDER25"]

            # Under 1.5 solo in contesti davvero chiusi
            if under15_strong:
                bet_tags.append("BET_UNDER15")

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="high",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_under_positive_message(profile)

            alert = BettingAlert(
                code="OVER_UNDER_SIGNAL",
                severity="high",
                message=message,
                tags=["OVER_UNDER", "STRONG_UNDER"] + bet_tags,
                bets=bets,
                meta={
                    "scenario": "STRONG_UNDER",
                    "shape": shape,
                    "pO15": pO15,
                    "pO25": pO25,
                    "lambda": lam,
                    "tightness_index": tight,
                    "cluster_ou15": c15,
                    "cluster_ou25": c25,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 4) UNDER_EDGE (bias under ma non estremo)
        # ================================
        if shape in {"UNDER_EDGE", "MIXED"} and (
            under25_strong
            or (over25_risk_under and not over15_strong)
        ):
            bet_tags = ["BET_UNDER25"]

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="medium",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_under_edge_message(profile)

            alert = BettingAlert(
                code="OVER_UNDER_SIGNAL",
                severity="medium",
                message=message,
                tags=["OVER_UNDER", "UNDER_EDGE"] + bet_tags,
                bets=bets,
                meta={
                    "scenario": "UNDER_EDGE",
                    "shape": shape,
                    "pO15": pO15,
                    "pO25": pO25,
                    "lambda": lam,
                    "tightness_index": tight,
                    "cluster_ou15": c15,
                    "cluster_ou25": c25,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 5) NESSUNO SCENARIO CHIARO
        # ================================
        print("    - Profilo Over/Under neutro; nessun alert generato (v2.0).")
        return alerts

    except Exception as e:
        print("    - ERRORE (OVER_UNDER_SIGNAL v2.0):", e)
        raise e