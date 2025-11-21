import pandas as pd
import math
from typing import List, Dict, Any
from .betting_alert_model import BettingAlert
from .bet_translation_engine import build_bet_suggestions


def safe_int(x, default: int = -1) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


# ============================================================================
# COSTRUZIONE PROFILO FAVORITA
# ============================================================================

def build_favorite_profile(t0: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Costruisce un profilo sintetico della favorita (home/away) a partire da:
      - picchetto tecnico (pic_p1 / pic_p2)
      - soft probabilities (sp1 / sp2)
      - tightness_index
      - lambda_total_form
      - cluster_1x2, cluster_ou25
    """
    try:
        bk_p1 = float(_f(t0.get("bk_p1")))
        bk_p2 = float(_f(t0.get("bk_p2")))
        bk_px = float(_f(t0.get("bk_px")))
        pic_p1 = float(_f(t0.get("pic_p1")))
        pic_p2 = float(_f(t0.get("pic_p2")))
        tight = float(_f(t0.get("tightness_index")))
        lam = float(_f(t0.get("lambda_total_form")))
        c1 = safe_int(t0.get("cluster_1x2"))
        c25 = safe_int(t0.get("cluster_ou25"))

        soft = ctx.get("soft_probs", {}) or {}
        sp1 = float(_f(soft.get("p1")))
        sp2 = float(_f(soft.get("p2")))

        # Se mancano le quote, non ha senso profilarla
        if any(math.isnan(x) for x in [bk_p1, bk_px, bk_p2]):
            return {"has_favorite": False}

        # NB: qui assumo che t0 contenga le probabilità implicite (non le quote)
        #     e che valore più alto = più favorita.
        is_home_fav = bk_p1 > max(bk_px, bk_p2)
        is_away_fav = bk_p2 > max(bk_px, bk_p1)

        if not (is_home_fav or is_away_fav):
            return {"has_favorite": False}

        fav_side = "home" if is_home_fav else "away"
        pic_fav = pic_p1 if is_home_fav else pic_p2
        soft_fav = sp1 if is_home_fav else sp2

        # -----------------------------
        #  Profili base favorita
        # -----------------------------
        very_strong = (pic_fav >= 0.65 and soft_fav >= 0.70 and c1 in {0, 4})
        strong = (pic_fav >= 0.55 and soft_fav >= 0.55 and c1 in {0, 3, 4})
        weak = (pic_fav < 0.45 or soft_fav < 0.40 or c1 in {1, 2, 5})

        if very_strong:
            strength = "VERY_STRONG"
        elif strong:
            strength = "STRONG"
        elif weak:
            strength = "WEAK"
        else:
            strength = "MEDIUM"

        # -----------------------------
        #  Profilo GOAL favorita
        # -----------------------------
        no_goal_risk = (
            (pic_fav <= 0.40) or
            (tight >= 0.70) or
            (lam < 2.0) or
            (c25 in {1, 4}) or
            (soft_fav < 0.20)
        )

        goal_high = (
            (pic_fav >= 0.55) and
            (tight <= 0.60) and
            (lam >= 2.5) and
            (c1 in {0, 4}) and
            (c25 in {2, 3})
        )

        # -----------------------------
        #  Profilo WIN favorita
        # -----------------------------
        win_risk = (
            (pic_fav < 0.45) or
            (tight >= 0.70) or
            (lam < 2.0) or
            (soft_fav < 0.35) or
            (c1 in {1, 2, 5})
        )

        win_high = (
            (pic_fav >= 0.60) and
            (tight <= 0.60) and
            (lam >= 2.6) and
            (soft_fav >= 0.65) and
            (c1 in {0, 4})
        )

        win_standard = (
            (pic_fav >= 0.50) or
            (lam >= 2.3) or
            (soft_fav >= 0.50) or
            (tight <= 0.65)
        )

        # -----------------------------
        #  Profilo MULTIGOAL favorita
        #   (1–3, 1–4, 1–5) via lambda_total_form
        # -----------------------------
        if lam >= 3.5:
            multigoal_band = "1_5"  # favorita spesso in 1–5 gol
        elif lam >= 3.0:
            multigoal_band = "1_4"  # favorita spesso in 1–4 gol
        elif lam >= 2.4:
            multigoal_band = "1_3"  # favorita spesso in 1–3 gol
        else:
            multigoal_band = None

        return {
            "has_favorite": True,
            "side": fav_side,
            "strength": strength,
            "pic_fav": pic_fav,
            "soft_fav": soft_fav,
            "tight": tight,
            "lambda": lam,
            "cluster_1x2": c1,
            "cluster_ou25": c25,
            "no_goal_risk": no_goal_risk,
            "goal_high": goal_high,
            "win_risk": win_risk,
            "win_high": win_high,
            "win_standard": win_standard,
            "multigoal_band": multigoal_band,
        }
    except Exception:
        return {"has_favorite": False}


# ============================================================================
# HELPERS PER MESSAGGI TESTUALI (NON TECNICI)
# ============================================================================

def _describe_strength(strength: str, fav_label: str) -> str:
    if strength == "VERY_STRONG":
        return f"una {fav_label} molto solida e nettamente avanti"
    if strength == "STRONG":
        return f"una {fav_label} solida e tendenzialmente favorita"
    if strength == "MEDIUM":
        return f"una {fav_label} leggermente avanti ma non blindata"
    return f"una {fav_label} favorita solo sulla carta"


def _describe_multigoal(multigoal_band: str | None, fav_label: str) -> str:
    if multigoal_band == "1_3":
        return f", con uno scenario frequente in cui la {fav_label} si ferma tra 1 e 3 gol"
    if multigoal_band == "1_4":
        return f", con partite spesso ricche dove la {fav_label} si muove in un range 1–4 gol"
    if multigoal_band == "1_5":
        return f", con molti precedenti dove la {fav_label} arriva facilmente anche a più gol (1–5)"
    return ""


def _build_negative_message(
    fav_label: str,
    strength: str,
    win_risk: bool,
    no_goal_risk: bool,
) -> str:
    parts: List[str] = []
    parts.append(f"Profilo negativo per la {fav_label}: ")

    if strength == "WEAK":
        parts.append("la squadra favorita appare fragile e poco affidabile. ")
    elif strength == "MEDIUM":
        parts.append("il vantaggio della favorita è molto sottile. ")
    else:
        parts.append("nonostante lo status di favorita, il contesto non è rassicurante. ")

    if win_risk:
        parts.append("C'è un rischio concreto che non riesca a portare a casa il risultato pieno. ")

    if no_goal_risk:
        parts.append("In molte partite simili la favorita fatica addirittura a trovare il gol. ")

    parts.append(
        "Meglio proteggersi con esiti a favore dell'altra squadra "
        "o valutare contesti da pochi gol."
    )
    return "".join(parts)


def _build_positive_message(
    fav_label: str,
    strength: str,
    multigoal_band: str | None,
) -> str:
    parts: List[str] = []
    parts.append(f"Profilo molto positivo per la {fav_label}: ")
    parts.append(_describe_strength(strength, fav_label).capitalize() + ". ")
    parts.append(
        "Lo storico delle partite simili è spesso favorevole sia alla vittoria "
        "sia ad almeno un gol della favorita"
    )
    parts.append(_describe_multigoal(multigoal_band, fav_label))
    parts.append(
        ". In ottica betting, è un quadro che sostiene segno favorevole e mercati "
        "legati ai gol della squadra favorita."
    )
    return "".join(parts)


def _build_multigoal_only_message(
    fav_label: str,
    strength: str,
    multigoal_band: str | None,
    win_risk: bool,
) -> str:
    parts: List[str] = []
    parts.append(f"Profilo multigol per la {fav_label}: ")
    parts.append(_describe_strength(strength, fav_label).capitalize() + ". ")
    parts.append(
        "Le partite storicamente simili mostrano spesso una favorita che trova "
        "la porta più di una volta"
    )
    parts.append(_describe_multigoal(multigoal_band, fav_label))
    parts.append(". ")

    if win_risk:
        parts.append(
            "Il risultato finale non è però scontato, quindi è prudente lavorare "
            "più sui gol squadra che sul segno fisso."
        )
    else:
        parts.append(
            "Il risultato resta tendenzialmente favorevole, ma non completamente "
            "blindato sul segno fisso."
        )

    return "".join(parts)


def _build_standard_message(
    fav_label: str,
    strength: str,
    win_risk: bool,
    goal_high: bool,
    multigoal_band: str | None,
) -> str:
    parts: List[str] = []
    parts.append(f"Profilo equilibrato per la {fav_label}: ")
    parts.append(_describe_strength(strength, fav_label).capitalize() + ". ")

    if goal_high or multigoal_band:
        parts.append("È ragionevole aspettarsi almeno un gol dalla favorita")
        parts.append(_describe_multigoal(multigoal_band, fav_label))
        parts.append(". ")

    if win_risk:
        parts.append(
            "Restano però margini di incertezza sull'esito finale, motivo per cui "
            "può essere preferibile proteggerla con la doppia chance. "
        )
    else:
        parts.append(
            "Il quadro sul risultato è complessivamente favorevole, pur senza "
            "essere totalmente blindato. "
        )

    parts.append(
        "Scenario adatto a combinare mercati sui gol della favorita e coperture "
        "sull'esito finale."
    )
    return "".join(parts)


# ============================================================================
# REGOLA PRINCIPALE
# ============================================================================

def rule_favorite_profile_signal(t0: pd.Series, ctx: Dict[str, Any]) -> List[BettingAlert]:
    """
    FAVORITE_PROFILE_SIGNAL (v2.0, versione pulita)
    Regola gerarchica che sintetizza:
      - se la favorita vince / è a rischio non vittoria
      - se la favorita segna / è a rischio no-goal
      - se la favorita ha profilo multigol (1–3, 1–4, 1–5)
    e propone le giocate più coerenti.

    Struttura:
    - bet_tags_raw: lista di codici tecnici (BET_*)
    - bet_suggestions: oggetto ricco con traduzioni, priorità e motivazioni
    - bets: lista di oggetti suggerimento (pronta per il FE)
    """
    try:
        alerts: List[BettingAlert] = []

        print("    - Costruzione profilo favorita (v2.0)...")

        profile = build_favorite_profile(t0, ctx)
        if not profile.get("has_favorite"):
            return alerts

        side = profile["side"]
        strength = profile["strength"]
        pic_fav = profile["pic_fav"]
        soft_fav = profile["soft_fav"]
        tight = profile["tight"]
        lam = profile["lambda"]
        c1 = profile["cluster_1x2"]
        c25 = profile["cluster_ou25"]
        no_goal_risk = profile["no_goal_risk"]
        goal_high = profile["goal_high"]
        win_risk = profile["win_risk"]
        win_high = profile["win_high"]
        win_standard = profile["win_standard"]
        multigoal_band = profile["multigoal_band"]

        is_home = (side == "home")
        fav_label = "squadra di casa" if is_home else "squadra ospite"

        dc_fav = "1X" if is_home else "X2"
        dc_opp = "X2" if is_home else "1X"
        win_code = "1" if is_home else "2"

        print("    - Valutazione scenario profilo favorita (v2.0)...")

        has_multigoal = multigoal_band is not None
        multi_13 = multigoal_band == "1_3"
        multi_14 = multigoal_band == "1_4"
        multi_15 = multigoal_band == "1_5"

        # ================================
        # 1) PROFILO NEGATIVO / TRAP
        # ================================
        if win_risk and no_goal_risk:
            # Pulito: via segnali deboli (UNDER15, LAY_FAV_SEGNA)
            bet_tags = [
                "BET_NO_BET_STRONG",   # indicazione gestionale (no-bet forte)
                "BET_DC_OPPOSITE",     # doppia chance contro favorita
                "BET_UNDER_1_5_FAV",   # favorita max 1 gol
            ]

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="low",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_negative_message(
                fav_label=fav_label,
                strength=strength,
                win_risk=win_risk,
                no_goal_risk=no_goal_risk,
            )

            alert = BettingAlert(
                code="FAVORITE_PROFILE_SIGNAL",
                severity="low",
                message=message,
                tags=["FAV_PROFILE", "NEGATIVE"] + bet_tags,
                bets=bets,
                meta={
                    "side": side,
                    "strength": strength,
                    "pic_fav": pic_fav,
                    "soft_fav": soft_fav,
                    "tight": tight,
                    "lambda": lam,
                    "cluster_1x2": c1,
                    "cluster_ou25": c25,
                    "multigoal_band": multigoal_band,
                    "scenario": "NEGATIVE_RISK_NOWIN_NOGOAL",
                    "suggested_dc": dc_opp,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        print("    - Favorita non in scenario negativo/trap.")

        # ================================
        # 2) PROFILO MOLTO POSITIVO
        # ================================
        if win_high and goal_high:
            bet_tags = [
                f"BET_{win_code}",     # segno 1 o 2
                "BET_DC_FAV",          # doppia chance a favore
                "BET_FAV_SEGNA",       # favorita segna
                "BET_FAV_OVER_0_5",    # favorita almeno 1 gol
            ]

            # multigol solo dove ha mostrato buona robustezza
            if multi_13:
                bet_tags.append("BET_FAV_1_3")
            elif multi_14:
                bet_tags += ["BET_FAV_1_3", "BET_FAV_1_4"]
            elif multi_15:
                bet_tags += ["BET_FAV_1_3", "BET_FAV_1_4", "BET_FAV_1_5"]

            # (Asian -0.25 rimossa: poco chiaro per l'utente e fuori backtest)

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="high",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_positive_message(
                fav_label=fav_label,
                strength=strength,
                multigoal_band=multigoal_band,
            )

            alert = BettingAlert(
                code="FAVORITE_PROFILE_SIGNAL",
                severity="high",
                message=message,
                tags=["FAV_PROFILE", "POSITIVE", "HIGH_CONFIDENCE"] + bet_tags,
                bets=bets,
                meta={
                    "side": side,
                    "strength": strength,
                    "pic_fav": pic_fav,
                    "soft_fav": soft_fav,
                    "tight": tight,
                    "lambda": lam,
                    "cluster_1x2": c1,
                    "cluster_ou25": c25,
                    "multigoal_band": multigoal_band,
                    "scenario": "STRONG_WIN_AND_GOAL",
                    "suggested_dc": dc_fav,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 3) PROFILO MULTIGOL FAVORITA
        # ================================
        if has_multigoal:
            bet_tags = [
                "BET_DC_FAV",
                "BET_FAV_OVER_0_5",
            ]

            if multi_13:
                bet_tags.append("BET_FAV_1_3")
            elif multi_14:
                bet_tags += ["BET_FAV_1_3", "BET_FAV_1_4"]
            elif multi_15:
                bet_tags += ["BET_FAV_1_3", "BET_FAV_1_4", "BET_FAV_1_5"]

            # segno se forte e senza grossi rischi
            if strength in {"VERY_STRONG", "STRONG"} and not win_risk:
                bet_tags.append(f"BET_{win_code}")

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="medium",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_multigoal_only_message(
                fav_label=fav_label,
                strength=strength,
                multigoal_band=multigoal_band,
                win_risk=win_risk,
            )

            alert = BettingAlert(
                code="FAVORITE_PROFILE_SIGNAL",
                severity="medium",
                message=message,
                tags=["FAV_PROFILE", "MULTIGOAL"] + bet_tags,
                bets=bets,
                meta={
                    "side": side,
                    "strength": strength,
                    "pic_fav": pic_fav,
                    "soft_fav": soft_fav,
                    "tight": tight,
                    "lambda": lam,
                    "cluster_1x2": c1,
                    "cluster_ou25": c25,
                    "multigoal_band": multigoal_band,
                    "scenario": f"MULTIGOAL_{multigoal_band}",
                    "suggested_dc": dc_fav,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        # ================================
        # 4) PROFILO INTERMEDIO / STANDARD
        # ================================
        if win_standard or goal_high or has_multigoal:
            bet_tags = [
                "BET_DC_FAV",
                "BET_FAV_OVER_0_5",
            ]

            if strength in {"VERY_STRONG", "STRONG"} and not win_risk:
                bet_tags.append(f"BET_{win_code}")

            if multigoal_band == "1_3":
                bet_tags.append("BET_FAV_1_3")

            bet_suggestions = build_bet_suggestions(
                bet_tags=bet_tags,
                severity="medium",
                side=side,
            )
            bets = bet_suggestions.get("suggestions", [])

            message = _build_standard_message(
                fav_label=fav_label,
                strength=strength,
                win_risk=win_risk,
                goal_high=goal_high,
                multigoal_band=multigoal_band,
            )

            alert = BettingAlert(
                code="FAVORITE_PROFILE_SIGNAL",
                severity="medium",
                message=message,
                tags=["FAV_PROFILE", "BALANCED"] + bet_tags,
                bets=bets,
                meta={
                    "side": side,
                    "strength": strength,
                    "pic_fav": pic_fav,
                    "soft_fav": soft_fav,
                    "tight": tight,
                    "lambda": lam,
                    "cluster_1x2": c1,
                    "cluster_ou25": c25,
                    "multigoal_band": multigoal_band,
                    "scenario": "STANDARD_PROFILE",
                    "suggested_dc": dc_fav,
                    "bet_tags_raw": bet_tags,
                    "bet_suggestions": bet_suggestions,
                },
            )
            alerts.append(alert)
            return alerts

        print("    - Profilo neutro per la favorita; nessun alert generato (v2.0).")
        return alerts

    except Exception as e:
        print("    - ERRORE (FAVORITE_PROFILE_SIGNAL v2.0):", e)
        raise e