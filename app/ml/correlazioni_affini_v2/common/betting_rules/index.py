from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import math
import numbers
from .betting_alert_model import BettingAlert
from .rule_favorite_profile_signal import rule_favorite_profile_signal
from .rule_over_signal import rule_over_signal
from .rule_mg_signal import rule_mg_fav_signal_v1


# ============================================================
#  BASE STRUCT
# ============================================================

@dataclass
class RuleFn:
    __call__: Callable[[pd.Series, Dict[str, Any]], List[BettingAlert]]

REGISTERED_RULES = [
    rule_favorite_profile_signal,
    rule_over_signal,
    rule_mg_fav_signal_v1
]

from .rule_mg_optimum_from_decision import build_mg_fav_optimum_alert_from_decision

def evaluate_decision_rules(decision: Dict[str, Any], meta: Optional[Dict[str, Any]] = None):
    """
    Regole basate sul JSON di decisione finale (stepZ_formatter).
    Esempio: MG optimum favorita.
    """
    alerts: List[Dict[str, Any]] = []

    try:
        alert = build_mg_fav_optimum_alert_from_decision(
            meta=meta or {},
            decision=decision,
        )
        if alert is not None:
            alerts.append(alert.to_output())
    except Exception as e:
        print("MG Optimum error:", e)

    return alerts

def evaluate_all_rules(t0: pd.Series, ctx: Optional[Dict[str, Any]] = None):
    """
    Pipeline gerarchica di applicazione regole:
      1. Esecuzione di tutte le regole registrate
      2. Salvataggio degli alert preliminari
      3. Applicazione delle PRIORITÀ tra alert

    Priorità attuali:
        - STRONG_FAIL_RISK  → rimuove   FAVORITA_NONSEGNA
        - OVER15_TRAP       → rimuove   OVER15_SIGNAL
    """

    if ctx is None:
        ctx = {}

    raw: List[BettingAlert] = []

    # -----------------------------------------------
    # PASSO 1: esegui tutte le regole registrate
    # -----------------------------------------------
    for fn in REGISTERED_RULES:
        try:
            res = fn(t0, ctx)
            if res:
                raw.extend(res)
        except Exception:
            pass

    # Convertiamo in dizionario formato finale
    base_outputs = [a.to_output() for a in raw]

    # Memorizziamo gli alert preliminari
    ctx["alerts_pre"] = base_outputs.copy()


    return base_outputs
