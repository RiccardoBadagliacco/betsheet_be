from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
import math
import numbers
from .betting_alert_model import BettingAlert
from .rule_favorite_profile_signal import rule_favorite_profile_signal
from .rule_over_signal import rule_over_signal
# ============================================================
#  BASE STRUCT
# ============================================================

@dataclass
class RuleFn:
    __call__: Callable[[pd.Series, Dict[str, Any]], List[BettingAlert]]

REGISTERED_RULES = [
    rule_favorite_profile_signal,
    rule_over_signal
]

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

    print("  - Esecuzione regole betting...")

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

    print(f"    - Totale alert preliminari: {len(base_outputs)}")


    return base_outputs
