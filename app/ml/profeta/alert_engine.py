# app/ml/profeta/engine/alert_engine.py

from pathlib import Path
import pandas as pd
import json

# ============================================================
# LOAD ENABLEMENT (STEP7)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ENABLEMENT_PATH = DATA_DIR / "step7_profeta_market_enablement.parquet"

ENABLEMENT = pd.read_parquet(ENABLEMENT_PATH)

# indicizzazione veloce
ENABLEMENT_MAP = {
    (r.control_state, r.goal_state): json.loads(r.enabled_markets)
    for r in ENABLEMENT.itertuples()
}

# ============================================================
# ALERT THRESHOLD (MINIMI, SOFT)
# ============================================================

MIN_CONF = {
    "HOME_WIN": 0.50,
    "DRAW": 0.28,
    "AWAY_WIN": 0.50,

    "OVER15": 0.70,   # 👈 corretto
    "OVER25": 0.55,
    "UNDER25": 0.60,

    "MG_HOME_1_4": 0.72,
    "MG_AWAY_1_4": 0.72,
}


MARKET_TO_PROB_KEY = {
    # 1X2
    "HOME_WIN": "p1",
    "DRAW": "px",
    "AWAY_WIN": "p2",

    # Over / Under
    "OVER15": "p_over_1_5",
    "OVER25": "p_over_2_5",
    "OVER35": "p_over_3_5",
    "UNDER25": "p_under_2_5",

    # Multigol Totali
    "MG_1_3": "p_mg_1_3",
    "MG_1_4": "p_mg_1_4",
    "MG_1_5": "p_mg_1_5",
    "MG_2_4": "p_mg_2_4",
    "MG_2_5": "p_mg_2_5",

    # Multigol Casa
    "MG_HOME_1_3": "p_mg_home_1_3",
    "MG_HOME_1_4": "p_mg_home_1_4",
    "MG_HOME_1_5": "p_mg_home_1_5",

    # Multigol Ospite
    "MG_AWAY_1_3": "p_mg_away_1_3",
    "MG_AWAY_1_4": "p_mg_away_1_4",
    "MG_AWAY_1_5": "p_mg_away_1_5",
}

# ============================================================
# CORE
# ============================================================

def run_alert_engine(
    markets: dict,
    control_state: str,
    goal_state: str,
):
    """
    markets: output ProfetaEngine
    control_state: STEP5A
    goal_state: STEP5B
    """

    alerts = []

    enabled = ENABLEMENT_MAP.get((control_state, goal_state), [])

    if not enabled:
        return alerts

    for market in enabled:
        prob_key = MARKET_TO_PROB_KEY.get(market)
        if not prob_key:
            continue

        p = markets.get(prob_key)
        if p is None:
            continue

        min_p = MIN_CONF.get(market, 0.0)

        if p >= min_p:
            alerts.append({
                "market": market,
                "confidence": round(p, 3),
                "state": f"{control_state} + {goal_state}",
                "reason": "Mercato abilitato dallo stato e probabilità coerente",
            })

    return alerts
