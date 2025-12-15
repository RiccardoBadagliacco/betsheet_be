# app/ml/profeta/state_engine.py

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

CONTROL_THR = json.load(open(DATA_DIR / "step5a_control_state_thresholds.json"))
GOAL_THR    = json.load(open(DATA_DIR / "step5b_goal_state_thresholds.json"))


def compute_control_state(markets):
    p1 = markets["p1"]
    p2 = markets["p2"]
    px = markets["px"]
    ld = markets["lambda_home"] - markets["lambda_away"]

    if px >= CONTROL_THR["px_draw"] and abs(p1 - p2) <= 0.10:
        return "DRAW_PRONE"

    if p1 >= CONTROL_THR["p1_dom"] and ld >= CONTROL_THR["lambda_diff"]:
        return "HOME_DOMINANT"

    if p2 >= CONTROL_THR["p2_dom"] and ld <= -CONTROL_THR["lambda_diff"]:
        return "AWAY_DOMINANT"

    return "BALANCED"


def compute_goal_state(markets):
    xg  = markets["xg_total"]
    o25 = markets["p_over_2_5"]
    o35 = markets["p_over_3_5"]
    u25 = markets["p_under_2_5"]

    if xg >= GOAL_THR["xg_wild"] or o35 >= GOAL_THR["o35_wild"]:
        return "WILD_GOALS"

    if xg <= GOAL_THR["xg_low"] and u25 >= GOAL_THR["u25_low"]:
        return "LOW_GOALS"

    if xg >= GOAL_THR["xg_high"] or o25 >= GOAL_THR["o25_high"]:
        return "HIGH_GOALS"

    return "MID_GOALS"