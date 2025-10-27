#!/usr/bin/env python3
# ==============================================================
# A/B backtest: baseline vs context scoring v4 + soglie log
# ==============================================================
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
import sqlite3
from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable
import json
DEBUG = str(os.getenv("DEBUG_CONTEXT", "0")).strip() == "1"

from typing import Dict

FAVORITE_DELTA_EXT = -3
FORM_DELTA = -3
TACTICAL_DELTA = -4
SEGMENT_DELTA = -5
CROSS_MARKET_DELTA = -3

TEAM_PROFILES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'team_profiles.json')
with open(TEAM_PROFILES_PATH, 'r', encoding='utf-8') as f:
    TEAM_PROFILES = json.load(f)

# ===============================================
# LOGGING PER ANALISI SOGLIE
# ===============================================
LOG_FILE = Path("threshold_analysis_log.csv")

# --------------------- rules --------------------- #
TARGET_MARKETS = {
    "O:0.5", "O:1.5", "O:2.5",
    "MG_Casa_1_3", "MG_Casa_1_4", "MG_Casa_1_5",
    "MG_Ospite_1_3", "MG_Ospite_1_4", "MG_Ospite_1_5",
}
FAVORITE_ODDS_MAX = 1.8
DELTA_FAVORITE = -2
DELTA_TIGHT_PENALTY = +2
DELTA_TIGHT_BOOST = -1
DELTA_H2H_BOOST = -1
DELTA_H2H_PENALTY = +1

def _favorite_boost(s): return min(s["odds_home"], s["odds_away"]) <= FAVORITE_ODDS_MAX
def _tight_penalty(s): return s["match_tightness"] > 1.5
def _tight_boost(s): return s["match_tightness"] < 0.5
def _h2h_boost(s): return s["h2h_avg_goals"] > 2.5
def _h2h_penalty(s): return 0 < s["h2h_avg_goals"] < 1.5


MARKET_RULES = {
    "O:0.5": [(_favorite_boost, DELTA_FAVORITE), (_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST)],
    "O:1.5": [(_favorite_boost, DELTA_FAVORITE), (_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST), (_h2h_penalty, DELTA_H2H_PENALTY)],
    "O:2.5": [(_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST)],
    "MG_Casa_1_3": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Casa_1_4": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Casa_1_5": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_3": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_4": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_5": [(_favorite_boost, DELTA_FAVORITE)],
}

def init_threshold_log():
    """Crea file CSV per logging se non esiste."""
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["market", "confidence", "correct"])

def log_threshold_row(market, confidence, correct):
    """Scrive una riga nel log soglie per analisi successiva."""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([market, confidence, int(correct)])
        
def build_signals_map(matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for m in matches:
        mid = m.get("id") or m.get("match_id") or m.get("fixture_id")
        if mid:
            out[mid] = compute_context_signals(m)
    return out

def _market_key_from_label(label: str) -> str | None:
    s = (label or "").lower()
    if s.startswith("over "):
        parts = s.replace("goal", "").split()
        return f"O:{parts[1]}" if len(parts) >= 2 else None
    if "multigol casa" in s:
        rng = s.split()[-1].replace("-", "_")
        return f"MG_Casa_{rng}"
    if "multigol ospite" in s:
        rng = s.split()[-1].replace("-", "_")
        return f"MG_Ospite_{rng}"
    return None

# --------------------- utils --------------------- #
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else float(default)
    except Exception:
        return float(default)
    
def compute_context_signals(match: Dict[str, Any]) -> Dict[str, float]:
    odds_home = _safe_float(match.get("AvgH"), 0)
    odds_away = _safe_float(match.get("AvgA"), 0)
    odds_draw = _safe_float(match.get("AvgD"), 0)
    ranking_gap = abs(odds_home - odds_away)

    delta_lambda = odds_home - odds_away
    momentum_home = _safe_float(match.get("FTHG"), 0) - _safe_float(match.get("FTAG"), 0)
    momentum_away = -momentum_home

    avg_over25 = _safe_float(match.get("Avg>2.5"), 0)
    h2h_avg_goals = 0.0
    if avg_over25 > 0:
        h2h_avg_goals = round(2.5 * (2.0 / avg_over25), 3)

    return {
        "odds_home": odds_home,
        "odds_away": odds_away,
        "odds_draw": odds_draw,
        "delta_lambda": round(delta_lambda, 3),
        "momentum_home": momentum_home,
        "momentum_away": momentum_away,
        "ranking_gap": ranking_gap,
        "match_tightness": ranking_gap,
        "h2h_avg_goals": h2h_avg_goals
    }

class FootballBacktest:
    def __init__(self, num_matches=500, use_context=False):
        from app.api.ml_football_exact import ExactSimpleFooballPredictor
        self.num_matches = num_matches
        self.db_path = './data/football_dataset.db'
        self.predictor = ExactSimpleFooballPredictor()
        self.use_context = use_context
        self.market_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    def load_random_matches(self):
        conn = sqlite3.connect(self.db_path)
        q = """
        SELECT 
            m.id,
            m.match_date as Date,
            ht.name as HomeTeam,
            at.name as AwayTeam,
            m.home_goals_ft as FTHG,
            m.away_goals_ft as FTAG,
            m.avg_home_odds as AvgH,
            m.avg_draw_odds as AvgD,
            m.avg_away_odds as AvgA,
            m.avg_over_25_odds as "Avg>2.5",
            m.avg_under_25_odds as "Avg<2.5"
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id  
        ORDER BY RANDOM()
        LIMIT ?
        """
        df = pd.read_sql_query(q, conn, params=[self.num_matches])
        conn.close()
        return df

    def calc_results(self, m):
        h, a = int(m["FTHG"]), int(m["FTAG"])
        tot = h + a
        r = {}
        for x in [0.5, 1.5, 2.5, 3.5]:
            r[f"Over {x} Goal"] = 1 if tot > x else 0
            r[f"Under {x} Goal"] = 1 if tot <= x else 0
        for lo, hi in [(1, 3), (1, 4), (1, 5)]:
            r[f"Multigol Casa {lo}-{hi}"] = 1 if lo <= h <= hi else 0
            r[f"Multigol Ospite {lo}-{hi}"] = 1 if lo <= a <= hi else 0
        r["1X2 Casa"] = 1 if h > a else 0
        r["1X2 Ospite"] = 1 if a > h else 0
        r["Doppia Chance 1X"] = 1 if h >= a else 0
        r["Doppia Chance 12"] = 1 if h != a else 0
        r["Doppia Chance X2"] = 1 if a >= h else 0
        return r

    def evaluate(self, rec, actual):
        m = rec["market"]
        return actual.get(m, 0) == 1

    def run(self, df, use_context=False):
        matches = [row.to_dict() for _, row in df.iterrows()]
        signals_map = build_signals_map(matches)
        if use_context:
            print("   ➕ Context attivo")
        else:
            print("   🧪 Baseline")

        from app.api.ml_football_exact import get_recommended_bets
        for idx, row in df.iterrows():
            if idx == 0:
                print(f"\n🔍 DEBUG - Prima--- partita: {row['HomeTeam']} vs {row['AwayTeam']}")
            actual = self.calc_results(row)
            pred = self.predictor.predict_match(df, idx)
            quotes = {
                '1': row["AvgH"], 'X': row["AvgD"], '2': row["AvgA"],
                'over_25': row["Avg>2.5"], 'under_25': row["Avg<2.5"]
            }

            # context integration
            if use_context:
                mid = row["id"]
                candidate_markets = [
                    'Over 0.5', 'Over 1.5', 'Over 2.5',
                    'Under 2.5', 'Under 3.5',
                    'Multigol Casa 1-3', 'Multigol Casa 1-4', 'Multigol Casa 1-5',
                    'Multigol Ospite 1-3', 'Multigol Ospite 1-4', 'Multigol Ospite 1-5',
                    'Doppia Chance 1X', 'Doppia Chance 12', 'Doppia Chance X2',
                    '1X2 Casa', '1X2 Ospite'
                ]
                proto = [{'match_id': mid, 'market': m, 'threshold': 0} for m in candidate_markets]
                directives = compute_context_directives(proto, signals_map).get(mid, {})
                pred.update(directives)

            recs = get_recommended_bets(pred, quotes)
            for rec in recs:
                self.market_stats[rec["market"]]["total"] += 1
                correct = self.evaluate(rec, actual)
                if correct:
                    self.market_stats[rec["market"]]["correct"] += 1
                # 👉 LOG PER ANALISI SOGLIE
                log_threshold_row(rec["market"], rec["confidence"], correct)

    def print_stats(self, title):
        print(f"\n📊 RISULTATI BACKTEST — {title}")
        tot = sum(s["total"] for s in self.market_stats.values())
        cor = sum(s["correct"] for s in self.market_stats.values())
        acc = round((cor/tot)*100, 2) if tot else 0
        print(f"Totale: {tot}  Corrette: {cor}  Acc%: {acc}")
        for m, s in sorted(self.market_stats.items()):
            t, c = s["total"], s["correct"]
            a = round((c/t)*100, 2) if t else 0
            print(f"{m:<24} {t:>6} {c:>6} {a:>6.2f}%")


def compare_backtest(n=2000):
    print("\n=== CONFRONTO BASELINE vs CONTEXT SCORING v4 (stesso seed/matches) ===")
    init_threshold_log()

    runner_b = FootballBacktest(n)
    df = runner_b.load_random_matches()
    df_copy = df.copy()

    # baseline
    runner_b.run(df, use_context=False)
    b_stats = runner_b.market_stats

    # context
    runner_c = FootballBacktest(n)
    runner_c.run(df_copy, use_context=True)
    c_stats = runner_c.market_stats

    # stampa tabella
    print(f"{'Market':<24} | {'BASELINE':<18} | {'CONTEXT':<18}")
    for m in sorted(set(b_stats) | set(c_stats)):
        bt = b_stats.get(m, {}).get("total", 0)
        bc = b_stats.get(m, {}).get("correct", 0)
        ct = c_stats.get(m, {}).get("total", 0)
        cc = c_stats.get(m, {}).get("correct", 0)
        ba = round(bc / bt * 100, 2) if bt else 0
        ca = round(cc / ct * 100, 2) if ct else 0
        print(f"{m:<24} | {bt:>4} {bc:>4} {ba:>6.2f}% | {ct:>4} {cc:>4} {ca:>6.2f}%")

    # aggregati
    total_b = sum(v["total"] for v in b_stats.values())
    correct_b = sum(v["correct"] for v in b_stats.values())
    total_c = sum(v["total"] for v in c_stats.values())
    correct_c = sum(v["correct"] for v in c_stats.values())
    acc_b = round(correct_b / total_b * 100, 2) if total_b else 0
    acc_c = round(correct_c / total_c * 100, 2) if total_c else 0

    print("-" * 72)
    print(f"{'TOTALE':<24} | {total_b:>4} {correct_b:>4} {acc_b:>6.2f}% | {total_c:>4} {correct_c:>4} {acc_c:>6.2f}%")
    
# ====================================================
# 🧠 Analisi contestuale avanzata (forma, tattica, segmenti)
# ====================================================

def _classify_match_segment(odds_home: float, odds_away: float) -> str:
    diff = abs(odds_home - odds_away)
    fav = min(odds_home, odds_away)
    if fav < 1.50:
        return "big_fav"
    elif diff < 0.2:
        return "balanced"
    elif odds_away < odds_home:
        return "away_fav"
    else:
        return "other"

def _form_boost(home_team: str, away_team: str) -> Dict[str, float]:
    boosts = {}
    fh = TEAM_PROFILES.get(home_team, {}).get("gf", 1.5)
    fa = TEAM_PROFILES.get(away_team, {}).get("gf", 1.5)
    if fh > 2.0 and fa > 1.5:
        boosts["Over 1.5 Goal"] = FORM_DELTA
        boosts["Multigol Casa 1-4"] = FORM_DELTA
        boosts["Multigol Ospite 1-4"] = FORM_DELTA
    return boosts

def _tactical_boost(home_team: str, away_team: str) -> Dict[str, float]:
    boosts = {}
    sh = TEAM_PROFILES.get(home_team, {}).get("style", "neutral")
    sa = TEAM_PROFILES.get(away_team, {}).get("style", "neutral")
    if sh == "attacking" and sa == "attacking":
        boosts["Over 1.5 Goal"] = TACTICAL_DELTA
        boosts["Over 2.5 Goal"] = TACTICAL_DELTA
        boosts["Multigol Casa 1-5"] = TACTICAL_DELTA
        boosts["Multigol Ospite 1-5"] = TACTICAL_DELTA
    elif sh == "defensive" and sa == "defensive":
        boosts["Over 2.5 Goal"] = +5
        boosts["Multigol Casa 1-5"] = +5
        boosts["Multigol Ospite 1-5"] = +5
    return boosts

def _segment_boost(segment: str) -> Dict[str, float]:
    boosts = {}
    if segment == "big_fav":
        boosts["Multigol Casa 1-4"] = SEGMENT_DELTA
        boosts["Over 0.5 Goal"] = SEGMENT_DELTA
    elif segment == "balanced":
        boosts["Multigol Casa 1-3"] = -2
        boosts["Multigol Ospite 1-3"] = -2
    elif segment == "away_fav":
        boosts["Over 0.5 Goal"] = -2
        boosts["Multigol Ospite 1-4"] = -2
    return boosts

def _cross_market_boost(prediction: dict) -> Dict[str, float]:
    boosts = {}
    o15 = prediction.get("O_1_5", 0)
    o05 = prediction.get("O_0_5", 0)
    if o15 > 0.88:
        boosts["Multigol Casa 1-4"] = CROSS_MARKET_DELTA
        boosts["Multigol Ospite 1-4"] = CROSS_MARKET_DELTA
    if o05 > 0.95:
        boosts["Multigol Casa 1-3"] = CROSS_MARKET_DELTA
        boosts["Multigol Ospite 1-3"] = CROSS_MARKET_DELTA
    return boosts

def compute_contextual_boosts(home_team: str, away_team: str, odds_home: float, odds_away: float, prediction: dict) -> Dict[str, float]:
    segment = _classify_match_segment(odds_home, odds_away)
    boosts = {}
    boosts.update(_segment_boost(segment))
    boosts.update(_form_boost(home_team, away_team))
    boosts.update(_tactical_boost(home_team, away_team))
    boosts.update(_cross_market_boost(prediction))
    return boosts


# --------------------- directives --------------------- #
def compute_context_directives(predictions: List[Dict[str, Any]], signals_map: Dict[str, Dict[str, float]]):
    out = {}
    for rec in predictions:
        mid = rec.get("match_id") or rec.get("id")
        market = rec.get("market")
        mk = _market_key_from_label(market)
        if not (mid and mk and mk in TARGET_MARKETS): 
            continue
        s = signals_map.get(mid)
        if not s:
            continue

        delta = 0
        for cond, val in MARKET_RULES.get(mk, []):
            if cond(s):
                delta += val

        if delta != 0:
            entry = out.setdefault(mid, {
                "ctx_delta_thresholds": {},
                "skip_recommendations": [],
                "force_recommendations": []
            })
            entry["ctx_delta_thresholds"][market] = float(delta)

        
        # =======================
        # 📈 BOOST CONTESTUALE ESTESO
        # =======================
        if delta == 0:  # Applica extra solo se non già gestito
            home_team = rec.get("HomeTeam") or rec.get("home_team") or ""
            away_team = rec.get("AwayTeam") or rec.get("away_team") or ""
            odds_home = s["odds_home"]
            odds_away = s["odds_away"]
            # usa il dict di predizioni grezze
            extra_boosts = compute_contextual_boosts(home_team, away_team, odds_home, odds_away, rec)
            if market in extra_boosts:
                delta += extra_boosts[market]

        if delta != 0:
            print(f"[CTX EXT] {market}: Δextra={delta:+} seg={_classify_match_segment(s['odds_home'], s['odds_away'])}")
                
    return out


if __name__ == "__main__":
    import random
    import numpy as np

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare_backtest(args.num)
    else:
        init_threshold_log()
        runner = FootballBacktest(args.num, use_context=True)
        df = runner.load_random_matches()
        runner.run(df, use_context=True)
        runner.print_stats("CONTEXT ONLY")
