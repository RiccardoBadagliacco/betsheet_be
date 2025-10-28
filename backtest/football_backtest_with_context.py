#!/usr/bin/env python3
# ==============================================================
# A/B backtest: baseline vs context scoring v4 + soglie log (refactor)
# ==============================================================
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
import sqlite3
from collections import defaultdict
import csv
from pathlib import Path

from app.api.ml_football_exact import ExactSimpleFooballPredictor, get_recommended_bets
from addons.context_scoring_v4 import build_signals_map, compute_context_directives
from addons.betting_config import CANDIDATE_MARKETS

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ===============================================
# LOGGING PER ANALISI SOGLIE
# ===============================================
LOG_FILE = Path("threshold_analysis_log.csv")

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

# ===============================================
# CLASS BACKTEST
# ===============================================
class FootballBacktest:
    def __init__(self, num_matches=500, use_context=False):
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
        print("   ➕ Context attivo" if use_context else "   🧪 Baseline")

        n = len(df)
        iterator = tqdm(range(n), desc="Backtest", unit="match") if HAS_TQDM else range(n)
        for idx in iterator:
            row = df.iloc[idx]
            actual = self.calc_results(row)
            pred = self.predictor.predict_match(df, idx)

            quotes = {
                '1': row["AvgH"], 'X': row["AvgD"], '2': row["AvgA"],
                'over_25': row["Avg>2.5"], 'under_25': row["Avg<2.5"]
            }

            if use_context:
                mid = row["id"]
                proto = [
                    {
                        'match_id': mid,
                        'market': m,
                        'threshold': 0,
                        'HomeTeam': row["HomeTeam"],
                        'AwayTeam': row["AwayTeam"],
                        'odds_home': row["AvgH"],
                        'odds_away': row["AvgA"]
                    }
                    for m in CANDIDATE_MARKETS
                ]
                directives = compute_context_directives(proto, signals_map).get(mid, {})
                pred.update(directives)

            recs = get_recommended_bets(pred, quotes)

            for rec in recs:
                self.market_stats[rec["market"]]["total"] += 1
                correct = self.evaluate(rec, actual)
                if correct:
                    self.market_stats[rec["market"]]["correct"] += 1
                log_threshold_row(rec["market"], rec["confidence"], correct)

            if not HAS_TQDM and (idx+1) % 100 == 0:
                print(f"Progress: {idx+1}/{n} matches...")

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

# ===============================================
# CONFRONTO BASELINE vs CONTEXT
# ===============================================
def compare_backtest(n=2000):
    print("\n=== CONFRONTO BASELINE vs CONTEXT SCORING v4 ===")
    init_threshold_log()
    runner_b = FootballBacktest(n)
    df = runner_b.load_random_matches()

    # 🟡 BASELINE
    runner_b.run(df, use_context=False)
    b_stats = runner_b.market_stats

    # 🟢 CONTEXT
    runner_c = FootballBacktest(n)
    runner_c.run(df.copy(), use_context=True)
    c_stats = runner_c.market_stats

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