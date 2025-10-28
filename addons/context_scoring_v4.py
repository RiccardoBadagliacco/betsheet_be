#!/usr/bin/env python3
# ==============================================================
# Context Scoring v4 — Refactor (2025)
# ==============================================================
# Obiettivi del refactor:
# - Modularizzare i boost contestuali con un registry di funzioni
# - Mantenere compatibilità con get_recommended_bets (ctx_delta_thresholds, skip/force)
# - Pulizia/robustezza: fallback sicuri, logging opzionale, nessuna duplicazione
# - Invariata la pipeline di backtest e l'integrazione con betting_recommendations
# ==============================================================

from __future__ import annotations

import os
import sys
import csv
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Callable
from collections import defaultdict

import pandas as pd

# Permetti import relativi al progetto (addons/*)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --------------------------------------------------------------
# Config & costanti
# --------------------------------------------------------------
DEBUG = str(os.getenv("DEBUG_CONTEXT", "0")).strip() == "1"

# Deltas principali (più negativo = soglia più facile da superare)
FAVORITE_DELTA_EXT = -3
FORM_DELTA = -3
TACTICAL_DELTA = -4
SEGMENT_DELTA = -5
CROSS_MARKET_DELTA = -3

# File con profili squadra
TEAM_PROFILES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'team_profiles.json')
TEAM_PROFILES: Dict[str, Dict[str, Any]] = {}
if os.path.exists(TEAM_PROFILES_PATH):
    with open(TEAM_PROFILES_PATH, 'r', encoding='utf-8') as f:
        TEAM_PROFILES = json.load(f)

# Log per analisi soglie
LOG_FILE = Path("threshold_analysis_log.csv")

# Import da addons
from addons.betting_config import TARGET_MARKETS, CANDIDATE_MARKETS
from addons.football_utils import market_key_from_label

# --------------------------------------------------------------
# Regole base per mercati (statiche)
# --------------------------------------------------------------
FAVORITE_ODDS_MAX = 1.8
DELTA_FAVORITE = -2
DELTA_TIGHT_PENALTY = +2
DELTA_TIGHT_BOOST = -1
DELTA_H2H_BOOST = -1
DELTA_H2H_PENALTY = +1

# Predicati-base
_def = lambda x: False

def _favorite_boost(s: Dict[str, float]) -> bool:
    return min(s.get("odds_home", 99), s.get("odds_away", 99)) <= FAVORITE_ODDS_MAX

def _tight_penalty(s: Dict[str, float]) -> bool:
    return s.get("match_tightness", 0) > 1.5

def _tight_boost(s: Dict[str, float]) -> bool:
    return s.get("match_tightness", 0) < 0.5

def _h2h_boost(s: Dict[str, float]) -> bool:
    return s.get("h2h_avg_goals", 0) > 2.5

def _h2h_penalty(s: Dict[str, float]) -> bool:
    g = s.get("h2h_avg_goals", 0)
    return 0 < g < 1.5

# Mappatura mercato -> regole statiche
MARKET_RULES: Dict[str, List[tuple[Callable[[Dict[str, float]], bool], int]]] = {
    "O:0.5":     [(_favorite_boost, DELTA_FAVORITE), (_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST)],
    "O:1.5":     [(_favorite_boost, DELTA_FAVORITE), (_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST), (_h2h_penalty, DELTA_H2H_PENALTY)],
    "O:2.5":     [(_tight_penalty, DELTA_TIGHT_PENALTY), (_h2h_boost, DELTA_H2H_BOOST)],
    "MG_Casa_1_3":   [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Casa_1_4":   [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Casa_1_5":   [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_3": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_4": [(_favorite_boost, DELTA_FAVORITE)],
    "MG_Ospite_1_5": [(_favorite_boost, DELTA_FAVORITE)],
}

# --------------------------------------------------------------
# Logging soglie (per tuning post-backtest)
# --------------------------------------------------------------

def init_threshold_log() -> None:
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["market", "confidence", "correct"])

def log_threshold_row(market: str, confidence: float, correct: bool) -> None:
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([market, confidence, int(bool(correct))])

# --------------------------------------------------------------
# Signals & helpers
# --------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else float(default)
    except Exception:
        return float(default)


def compute_context_signals(match: Dict[str, Any]) -> Dict[str, float]:
    """Deriva segnali semplici dal match (quote, tightness, pseudo-H2H)."""
    odds_home = _safe_float(match.get("AvgH"), 0)
    odds_away = _safe_float(match.get("AvgA"), 0)
    odds_draw = _safe_float(match.get("AvgD"), 0)
    ranking_gap = abs(odds_home - odds_away)

    delta_lambda = odds_home - odds_away
    momentum_home = _safe_float(match.get("FTHG"), 0) - _safe_float(match.get("FTAG"), 0)
    momentum_away = -momentum_home

    # Grezza proxy H2H tramite quota over 2.5: più bassa => più goal attesi
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
        "h2h_avg_goals": h2h_avg_goals,
    }


def build_signals_map(matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m in matches:
        mid = m.get("id") or m.get("match_id") or m.get("fixture_id")
        if mid:
            out[mid] = compute_context_signals(m)
    return out

# --------------------------------------------------------------
# Boost modulari (registry)
# --------------------------------------------------------------

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


def boost_segment(signals: Dict[str, float], home_team: str, away_team: str, prediction: Dict[str, Any]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    segment = _classify_match_segment(signals.get("odds_home", 9.9), signals.get("odds_away", 9.9))
    if segment == "big_fav":
        boosts["Multigol Casa 1-4"] = SEGMENT_DELTA
        boosts["Over 0.5 Goal"] = SEGMENT_DELTA
    elif segment == "balanced":
        boosts["Multigol Casa 1-3"] = -2
        boosts["Multigol Ospite 1-3"] = -2
    elif segment == "away_fav":
        boosts["Multigol Ospite 1-4"] = -2
        boosts["Over 0.5 Goal"] = -2
    return boosts


def boost_form(signals: Dict[str, float], home_team: str, away_team: str, prediction: Dict[str, Any]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    fh = TEAM_PROFILES.get(home_team, {}).get("gf", 1.5)
    fa = TEAM_PROFILES.get(away_team, {}).get("gf", 1.5)
    if fh > 2.0 and fa > 1.5:
        for market in ("Over 1.5 Goal", "Multigol Casa 1-4", "Multigol Ospite 1-4"):
            boosts[market] = FORM_DELTA
    return boosts


def boost_tactical(signals: Dict[str, float], home_team: str, away_team: str, prediction: Dict[str, Any]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    sh = TEAM_PROFILES.get(home_team, {}).get("style", "neutral")
    sa = TEAM_PROFILES.get(away_team, {}).get("style", "neutral")
    if sh == sa == "attacking":
        for mkt in ("Over 1.5 Goal", "Over 2.5 Goal", "Multigol Casa 1-5", "Multigol Ospite 1-5"):
            boosts[mkt] = TACTICAL_DELTA
    elif sh == sa == "defensive":
        for mkt in ("Over 2.5 Goal", "Multigol Casa 1-5", "Multigol Ospite 1-5"):
            boosts[mkt] = +5
    return boosts


def boost_cross_market(signals: Dict[str, float], home_team: str, away_team: str, prediction: Dict[str, Any]) -> Dict[str, float]:
    boosts: Dict[str, float] = {}
    o15 = float(prediction.get("O_1_5", 0))
    o05 = float(prediction.get("O_0_5", 0))
    if o15 > 0.88:
        boosts["Multigol Casa 1-4"] = CROSS_MARKET_DELTA
        boosts["Multigol Ospite 1-4"] = CROSS_MARKET_DELTA
    if o05 > 0.95:
        boosts["Multigol Casa 1-3"] = CROSS_MARKET_DELTA
        boosts["Multigol Ospite 1-3"] = CROSS_MARKET_DELTA
    return boosts

# Registry centrale dei boost
BOOST_REGISTRY: List[Callable[[Dict[str, float], str, str, Dict[str, Any]], Dict[str, float]]] = [
    boost_segment,
    boost_form,
    boost_tactical,
    boost_cross_market,
]


def compute_contextual_boosts(home_team: str, away_team: str, odds_home: float, odds_away: float, prediction: Dict[str, Any]) -> Dict[str, float]:
    signals = {"odds_home": float(odds_home or 9.9), "odds_away": float(odds_away or 9.9)}
    total: Dict[str, float] = {}
    for fn in BOOST_REGISTRY:
        b = fn(signals, home_team, away_team, prediction)
        for k, v in b.items():
            total[k] = total.get(k, 0.0) + float(v)
    if DEBUG and total:
        print(f"[CTX BOOST] {home_team}-{away_team} :: {total}")
    return total

# --------------------------------------------------------------
# Context directives (output compatibile con betting_recommendations)
# --------------------------------------------------------------

def compute_context_directives(predictions: List[Dict[str, Any]], signals_map: Dict[str, Dict[str, float]]):
    """Restituisce, per ciascun match_id, un dict con:
       - ctx_delta_thresholds: { label -> delta }
       - skip_recommendations: [label,...]
       - force_recommendations: [label,...]
    """
    out: Dict[str, Dict[str, Any]] = {}

    for rec in predictions:
        mid = rec.get("match_id") or rec.get("id") or rec.get("fixture_id")
        market = rec.get("market")
        if not (mid and market):
            continue

        mk = market_key_from_label(market)
        if not mk or mk not in TARGET_MARKETS:
            # Non è un mercato target: salta senza errori
            continue

        s = signals_map.get(mid)
        if not s:
            continue

        # 1) Delta statico da MARKET_RULES
        delta = 0
        for cond, val in MARKET_RULES.get(mk, []):
            try:
                if cond(s):
                    delta += val
            except Exception:
                # safety: non interrompere l'intero ciclo per un solo errore
                continue

        # 2) Extra boost contestuale esteso se non già gestito da regole statiche
        #    (applichiamo solo se la stessa label è nel boost)
        home_team = rec.get("HomeTeam") or rec.get("home_team") or ""
        away_team = rec.get("AwayTeam") or rec.get("away_team") or ""
        odds_home = s.get("odds_home", 9.9)
        odds_away = s.get("odds_away", 9.9)
        extra_boosts = compute_contextual_boosts(home_team, away_team, odds_home, odds_away, rec)
        if market in extra_boosts:
            delta += extra_boosts[market]

        if delta != 0:
            entry = out.setdefault(mid, {
                "ctx_delta_thresholds": {},
                "skip_recommendations": [],
                "force_recommendations": [],
            })
            entry["ctx_delta_thresholds"][market] = float(delta)
            if DEBUG:
                print(f"[CTX] {mid} :: {market}: Δ={delta:+}")

    return out

# --------------------------------------------------------------
# Backtest runner (compatibile con versione precedente)
# --------------------------------------------------------------

class FootballBacktest:
    def __init__(self, num_matches: int = 500, use_context: bool = False):
        from app.api.ml_football_exact import ExactSimpleFooballPredictor
        self.num_matches = num_matches
        self.db_path = './data/football_dataset.db'
        self.predictor = ExactSimpleFooballPredictor()
        self.use_context = use_context
        self.market_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        
    def load_full_dataset(self) -> pd.DataFrame:
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
        ORDER BY m.match_date ASC
        """
        df = pd.read_sql_query(q, conn)
        conn.close()
        return df


    def load_random_matches(self, all_matches: bool = False) -> pd.DataFrame:
        df_full = self.load_full_dataset()

        # 🔸 Ordina le partite per data
        df_full = df_full.sort_values("Date")

        # Se vuoi tutto il dataset (per ELO)
        if all_matches:
            return df_full

        # 🔸 Prendi solo le ultime N partite per il test
        sampled_df = df_full.tail(self.num_matches)

        # 🔸 Mantieni l’intero dataset per lo storico ELO
        self.df_full = df_full
        return sampled_df

    @staticmethod
    def calc_results(m: pd.Series) -> Dict[str, int]:
        h, a = int(m["FTHG"]), int(m["FTAG"])  # full-time goals
        tot = h + a
        r: Dict[str, int] = {}
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

    @staticmethod
    def evaluate(rec: Dict[str, Any], actual: Dict[str, int]) -> bool:
        return actual.get(rec["market"], 0) == 1

    def run(self, df: pd.DataFrame, use_context: bool = False) -> None:
        from addons.betting_recommendations import get_recommended_bets
        
        matches = [row.to_dict() for _, row in df.iterrows()]
        signals_map = build_signals_map(matches)
        print("   ➕ Context attivo " if use_context else "   🧪 Baseline")

        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Processing matches", unit="match")
        except ImportError:
            iterator = df.iterrows()

        for idx, row in iterator:
            actual = self.calc_results(row)
            pred = self.predictor.predict_match(self.df_full, idx)
            
            # 🧠 Skip partite con poco storico
            if pred is None:
                # opzionale: loggare o contare le partite saltate
                if not hasattr(self, "skipped_matches"):
                    self.skipped_matches = 0
                self.skipped_matches += 1
                if self.skipped_matches <= 10:  # evita flooding in console
                    print(f"[SKIP] Partita {row['HomeTeam']} vs {row['AwayTeam']} saltata per storico insufficiente")
                continue

            quotes = {
                '1': row["AvgH"], 'X': row["AvgD"], '2': row["AvgA"],
                'over_25': row["Avg>2.5"], 'under_25': row["Avg<2.5"]
            }

            if use_context:
                mid = row.get("id")
                if mid is not None:
                    proto = [{'match_id': mid, 'market': m, 'threshold': 0} for m in CANDIDATE_MARKETS]
                    directives_all = compute_context_directives(proto, signals_map)
                    directives = directives_all.get(mid, {})
                    # Inietta le direttive nella predizione
                    pred.update(directives)
            recs = get_recommended_bets(pred, quotes)
            for rec in recs:
                self.market_stats[rec["market"]]["total"] += 1
                if self.evaluate(rec, actual):
                    self.market_stats[rec["market"]]["correct"] += 1
                # Log per analisi soglie
                log_threshold_row(rec["market"], rec["confidence"], actual.get(rec["market"], 0) == 1)

    def print_stats(self, title: str) -> None:
        print(f"\n📊 RISULTATI BACKTEST — {title}")
        tot = sum(s["total"] for s in self.market_stats.values())
        cor = sum(s["correct"] for s in self.market_stats.values())
        acc = round((cor/tot)*100, 2) if tot else 0.0
        print(f"Totale: {tot}  Corrette: {cor}  Acc%: {acc}")
        for m, s in sorted(self.market_stats.items()):
            t, c = s["total"], s["correct"]
            a = round((c/t)*100, 2) if t else 0.0
            print(f"{m:<24} {t:>6} {c:>6} {a:>6.2f}%")
        if hasattr(self, "skipped_matches") and self.skipped_matches > 0:
            print(f"📉 Partite scartate per storico insufficiente: {self.skipped_matches}")

# --------------------------------------------------------------
# CLI: confronto baseline vs context
# --------------------------------------------------------------

def compare_backtest(n: int = 2000) -> None:
    print("\n=== CONFRONTO BASELINE vs CONTEXT SCORING v4 (refactor) ===")
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

    print(f"{'Market':<24} | {'BASELINE':<18} | {'CONTEXT':<18}")
    for m in sorted(set(b_stats) | set(c_stats)):
        bt = b_stats.get(m, {}).get("total", 0)
        bc = b_stats.get(m, {}).get("correct", 0)
        ct = c_stats.get(m, {}).get("total", 0)
        cc = c_stats.get(m, {}).get("correct", 0)
        ba = round(bc / bt * 100, 2) if bt else 0
        ca = round(cc / ct * 100, 2) if ct else 0
        print(f"{m:<24} | {bt:>4} {bc:>4} {ba:>6.2f}% | {ct:>4} {cc:>4} {ca:>6.2f}%")

    total_b = sum(v["total"] for v in b_stats.values())
    correct_b = sum(v["correct"] for v in b_stats.values())
    total_c = sum(v["total"] for v in c_stats.values())
    correct_c = sum(v["correct"] for v in c_stats.values())
    acc_b = round(correct_b / total_b * 100, 2) if total_b else 0
    acc_c = round(correct_c / total_c * 100, 2) if total_c else 0

    print("-" * 72)
    print(f"{'TOTALE':<24} | {total_b:>4} {correct_b:>4} {acc_b:>6.2f}% | {total_c:>4} {correct_c:>4} {acc_c:>6.2f}%")


if __name__ == "__main__":
    import argparse
    from addons.compute_team_profiles import annotate_pre_match_elo

    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare_backtest(args.num)
    else:
        init_threshold_log()

        # ✅ 1. Crea un runner per caricare l’intero dataset
        runner = FootballBacktest(args.num, use_context=True)

        # ✅ 2. Carica TUTTE le partite disponibili (NON solo num)
        df_full = runner.load_random_matches(all_matches=True)  # 👈 ti spiego sotto

        # ✅ 3. Calcola l’ELO cronologicamente su tutto il dataset
        df_full = annotate_pre_match_elo(df_full)

        # ✅ 4. Seleziona solo le ultime N partite per fare il backtest
        df = df_full.tail(args.num)

        # ✅ 5. Passa df_full al runner per predire con ELO corretto
        runner.df_full = df_full

        # ✅ 6. Esegui il backtest
        runner.run(df, use_context=True)
        runner.print_stats("Context v4 Refactor")