#!/usr/bin/env python3
"""
ml_football_exact.py — Refactor 2025
===================================

Obiettivi del refactor:
- Eliminare duplicazioni (predict_match definita due volte)
- Separare helpers riutilizzabili (fixture row, cleaning, fallback odds)
- Introdurre caching leggero per predictor/dataset per lega
- Mantenere compatibilità API ed integrazione con betting_recommendations
- Migliorare robustezza (type hints, error handling, input sanitization)

Dipendenze interne:
- addons.football_utils: get_team_features, remove_vig, estimate_lambdas_from_market,
  estimate_lambdas_from_stats, calculate_probabilities
- addons.betting_recommendations: get_recommended_bets
- addons.context_scoring_v4: opzionale (usato dai servizi esterni / backtest)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from addons.betting_config import MARKET_KEY_MAP
import os

# --- addons imports (funzioni condivise) ---
from addons.football_utils import (
    get_team_features,
    remove_vig,
    estimate_lambdas_from_market,
    estimate_lambdas_from_stats,
    calculate_probabilities,
)
from addons.betting_recommendations import get_recommended_bets
from addons.compute_team_profiles import annotate_pre_match_elo,build_team_profiles

# ==============================================================
# Router FastAPI
# ==============================================================
router = APIRouter()

# ==============================================================
# Helpers e utilità locali
# ==============================================================
DEBUG = str(os.getenv("DEBUG_CONTEXT", "0")).strip() == "1"

def _convert_numpy_types(obj: Any) -> Any:
    """Converte tipi numpy in tipi Python nativi (JSON-safe)."""
    if obj is None:
        return None
    try:
        import numpy as _np
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, ( _np.integer, _np.int32, _np.int64 )):
            return int(obj)
        if isinstance(obj, ( _np.floating, _np.float32, _np.float64 )):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(v) for v in obj]
    return obj


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _fallback_odds_map() -> Dict[str, float]:
    """Quote di fallback conservative nel caso manchino dal DB."""
    return {
        'AvgH': 2.00,
        'AvgD': 3.20,
        'AvgA': 3.80,
        'Avg>2.5': 1.80,
        'Avg<2.5': 2.00,
    }


def _build_fixture_row(fixture) -> Dict[str, Any]:
    """Converte un oggetto Fixture in una riga compatibile col DataFrame storico.
    Applica fallback odds se mancanti.
    """
    fh = getattr(fixture, 'avg_home_odds', None)
    fd = getattr(fixture, 'avg_draw_odds', None)
    fa = getattr(fixture, 'avg_away_odds', None)
    fo = getattr(fixture, 'avg_over_25_odds', None)
    fu = getattr(fixture, 'avg_under_25_odds', None)
    fb = _fallback_odds_map()

    return {
        'Date': pd.to_datetime(getattr(fixture, 'match_date', None)),
        'HomeTeam': fixture.home_team.name if getattr(fixture, 'home_team', None) else 'Unknown',
        'AwayTeam': fixture.away_team.name if getattr(fixture, 'away_team', None) else 'Unknown',
        'FTHG': getattr(fixture, 'home_goals_ft', None),
        'FTAG': getattr(fixture, 'away_goals_ft', None),
        'AvgH': fh if fh is not None else fb['AvgH'],
        'AvgD': fd if fd is not None else fb['AvgD'],
        'AvgA': fa if fa is not None else fb['AvgA'],
        'Avg>2.5': fo if fo is not None else fb['Avg>2.5'],
        'Avg<2.5': fu if fu is not None else fb['Avg<2.5'],
    }


# ==============================================================
# Predictor (solo logica di previsione)
# ==============================================================

@dataclass
class PredictorConfig:
    global_window: int = 10
    venue_window: int = 5
    market_weight: float = 0.60


class ExactSimpleFootballPredictor:
    """Replica del SimpleFootballPredictor adattata a DataFrame DB.
    Mantiene la stessa logica di calcolo (features, lambdas, Poisson).
    """

    def __init__(self, config: PredictorConfig | None = None):
        self.cfg = config or PredictorConfig()
        self.data: Optional[pd.DataFrame] = None
        # ✅ Compatibilità + comportamento aggiornato
        self.use_context_scoring: bool = True
        self.model_version: str = "CONTEXT_SCORING_V4"

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------
    def load_data(self, db: Session, league_code: str) -> pd.DataFrame:
        """Carica i match della lega dal DB e li normalizza."""
        from app.db.models_football import Match, Team, League, Season

        q = (
            db.query(Match)
              .join(Season)
              .join(League)
              .filter(League.code == league_code)
              .order_by(Match.match_date)
        )
        matches = q.all()
        if not matches:
            raise ValueError(f"No matches found for league {league_code}")

        rows: List[Dict[str, Any]] = []
        for m in matches:
            rows.append({
                'Date': m.match_date,
                'HomeTeam': m.home_team.name if m.home_team else 'Unknown',
                'AwayTeam': m.away_team.name if m.away_team else 'Unknown',
                'FTHG': m.home_goals_ft,
                'FTAG': m.away_goals_ft,
                'AvgH': m.avg_home_odds,
                'AvgD': m.avg_draw_odds,
                'AvgA': m.avg_away_odds,
                'Avg>2.5': m.avg_over_25_odds,
                'Avg<2.5': m.avg_under_25_odds,
            })
        df = pd.DataFrame(rows)

        # Normalizzazione
        df['Date'] = pd.to_datetime(df['Date'])
        df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
        df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()
        for col in ['FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
        
        df = annotate_pre_match_elo(df)
        
        self.team_profile = build_team_profiles(df)

        self.df_full = df
        return df

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------
    def predict_match(self, df: pd.DataFrame, match_idx: int) -> Dict[str, Any]:
        """
        Minimal predictor — ONLY Over/Under goals markets:
        - O/U 0.5
        - O/U 1.5
        - O/U 2.5
        No context, no multigol, no exotic logic.
        """

        match = df.iloc[match_idx]
        current_date = match['Date']

        # --- Team history (rolling features)
        home_features = get_team_features(df, match['HomeTeam'], current_date, is_home=True)
        away_features = get_team_features(df, match['AwayTeam'], current_date, is_home=False)

        if not home_features.get('valid') or not away_features.get('valid'):
            return None  # not enough history

        # --- Poisson lambdas from market + stats
        # 1) market lambda (from 1X2 + OU)
        market_lambdas = (1.3, 1.1)
        if pd.notna(match.get('AvgH')) and pd.notna(match.get('AvgD')) and pd.notna(match.get('AvgA')):
            odds_1x2 = {'H': match['AvgH'], 'D': match['AvgD'], 'A': match['AvgA']}
            p_1x2 = remove_vig(odds_1x2)

            odds_ou = {'over': match.get('Avg>2.5', 1.9), 'under': match.get('Avg<2.5', 1.9)}
            p_ou = remove_vig(odds_ou)

            market_lambdas = estimate_lambdas_from_market(p_1x2, p_ou)

        # 2) Stats λ
        if 'elo_home_pre' not in df.columns or 'elo_away_pre' not in df.columns:
            df = annotate_pre_match_elo(df)

        row_elo = df.iloc[match_idx]
        elo_home = row_elo.get('elo_home_pre', 1500)
        elo_away = row_elo.get('elo_away_pre', 1500)

        lambda_home_stats, lambda_away_stats = estimate_lambdas_from_stats(
            home_features,
            away_features,
            match['HomeTeam'], match['AwayTeam'],
            elo_home_pre=elo_home,
            elo_away_pre=elo_away
        )

        # Blend
        w = self.cfg.market_weight
        lambda_home = w * market_lambdas[0] + (1 - w) * lambda_home_stats
        lambda_away = w * market_lambdas[1] + (1 - w) * lambda_away_stats

        # --- Poisson probabilities (we only keep the essentials)
        probs = calculate_probabilities(lambda_home, lambda_away)

        result = {
            "match_idx": match_idx,
            "date": current_date,
            "home_team": match["HomeTeam"],
            "away_team": match["AwayTeam"],

            "lambda_home": float(lambda_home),
            "lambda_away": float(lambda_away),

            # ✅ Only the Over/Under markets we want
            "Over 0.5": float(probs.get("Over 0.5 Goal", 0)),
            "Under 0.5": float(probs.get("Under 0.5 Goal", 1)),

            "Over 1.5": float(probs.get("Over 1.5 Goal", 0)),
            "Under 1.5": float(probs.get("Under 1.5 Goal", 1)),

            "Over 2.5": float(probs.get("Over 2.5 Goal", 0)),
            "Under 2.5": float(probs.get("Under 2.5 Goal", 1)),
        }

        # ✅ include odds if present (for logging)
        if pd.notna(match.get('AvgH')):
            result.update({
                "odds_1": float(match.get("AvgH")),
                "odds_X": float(match.get("AvgD")),
                "odds_2": float(match.get("AvgA")),
                "odds_over_25": float(match.get("Avg>2.5")),
                "odds_under_25": float(match.get("Avg<2.5")),
            })

        # ✅ actual outcome if historical (for backtest)
        if pd.notna(match.get("FTHG")):
            h, a = int(match["FTHG"]), int(match["FTAG"])
            tot = h + a
            result["actual_total_goals"] = tot
            result["actual_over_0.5"] = int(tot > 0.5)
            result["actual_over_1.5"] = int(tot > 1.5)
            result["actual_over_2.5"] = int(tot > 2.5)

        return result
    
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
        from app.api.ml_football_exact import ExactSimpleFootballPredictor
        self.num_matches = num_matches
        self.db_path = './data/football_dataset.db'
        self.predictor = ExactSimpleFootballPredictor()
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
    
def analyze_favorite_anomalies(df: pd.DataFrame) -> None:
    """
    Analizza le partite in cui la favorita non segna o segna >=5 gol.
    Determina la favorita in base alle quote 1X2.
    """
    if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgH", "AvgA"}.issubset(df.columns):
        print("⚠️ Impossibile analizzare anomalie: colonne mancanti.")
        return

    df = df.copy()
    df["favorite_side"] = df.apply(lambda r: "home" if r["AvgH"] < r["AvgA"] else "away", axis=1)
    df["fav_goals"] = df.apply(lambda r: r["FTHG"] if r["favorite_side"] == "home" else r["FTAG"], axis=1)
    df["fav_odds"] = df.apply(lambda r: r["AvgH"] if r["favorite_side"] == "home" else r["AvgA"], axis=1)
    df["underdog_goals"] = df.apply(lambda r: r["FTAG"] if r["favorite_side"] == "home" else r["FTHG"], axis=1)

    no_goal = df[df["fav_goals"] == 0]
    over_5 = df[df["fav_goals"] >= 5]

    total = len(df)
    print("\n⚽️ ANALISI FAVORITA — CASI ANOMALI")
    print("──────────────────────────────────────────────")
    print(f"Totale partite analizzate: {total}")
    print(f"Favorita non segna: {len(no_goal)} ({len(no_goal)/total*100:.2f}%)")
    print(f"Favorita segna ≥5 gol: {len(over_5)} ({len(over_5)/total*100:.2f}%)")

    # Breakdown per fascia di quota
    df["fav_bin"] = pd.cut(df["fav_odds"], [1.0, 1.4, 1.7, 2.0, 2.5, 3.0, 10.0])
    mean_goals = df.groupby("fav_bin")["fav_goals"].mean().round(2)
    print("\n📊 Media gol favorita per fascia quote:")
    print(mean_goals)

    # Leghe più colpite
    if "league_code" in df.columns:
        print("\n🏆 Leghe con più favorite a zero gol:")
        print(no_goal["league_code"].value_counts().head())

    # 🔎 Extra debug: salvataggio CSV opzionale
    out_path = "reports/favorite_anomalies.csv"
    import os
    os.makedirs("reports", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n📁 Report completo salvato in: {out_path}")


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
        try:
            analyze_favorite_anomalies(runner.df_full)
        except Exception as e:
            print(f"⚠️ Errore durante analisi favorita: {e}")

    def predict_matches(self, df: pd.DataFrame, start_idx: int = 50) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx in range(start_idx, len(df)):
            try:
                results.append(self.predict_match(df, idx))
            except Exception:
                # Non bloccare l'intero batch per un match
                continue
        return results

# ==============================================================
# Caching predictor per lega (riduce tempi di I/O)
# ==============================================================

_LEAGUE_PREDICTORS: Dict[str, ExactSimpleFootballPredictor] = {}


def _get_model_for_league(league_code: str) -> ExactSimpleFootballPredictor:
    pred = _LEAGUE_PREDICTORS.get(league_code)
    if pred is None:
        pred = ExactSimpleFootballPredictor()
        _LEAGUE_PREDICTORS[league_code] = pred
    return pred

# ==============================================================
# Connessione DB (read-only helper)
# ==============================================================

def get_football_db():
    """Crea una sessione SQLite locale per il dataset football (read-only scope)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    FOOTBALL_DATABASE_URL = "sqlite:///./data/football_dataset.db"
    engine = create_engine(FOOTBALL_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================================================
# Endpoints API
# ==============================================================

@router.post("/exact_predict_fixture/{fixture_id}")
async def exact_predict_fixture(
    fixture_id: str,
    db: Session = Depends(get_football_db)
):
    """
    Predice un fixture usando il modello Exact + Context Scoring v4
    e restituisce raccomandazioni già integrate nella pipeline.
    """
    try:
        from app.db.models_football import Fixture
        from uuid import UUID
        import pandas as pd

        # ✅ Normalizza fixture_id
        try:
            if len(fixture_id) == 32 and '-' not in fixture_id:
                fixture_uuid = UUID(f"{fixture_id[:8]}-{fixture_id[8:12]}-{fixture_id[12:16]}-{fixture_id[16:20]}-{fixture_id[20:]}")
            elif len(fixture_id) == 36:
                fixture_uuid = UUID(fixture_id)
            else:
                fixture_uuid = fixture_id
        except ValueError:
            fixture_uuid = fixture_id

        # ✅ Recupera fixture
        fixture = db.query(Fixture).filter(Fixture.id == fixture_uuid).first()
        if not fixture:
            raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found")
        if not fixture.league_code:
            raise HTTPException(status_code=400, detail="Fixture missing league_code")

        # ✅ Carica modello e dati lega
        predictor = _get_model_for_league(fixture.league_code)
        df = predictor.load_data(db, fixture.league_code)

        # ✅ Converte fixture in riga, aggiunge agli storici e predice
        row = _build_fixture_row(fixture)
        extended_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        match_idx = len(extended_df) - 1
        
        result = predictor.predict_match(extended_df, match_idx)
        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient team history for prediction")

        # ✅ Compila metadati
        result["fixture_id"] = fixture_id
        result["league_code"] = fixture.league_code
        result["fixture_date"] = fixture.match_date.isoformat() if getattr(fixture, 'match_date', None) else None
        result["note"] = "Prediction using Exact Model + Context Scoring v4"

        # ✅ Pulizia numpy types
        clean_prediction = {k: _convert_numpy_types(v) for k, v in result.items()}

        return {
            "success": True,
            "fixture_id": fixture_id,
            "league_code": fixture.league_code,
            "prediction": clean_prediction,
            "betting_recommendations": clean_prediction.get("betting_recommendations", []),
            "context_directives": clean_prediction.get("context_directives", {}),
            "model_info": {
                "version": predictor.model_version,
                "data_source": "database",
                "training_matches": int(len(df)),
                "context_scoring": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exact_predict_match/{league_code}")
async def exact_predict_match(
    league_code: str,
    home_team: str,
    away_team: str,
    match_date: str | None = None,
    db: Session = Depends(get_football_db)
):
    """[LEGACY] Predice un match dato da nomi squadre (+/- data).
    Preferire /exact_predict_fixture in produzione.
    """
    try:
        predictor = _get_model_for_league(league_code)
        df = predictor.load_data(db, league_code)

        # Trova match storico se esiste
        if match_date:
            date_parsed = datetime.strptime(match_date, '%Y-%m-%d').date()
            matches = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) & (df['Date'].dt.date == date_parsed)]
        else:
            matches = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)].tail(1)

        if len(matches) == 0:
            # Mock future fixture
            mock = {
                'Date': pd.to_datetime(match_date or datetime.utcnow().date()),
                'HomeTeam': home_team.strip(),
                'AwayTeam': away_team.strip(),
                'FTHG': np.nan,
                'FTAG': np.nan,
                'AvgH': 2.0,
                'AvgD': 3.2,
                'AvgA': 3.8,
                'Avg>2.5': 1.8,
                'Avg<2.5': 2.0,
            }
            extended_df = pd.concat([df, pd.DataFrame([mock])], ignore_index=True)
            match_idx = len(extended_df) - 1
            prediction = predictor.predict_match(extended_df, match_idx)
            prediction['note'] = 'Future fixture prediction using mock data'
        else:
            match_idx = matches.index[0]
            prediction = predictor.predict_match(df, match_idx)
            prediction['note'] = 'Historical match prediction'

        clean_prediction = {k: _convert_numpy_types(v) for k, v in prediction.items()}
        betting_recs = get_recommended_bets(clean_prediction)
        clean_recs = [_convert_numpy_types(r) for r in betting_recs]

        response = {
            "success": True,
            "prediction": clean_prediction,
            "betting_recommendations": clean_recs,
            "model_info": {
                "version": "EXACT_REPLICA",
                "data_source": "database",
                "matches_loaded": int(len(df)),
            },
        }
        return _convert_numpy_types(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all_fixtures_recommendations")
async def get_all_fixtures_recommendations(
    league_code: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_football_db),
    use_context: bool = True,
):
    """Raccomandazioni betting per i fixture in finestra temporale (± range standard)."""
    try:
        from app.db.models_football import Fixture
        from datetime import timedelta

        # Query base
        query = db.query(Fixture)
        if league_code:
            query = query.filter(Fixture.league_code == league_code)

        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ahead = now + timedelta(days=30)

        fixtures = (
            query.filter(Fixture.match_date >= week_ago, Fixture.match_date <= month_ahead)
                 .order_by(Fixture.match_date)
                 .limit(limit)
                 .all()
        )
        if not fixtures:
            return {
                "success": True,
                "fixtures": [],
                "total_fixtures": 0,
                "message": "No fixtures found in the specified date range",
            }

        # Predictor cache per lega
        league_predictors: Dict[str, ExactSimpleFootballPredictor] = {}
        fixture_recommendations: List[Dict[str, Any]] = []
        processed_leagues: set[str] = set()

        for i, fixture in enumerate(fixtures, start=1):
            try:
                lc = fixture.league_code
                if lc not in league_predictors:
                    predictor = _get_model_for_league(lc)
                    df = predictor.load_data(db, lc)
                    predictor.data = df
                    league_predictors[lc] = predictor
                    processed_leagues.add(lc)

                predictor = league_predictors[lc]
                df = predictor.data

                row = _build_fixture_row(fixture)
                extended_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                
                from addons.compute_team_profiles import annotate_pre_match_elo
                
                try:
                    extended_df = annotate_pre_match_elo(extended_df)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    raise  # opzionale: per non silenziarlo

                match_idx = len(extended_df) - 1

                try:
                    prediction = predictor.predict_match(extended_df, match_idx)
                    print(f"[{i}/{len(fixtures)}] Predicted fixture {fixture.id} ({fixture.home_team.name} vs {fixture.away_team.name})")
                except Exception as e:
                    if DEBUG:
                        print(f"  - DEBUG: Prediction failed for fixture {fixture.id}: {e}")
                    import traceback; traceback.print_exc()
                    raise  # opzionale: per non silenziarlo

                clean_prediction = _convert_numpy_types(prediction)

                real_quotes: Optional[Dict[str, float]] = None
                if fixture.avg_home_odds and fixture.avg_away_odds:
                    real_quotes = {
                        '1': fixture.avg_home_odds,
                        '2': fixture.avg_away_odds,
                        'X': fixture.avg_draw_odds or 3.2,
                    }
                if DEBUG:
                    print(f"  - Generating recommendations with real quotes: {real_quotes}")
                    print(f"[DEBUG λ] {prediction.get('home_team')} vs {prediction.get('away_team')} → "
                          f"λ_home={prediction.get('lambda_home'):.2f}, λ_away={prediction.get('lambda_away'):.2f}")
                    print(f"[DEBUG FAVORITE] favorite_team={prediction.get('favorite_team')}")
                recs = get_recommended_bets(clean_prediction, quotes=real_quotes)
                clean_recs = [_convert_numpy_types(r) for r in recs]

                item = {
                    "fixture_id": str(fixture.id),
                    "match_date": fixture.match_date.isoformat() if getattr(fixture, 'match_date', None) else None,
                    "match_time": getattr(fixture, 'match_time', None),
                    "league_code": lc,
                    "home_team": fixture.home_team.name if getattr(fixture, 'home_team', None) else "Unknown",
                    "away_team": fixture.away_team.name if getattr(fixture, 'away_team', None) else "Unknown",
                    "odds_1x2": {
                        "home": fixture.avg_home_odds,
                        "draw": fixture.avg_draw_odds,
                        "away": fixture.avg_away_odds,
                        "over_25": fixture.avg_over_25_odds,
                        "under_25": fixture.avg_under_25_odds,
                    },
                    "betting_recommendations": clean_recs,
                    "total_recommendations": len(clean_recs),
                    "confidence_stats": {
                        "avg_confidence": round(sum(r['confidence'] for r in clean_recs) / len(clean_recs), 1) if clean_recs else 0,
                        "high_confidence": len([r for r in clean_recs if r['confidence'] >= 70]),
                        "medium_confidence": len([r for r in clean_recs if 60 <= r['confidence'] < 70]),
                    },
                }
                fixture_recommendations.append(item)

            except Exception as e:
                fixture_recommendations.append({
                    "fixture_id": str(getattr(fixture, 'id', 'unknown')),
                    "match_date": getattr(fixture, 'match_date', None),
                    "league_code": getattr(fixture, 'league_code', None),
                    "home_team": fixture.home_team.name if getattr(fixture, 'home_team', None) else "Unknown",
                    "away_team": fixture.away_team.name if getattr(fixture, 'away_team', None) else "Unknown",
                    "error": str(e),
                    "betting_recommendations": [],
                    "total_recommendations": 0,
                })
                continue

        total_recommendations = sum(f.get('total_recommendations', 0) for f in fixture_recommendations)
        successful_fixtures = len([f for f in fixture_recommendations if 'error' not in f])

        return {
            "success": True,
            "fixtures": fixture_recommendations,
            "total_fixtures": len(fixtures),
            "successful_predictions": successful_fixtures,
            "failed_predictions": len(fixtures) - successful_fixtures,
            "total_recommendations": total_recommendations,
            "processed_leagues": list(processed_leagues),
            "model_info": {
                "version": "V3_COMPLETE",
                "features": "Aggressive thresholds + New Multigol markets",
                "data_source": "database",
            },
            "filter_info": {
                "league_code": league_code,
                "date_range": f"{week_ago.strftime('%Y-%m-%d')} to {month_ahead.strftime('%Y-%m-%d')}",
                "limit": limit,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
