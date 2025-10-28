"""
football_utils.py
================

Funzioni condivise per calcolo probabilità, features di squadra e lambdas.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from addons.compute_team_profiles import TEAM_ELO, TEAM_PROFILE
from typing import Dict, Tuple, Optional

try:
    from scipy.optimize import minimize
    from scipy.stats import poisson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def get_team_features(df: pd.DataFrame, team: str, current_date: datetime, is_home: bool = True, min_history: int = 5) -> Dict[str, float]:
    """
    Estrae statistiche di gol recenti per una squadra.
    Se lo storico è insufficiente, restituisce valid=False.
    """
    
    def exp_weighted_mean(values: np.ndarray, alpha: float = 0.6) -> float:
        """Media pesata esponenzialmente: le partite più recenti pesano di più."""
        if len(values) == 0:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(values)))
        weights = weights ** alpha
        weights /= weights.sum()
        return np.dot(values, weights)

    historical = df[df['Date'] < current_date].copy()
    
    home_matches = historical[historical['HomeTeam'] == team].copy()
    away_matches = historical[historical['AwayTeam'] == team].copy()

    # goals for/against
    if len(home_matches) > 0:
        home_matches = home_matches.assign(
            goals_for=lambda df: df['FTHG'],
            goals_against=lambda df: df['FTAG']
        )
    else:
        home_matches = pd.DataFrame(columns=['Date', 'goals_for', 'goals_against'])

    if len(away_matches) > 0:
        away_matches = away_matches.assign(
            goals_for=lambda df: df['FTAG'],
            goals_against=lambda df: df['FTHG']
        )
    else:
        away_matches = pd.DataFrame(columns=['Date', 'goals_for', 'goals_against'])

    all_matches = []
    if not home_matches.empty:
        all_matches.append(home_matches[['Date', 'goals_for', 'goals_against']])
    if not away_matches.empty:
        all_matches.append(away_matches[['Date', 'goals_for', 'goals_against']])

    if not all_matches:
        return {'valid': False, 'total_matches': 0}

    team_matches = pd.concat(all_matches).sort_values('Date').tail(10)
    total_count = len(team_matches)

    # ❌ Se storico troppo corto, invalidiamo la squadra
    if total_count < min_history:
        return {'valid': False, 'total_matches': total_count}

    venue_matches = home_matches.tail(5) if is_home else away_matches.tail(5)

    features = {
        'goals_for_avg': exp_weighted_mean(team_matches['goals_for'].values),
        'goals_against_avg': exp_weighted_mean(team_matches['goals_against'].values),
        'total_matches': total_count,
        'valid': True
    }

    if len(venue_matches) > 0:
        features['venue_goals_for'] = exp_weighted_mean(venue_matches['goals_for'].values)
        features['venue_goals_against'] = exp_weighted_mean(venue_matches['goals_against'].values)
    else:
        # fallback: pesiamo comunque la media globale
        features['venue_goals_for'] = features['goals_for_avg']
        features['venue_goals_against'] = features['goals_against_avg']

    return features

def remove_vig(odds: Dict[str, float]) -> Dict[str, float]:
    if 'H' in odds and 'D' in odds and 'A' in odds:
        total_implied = 1/odds['H'] + 1/odds['D'] + 1/odds['A']
        return {
            'H': (1/odds['H']) / total_implied,
            'D': (1/odds['D']) / total_implied,
            'A': (1/odds['A']) / total_implied
        }
    elif 'over' in odds and 'under' in odds:
        total_implied = 1/odds['over'] + 1/odds['under']
        return {
            'over': (1/odds['over']) / total_implied,
            'under': (1/odds['under']) / total_implied
        }
    return odds

def estimate_lambdas_from_market(p_1x2: Dict, p_ou: Dict) -> Tuple[float, float]:
    p_over_25 = p_ou.get('over', 0.5)
    if p_over_25 > 0.5:
        total_goals_est = 2.8
    else:
        total_goals_est = 2.2
    p_home = p_1x2.get('H', 0.4)
    p_away = p_1x2.get('A', 0.3)
    home_factor = 1.1 if p_home > p_away else 0.9
    lambda_home = (total_goals_est / 2) * home_factor
    lambda_away = (total_goals_est / 2) * (2 - home_factor)
    return max(0.1, lambda_home), max(0.1, lambda_away)

def estimate_lambdas_from_stats(
        home_features: Dict, away_features: Dict,
        home_team_name: Optional[str] = None, away_team_name: Optional[str] = None,
        matchday: Optional[int] = None,
        elo_home_pre: Optional[float] = None, elo_away_pre: Optional[float] = None,
        team_profile: Optional[Dict[str, Dict]] = None
    ) -> Tuple[float, float]:
    # Base lambda
    home_attack = home_features.get('goals_for_avg', 1.0)
    home_defense = home_features.get('goals_against_avg', 1.0)
    away_attack = away_features.get('goals_for_avg', 1.0)
    away_defense = away_features.get('goals_against_avg', 1.0)

    # base
    home_boost = 1.15
    lambda_home = ((home_attack + away_defense) / 2.0) * home_boost
    lambda_away = (away_attack + home_defense) / 2.0

    # venue
    venue_home = home_features.get('venue_goals_for', home_attack) / (home_attack + 0.1)
    venue_away = away_features.get('venue_goals_for', away_attack) / (away_attack + 0.1)
    lambda_home *= venue_home
    lambda_away *= venue_away

    # ELO pre-match (preferito)
    if elo_home_pre is not None and elo_away_pre is not None:
        elo_diff = elo_home_pre - elo_away_pre
        scale = elo_diff / 1000.0
        lambda_home *= (1.0 + scale)
        lambda_away *= (1.0 - scale)

    # stagionalità
    if matchday is not None:
        season_factor = 1.05 if matchday > 20 else 1.0
        lambda_home *= season_factor
        lambda_away *= season_factor

    # stile squadra (se fornito un profilo)
    if team_profile and home_team_name:
        style_home = team_profile.get(home_team_name, {}).get('style', 'neutral')
        if style_home == 'attacking':
            lambda_home *= 1.05
        elif style_home == 'defensive':
            lambda_home *= 0.95
    if team_profile and away_team_name:
        style_away = team_profile.get(away_team_name, {}).get('style', 'neutral')
        if style_away == 'attacking':
            lambda_away *= 1.05
        elif style_away == 'defensive':
            lambda_away *= 0.95

    return max(0.1, lambda_home), max(0.1, lambda_away)

def calculate_probabilities(lambda_home: float, lambda_away: float) -> Dict[str, float]:
    probs = {}
    matrix = np.zeros((6, 6))
    for h in range(6):
        for a in range(6):
            if HAS_SCIPY:
                matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            else:
                matrix[h, a] = (lambda_home**h * np.exp(-lambda_home) / np.math.factorial(h)) * \
                              (lambda_away**a * np.exp(-lambda_away) / np.math.factorial(a))
    matrix = matrix / np.sum(matrix)
    probs['O_0_5'] = 1 - matrix[0, 0]
    under_15 = matrix[0, 0] + matrix[1, 0] + matrix[0, 1]
    probs['O_1_5'] = 1 - under_15
    probs['MG_Casa_1_3'] = np.sum(matrix[1:4, :])
    probs['MG_Casa_1_4'] = np.sum(matrix[1:5, :])
    probs['MG_Casa_1_5'] = np.sum(matrix[1:6, :])
    probs['MG_Ospite_1_3'] = np.sum(matrix[:, 1:4])
    probs['MG_Ospite_1_4'] = np.sum(matrix[:, 1:5])
    probs['MG_Ospite_1_5'] = np.sum(matrix[:, 1:6])
    prob_h = prob_d = prob_a = 0.0
    for h in range(6):
        for a in range(6):
            if h > a:
                prob_h += matrix[h, a]
            elif h == a:
                prob_d += matrix[h, a]
            else:
                prob_a += matrix[h, a]
    probs['1X2_H'] = prob_h
    probs['1X2_D'] = prob_d
    probs['1X2_A'] = prob_a
    flat_probs = matrix.flatten()
    top_indices = np.argsort(flat_probs)[-4:][::-1]
    top_scorelines = []
    for idx in top_indices:
        h, a = np.unravel_index(idx, matrix.shape)
        scoreline = f"{h}-{a}"
        prob = matrix[h, a]
        top_scorelines.append((scoreline, prob))
    probs['top_scorelines'] = top_scorelines
    return probs


def market_key_from_label(label: str) -> str | None:
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


def weighted_goal_average(matches: pd.DataFrame, team: str, is_home: bool, decay: float = 0.7, max_matches: int = 10):
    """
    Calcola la media gol ponderata esponenzialmente (forma recente).
    decay < 1 → partite recenti pesano di più.
    """
    if is_home:
        goals = matches[matches['HomeTeam'] == team]['FTHG']
    else:
        goals = matches[matches['AwayTeam'] == team]['FTAG']
    
    goals = goals.tail(max_matches).to_numpy()
    if len(goals) == 0:
        return 0

    weights = np.array([decay**i for i in range(len(goals)-1, -1, -1)])
    weights /= weights.sum()
    return float(np.dot(goals, weights))

