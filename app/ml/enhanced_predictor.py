#!/usr/bin/env python3
"""
Enhanced Football Predictor with Advanced Form Analysis
======================================================

Miglioramento del modello base con:
1. Weighted recent form analysis (últimos 5 matches)
2. Home/Away performance separation
3. Goal trend analysis
4. Confidence calibration

Obiettivo: Migliorare accuracy 1X2 da 39.7% a 45%+
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy.orm import Session
from app.db.models import Match, Team, Season, League

try:
    from scipy.optimize import minimize
    from scipy.stats import poisson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class EnhancedFooballPredictor:
    """Enhanced Football Predictor with Advanced Form Analysis"""
    
    def __init__(self):
        self.attack_weight = 0.7
        self.defense_weight = 0.3  
        self.market_weight = 0.6  # Will be optimized
        self.form_weight = 0.8    # NEW: Form importance
        self.form_decay = 0.85    # NEW: Recent matches weight more
        self.home_advantage = 0.1 # NEW: Home advantage factor
        
        # Enhanced parameters
        self.recent_matches = 5   # Number of recent matches to analyze
        self.min_matches = 3      # Minimum matches for reliable stats
        
    def calculate_enhanced_form(self, team_matches: pd.DataFrame, is_home: bool) -> Dict:
        """Calculate enhanced form metrics with recency weighting"""
        
        if len(team_matches) < self.min_matches:
            return {
                'goals_scored_avg': 1.0,
                'goals_conceded_avg': 1.0,
                'form_factor': 0.5,
                'scoring_efficiency': 0.5,
                'defensive_solidity': 0.5
            }
        
        # Get recent matches with exponential weighting
        recent = team_matches.head(self.recent_matches).copy()
        
        # Create weights (more recent = higher weight)
        weights = np.array([self.form_decay ** i for i in range(len(recent))])
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted averages
        if is_home:
            goals_scored = recent['FTHG'].values
            goals_conceded = recent['FTAG'].values
        else:
            goals_scored = recent['FTAG'].values  
            goals_conceded = recent['FTHG'].values
            
        # Weighted statistics
        goals_scored_avg = np.average(goals_scored, weights=weights)
        goals_conceded_avg = np.average(goals_conceded, weights=weights)
        
        # Form factor (wins/draws ratio in recent matches)
        if is_home:
            results = []
            for _, match in recent.iterrows():
                if match['FTHG'] > match['FTAG']:
                    results.append(1.0)  # Win
                elif match['FTHG'] == match['FTAG']:
                    results.append(0.5)  # Draw
                else:
                    results.append(0.0)  # Loss
        else:
            results = []
            for _, match in recent.iterrows():
                if match['FTAG'] > match['FTHG']:
                    results.append(1.0)  # Win
                elif match['FTAG'] == match['FTHG']:
                    results.append(0.5)  # Draw
                else:
                    results.append(0.0)  # Loss
                    
        form_factor = np.average(results, weights=weights)
        
        # Advanced metrics
        scoring_efficiency = min(goals_scored_avg / 2.0, 1.0)  # Normalized to [0,1]
        defensive_solidity = max(0.0, 1.0 - goals_conceded_avg / 2.0)  # Inverse of conceded
        
        return {
            'goals_scored_avg': goals_scored_avg,
            'goals_conceded_avg': goals_conceded_avg, 
            'form_factor': form_factor,
            'scoring_efficiency': scoring_efficiency,
            'defensive_solidity': defensive_solidity
        }
    
    def calculate_enhanced_lambda(self, home_stats: Dict, away_stats: Dict, market_odds: Dict) -> Tuple[float, float]:
        """Calculate enhanced lambda with form factors and home advantage"""
        
        # Base lambda from form and efficiency
        home_attack = home_stats['scoring_efficiency'] * (1 + self.home_advantage)
        home_defense = home_stats['defensive_solidity']
        away_attack = away_stats['scoring_efficiency'] 
        away_defense = away_stats['defensive_solidity'] * (1 + self.home_advantage * 0.5)
        
        # Enhanced lambda calculation
        lambda_home_form = home_attack * (1 - away_defense) * 2.5
        lambda_away_form = away_attack * (1 - home_defense) * 2.5
        
        # Market-based lambda (if odds available)
        if market_odds and all(key in market_odds for key in ['home', 'draw', 'away']):
            # Convert odds to implied probabilities
            prob_home = 1.0 / market_odds['home']
            prob_draw = 1.0 / market_odds['draw'] 
            prob_away = 1.0 / market_odds['away']
            
            # Normalize probabilities (remove bookmaker margin)
            total_prob = prob_home + prob_draw + prob_away
            prob_home /= total_prob
            prob_draw /= total_prob
            prob_away /= total_prob
            
            # Estimate lambda from market probabilities
            # Using reverse Poisson approximation
            lambda_home_market = max(0.1, -np.log(prob_draw))
            lambda_away_market = max(0.1, -np.log(prob_draw) * prob_away / prob_home)
        else:
            lambda_home_market = 1.3  # Default values
            lambda_away_market = 1.1
        
        # Blend form-based and market-based lambdas
        lambda_home = (self.form_weight * lambda_home_form + 
                      (1 - self.form_weight) * lambda_home_market)
        lambda_away = (self.form_weight * lambda_away_form + 
                      (1 - self.form_weight) * lambda_away_market)
        
        return max(0.1, lambda_home), max(0.1, lambda_away)
    
    def predict_enhanced_match(self, df: pd.DataFrame, home_team: str, away_team: str, 
                             match_date: datetime = None, market_odds: Dict = None) -> Dict:
        """Enhanced match prediction with form analysis"""
        
        # Filter matches up to the prediction date
        if match_date:
            df = df[df['Date'] < match_date].copy()
        
        # Get recent matches for both teams
        home_matches = df[df['HomeTeam'] == home_team].tail(20).sort_values('Date', ascending=False)
        away_matches = df[df['AwayTeam'] == away_team].tail(20).sort_values('Date', ascending=False)
        
        # Calculate enhanced form statistics  
        home_stats = self.calculate_enhanced_form(home_matches, is_home=True)
        away_stats = self.calculate_enhanced_form(away_matches, is_home=False)
        
        # Calculate enhanced lambdas
        lambda_home, lambda_away = self.calculate_enhanced_lambda(
            home_stats, away_stats, market_odds
        )
        
        # Generate probability matrix using Poisson
        max_goals = 8
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if HAS_SCIPY:
                    prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                else:
                    # Simple approximation without scipy
                    prob_matrix[i, j] = (np.exp(-lambda_home) * lambda_home**i / np.math.factorial(i)) * \
                                       (np.exp(-lambda_away) * lambda_away**j / np.math.factorial(j))
        
        # Calculate market probabilities
        prob_home = np.sum(prob_matrix[1:, 0]) + np.sum([prob_matrix[i, j] 
                                                        for i in range(1, max_goals + 1) 
                                                        for j in range(1, max_goals + 1) if i > j])
        prob_draw = np.sum([prob_matrix[i, i] for i in range(max_goals + 1)])
        prob_away = 1.0 - prob_home - prob_draw
        
        # Over/Under probabilities
        over_05 = 1.0 - prob_matrix[0, 0]
        over_15 = 1.0 - prob_matrix[0, 0] - prob_matrix[0, 1] - prob_matrix[1, 0]
        over_25 = np.sum([prob_matrix[i, j] for i in range(max_goals + 1) 
                         for j in range(max_goals + 1) if i + j > 2])
        
        # BTTS probabilities  
        btts_no = prob_matrix[0, :].sum() + prob_matrix[:, 0].sum() - prob_matrix[0, 0]
        btts_yes = 1.0 - btts_no
        
        # Top scorelines
        scorelines = []
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scorelines.append((f"{i}-{j}", prob_matrix[i, j]))
        scorelines.sort(key=lambda x: x[1], reverse=True)
        
        # Enhanced result with form context
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date.isoformat() if match_date else datetime.now().isoformat(),
            
            # Core predictions
            'lambda_home': float(lambda_home),
            'lambda_away': float(lambda_away),
            'prob_home': float(prob_home),
            'prob_draw': float(prob_draw), 
            'prob_away': float(prob_away),
            
            # Over/Under
            'over_05': float(over_05),
            'over_15': float(over_15),
            'over_25': float(over_25),
            
            # BTTS
            'btts_yes': float(btts_yes),
            'btts_no': float(btts_no),
            
            # Enhanced form context
            'home_form_factor': float(home_stats['form_factor']),
            'away_form_factor': float(away_stats['form_factor']),
            'home_scoring_efficiency': float(home_stats['scoring_efficiency']),
            'away_scoring_efficiency': float(away_stats['scoring_efficiency']),
            'home_defensive_solidity': float(home_stats['defensive_solidity']),
            'away_defensive_solidity': float(away_stats['defensive_solidity']),
            
            # Match context
            'confidence_score': float(min(home_stats['form_factor'] + away_stats['form_factor'], 1.0)),
            'prediction_quality': 'HIGH' if len(home_matches) >= 10 and len(away_matches) >= 10 else 'MEDIUM' if len(home_matches) >= 5 and len(away_matches) >= 5 else 'LOW',
            
            # Top scorelines
            'top_scorelines': scorelines[:4],
            'model_version': 'ENHANCED_v1.0'
        }
        
        return result

# Usage example and integration point
def create_enhanced_predictor_endpoint():
    """Create enhanced predictor for API integration"""
    return EnhancedFooballPredictor()

if __name__ == "__main__":
    # Test the enhanced predictor
    predictor = EnhancedFooballPredictor()
    print("🚀 Enhanced Football Predictor initialized!")
    print(f"📊 Form weight: {predictor.form_weight}")
    print(f"⚡ Form decay: {predictor.form_decay}")  
    print(f"🏠 Home advantage: {predictor.home_advantage}")