#!/usr/bin/env python3
"""
Simplified Football Prediction Model
===================================

A working version of the football prediction model that focuses on core functionality
without complex configuration issues.

Usage:
    python simple_football_model.py --data leagues_csv_unified/Italy_I1_Serie_A_ALL_SEASONS.csv
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy.stats import poisson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Using simplified calculations.")

class SimpleFooballPredictor:
    """Simplified football prediction model."""
    
    def __init__(self):
        """Initialize with default settings."""
        self.global_window = 10  # Last N matches for features
        self.venue_window = 5    # Last N home/away matches
        self.market_weight = 0.6 # Weight for market vs stats (60% market, 40% stats)
        
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and clean match data."""
        print(f"Loading data from {path}...")
        df = pd.read_csv(path)
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        # Clean team names
        df['HomeTeam'] = df['HomeTeam'].str.strip()
        df['AwayTeam'] = df['AwayTeam'].str.strip()
        
        # Convert numeric columns
        numeric_cols = ['FTHG', 'FTAG', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
        return df
    
    def get_team_features(self, df: pd.DataFrame, team: str, current_date: datetime, 
                         is_home: bool = True) -> Dict[str, float]:
        """Get rolling features for a team."""
        # Get historical matches before current date
        historical = df[df['Date'] < current_date].copy()
        
        # Team matches (home and away)
        home_matches = historical[historical['HomeTeam'] == team].copy()
        away_matches = historical[historical['AwayTeam'] == team].copy()
        
        # Calculate goals for/against
        if len(home_matches) > 0:
            home_matches['goals_for'] = home_matches['FTHG']
            home_matches['goals_against'] = home_matches['FTAG']
            
        if len(away_matches) > 0:
            away_matches['goals_for'] = away_matches['FTAG'] 
            away_matches['goals_against'] = away_matches['FTHG']
        
        # Combine all matches
        all_matches = []
        if len(home_matches) > 0:
            all_matches.append(home_matches[['Date', 'goals_for', 'goals_against']].copy())
        if len(away_matches) > 0:
            all_matches.append(away_matches[['Date', 'goals_for', 'goals_against']].copy())
        
        if not all_matches:
            return {'goals_for_avg': 1.0, 'goals_against_avg': 1.0, 'total_matches': 0}
        
        team_matches = pd.concat(all_matches).sort_values('Date').tail(self.global_window)
        
        # Venue specific matches
        if is_home:
            venue_matches = home_matches[['Date', 'goals_for', 'goals_against']].tail(self.venue_window)
        else:
            venue_matches = away_matches[['Date', 'goals_for', 'goals_against']].tail(self.venue_window)
        
        # Calculate averages
        features = {
            'goals_for_avg': team_matches['goals_for'].mean() if len(team_matches) > 0 else 1.0,
            'goals_against_avg': team_matches['goals_against'].mean() if len(team_matches) > 0 else 1.0,
            'total_matches': len(team_matches)
        }
        
        # Add venue-specific features
        if len(venue_matches) > 0:
            features['venue_goals_for'] = venue_matches['goals_for'].mean()
            features['venue_goals_against'] = venue_matches['goals_against'].mean()
        else:
            features['venue_goals_for'] = features['goals_for_avg']
            features['venue_goals_against'] = features['goals_against_avg']
        
        return features
    
    def remove_vig(self, odds: Dict[str, float]) -> Dict[str, float]:
        """Remove bookmaker margin from odds."""
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
    
    def estimate_lambdas_from_market(self, p_1x2: Dict, p_ou: Dict) -> Tuple[float, float]:
        """Estimate Poisson lambdas from market probabilities."""
        # Simple estimation based on over/under 2.5
        p_over_25 = p_ou.get('over', 0.5)
        
        # Rough estimation: if P(goals > 2.5) = p, then total_goals ≈ 2.5 + adjustment
        if p_over_25 > 0.5:
            total_goals_est = 2.8  # High scoring expected
        else:
            total_goals_est = 2.2  # Lower scoring expected
        
        # Split between home and away based on 1X2 probabilities
        p_home = p_1x2.get('H', 0.4)
        p_away = p_1x2.get('A', 0.3)
        
        # Home advantage factor
        home_factor = 1.1 if p_home > p_away else 0.9
        
        lambda_home = (total_goals_est / 2) * home_factor
        lambda_away = (total_goals_est / 2) * (2 - home_factor)
        
        return max(0.1, lambda_home), max(0.1, lambda_away)
    
    def estimate_lambdas_from_stats(self, home_features: Dict, away_features: Dict) -> Tuple[float, float]:
        """Estimate Poisson lambdas from team statistics."""
        # Simple model: team attack vs opponent defense
        home_attack = home_features.get('goals_for_avg', 1.0)
        home_defense = home_features.get('goals_against_avg', 1.0)
        away_attack = away_features.get('goals_for_avg', 1.0)
        away_defense = away_features.get('goals_against_avg', 1.0)
        
        # Home advantage
        home_boost = 1.15
        
        # Estimate lambdas
        lambda_home = ((home_attack + away_defense) / 2) * home_boost
        lambda_away = (away_attack + home_defense) / 2
        
        # Venue adjustments
        venue_home = home_features.get('venue_goals_for', home_attack) / (home_attack + 0.1)
        venue_away = away_features.get('venue_goals_for', away_attack) / (away_attack + 0.1)
        
        lambda_home *= venue_home
        lambda_away *= venue_away
        
        return max(0.1, lambda_home), max(0.1, lambda_away)
    
    def calculate_probabilities(self, lambda_home: float, lambda_away: float) -> Dict[str, float]:
        """Calculate market probabilities from Poisson parameters."""
        probs = {}
        
        # Generate scoreline matrix (up to 5 goals each)
        matrix = np.zeros((6, 6))
        for h in range(6):
            for a in range(6):
                if HAS_SCIPY:
                    matrix[h, a] = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                else:
                    # Simple approximation without scipy
                    matrix[h, a] = (lambda_home**h * np.exp(-lambda_home) / np.math.factorial(h)) * \
                                  (lambda_away**a * np.exp(-lambda_away) / np.math.factorial(a))
        
        # Normalize
        matrix = matrix / np.sum(matrix)
        
        # Over/Under 0.5
        probs['O_0_5'] = 1 - matrix[0, 0]
        
        # Over/Under 1.5
        under_15 = matrix[0, 0] + matrix[1, 0] + matrix[0, 1]
        probs['O_1_5'] = 1 - under_15
        
        # Multigol Casa 1-3
        probs['MG_Casa_1_3'] = np.sum(matrix[1:4, :])
        
        # Multigol Casa 1-4
        probs['MG_Casa_1_4'] = np.sum(matrix[1:5, :])
        
        # Multigol Casa 1-5  
        probs['MG_Casa_1_5'] = np.sum(matrix[1:6, :])
        
        # Multigol Ospite 1-3
        probs['MG_Ospite_1_3'] = np.sum(matrix[:, 1:4])
        
        # Multigol Ospite 1-4
        probs['MG_Ospite_1_4'] = np.sum(matrix[:, 1:5])
        
        # Multigol Ospite 1-5
        probs['MG_Ospite_1_5'] = np.sum(matrix[:, 1:6])
        
        # 1X2
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
        
        # Top scorelines
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
    
    def predict_match(self, df: pd.DataFrame, match_idx: int) -> Dict:
        """Predict a single match."""
        match = df.iloc[match_idx]
        
        # Get team features
        home_features = self.get_team_features(df, match['HomeTeam'], match['Date'], is_home=True)
        away_features = self.get_team_features(df, match['AwayTeam'], match['Date'], is_home=False)
        
        # Market probabilities (if odds available)
        market_lambdas = (1.3, 1.1)  # Default values
        if pd.notna(match.get('AvgH')) and pd.notna(match.get('AvgD')) and pd.notna(match.get('AvgA')):
            odds_1x2 = {'H': match['AvgH'], 'D': match['AvgD'], 'A': match['AvgA']}
            p_1x2 = self.remove_vig(odds_1x2)
            
            odds_ou = {'over': match.get('Avg>2.5', 1.9), 'under': match.get('Avg<2.5', 1.9)}
            p_ou = self.remove_vig(odds_ou)
            
            market_lambdas = self.estimate_lambdas_from_market(p_1x2, p_ou)
        
        # Statistical lambdas
        stats_lambdas = self.estimate_lambdas_from_stats(home_features, away_features)
        
        # Combine with weights
        lambda_home = self.market_weight * market_lambdas[0] + (1 - self.market_weight) * stats_lambdas[0]
        lambda_away = self.market_weight * market_lambdas[1] + (1 - self.market_weight) * stats_lambdas[1]
        
        # Calculate probabilities
        probs = self.calculate_probabilities(lambda_home, lambda_away)
        
        # Build result
        result = {
            'match_idx': match_idx,
            'date': match['Date'],
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam'],
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'lambda_home_market': market_lambdas[0],
            'lambda_away_market': market_lambdas[1],
            'lambda_home_stats': stats_lambdas[0],
            'lambda_away_stats': stats_lambdas[1],
            'home_matches_count': home_features['total_matches'],
            'away_matches_count': away_features['total_matches'],
            **probs
        }
        
        # Add actual results if available
        if pd.notna(match.get('FTHG')) and pd.notna(match.get('FTAG')):
            result['actual_home_goals'] = int(match['FTHG'])
            result['actual_away_goals'] = int(match['FTAG'])
            result['actual_total_goals'] = int(match['FTHG']) + int(match['FTAG'])
            result['actual_scoreline'] = f"{int(match['FTHG'])}-{int(match['FTAG'])}"
        
        return result
    
    def predict_matches(self, df: pd.DataFrame, start_idx: int = 50) -> List[Dict]:
        """Predict multiple matches."""
        results = []
        
        print(f"Predicting matches from index {start_idx} to {len(df)}...")
        
        for idx in range(start_idx, len(df)):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(df)}")
            
            try:
                result = self.predict_match(df, idx)
                results.append(result)
            except Exception as e:
                print(f"Error predicting match {idx}: {e}")
                continue
        
        print(f"Successfully predicted {len(results)} matches")
        return results
    
    def generate_report(self, results: List[Dict], output_path: str = None) -> str:
        """Generate simple HTML report."""
        # Calculate basic metrics
        total_matches = len(results)
        
        # Count matches with actual results
        actual_results = [r for r in results if 'actual_total_goals' in r]
        
        html = f"""
        <html>
        <head><title>Football Predictions Report</title></head>
        <body>
        <h1>Football Predictions Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Summary</h2>
        <ul>
        <li>Total predictions: {total_matches}</li>
        <li>Matches with actual results: {len(actual_results)}</li>
        </ul>
        
        <h2>Sample Predictions</h2>
        <table border="1" style="border-collapse: collapse;">
        <tr>
            <th>Date</th><th>Match</th><th>λH</th><th>λA</th>
            <th>O:0.5</th><th>O:1.5</th><th>Top Scoreline</th><th>Actual</th>
        </tr>
        """
        
        for result in results[:20]:  # Show first 20
            top_score = result['top_scorelines'][0] if result['top_scorelines'] else ('N/A', 0)
            actual = result.get('actual_scoreline', 'N/A')
            
            html += f"""
            <tr>
                <td>{result['date'].strftime('%Y-%m-%d')}</td>
                <td>{result['home_team']} vs {result['away_team']}</td>
                <td>{result['lambda_home']:.2f}</td>
                <td>{result['lambda_away']:.2f}</td>
                <td>{result['O_0_5']:.3f}</td>
                <td>{result['O_1_5']:.3f}</td>
                <td>{top_score[0]} ({top_score[1]:.3f})</td>
                <td>{actual}</td>
            </tr>
            """
        
        html += """
        </table>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Report saved to {output_path}")
        
        return html

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Simple Football Prediction Model')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--out', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--report', default='report.html', help='Output HTML report')
    parser.add_argument('--start', type=int, default=50, help='Starting match index')
    parser.add_argument('--sample', type=int, help='Predict only N matches (for testing)')
    
    args = parser.parse_args()
    
    # Initialize model
    print("🏆 Simple Football Prediction Model")
    print("=" * 50)
    
    predictor = SimpleFooballPredictor()
    
    # Load data
    df = predictor.load_data(args.data)
    
    # Limit for testing
    if args.sample:
        end_idx = min(args.start + args.sample, len(df))
        print(f"Sample mode: predicting {args.sample} matches ({args.start} to {end_idx})")
    else:
        end_idx = len(df)
    
    # Make predictions
    if args.sample:
        # For sample mode, just predict the specified number
        sample_results = []
        for idx in range(args.start, min(args.start + args.sample, len(df))):
            try:
                result = predictor.predict_match(df, idx)
                sample_results.append(result)
            except Exception as e:
                print(f"Error predicting match {idx}: {e}")
        results = sample_results
    else:
        # Full prediction
        results = predictor.predict_matches(df, args.start)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Select output columns
    output_cols = [
        'match_idx', 'date', 'home_team', 'away_team',
        'lambda_home', 'lambda_away',
        'O_0_5', 'O_1_5', 'MG_Casa_1_3', 'MG_Casa_1_4', 'MG_Casa_1_5', 'MG_Ospite_1_3', 'MG_Ospite_1_4', 'MG_Ospite_1_5',
        '1X2_H', '1X2_D', '1X2_A'
    ]
    
    # Add actual results if available
    if 'actual_scoreline' in results_df.columns:
        output_cols.extend(['actual_home_goals', 'actual_away_goals', 'actual_scoreline'])
    
    # Add top scorelines
    if 'top_scorelines' in results_df.columns:
        for i in range(4):
            col_name = f'top_scoreline_{i+1}'
            results_df[col_name] = results_df['top_scorelines'].apply(
                lambda x: f"{x[i][0]} ({x[i][1]:.3f})" if len(x) > i else ""
            )
            output_cols.append(col_name)
    
    # Save CSV
    output_df = results_df[output_cols].copy()
    output_df.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")
    
    # Generate report
    predictor.generate_report(results, args.report)
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Predictions made: {len(results)}")
    if results:
        print(f"Date range: {results[0]['date'].strftime('%Y-%m-%d')} to {results[-1]['date'].strftime('%Y-%m-%d')}")
        
        # Sample prediction
        sample = results[0]
        print(f"\nSample prediction:")
        print(f"  {sample['home_team']} vs {sample['away_team']}")
        print(f"  λ_home: {sample['lambda_home']:.2f}, λ_away: {sample['lambda_away']:.2f}")
        print(f"  O:0.5 = {sample['O_0_5']:.3f}, O:1.5 = {sample['O_1_5']:.3f}")
        if 'actual_scoreline' in sample:
            print(f"  Actual result: {sample['actual_scoreline']}")
    
    print("\n✅ Prediction complete!")

if __name__ == "__main__":
    main()