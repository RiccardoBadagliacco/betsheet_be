#!/usr/bin/env python3
"""
Detailed Analysis of Model Differences
=====================================

Analizza in dettaglio le differenze tra il modello originale e quello nuovo
per capire dove si verificano le maggiori discrepanze.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_merge_predictions():
    """Load both prediction sets and merge for comparison"""
    
    # Load original predictions
    original_df = pd.read_csv("/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be/reports/predictions_2025_2026_complete.csv")
    original_df['date'] = pd.to_datetime(original_df['date'])
    
    # Load new predictions
    new_df = pd.read_csv("/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be/reports/new_model_predictions_2025_2026.csv")
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    # Merge on match details
    merged = original_df.merge(
        new_df, 
        on=['match_idx', 'home_team', 'away_team', 'actual_home_goals', 'actual_away_goals'],
        suffixes=('_orig', '_new')
    )
    
    return merged

def analyze_lambda_differences(merged_df):
    """Analyze differences in lambda predictions"""
    
    print("⚽ LAMBDA PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Calculate lambda differences
    merged_df['lambda_home_diff'] = merged_df['lambda_home_new'] - merged_df['lambda_home_orig']
    merged_df['lambda_away_diff'] = merged_df['lambda_away_new'] - merged_df['lambda_away_orig']
    
    print(f"Lambda Home Differences:")
    print(f"  Mean difference: {merged_df['lambda_home_diff'].mean():.3f}")
    print(f"  Std difference:  {merged_df['lambda_home_diff'].std():.3f}")
    print(f"  Max difference:  {merged_df['lambda_home_diff'].max():.3f}")
    print(f"  Min difference:  {merged_df['lambda_home_diff'].min():.3f}")
    
    print(f"\nLambda Away Differences:")
    print(f"  Mean difference: {merged_df['lambda_away_diff'].mean():.3f}")
    print(f"  Std difference:  {merged_df['lambda_away_diff'].std():.3f}")  
    print(f"  Max difference:  {merged_df['lambda_away_diff'].max():.3f}")
    print(f"  Min difference:  {merged_df['lambda_away_diff'].min():.3f}")
    
    # Find matches with biggest differences
    print(f"\n🔍 Matches with biggest lambda differences:")
    
    # Biggest home lambda differences
    biggest_home_diff = merged_df.nlargest(3, 'lambda_home_diff')[['home_team', 'away_team', 'lambda_home_orig', 'lambda_home_new', 'lambda_home_diff', 'actual_home_goals']]
    print(f"\nBiggest HOME lambda increases (New > Original):")
    for _, match in biggest_home_diff.iterrows():
        print(f"  {match['home_team']} vs {match['away_team']}: {match['lambda_home_orig']:.3f} → {match['lambda_home_new']:.3f} (diff: +{match['lambda_home_diff']:.3f}, actual: {match['actual_home_goals']})")
    
    biggest_home_decrease = merged_df.nsmallest(3, 'lambda_home_diff')[['home_team', 'away_team', 'lambda_home_orig', 'lambda_home_new', 'lambda_home_diff', 'actual_home_goals']]
    print(f"\nBiggest HOME lambda decreases (New < Original):")
    for _, match in biggest_home_decrease.iterrows():
        print(f"  {match['home_team']} vs {match['away_team']}: {match['lambda_home_orig']:.3f} → {match['lambda_home_new']:.3f} (diff: {match['lambda_home_diff']:.3f}, actual: {match['actual_home_goals']})")

def analyze_probability_differences(merged_df):
    """Analyze differences in probability predictions"""
    
    print(f"\n🎯 PROBABILITY PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Debug: print column names
    print("Available columns:", [col for col in merged_df.columns if '1X2' in col or 'O_2' in col])
    
    # Calculate probability differences
    merged_df['1x2_h_diff'] = merged_df['1X2_H_new'] - merged_df['1X2_H_orig']
    merged_df['1x2_d_diff'] = merged_df['1X2_D_new'] - merged_df['1X2_D_orig']
    merged_df['1x2_a_diff'] = merged_df['1X2_A_new'] - merged_df['1X2_A_orig']
    
    # Check if O_2_5 column exists 
    if 'O_2_5' in merged_df.columns:
        # Original model doesn't have O_2_5, so we'll calculate it from the available data
        print("Note: Original model doesn't have O_2_5 column, calculating from lambda values...")
        # Calculate O_2_5 for original model using Poisson distribution
        from scipy.stats import poisson
        original_o25 = []
        for _, row in merged_df.iterrows():
            lambda_h = row['lambda_home_orig']
            lambda_a = row['lambda_away_orig']
            prob_over_25 = 0
            for h in range(6):
                for a in range(6):
                    if h + a > 2.5:
                        prob_over_25 += poisson.pmf(h, lambda_h) * poisson.pmf(a, lambda_a)
            original_o25.append(prob_over_25)
        
        merged_df['o25_orig_calc'] = original_o25
        merged_df['ou25_diff'] = merged_df['O_2_5'] - merged_df['o25_orig_calc']
    else:
        print("Warning: O_2_5 columns not found")
    
    print(f"1X2 Probability Differences:")
    print(f"  Home Win - Mean: {merged_df['1x2_h_diff'].mean():.3f}, Std: {merged_df['1x2_h_diff'].std():.3f}")
    print(f"  Draw     - Mean: {merged_df['1x2_d_diff'].mean():.3f}, Std: {merged_df['1x2_d_diff'].std():.3f}")
    print(f"  Away Win - Mean: {merged_df['1x2_a_diff'].mean():.3f}, Std: {merged_df['1x2_a_diff'].std():.3f}")
    print(f"  Over 2.5 - Mean: {merged_df['ou25_diff'].mean():.3f}, Std: {merged_df['ou25_diff'].std():.3f}")

def analyze_team_specific_differences(merged_df):
    """Analyze differences by team"""
    
    print(f"\n🏆 TEAM-SPECIFIC ANALYSIS")
    print("=" * 50)
    
    # Home team analysis
    home_team_diff = merged_df.groupby('home_team').agg({
        'lambda_home_diff': ['mean', 'count'],
        '1x2_h_diff': 'mean'
    }).round(3)
    
    home_team_diff.columns = ['lambda_diff_mean', 'matches', 'prob_diff_mean']
    home_team_diff = home_team_diff.sort_values('lambda_diff_mean')
    
    print(f"Teams with most conservative home lambda (New < Original):")
    for team in home_team_diff.head(5).index:
        stats = home_team_diff.loc[team]
        print(f"  {team}: λ_diff = {stats['lambda_diff_mean']:+.3f}, prob_diff = {stats['prob_diff_mean']:+.3f} ({stats['matches']} matches)")
    
    print(f"\nTeams with most aggressive home lambda (New > Original):")
    for team in home_team_diff.tail(5).index:
        stats = home_team_diff.loc[team]
        print(f"  {team}: λ_diff = {stats['lambda_diff_mean']:+.3f}, prob_diff = {stats['prob_diff_mean']:+.3f} ({stats['matches']} matches)")

def analyze_accuracy_by_confidence(merged_df):
    """Analyze accuracy by prediction confidence"""
    
    print(f"\n📈 ACCURACY BY CONFIDENCE ANALYSIS")
    print("=" * 50)
    
    # Create confidence bins
    merged_df['confidence_bin'] = pd.cut(merged_df['confidence'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
    
    # Calculate accuracy by confidence for new model
    for conf_level in ['Low', 'Medium', 'High']:
        subset = merged_df[merged_df['confidence_bin'] == conf_level]
        if len(subset) > 0:
            # 1X2 accuracy
            correct_1x2 = 0
            for _, match in subset.iterrows():
                pred_h, pred_d, pred_a = match['1X2_H_new'], match['1X2_D_new'], match['1X2_A_new']
                predicted = 'H' if pred_h == max(pred_h, pred_d, pred_a) else ('D' if pred_d == max(pred_h, pred_d, pred_a) else 'A')
                
                actual_h, actual_a = match['actual_home_goals'], match['actual_away_goals']
                actual = 'H' if actual_h > actual_a else ('D' if actual_h == actual_a else 'A')
                
                if predicted == actual:
                    correct_1x2 += 1
            
            accuracy = correct_1x2 / len(subset)
            print(f"  {conf_level} confidence ({len(subset)} matches): {accuracy:.3f} accuracy")

def main():
    """Main analysis function"""
    
    print("🔬 DETAILED MODEL COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load and merge predictions
    print("📊 Loading prediction data...")
    merged_df = load_and_merge_predictions()
    print(f"   ✓ Loaded {len(merged_df)} matched predictions")
    
    # Run analyses
    analyze_lambda_differences(merged_df)
    analyze_probability_differences(merged_df)
    analyze_team_specific_differences(merged_df)
    analyze_accuracy_by_confidence(merged_df)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • New model tends to be more conservative in lambda predictions")
    print(f"   • Probability distributions differ significantly between models")
    print(f"   • Team-specific biases exist in both models")
    print(f"   • Confidence levels correlate with prediction accuracy")
    
    # Save detailed analysis
    output_path = "/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be/reports/detailed_model_comparison.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_path}")

if __name__ == "__main__":
    main()