#!/usr/bin/env python3
"""
EXACT REPLICA BACKTEST - Serie A 2025-2026
==========================================

This backtest uses the EXACT replica of the original SimpleFooballPredictor
to ensure identical performance between CSV and database versions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import sys
import os

# Add the app directory to Python path
sys.path.append('/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be')

# Database session will be created directly using football database
from app.api.ml_football_exact import ExactSimpleFooballPredictor

def generate_exact_database_predictions() -> pd.DataFrame:
    """Generate predictions using EXACT replica of original model"""
    
    print("🤖 Training and predicting with EXACT REPLICA database model...")
    
    # Initialize database session - use football database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    FOOTBALL_DATABASE_URL = "sqlite:///./data/football_dataset.db"
    engine = create_engine(FOOTBALL_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Initialize the EXACT model
        predictor = ExactSimpleFooballPredictor()
        
        print("   Loading data from database...")
        df = predictor.load_data(db, "I1")
        print(f"   ✓ Loaded {len(df)} matches")
        
        # Find 2025-2026 season matches
        season_matches = df[df['Date'] >= '2025-08-01'].copy()
        print(f"   📅 Found {len(season_matches)} matches in 2025-2026 season")
        
        # Generate predictions for each match
        predictions = []
        print("   🔮 Generating predictions using EXACT original method...")
        
        for i, match_idx in enumerate(season_matches.index):
            try:
                prediction = predictor.predict_match(df, match_idx)
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i+1}/{len(season_matches)}")
                    
            except Exception as e:
                print(f"   Error predicting match {match_idx}: {e}")
                continue
        
        print(f"   ✓ Generated {len(predictions)} predictions using EXACT original method")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        return predictions_df
        
    finally:
        db.close()

def calculate_exact_metrics(predictions_df: pd.DataFrame) -> Dict:
    """Calculate metrics using EXACT same method as original"""
    
    metrics = {
        'total_matches': 0,
        '1x2_correct': 0,
        'ou25_correct': 0,
        'mg_home_correct': 0,
        'mg_away_correct': 0,
        'kelly_1x2': 0,
        'kelly_ou25': 0
    }
    
    for _, match in predictions_df.iterrows():
        # Skip if no actual results available
        home_goals_col = 'actual_home_goals' if 'actual_home_goals' in match else 'FTHG'
        away_goals_col = 'actual_away_goals' if 'actual_away_goals' in match else 'FTAG'
        
        if home_goals_col not in match or away_goals_col not in match:
            continue
        if pd.isna(match[home_goals_col]) or pd.isna(match[away_goals_col]):
            continue
            
        home_goals = match[home_goals_col]
        away_goals = match[away_goals_col]
        total_goals = home_goals + away_goals
        
        metrics['total_matches'] += 1
        
        # 1X2 Analysis
        if home_goals > away_goals:
            actual_1x2 = 'H'
        elif away_goals > home_goals:
            actual_1x2 = 'A'
        else:
            actual_1x2 = 'D'
        
        # Find predicted 1X2
        pred_h = match['1X2_H']
        pred_d = match['1X2_D'] 
        pred_a = match['1X2_A']
        
        predicted_1x2 = 'H' if pred_h == max(pred_h, pred_d, pred_a) else ('D' if pred_d == max(pred_h, pred_d, pred_a) else 'A')
        
        if predicted_1x2 == actual_1x2:
            metrics['1x2_correct'] += 1
        
        # Over/Under 2.5 Analysis
        actual_ou25 = 'Over' if total_goals > 2.5 else 'Under'
        pred_over25 = match.get('O_2_5', 0.5)
        predicted_ou25 = 'Over' if pred_over25 > 0.5 else 'Under'
        
        if predicted_ou25 == actual_ou25:
            metrics['ou25_correct'] += 1
        
        # Multigol Analysis (1-3 goals)
        mg_home_actual = 1 <= home_goals <= 3
        mg_away_actual = 1 <= away_goals <= 3
        
        pred_mg_home = match.get('MG_Casa_1_3', 0.5)
        pred_mg_away = match.get('MG_Ospite_1_3', 0.5)
        
        pred_mg_home_bool = pred_mg_home > 0.5
        pred_mg_away_bool = pred_mg_away > 0.5
        
        if pred_mg_home_bool == mg_home_actual:
            metrics['mg_home_correct'] += 1
        
        if pred_mg_away_bool == mg_away_actual:
            metrics['mg_away_correct'] += 1
        
        # Kelly Criterion for 1X2 (simplified)
        odds_h = match.get('odds_1', 2.0)
        odds_d = match.get('odds_X', 3.0)  
        odds_a = match.get('odds_2', 3.0)
        
        if predicted_1x2 == 'H' and actual_1x2 == 'H':
            metrics['kelly_1x2'] += (odds_h - 1) * 0.1
        elif predicted_1x2 == 'H' and actual_1x2 != 'H':
            metrics['kelly_1x2'] -= 0.1
        elif predicted_1x2 == 'D' and actual_1x2 == 'D':
            metrics['kelly_1x2'] += (odds_d - 1) * 0.1
        elif predicted_1x2 == 'D' and actual_1x2 != 'D':
            metrics['kelly_1x2'] -= 0.1
        elif predicted_1x2 == 'A' and actual_1x2 == 'A':
            metrics['kelly_1x2'] += (odds_a - 1) * 0.1
        elif predicted_1x2 == 'A' and actual_1x2 != 'A':
            metrics['kelly_1x2'] -= 0.1
        
        # Over/Under Kelly (assume odds of 1.9)
        if predicted_ou25 == actual_ou25:
            metrics['kelly_ou25'] += 0.9 * 0.1
        else:
            metrics['kelly_ou25'] -= 0.1
    
    # Calculate percentages
    if metrics['total_matches'] > 0:
        metrics['1x2_accuracy'] = metrics['1x2_correct'] / metrics['total_matches']
        metrics['ou25_accuracy'] = metrics['ou25_correct'] / metrics['total_matches']
        metrics['mg_home_accuracy'] = metrics['mg_home_correct'] / metrics['total_matches']
        metrics['mg_away_accuracy'] = metrics['mg_away_correct'] / metrics['total_matches']
        metrics['total_kelly'] = metrics['kelly_1x2'] + metrics['kelly_ou25']
    
    return metrics

def compare_lambda_accuracy_exact(original_df: pd.DataFrame, exact_db_predictions_df: pd.DataFrame) -> Dict:
    """Compare lambda accuracy using EXACT matching"""
    
    print("⚽ Comparing lambda predictions...")
    
    lambda_stats = {
        'matches_compared': 0,
        'orig_home_mae': 0,
        'orig_away_mae': 0, 
        'exact_db_home_mae': 0,
        'exact_db_away_mae': 0
    }
    
    for _, orig_row in original_df.iterrows():
        # Find exact match in database predictions
        matching_db = exact_db_predictions_df[
            (exact_db_predictions_df['home_team'] == orig_row['home_team']) &
            (exact_db_predictions_df['away_team'] == orig_row['away_team']) &
            (pd.to_datetime(exact_db_predictions_df['date']).dt.date == orig_row['date'].date())
        ]
        
        if len(matching_db) == 1:
            db_row = matching_db.iloc[0]
            
            actual_home = orig_row['actual_home_goals']
            actual_away = orig_row['actual_away_goals']
            
            # Original model lambdas
            orig_lambda_h = orig_row['lambda_home']
            orig_lambda_a = orig_row['lambda_away']
            
            # Exact database model lambdas  
            exact_db_lambda_h = db_row['lambda_home']
            exact_db_lambda_a = db_row['lambda_away']
            
            # Calculate MAE
            lambda_stats['orig_home_mae'] += abs(orig_lambda_h - actual_home)
            lambda_stats['orig_away_mae'] += abs(orig_lambda_a - actual_away)
            lambda_stats['exact_db_home_mae'] += abs(exact_db_lambda_h - actual_home)
            lambda_stats['exact_db_away_mae'] += abs(exact_db_lambda_a - actual_away)
            lambda_stats['matches_compared'] += 1
    
    # Calculate averages
    if lambda_stats['matches_compared'] > 0:
        n = lambda_stats['matches_compared']
        lambda_stats['orig_home_mae'] /= n
        lambda_stats['orig_away_mae'] /= n
        lambda_stats['exact_db_home_mae'] /= n
        lambda_stats['exact_db_away_mae'] /= n
    
    print(f"   ✓ Compared {lambda_stats['matches_compared']} matches")
    return lambda_stats

def print_exact_comparison_report(orig_metrics: Dict, exact_db_metrics: Dict, lambda_stats: Dict):
    """Print comparison report using EXACT same calculations"""
    
    print("\n" + "="*80)
    print("🏆 EXACT REPLICA BACKTEST COMPARISON - Serie A 2025-2026")
    print("   (Using EXACT same code as original CSV model)")
    print("="*80)
    
    print(f"\n📊 BASIC STATISTICS")
    print(f"   Original CSV Model:     {orig_metrics['total_matches']} matches")
    print(f"   Exact Database Model:   {exact_db_metrics['total_matches']} matches")
    print(f"   Lambda Comparison:      {lambda_stats['matches_compared']} matches")
    
    print(f"\n🎯 PREDICTION ACCURACY (EXACT COMPARISON)")
    print(f"   Metric                    Original    ExactDB     Difference")
    print(f"   {'─'*60}")
    print(f"   1X2 Accuracy             {orig_metrics['1x2_accuracy']:.3f}     {exact_db_metrics['1x2_accuracy']:.3f}     {exact_db_metrics['1x2_accuracy'] - orig_metrics['1x2_accuracy']:+.3f}")
    print(f"   Over/Under 2.5 Accuracy  {orig_metrics['ou25_accuracy']:.3f}     {exact_db_metrics['ou25_accuracy']:.3f}     {exact_db_metrics['ou25_accuracy'] - orig_metrics['ou25_accuracy']:+.3f}")
    print(f"   Multigol Home Accuracy   {orig_metrics['mg_home_accuracy']:.3f}     {exact_db_metrics['mg_home_accuracy']:.3f}     {exact_db_metrics['mg_home_accuracy'] - orig_metrics['mg_home_accuracy']:+.3f}")
    print(f"   Multigol Away Accuracy   {orig_metrics['mg_away_accuracy']:.3f}     {exact_db_metrics['mg_away_accuracy']:.3f}     {exact_db_metrics['mg_away_accuracy'] - orig_metrics['mg_away_accuracy']:+.3f}")
    
    print(f"\n⚽ LAMBDA PREDICTION QUALITY (Lower MAE = Better)")
    print(f"   Lambda Metric            Original    ExactDB     Difference")
    print(f"   {'─'*60}")
    print(f"   Home Lambda MAE          {lambda_stats['orig_home_mae']:.3f}     {lambda_stats['exact_db_home_mae']:.3f}     {lambda_stats['exact_db_home_mae'] - lambda_stats['orig_home_mae']:+.3f}")
    print(f"   Away Lambda MAE          {lambda_stats['orig_away_mae']:.3f}     {lambda_stats['exact_db_away_mae']:.3f}     {lambda_stats['exact_db_away_mae'] - lambda_stats['orig_away_mae']:+.3f}")
    
    print(f"\n💰 KELLY CRITERION PERFORMANCE")
    print(f"   Kelly Metric             Original    ExactDB     Difference")
    print(f"   {'─'*60}")
    print(f"   1X2 Kelly Profit         {orig_metrics['kelly_1x2']:+.2f}      {exact_db_metrics['kelly_1x2']:+.2f}      {exact_db_metrics['kelly_1x2'] - orig_metrics['kelly_1x2']:+.2f}")
    print(f"   O/U Kelly Profit         {orig_metrics['kelly_ou25']:+.2f}      {exact_db_metrics['kelly_ou25']:+.2f}      {exact_db_metrics['kelly_ou25'] - orig_metrics['kelly_ou25']:+.2f}")
    print(f"   Total Kelly Profit       {orig_metrics['total_kelly']:+.2f}      {exact_db_metrics['total_kelly']:+.2f}      {exact_db_metrics['total_kelly'] - orig_metrics['total_kelly']:+.2f}")
    
    # Final assessment
    accuracy_improvements = 0
    if exact_db_metrics['1x2_accuracy'] >= orig_metrics['1x2_accuracy']: accuracy_improvements += 1
    if exact_db_metrics['ou25_accuracy'] >= orig_metrics['ou25_accuracy']: accuracy_improvements += 1
    if exact_db_metrics['mg_home_accuracy'] >= orig_metrics['mg_home_accuracy']: accuracy_improvements += 1
    if exact_db_metrics['mg_away_accuracy'] >= orig_metrics['mg_away_accuracy']: accuracy_improvements += 1
    
    lambda_improvements = 0
    if lambda_stats['exact_db_home_mae'] <= lambda_stats['orig_home_mae']: lambda_improvements += 1
    if lambda_stats['exact_db_away_mae'] <= lambda_stats['orig_away_mae']: lambda_improvements += 1
    
    profit_better = exact_db_metrics['total_kelly'] >= orig_metrics['total_kelly']
    
    overall_score = accuracy_improvements + lambda_improvements + (1 if profit_better else 0)
    
    if overall_score >= 6:
        verdict = "🟢 EXACT DATABASE MODEL PERFORMS IDENTICALLY"
    elif overall_score >= 4:
        verdict = "🟡 EXACT DATABASE MODEL PERFORMS SIMILARLY"
    else:
        verdict = "🔴 EXACT DATABASE MODEL NEEDS FURTHER INVESTIGATION"
    
    print(f"\n🔍 FINAL ASSESSMENT")
    print(f"   Accuracy Improvements:   {accuracy_improvements}/4 metrics")
    print(f"   Lambda Improvements:     {lambda_improvements}/2 metrics")
    print(f"   Profit Performance:      {'✓ Better' if profit_better else '✗ Worse'}")
    print(f"   Overall Score:           {overall_score}/7")
    print(f"   Final Verdict:           {verdict}")
    print("="*80)

def main():
    """Main backtest execution"""
    
    print("🚀 EXACT REPLICA BACKTEST: Database Model vs Original CSV")
    print("   Using EXACT same code as original SimpleFooballPredictor")
    print("="*70)
    
    # Load original CSV predictions
    print("📁 Loading original CSV predictions...")
    original_df = pd.read_csv('/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be/reports/predictions_2025_2026_complete.csv')
    original_df['date'] = pd.to_datetime(original_df['date'])
    print(f"   ✓ Loaded {len(original_df)} original predictions")
    
    # Generate exact database predictions 
    exact_db_predictions_df = generate_exact_database_predictions()
    
    # Calculate performance metrics
    print("📈 Calculating performance metrics...")
    orig_metrics = calculate_exact_metrics(original_df)
    exact_db_metrics = calculate_exact_metrics(exact_db_predictions_df)
    
    # Compare lambda predictions
    lambda_stats = compare_lambda_accuracy_exact(original_df, exact_db_predictions_df)
    
    # Print comparison report
    print_exact_comparison_report(orig_metrics, exact_db_metrics, lambda_stats)
    
    # Save exact database predictions
    exact_db_predictions_df.to_csv('exact_replica_db_predictions_2025_2026.csv', index=False)
    print(f"\n💾 Exact replica database predictions saved to: exact_replica_db_predictions_2025_2026.csv")

if __name__ == "__main__":
    main()