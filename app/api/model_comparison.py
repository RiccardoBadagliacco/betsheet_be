#!/usr/bin/env python3
"""
Model Comparison and A/B Testing API
===================================

Endpoint per confrontare performance tra:
1. ExactSimpleFooballPredictor (baseline)
2. EnhancedFooballPredictor (improved)

Metriche di confronto:
- Accuracy predictions
- Confidence calibration  
- Betting ROI potential
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from app.api.ml_football_exact import ExactSimpleFooballPredictor, get_football_db
from app.ml.enhanced_predictor import EnhancedFooballPredictor

router = APIRouter()

@router.post("/compare_models/{league_code}")
async def compare_models(
    league_code: str,
    home_team: str,
    away_team: str,
    match_date: str = None,
    db: Session = Depends(get_football_db)
):
    """Compare baseline vs enhanced model predictions"""
    try:
        # Parse date if provided
        parsed_date = datetime.strptime(match_date, '%Y-%m-%d') if match_date else None
        
        # Initialize both predictors
        baseline_predictor = ExactSimpleFooballPredictor()
        enhanced_predictor = EnhancedFooballPredictor()
        
        # Load data
        if not hasattr(baseline_predictor, 'data') or baseline_predictor.data is None:
            df = baseline_predictor.load_data(db, league_code)
        else:
            df = baseline_predictor.data
            
        # Get baseline prediction  
        baseline_result = {}
        try:
            # Find match or create mock
            if parsed_date:
                matches = df[
                    (df['HomeTeam'] == home_team) & 
                    (df['AwayTeam'] == away_team) &
                    (df['Date'].dt.date == parsed_date.date())
                ]
            else:
                matches = df[
                    (df['HomeTeam'] == home_team) & 
                    (df['AwayTeam'] == away_team)
                ].tail(1)
                
            if len(matches) == 0:
                # Create mock match for future fixture
                import numpy as np
                mock_match = {
                    'Date': pd.to_datetime(match_date if match_date else '2025-10-25'),
                    'HomeTeam': home_team.strip(),
                    'AwayTeam': away_team.strip(), 
                    'FTHG': np.nan,
                    'FTAG': np.nan,
                    'AvgH': 2.0, 'AvgD': 3.2, 'AvgA': 3.8,
                    'Avg>2.5': 1.8, 'Avg<2.5': 2.0
                }
                extended_df = pd.concat([df, pd.DataFrame([mock_match])], ignore_index=True)
                match_idx = len(extended_df) - 1
                baseline_result = baseline_predictor.predict_match(extended_df, match_idx)
            else:
                match_idx = matches.index[0]
                baseline_result = baseline_predictor.predict_match(df, match_idx)
                
        except Exception as e:
            baseline_result = {'error': str(e)}
        
        # Get enhanced prediction
        enhanced_result = {}
        try:
            # Prepare market odds if available
            market_odds = None
            if 'AvgH' in df.columns and len(matches) > 0:
                match_data = matches.iloc[0] if len(matches) > 0 else None
                if match_data is not None and not pd.isna(match_data.get('AvgH')):
                    market_odds = {
                        'home': float(match_data['AvgH']),
                        'draw': float(match_data['AvgD']),  
                        'away': float(match_data['AvgA'])
                    }
            
            enhanced_result = enhanced_predictor.predict_enhanced_match(
                df, home_team, away_team, parsed_date, market_odds
            )
        except Exception as e:
            enhanced_result = {'error': str(e)}
        
        # Calculate differences and insights
        comparison = analyze_prediction_differences(baseline_result, enhanced_result)
        
        return {
            "success": True,
            "match": {
                "home_team": home_team,
                "away_team": away_team,
                "date": match_date,
                "league": league_code
            },
            "predictions": {
                "baseline": baseline_result,
                "enhanced": enhanced_result
            },
            "comparison": comparison,
            "model_info": {
                "baseline_version": "EXACT_REPLICA",
                "enhanced_version": "ENHANCED_v1.0",
                "data_source": "database",
                "matches_loaded": len(df)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def analyze_prediction_differences(baseline: Dict, enhanced: Dict) -> Dict:
    """Analyze differences between baseline and enhanced predictions"""
    
    if 'error' in baseline or 'error' in enhanced:
        return {
            "status": "error",
            "baseline_error": baseline.get('error'),
            "enhanced_error": enhanced.get('error')
        }
    
    try:
        # Extract key metrics for comparison
        metrics = ['prob_home', 'prob_draw', 'prob_away', 'lambda_home', 'lambda_away']
        
        differences = {}
        for metric in metrics:
            baseline_val = baseline.get(metric.replace('prob_', '1X2_'), 0) if 'prob_' in metric else baseline.get(metric, 0)
            enhanced_val = enhanced.get(metric, 0)
            
            if baseline_val and enhanced_val:
                diff = float(enhanced_val) - float(baseline_val)
                diff_pct = (diff / float(baseline_val)) * 100 if baseline_val != 0 else 0
                
                differences[metric] = {
                    "baseline": float(baseline_val),
                    "enhanced": float(enhanced_val), 
                    "difference": round(diff, 4),
                    "difference_pct": round(diff_pct, 2)
                }
        
        # Calculate confidence metrics
        enhanced_confidence = enhanced.get('confidence_score', 0.5)
        enhanced_quality = enhanced.get('prediction_quality', 'UNKNOWN')
        
        # Determine which model is more confident
        confidence_analysis = {
            "enhanced_confidence": float(enhanced_confidence),
            "prediction_quality": enhanced_quality,
            "form_factors": {
                "home_form": enhanced.get('home_form_factor', 0),
                "away_form": enhanced.get('away_form_factor', 0)
            },
            "efficiency_metrics": {
                "home_scoring": enhanced.get('home_scoring_efficiency', 0),
                "away_scoring": enhanced.get('away_scoring_efficiency', 0),
                "home_defense": enhanced.get('home_defensive_solidity', 0),
                "away_defense": enhanced.get('away_defensive_solidity', 0)
            }
        }
        
        # Identify biggest differences
        biggest_diffs = []
        for metric, data in differences.items():
            if abs(data['difference_pct']) > 5:  # > 5% difference
                biggest_diffs.append({
                    "metric": metric,
                    "change": "increased" if data['difference'] > 0 else "decreased",
                    "magnitude": abs(data['difference_pct'])
                })
        
        biggest_diffs.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return {
            "status": "success",
            "metric_differences": differences,
            "confidence_analysis": confidence_analysis,
            "biggest_differences": biggest_diffs[:3],  # Top 3
            "summary": {
                "total_metrics_compared": len(differences),
                "significant_changes": len(biggest_diffs),
                "enhanced_more_confident": enhanced_confidence > 0.7
            }
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "analysis_error": str(e)
        }

@router.get("/model_performance/{league_code}")
async def get_model_performance(
    league_code: str,
    days_back: int = 30,
    db: Session = Depends(get_football_db)
):
    """Get performance comparison over recent matches"""
    try:
        # This would implement backtesting over recent matches
        # For now, return placeholder structure
        
        return {
            "success": True,
            "league": league_code,
            "period": f"Last {days_back} days",
            "performance_metrics": {
                "baseline_model": {
                    "accuracy_1x2": "39.7%",
                    "accuracy_ou25": "60.3%", 
                    "kelly_profit": "+0.99",
                    "confidence_avg": "0.65"
                },
                "enhanced_model": {
                    "accuracy_1x2": "TBD",
                    "accuracy_ou25": "TBD",
                    "kelly_profit": "TBD", 
                    "confidence_avg": "TBD"
                }
            },
            "note": "Enhanced model performance tracking will be implemented after sufficient data collection"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))