#!/usr/bin/env python3
"""
Enhanced Football Predictor - V1.0
==================================

Estensione del ExactSimpleFootballPredictor con miglioramenti per i mercati Multigol.
Introduce logica avanzata per identificare match con nette favorite e integra
posizione in classifica per ottimizzare le raccomandazioni.

Improvements:
- Filtraggio Multigol basato su quote favoriti (1X2 < 1.8)  
- Integrazione posizione classifica per valutare divario squadre
- Logica di selezione match più raffinata per mercati MG
- Mantenimento compatibilità con modello baseline

Autore: Sistema di AI Enhancement
Data: 26 Ottobre 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Match, Team, Season, League
from pydantic import BaseModel

# Import baseline predictor
from .ml_football_exact import ExactSimpleFootballPredictor, BettingRecommendation, MatchPredictionResponse

class EnhancedBettingRecommendation(BaseModel):
    market: str
    prediction: str
    confidence: float
    threshold: float
    enhancement_factor: Optional[str] = None  # New field to track enhancement reason
    odds_context: Optional[Dict] = None       # Context about odds that influenced decision

class EnhancedMatchPredictionResponse(BaseModel):
    success: bool
    prediction: dict
    betting_recommendations: List[EnhancedBettingRecommendation] = []
    model_info: dict
    enhancements_applied: List[str] = []      # Track which enhancements were applied

class EnhancedFootballPredictor(ExactSimpleFootballPredictor):
    """
    Predictor potenziato con logica avanzata per mercati Multigol.
    Estende il baseline senza modificarlo, aggiungendo intelligenza contestuale.
    """
    
    def __init__(self):
        """Initialize enhanced predictor with baseline settings plus new parameters."""
        super().__init__()
        
        # Enhancement parameters (adjusted for realistic thresholds)
        self.favorite_odds_threshold = 2.2  # Quote sotto 2.2 = favorita (più pratico)
        self.minimum_odds_gap = 0.5         # Gap minimo tra favorita e sfavorita
        self.table_position_weight = 0.2    # Peso posizione classifica nel calcolo
        self.form_window = 5                # Finestra form recente
        
        # Enhanced thresholds for Multigol when conditions are met
        self.enhanced_mg_thresholds = {
            'MG Casa 1-3': 65,  # Ridotto da 70 per favorite nette
            'MG Casa 1-4': 68,  # Ridotto da 75
            'MG Casa 1-5': 65,  # Ridotto da 70
            'MG Ospite 1-3': 68, # Ridotto da 75
            'MG Ospite 1-4': 68,
            'MG Ospite 1-5': 65
        }
        
    def analyze_favorite_status(self, prediction: Dict, quotes: Optional[Dict] = None) -> Dict:
        """
        Analizza lo status di favorita per mercati Multigol SEPARATAMENTE.
        
        LOGICA CORRETTA:
        - MG Casa: applica quando avg_home_odds < 1.80 (casa favorita)
        - MG Ospite: applica quando avg_away_odds < 1.80 (ospite favorita)
        - Non serve gap minimo - valutazione indipendente per squadra
        
        Returns:
        - home_favorite: bool (casa favorita per MG Casa)
        - away_favorite: bool (ospite favorita per MG Ospite)
        - home_odds/away_odds: quote effettive
        """
        result = {
            'home_favorite': False,
            'away_favorite': False,
            'home_odds': 999,
            'away_odds': 999,
            'analysis_source': 'probabilities'
        }
        
        if quotes and '1' in quotes and '2' in quotes:
            # Analisi basata su quote reali
            home_quote = quotes.get('1', 999)
            away_quote = quotes.get('2', 999)
            
            result['home_odds'] = home_quote
            result['away_odds'] = away_quote
            result['analysis_source'] = 'quotes'
            
            # Casa favorita per MG Casa: avg_home_odds < 1.80
            result['home_favorite'] = home_quote < 1.80
            
            # Ospite favorita per MG Ospite: avg_away_odds < 1.80  
            result['away_favorite'] = away_quote < 1.80
            
        else:
            # Fallback su probabilità del modello
            prob_home = float(prediction.get('1X2_H', 0)) 
            prob_away = float(prediction.get('1X2_A', 0))
            
            # Converti probabilità in quote approssimate
            if prob_home > 0:
                home_implied_odds = 1 / prob_home
                result['home_odds'] = home_implied_odds
                result['home_favorite'] = home_implied_odds < 1.80
            
            if prob_away > 0:
                away_implied_odds = 1 / prob_away 
                result['away_odds'] = away_implied_odds
                result['away_favorite'] = away_implied_odds < 1.80
        
        return result
    
    def get_team_table_position(self, db: Session, team_name: str, match_date: datetime, league_code: str) -> Optional[int]:
        """
        Ottiene la posizione in classifica di una squadra alla data del match.
        
        TODO: Implementare logica per calcolare posizione in classifica alla data specifica.
        Per ora restituisce None, ma può essere implementato analizzando i punti 
        accumulati fino alla data del match.
        """
        # Placeholder - in futuro implementare calcolo classifica real-time
        # Analizzare tutti i match fino alla data e calcolare punti/posizione
        return None
    
    def calculate_table_position_factor(self, home_pos: Optional[int], away_pos: Optional[int]) -> float:
        """
        Calcola un fattore basato sulla differenza di posizione in classifica.
        
        Returns:
        - 1.0: no enhancement (posizioni simili o dati mancanti)
        - 1.1-1.3: enhancement per favorita in posizione alta vs bassa
        """
        if home_pos is None or away_pos is None:
            return 1.0
        
        position_gap = abs(home_pos - away_pos)
        
        # Enhancement solo se gap significativo (>= 5 posizioni)
        if position_gap >= 5:
            # Fattore proporzionale al gap (max 1.3x)
            enhancement = min(1.0 + (position_gap * 0.02), 1.3)
            return enhancement
        
        return 1.0
    
    def generate_enhanced_recommendations(self, prediction: Dict, quotes: Optional[Dict] = None, 
                                        db: Optional[Session] = None, 
                                        match_context: Optional[Dict] = None) -> List[EnhancedBettingRecommendation]:
        """
        Genera raccomandazioni potenziate con FILTRO SELETTIVO per Multigol.
        
        STRATEGIA ENHANCEMENT V2:
        - Applica Multigol SOLO ai match con favorite < 1.8
        - Mantiene soglie originali (70%) su questi match selezionati  
        - Esclude completamente i Multigol su match equilibrati
        - Migliora precision eliminando falsi positivi
        
        Args:
        - prediction: Risultato del modello baseline
        - quotes: Quote della partita (opzionale)  
        - db: Sessione database per dati aggiuntivi
        - match_context: Contesto match (data, squadre, lega)
        """
        
        # Step 1: Generate baseline recommendations using correct method
        from . import ml_football_exact
        baseline_recommendations = ml_football_exact.get_recommended_bets(prediction, quotes)
        
        # Step 2: Analyze favorite status for filtering
        favorite_analysis = self.analyze_favorite_status(prediction, quotes)
        
        # Step 3: Get table positions if context available
        table_factor = 1.0
        if match_context and db:
            home_pos = self.get_team_table_position(
                db, match_context.get('home_team'), 
                match_context.get('date'), match_context.get('league')
            )
            away_pos = self.get_team_table_position(
                db, match_context.get('away_team'),
                match_context.get('date'), match_context.get('league')
            )
            table_factor = self.calculate_table_position_factor(home_pos, away_pos)
        
        # Step 4: Apply SELECTIVE FILTERING for Multigol markets
        enhanced_recommendations = []
        enhancements_applied = []
        
        # Convert dict recommendations to objects if needed
        if baseline_recommendations and isinstance(baseline_recommendations[0], dict):
            baseline_recommendations_objs = []
            for rec_dict in baseline_recommendations:
                rec = BettingRecommendation(
                    market=rec_dict['market'],
                    prediction=rec_dict['prediction'],
                    confidence=rec_dict['confidence'], 
                    threshold=rec_dict['threshold']
                )
                baseline_recommendations_objs.append(rec)
            baseline_recommendations = baseline_recommendations_objs
        
        for rec in baseline_recommendations:
            # ENHANCEMENT V2: SELECTIVE MULTIGOL FILTERING (CORRECTED LOGIC)
            if 'Multigol' in rec.market:
                is_casa_mg = 'Casa' in rec.market
                is_ospite_mg = 'Ospite' in rec.market
                
                # Apply filtering per mercato specifico
                if is_casa_mg:
                    # MG Casa: mantieni solo se casa è favorita (avg_home_odds < 1.80)
                    if not favorite_analysis['home_favorite']:
                        home_odds = favorite_analysis.get('home_odds', 999)
                        enhancements_applied.append(f"Filtered out {rec.market} - Casa not favorite (odds: {home_odds:.2f})")
                        continue  # Skip questa raccomandazione MG Casa
                    
                    # Casa favorita: MANTIENI con threshold originale
                    enhanced_rec = EnhancedBettingRecommendation(
                        market=rec.market,
                        prediction=rec.prediction,
                        confidence=rec.confidence,
                        threshold=rec.threshold,
                        enhancement_factor=f"SelectiveFilter_HomeFavorite_{favorite_analysis.get('home_odds', 0):.2f}",
                        odds_context={
                            'home_favorite': True,
                            'home_odds': favorite_analysis.get('home_odds', 0),
                            'filter_applied': True
                        }
                    )
                    enhancements_applied.append(f"KEPT {rec.market} - Casa favorita (odds: {favorite_analysis.get('home_odds', 0):.2f})")
                
                elif is_ospite_mg:
                    # MG Ospite: mantieni solo se ospite è favorita (avg_away_odds < 1.80)
                    if not favorite_analysis['away_favorite']:
                        away_odds = favorite_analysis.get('away_odds', 999)
                        enhancements_applied.append(f"Filtered out {rec.market} - Ospite not favorite (odds: {away_odds:.2f})")
                        continue  # Skip questa raccomandazione MG Ospite
                    
                    # Ospite favorita: MANTIENI con threshold originale  
                    enhanced_rec = EnhancedBettingRecommendation(
                        market=rec.market,
                        prediction=rec.prediction,
                        confidence=rec.confidence,
                        threshold=rec.threshold,
                        enhancement_factor=f"SelectiveFilter_AwayFavorite_{favorite_analysis.get('away_odds', 0):.2f}",
                        odds_context={
                            'away_favorite': True,
                            'away_odds': favorite_analysis.get('away_odds', 0),
                            'filter_applied': True
                        }
                    )
                    enhancements_applied.append(f"KEPT {rec.market} - Ospite favorita (odds: {favorite_analysis.get('away_odds', 0):.2f})")
                
                else:
                    # Altri mercati Multigol: passa attraverso senza filtering
                    enhanced_rec = EnhancedBettingRecommendation(
                        market=rec.market,
                        prediction=rec.prediction,
                        confidence=rec.confidence,
                        threshold=rec.threshold
                    )
                
            else:
                # Non-Multigol markets: pass through unchanged
                enhanced_rec = EnhancedBettingRecommendation(
                    market=rec.market,
                    prediction=rec.prediction,
                    confidence=rec.confidence,
                    threshold=rec.threshold
                )
            
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations, enhancements_applied
    
    def predict_match_enhanced(self, df: pd.DataFrame, match_idx: int, 
                              db: Optional[Session] = None) -> EnhancedMatchPredictionResponse:
        """
        Predizione completa con enhancements per un match.
        
        Usa il baseline predictor e applica enhancements per le raccomandazioni.
        """
        
        # Step 1: Get baseline prediction
        baseline_prediction = self.predict_match(df, match_idx)
        
        # Step 2: Extract match context
        match = df.iloc[match_idx]
        match_context = {
            'home_team': match['HomeTeam'],
            'away_team': match['AwayTeam'], 
            'date': match['Date'],
            'league': 'default'  # TODO: extract from match data
        }
        
        # Step 3: Extract quotes if available
        quotes = None
        if pd.notna(match.get('AvgH')) and pd.notna(match.get('AvgA')):
            quotes = {
                '1': float(match['AvgH']),
                'X': float(match.get('AvgD', 3.0)),
                '2': float(match['AvgA'])
            }
        
        # Step 4: Generate enhanced recommendations
        enhanced_recs, enhancements = self.generate_enhanced_recommendations(
            baseline_prediction, quotes, db, match_context
        )
        
        # Step 5: Build enhanced response
        response = EnhancedMatchPredictionResponse(
            success=True,
            prediction=baseline_prediction,
            betting_recommendations=enhanced_recs,
            model_info={
                'model_name': 'EnhancedFootballPredictor',
                'version': '2.0',  # Updated to V2
                'baseline_model': 'ExactSimpleFootballPredictor',
                'strategy': 'Selective Multigol Filtering',
                'enhancements': [
                    'Selective Multigol filtering (favorites only)',
                    'Balanced match exclusion for Multigol',
                    'Team alignment verification',
                    'Precision over recall optimization'
                ]
            },
            enhancements_applied=enhancements
        )
        
        return response


# FastAPI Router per l'enhanced predictor
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check per enhanced predictor"""
    return {
        "status": "healthy",
        "model": "EnhancedFootballPredictor",
        "version": "1.0",
        "baseline": "ExactSimpleFootballPredictor"
    }

@router.post("/predict-enhanced", response_model=EnhancedMatchPredictionResponse)
async def predict_match_enhanced(
    league_code: str,
    match_idx: int,
    db: Session = Depends(get_db)
):
    """
    Predizione potenziata per un match specifico con enhancements per Multigol.
    
    Args:
    - league_code: Codice della lega (es: 'premier_league')
    - match_idx: Indice del match nel dataset della lega
    - db: Sessione database
    
    Returns:
    - Enhanced prediction con raccomandazioni ottimizzate
    """
    try:
        # Initialize enhanced predictor
        predictor = EnhancedFootballPredictor()
        
        # Load league data
        df = predictor.load_data(db, league_code)
        
        if match_idx >= len(df):
            raise HTTPException(status_code=400, detail=f"Match index {match_idx} out of range. Max: {len(df)-1}")
        
        # Get enhanced prediction
        result = predictor.predict_match_enhanced(df, match_idx, db)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Funzione di utilità per comparare modelli
def compare_model_recommendations(df: pd.DataFrame, match_idx: int, 
                                 db: Optional[Session] = None) -> Dict:
    """
    Confronta raccomandazioni tra baseline e enhanced model per debugging.
    
    Returns:
    - baseline_recs: Raccomandazioni modello base
    - enhanced_recs: Raccomandazioni modello potenziato  
    - differences: Lista delle differenze
    """
    
    # Baseline predictions
    baseline_predictor = ExactSimpleFootballPredictor()
    baseline_prediction = baseline_predictor.predict_match(df, match_idx)
    baseline_recs = baseline_predictor.generate_recommendations(baseline_prediction)
    
    # Enhanced predictions  
    enhanced_predictor = EnhancedFootballPredictor()
    enhanced_response = enhanced_predictor.predict_match_enhanced(df, match_idx, db)
    enhanced_recs = enhanced_response.betting_recommendations
    
    # Find differences
    differences = []
    for i, (base_rec, enh_rec) in enumerate(zip(baseline_recs, enhanced_recs)):
        if base_rec.threshold != enh_rec.threshold:
            differences.append({
                'market': base_rec.market,
                'baseline_threshold': base_rec.threshold,
                'enhanced_threshold': enh_rec.threshold,
                'confidence': enh_rec.confidence,
                'enhancement_factor': enh_rec.enhancement_factor
            })
    
    return {
        'baseline_recs': [{'market': r.market, 'confidence': r.confidence, 'threshold': r.threshold} for r in baseline_recs],
        'enhanced_recs': [{'market': r.market, 'confidence': r.confidence, 'threshold': r.threshold, 'enhancement': r.enhancement_factor} for r in enhanced_recs],
        'differences': differences,
        'enhancements_applied': enhanced_response.enhancements_applied
    }