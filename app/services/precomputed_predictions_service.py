"""
Servizio per gestire le predizioni precompilate delle fixture
==========================================================

Questo servizio ottimizza le performance calcolando una volta sola le predizioni
quando vengono scaricate le fixture, invece di calcolarle ogni volta che vengono richieste.
"""

import json
import os
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path
from sqlalchemy.orm import Session

from app.db.database_football import get_football_db
from app.db.models_football import Fixture

logger = logging.getLogger(__name__)

class PrecomputedPredictionsService:
    """Servizio per gestire le predizioni precompilate"""
    
    def __init__(self, predictions_file: str = "data/precomputed_predictions.json"):
        self.predictions_file = predictions_file
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Crea la directory data se non esiste"""
        data_dir = Path(self.predictions_file).parent
        data_dir.mkdir(exist_ok=True)
    
    def generate_fixture_key(self, fixture: Fixture) -> str:
        """Genera una chiave univoca per identificare una fixture"""
        home_team = fixture.home_team.name if fixture.home_team else "Unknown"
        away_team = fixture.away_team.name if fixture.away_team else "Unknown"
        match_date = fixture.match_date.isoformat() if fixture.match_date else "no-date"
        league_code = fixture.league_code or "unknown"
        
        return f"{league_code}_{home_team}_{away_team}_{match_date}"
    
    def compute_all_predictions(self, db: Session) -> Dict[str, Any]:
        """
        Calcola le predizioni per tutte le fixture future nel database
        
        Returns:
            Dizionario con chiave fixture -> predizioni
        """
        logger.info("Iniziando calcolo predizioni precompilate...")
        
        # Ottieni tutte le fixture future (senza risultato)
        fixtures = db.query(Fixture).filter(
            Fixture.home_goals_ft.is_(None),  # Nessun risultato finale
            Fixture.match_date >= date.today()  # Solo fixture future
        ).all()
        
        logger.info(f"Trovate {len(fixtures)} fixture future da processare")
        
        predictions = {}
        processed_leagues = set()
        league_predictors = {}
        
        for i, fixture in enumerate(fixtures):
            try:
                fixture_key = self.generate_fixture_key(fixture)
                league_code = fixture.league_code
                
                if not league_code:
                    logger.warning(f"Fixture senza league_code: {fixture_key}")
                    continue
                
                # Log progresso ogni 10 fixture
                if (i + 1) % 10 == 0:
                    logger.info(f"Processando fixture {i + 1}/{len(fixtures)}")
                
                # Calcola predizione per questa fixture
                home_team = fixture.home_team.name if fixture.home_team else None
                away_team = fixture.away_team.name if fixture.away_team else None
                
                if not home_team or not away_team:
                    logger.warning(f"Fixture con team mancanti: {fixture_key}")
                    continue
                
                # Carica predictor per questa lega se non già fatto
                if league_code not in league_predictors:
                    try:
                        # Import inside function to avoid circular imports
                        from app.api.model_management import get_model_for_league
                        predictor = get_model_for_league(league_code)
                        df = predictor.load_data(db, league_code)
                        predictor.data = df  # Salva i dati nel predictor per riuso
                        league_predictors[league_code] = predictor
                        processed_leagues.add(league_code)
                        logger.info(f"Caricati dati per lega {league_code}")
                    except Exception as e:
                        logger.error(f"Errore caricamento dati lega {league_code}: {e}")
                        continue
                
                predictor = league_predictors[league_code]
                df = predictor.data  # I dati sono già stati caricati
                
                # Crea una row fixture per la predizione (come negli altri endpoint)
                import pandas as pd
                fixture_data = {
                    'Date': pd.to_datetime(fixture.match_date),
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'FTHG': fixture.home_goals_ft,  # None per fixture future
                    'FTAG': fixture.away_goals_ft,  # None per fixture future
                    'AvgH': fixture.avg_home_odds or 2.0,
                    'AvgD': fixture.avg_draw_odds or 3.2,
                    'AvgA': fixture.avg_away_odds or 3.8,
                    'Avg>2.5': fixture.avg_over_25_odds or 1.8,
                    'Avg<2.5': fixture.avg_under_25_odds or 2.0,
                }
                
                # Helper function to convert types for JSON serialization
                def convert_for_json(obj):
                    if hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    if hasattr(obj, 'item'):  # numpy types
                        return obj.item()
                    return obj
                
                # Aggiungi la fixture al dataframe per la predizione
                extended_df = pd.concat([df, pd.DataFrame([fixture_data])], ignore_index=True)
                match_idx = len(extended_df) - 1  # L'ultima riga è la nostra fixture
                
                # Calcola predizione usando il metodo corretto
                try:
                    prediction_result = predictor.predict_match(extended_df, match_idx)
                    
                    # Pulisci predizione in modo completo
                    from app.api.ml_football_exact import _convert_numpy_types, get_recommended_bets
                    clean_prediction = {}
                    for key, value in prediction_result.items():
                        clean_prediction[key] = _convert_numpy_types(value)
                    
                    # Calcola raccomandazioni betting con quote reali
                    real_quotes = None
                    if fixture.avg_home_odds and fixture.avg_away_odds:
                        real_quotes = {
                            '1': fixture.avg_home_odds,
                            '2': fixture.avg_away_odds,
                            'X': fixture.avg_draw_odds or 3.2
                        }
                    
                    betting_recommendations_raw = get_recommended_bets(clean_prediction, quotes=real_quotes)
                    
                    # Pulisci anche le raccomandazioni
                    betting_recommendations = []
                    for rec in betting_recommendations_raw:
                        clean_rec = {}
                        for k, v in rec.items():
                            clean_rec[k] = _convert_numpy_types(v)
                        betting_recommendations.append(clean_rec)
                    
                    # Salva predizione completa
                    # Converte tutto per JSON
                    predictions[fixture_key] = {
                        'fixture_info': {
                            'home_team': home_team,
                            'away_team': away_team,
                            'league_code': league_code,
                            'match_date': fixture.match_date.isoformat() if fixture.match_date else None,
                            'match_time': convert_for_json(fixture.match_time),
                        },
                        'prediction': clean_prediction,
                        'betting_recommendations': betting_recommendations,
                        'odds': {
                            'home': convert_for_json(fixture.avg_home_odds),
                            'draw': convert_for_json(fixture.avg_draw_odds),
                            'away': convert_for_json(fixture.avg_away_odds),
                            'over_2_5': convert_for_json(fixture.avg_over_25_odds),
                            'under_2_5': convert_for_json(fixture.avg_under_25_odds)
                        },
                        'computed_at': datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Errore calcolo predizione per {fixture_key}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Errore processamento fixture {i}: {e}")
                continue
        
        logger.info(f"Completato calcolo predizioni: {len(predictions)} predizioni generate")
        return predictions
    
    def save_predictions(self, predictions: Dict[str, Any]):
        """Salva le predizioni nel file JSON"""
        try:
            # Aggiungi metadata
            data_to_save = {
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'total_predictions': len(predictions),
                    'version': '1.0'
                },
                'predictions': predictions
            }
            
            with open(self.predictions_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Predizioni salvate in {self.predictions_file} ({len(predictions)} predizioni)")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio predizioni: {e}")
            raise
    
    def load_predictions(self) -> Dict[str, Any]:
        """Carica le predizioni dal file JSON"""
        try:
            if not os.path.exists(self.predictions_file):
                logger.warning(f"File predizioni non trovato: {self.predictions_file}")
                return {}
            
            with open(self.predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            predictions = data.get('predictions', {})
            metadata = data.get('metadata', {})
            
            logger.info(f"Caricate {len(predictions)} predizioni precompilate "
                       f"(generate il {metadata.get('generated_at', 'unknown')})")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Errore nel caricamento predizioni: {e}")
            return {}
    
    def get_prediction_for_match(self, home_team: str, away_team: str, 
                                league_code: str, match_date: str = None) -> Optional[Dict[str, Any]]:
        """
        Ottieni predizione precompilata per una specifica partita
        
        Args:
            home_team: Nome squadra casa
            away_team: Nome squadra trasferta  
            league_code: Codice lega
            match_date: Data partita (opzionale)
            
        Returns:
            Predizione precompilata o None se non trovata
        """
        predictions = self.load_predictions()
        
        # Prova prima con la data esatta se fornita
        if match_date:
            key = f"{league_code}_{home_team}_{away_team}_{match_date}"
            if key in predictions:
                return predictions[key]
        
        # Altrimenti cerca senza data (meno preciso ma funziona)
        for key, prediction in predictions.items():
            pred_info = prediction.get('fixture_info', {})
            if (pred_info.get('home_team') == home_team and 
                pred_info.get('away_team') == away_team and
                pred_info.get('league_code') == league_code):
                return prediction
        
        return None
    
    def get_all_predictions_for_league(self, league_code: str) -> List[Dict[str, Any]]:
        """Ottieni tutte le predizioni precompilate per una lega"""
        predictions = self.load_predictions()
        
        league_predictions = []
        for key, prediction in predictions.items():
            pred_info = prediction.get('fixture_info', {})
            if pred_info.get('league_code') == league_code:
                league_predictions.append(prediction)
        
        return league_predictions
    
    def compute_and_save_all_predictions(self):
        """Calcola e salva tutte le predizioni - funzione di utilità"""
        try:
            db: Session = next(get_football_db())
            try:
                predictions = self.compute_all_predictions(db)
                self.save_predictions(predictions)
                return {
                    'success': True,
                    'total_predictions': len(predictions),
                    'file_path': self.predictions_file
                }
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Errore nel calcolo e salvataggio predizioni: {e}")
            raise
    
    def is_predictions_file_recent(self, max_age_hours: int = 24) -> bool:
        """Verifica se il file delle predizioni è recente"""
        try:
            if not os.path.exists(self.predictions_file):
                return False
            
            file_time = datetime.fromtimestamp(os.path.getmtime(self.predictions_file))
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            return age_hours <= max_age_hours
            
        except Exception as e:
            logger.error(f"Errore verifica età file: {e}")
            return False