#!/usr/bin/env python3
"""
BACKTEST DEFINITIVO CON DATI REALI
Testa il modello ExactSimpleFooballPredictor su dati storici reali
includendo TUTTI i mercati: Over/Under, 1X2, Doppia Chance, Match Goals (MG)
"""

import sys
import os
sys.path.append('/Users/riccardobadagliacco/Documents/Sviluppo/BetSheet/betsheet_be')

import pandas as pd
import sqlite3
from collections import defaultdict
from datetime import datetime
from app.api.ml_football_exact import ExactSimpleFooballPredictor, get_recommended_bets

class RealFootballBacktest:
    
    def __init__(self):
        self.db_path = './data/football_dataset.db'
        self.predictor = ExactSimpleFooballPredictor()
        self.market_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
        
    def load_matches(self, limit=500):
        """Carica partite reali dal database football_dataset.db"""
        print(f"📂 Loading matches from {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        
        # Query completa con JOIN per ottenere nomi team e league info
        query = '''
        SELECT 
            m.id,
            m.match_date,
            ht.name as home_team,
            at.name as away_team,
            m.home_goals_ft,
            m.away_goals_ft,
            l.name as league_name,
            l.country,
            m.avg_home_odds,
            m.avg_draw_odds, 
            m.avg_away_odds,
            m.avg_over_25_odds,
            m.avg_under_25_odds,
            s.year as season_year
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id  
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        WHERE m.home_goals_ft IS NOT NULL
            AND m.away_goals_ft IS NOT NULL
            AND m.avg_home_odds IS NOT NULL
            AND m.avg_draw_odds IS NOT NULL
            AND m.avg_away_odds IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[limit])
        conn.close()
        
        print(f"✅ Loaded {len(df)} matches")
        if len(df) > 0:
            print(f"📊 Countries: {df['country'].value_counts().head().to_dict()}")
            print(f"🏆 Leagues: {df['league_name'].value_counts().head().to_dict()}")
            print(f"📅 Season years: {sorted(df['season_year'].unique())}")
        
        return df
    
    def calculate_actual_results(self, match):
        """Calcola i risultati reali per tutti i mercati betting"""
        home_goals = int(match['home_goals_ft'])
        away_goals = int(match['away_goals_ft'])
        total_goals = home_goals + away_goals
        
        results = {}
        
        # Over/Under markets
        results['Over 0.5 Goal'] = 1 if total_goals > 0.5 else 0
        results['Over 1.5 Goal'] = 1 if total_goals > 1.5 else 0
        results['Over 2.5 Goal'] = 1 if total_goals > 2.5 else 0
        results['Under 2.5 Goal'] = 1 if total_goals <= 2.5 else 0
        results['Under 3.5 Goal'] = 1 if total_goals <= 3.5 else 0
        
        # 1X2 markets
        if home_goals > away_goals:
            result_1x2 = '1'
        elif home_goals < away_goals:
            result_1x2 = '2'
        else:
            result_1x2 = 'X'
        
        results['1X2'] = result_1x2
        results['Doppia Chance 1X'] = 1 if result_1x2 in ['1', 'X'] else 0
        results['Doppia Chance 12'] = 1 if result_1x2 in ['1', '2'] else 0
        results['Doppia Chance X2'] = 1 if result_1x2 in ['X', '2'] else 0
        
        # Match Goals (MG) - Casa (Home)
        results['Multigol Casa 1-3'] = 1 if 1 <= home_goals <= 3 else 0
        results['Multigol Casa 1-4'] = 1 if 1 <= home_goals <= 4 else 0
        results['Multigol Casa 1-5'] = 1 if 1 <= home_goals <= 5 else 0
        results['Multigol Casa 2-5'] = 1 if 2 <= home_goals <= 5 else 0
        results['Multigol Casa 3-6'] = 1 if 3 <= home_goals <= 6 else 0
        results['Multigol Casa 4+'] = 1 if home_goals >= 4 else 0
        
        # Match Goals (MG) - Ospite (Away)
        results['Multigol Ospite 1-3'] = 1 if 1 <= away_goals <= 3 else 0
        results['Multigol Ospite 1-4'] = 1 if 1 <= away_goals <= 4 else 0
        results['Multigol Ospite 1-5'] = 1 if 1 <= away_goals <= 5 else 0
        results['Multigol Ospite 2-4'] = 1 if 2 <= away_goals <= 4 else 0
        results['Multigol Ospite 3-5'] = 1 if 3 <= away_goals <= 5 else 0
        results['Multigol Ospite 4+'] = 1 if away_goals >= 4 else 0
        
        return results
    
    def evaluate_recommendation(self, recommendation, actual_results):
        """Valuta se una raccomandazione è corretta"""
        market = recommendation['market']
        prediction = recommendation['prediction']
        
        # Mercati binari (Over/Under, Doppia Chance, MG)
        if market in actual_results:
            if isinstance(actual_results[market], int):
                return actual_results[market] == 1
            elif market == '1X2':
                return actual_results[market] == prediction
        
        return False
    
    def run_backtest(self, num_matches=300):
        """Esegue il backtest completo"""
        print("\n🧪 REAL FOOTBALL BETTING BACKTEST")
        print("=" * 50)
        print(f"🎯 Testing {num_matches} matches with ALL markets")
        print("📊 Markets: Over/Under, 1X2, Doppia Chance, Match Goals")
        print("=" * 50)
        
        # Carica matches
        matches_df = self.load_matches(limit=num_matches)
        
        if len(matches_df) == 0:
            print("❌ No matches loaded. Check database connection.")
            return
        
        total_recommendations = 0
        total_correct = 0
        processed_matches = 0
        
        print(f"\n🔄 Processing {len(matches_df)} matches...")
        
        for idx, match in matches_df.iterrows():
            
            if idx % 50 == 0:
                print(f"  Progress: {idx}/{len(matches_df)} matches")
            
            try:
                # Calcola risultati reali
                actual_results = self.calculate_actual_results(match)
                
                # Prepara dati per il predictor (formato compatibile)
                match_data = {
                    'Date': pd.to_datetime(match['match_date']),
                    'HomeTeam': match['home_team'],
                    'AwayTeam': match['away_team'],
                    'B365H': match['avg_home_odds'] or 2.0,
                    'B365D': match['avg_draw_odds'] or 3.0,
                    'B365A': match['avg_away_odds'] or 2.5,
                    'Avg>2.5': match['avg_over_25_odds'] or 1.8,
                    'Avg<2.5': match['avg_under_25_odds'] or 2.0
                }
                
                # Genera predizione
                prediction = self.predictor.predict_match(match_data)
                
                # Quote per get_recommended_bets
                quotes = {
                    '1': match_data['B365H'],
                    'X': match_data['B365D'], 
                    '2': match_data['B365A'],
                    'over_25': match_data['Avg>2.5'],
                    'under_25': match_data['Avg<2.5']
                }
                
                # Ottieni raccomandazioni
                recommendations = get_recommended_bets(prediction, quotes)
                
                # Valuta ogni raccomandazione
                for rec in recommendations:
                    market = rec['market']
                    is_correct = self.evaluate_recommendation(rec, actual_results)
                    
                    # Aggiorna statistiche
                    self.market_stats[market]['total'] += 1
                    if is_correct:
                        self.market_stats[market]['correct'] += 1
                        total_correct += 1
                    else:
                        self.market_stats[market]['incorrect'] += 1
                    
                    total_recommendations += 1
                
                processed_matches += 1
                
            except Exception as e:
                print(f"⚠️ Error processing match {idx}: {str(e)}")
                continue
        
        # Mostra risultati finali
        self.print_results(processed_matches, total_recommendations, total_correct)
    
    def print_results(self, processed_matches, total_recommendations, total_correct):
        """Stampa i risultati finali del backtest"""
        print(f"\n📊 BACKTEST RESULTS")
        print("=" * 60)
        
        # Performance per mercato
        print("🎯 ACCURACY BY MARKET:")
        print("-" * 60)
        
        markets_by_type = {
            'Over/Under': [],
            'Doppia Chance': [],
            'Match Goals Casa': [],
            'Match Goals Ospite': [],
            '1X2': []
        }
        
        # Categorizza mercati
        for market, stats in self.market_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                
                if 'Over' in market or 'Under' in market:
                    markets_by_type['Over/Under'].append((market, accuracy, stats))
                elif 'Doppia Chance' in market:
                    markets_by_type['Doppia Chance'].append((market, accuracy, stats))
                elif 'Multigol Casa' in market:
                    markets_by_type['Match Goals Casa'].append((market, accuracy, stats))
                elif 'Multigol Ospite' in market:
                    markets_by_type['Match Goals Ospite'].append((market, accuracy, stats))
                elif '1X2' in market:
                    markets_by_type['1X2'].append((market, accuracy, stats))
        
        # Stampa per categoria
        for category, markets in markets_by_type.items():
            if markets:
                print(f"\n🔥 {category.upper()}:")
                markets.sort(key=lambda x: x[1], reverse=True)  # Ordina per accuracy
                for market, accuracy, stats in markets:
                    correct = stats['correct']
                    total = stats['total']
                    print(f"  {market:<25}: {accuracy:5.1f}% ({correct}/{total})")
        
        # Performance globale
        overall_accuracy = (total_correct / total_recommendations * 100) if total_recommendations > 0 else 0
        
        print(f"\n🎯 OVERALL PERFORMANCE")
        print("=" * 30)
        print(f"Matches processed: {processed_matches}")
        print(f"Total recommendations: {total_recommendations}")
        print(f"Correct recommendations: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        
        # Valutazione finale
        if overall_accuracy >= 70:
            print(f"\n🚀 EXCELLENT performance! Model ready for production.")
        elif overall_accuracy >= 60:
            print(f"\n👍 GOOD performance. Model shows promise.")
        elif overall_accuracy >= 50:
            print(f"\n⚠️  AVERAGE performance. Model needs improvement.")
        else:
            print(f"\n❌ POOR performance. Model needs major revision.")
        
        print(f"\n✅ REAL BACKTEST COMPLETED")
        print(f"🎯 {overall_accuracy:.1f}% accuracy on {processed_matches} real matches")
        print("📊 All markets tested: Over/Under, 1X2, DC, Match Goals")

def main():
    """Entry point"""
    backtest = RealFootballBacktest()
    backtest.run_backtest(num_matches=300)

if __name__ == "__main__":
    main()