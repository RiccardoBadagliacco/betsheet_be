#!/usr/bin/env python3
"""
BACKTEST DEFINITIVO per ExactSimpleFooballPredictor
=================================================
Testa il modello su 50 partite random dal football_dataset.db
Analizza performance su tutti i mercati: 1X2, Over/Under, MG Casa/Ospite
"""

import sys
import os
# Add parent directory to path to import from app/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import sqlite3
from collections import defaultdict
from datetime import datetime
from app.api.ml_football_exact import ExactSimpleFooballPredictor, get_recommended_bets

class FootballBacktest:
    
    def __init__(self, num_matches=50):
        self.num_matches = num_matches
        self.db_path = '../data/football_dataset.db'
        self.predictor = ExactSimpleFooballPredictor()
        self.market_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
        self.failed_matches = 0
        self.detailed_results = []  # Per salvare risultati dettagliati
        
    def load_random_matches(self):
        """Carica partite random dal database con tutte le info necessarie"""
        print(f"📂 Caricando {self.num_matches} partite random dal database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Query con JOIN per ottenere nomi team e info complete
        query = '''
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
            m.avg_under_25_odds as "Avg<2.5",
            l.name as league_name,
            c.name as country,
            s.name as season_year
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id  
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        JOIN countries c ON l.country_id = c.id
        WHERE m.home_goals_ft IS NOT NULL
            AND m.away_goals_ft IS NOT NULL
            AND m.avg_home_odds IS NOT NULL
            AND m.avg_draw_odds IS NOT NULL
            AND m.avg_away_odds IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[self.num_matches])
        conn.close()
        
        # Converti Date in datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"✅ Caricate {len(df)} partite")
        if len(df) > 0:
            print(f"🌍 Paesi: {list(df['country'].value_counts().head(3).index)}")
            print(f"🏆 Leghe: {list(df['league_name'].value_counts().head(3).index)}")
            print(f"📅 Stagioni: {sorted(df['season_year'].unique())}")
        
        return df
    
    def calculate_actual_results(self, match):
        """Calcola risultati reali per tutti i mercati"""
        home_goals = int(match['FTHG'])
        away_goals = int(match['FTAG'])
        total_goals = home_goals + away_goals
        
        results = {}
        
        # === OVER/UNDER MARKETS ===
        results['Over 0.5 Goal'] = 1 if total_goals > 0.5 else 0
        results['Over 1.5 Goal'] = 1 if total_goals > 1.5 else 0
        results['Over 2.5 Goal'] = 1 if total_goals > 2.5 else 0
        results['Over 3.5 Goal'] = 1 if total_goals > 3.5 else 0
        results['Under 0.5 Goal'] = 1 if total_goals <= 0.5 else 0
        results['Under 1.5 Goal'] = 1 if total_goals <= 1.5 else 0
        results['Under 2.5 Goal'] = 1 if total_goals <= 2.5 else 0
        results['Under 3.5 Goal'] = 1 if total_goals <= 3.5 else 0
        
        # === 1X2 MARKETS ===
        if home_goals > away_goals:
            result_1x2 = '1'  # Casa
        elif home_goals < away_goals:
            result_1x2 = '2'  # Ospite
        else:
            result_1x2 = 'X'  # Pareggio
        
        results['1X2'] = result_1x2
        results['1X2 Casa'] = 1 if result_1x2 == '1' else 0
        results['1X2 Pareggio'] = 1 if result_1x2 == 'X' else 0
        results['1X2 Ospite'] = 1 if result_1x2 == '2' else 0
        
        # === DOPPIA CHANCE ===
        results['Doppia Chance 1X'] = 1 if result_1x2 in ['1', 'X'] else 0
        results['Doppia Chance 12'] = 1 if result_1x2 in ['1', '2'] else 0
        results['Doppia Chance X2'] = 1 if result_1x2 in ['X', '2'] else 0
        
        # === MATCH GOALS CASA ===
        results['Multigol Casa 1-3'] = 1 if 1 <= home_goals <= 3 else 0
        results['Multigol Casa 1-4'] = 1 if 1 <= home_goals <= 4 else 0
        results['Multigol Casa 1-5'] = 1 if 1 <= home_goals <= 5 else 0
        results['Multigol Casa 2-5'] = 1 if 2 <= home_goals <= 5 else 0
        results['Multigol Casa 3-6'] = 1 if 3 <= home_goals <= 6 else 0
        results['Multigol Casa 4+'] = 1 if home_goals >= 4 else 0
        
        # === MATCH GOALS OSPITE ===
        results['Multigol Ospite 1-3'] = 1 if 1 <= away_goals <= 3 else 0
        results['Multigol Ospite 1-4'] = 1 if 1 <= away_goals <= 4 else 0
        results['Multigol Ospite 1-5'] = 1 if 1 <= away_goals <= 5 else 0
        results['Multigol Ospite 2-4'] = 1 if 2 <= away_goals <= 4 else 0
        results['Multigol Ospite 3-5'] = 1 if 3 <= away_goals <= 5 else 0
        results['Multigol Ospite 4+'] = 1 if away_goals >= 4 else 0
        
        # === BTTS (Both Teams To Score) ===
        results['BTTS Si'] = 1 if home_goals > 0 and away_goals > 0 else 0
        results['BTTS No'] = 1 if home_goals == 0 or away_goals == 0 else 0
        
        return results
    
    def evaluate_recommendation(self, recommendation, actual_results):
        """Valuta se una raccomandazione è corretta"""
        market = recommendation['market']
        prediction = recommendation['prediction']
        
        # Mappatura nomi mercati per compatibilità
        market_mapping = {
            # 1X2 mappings
            '1X2 Casa': '1X2 Casa',
            '1X2 Pareggio': '1X2 Pareggio', 
            '1X2 Ospite': '1X2 Ospite',
            # Over/Under mappings (nomi potrebbero variare)
            'Over 0.5': 'Over 0.5 Goal',
            'Over 1.5': 'Over 1.5 Goal',
            'Over 2.5': 'Over 2.5 Goal',
            'Over 3.5': 'Over 3.5 Goal',
            'Under 0.5': 'Under 0.5 Goal',
            'Under 1.5': 'Under 1.5 Goal',
            'Under 2.5': 'Under 2.5 Goal', 
            'Under 3.5': 'Under 3.5 Goal',
        }
        
        # Usa mapping se disponibile, altrimenti nome originale
        actual_market = market_mapping.get(market, market)
        
        # Verifica risultato
        if actual_market in actual_results:
            if isinstance(actual_results[actual_market], int):
                # Mercati binari (0/1)
                return actual_results[actual_market] == 1
            elif actual_market == '1X2':
                # Mercato 1X2 speciale
                return actual_results[actual_market] == prediction
        
        # Se il mercato non è trovato, ritorna False
        return False
    
    def run_backtest(self):
        """Esegue il backtest completo"""
        print("\n🧪 FOOTBALL BETTING BACKTEST - MASSIVE SCALE")
        print("=" * 60)
        print(f"🎯 Testing su {self.num_matches} partite random dal football_dataset.db")
        print("📊 Mercati: 1X2, Over/Under, Doppia Chance, Match Goals, BTTS")
        print("⏰ Questa analisi richiederà diversi minuti...")
        print("📈 Validazione definitiva del modello su scala massiva")
        print("=" * 60)
        
        # Carica partite
        matches_df = self.load_random_matches()
        
        if len(matches_df) == 0:
            print("❌ Nessuna partita caricata. Controlla database.")
            return
        
        total_recommendations = 0
        total_correct = 0
        
        print(f"\n🔄 Processando {len(matches_df)} partite...")
        
        for idx in range(len(matches_df)):
            
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(matches_df)} partite ({(idx/len(matches_df)*100):.1f}%)")
            
            try:
                match = matches_df.iloc[idx]
                
                # Calcola risultati reali
                actual_results = self.calculate_actual_results(match)
                
                # *** CHIAVE: Il modello si aspetta DataFrame e indice! ***
                prediction = self.predictor.predict_match(matches_df, idx)
                
                # Prepara quote per get_recommended_bets
                quotes = {
                    '1': match.get('AvgH', 2.0),
                    'X': match.get('AvgD', 3.0),
                    '2': match.get('AvgA', 2.5),
                    'over_25': match.get('Avg>2.5', 1.8),
                    'under_25': match.get('Avg<2.5', 2.0)
                }
                
                # Ottieni raccomandazioni
                recommendations = get_recommended_bets(prediction, quotes)
                
                # Debug prima partita
                if idx == 0:
                    print(f"\\n🔍 DEBUG - Prima partita: {match['HomeTeam']} vs {match['AwayTeam']}")
                    print(f"    Risultato: {match['FTHG']}-{match['FTAG']}")
                    print(f"    Raccomandazioni generate: {len(recommendations)}")
                    if recommendations:
                        for i, rec in enumerate(recommendations[:3]):
                            print(f"      {i+1}. {rec['market']}: {rec['confidence']:.1f}%")
                
                # Valuta ogni raccomandazione
                match_recommendations = []
                match_correct = 0
                
                for rec in recommendations:
                    market = rec['market']
                    is_correct = self.evaluate_recommendation(rec, actual_results)
                    
                    # Salva dettagli raccomandazione
                    match_recommendations.append({
                        'market': market,
                        'prediction': rec['prediction'],
                        'confidence': rec['confidence'],
                        'correct': is_correct
                    })
                    
                    # Aggiorna statistiche
                    self.market_stats[market]['total'] += 1
                    if is_correct:
                        self.market_stats[market]['correct'] += 1
                        total_correct += 1
                        match_correct += 1
                    else:
                        self.market_stats[market]['incorrect'] += 1
                    
                    total_recommendations += 1
                
                # Salva risultati dettagliati per Excel
                self.detailed_results.append({
                    'Match_ID': idx,
                    'Date': match['Date'],
                    'HomeTeam': match['HomeTeam'],
                    'AwayTeam': match['AwayTeam'],
                    'League': match['league_name'],
                    'Country': match['country'],
                    'Season': match['season_year'],
                    'Result_Home': match['FTHG'],
                    'Result_Away': match['FTAG'],
                    'Total_Goals': match['FTHG'] + match['FTAG'],
                    'Match_Result': '1' if match['FTHG'] > match['FTAG'] else ('X' if match['FTHG'] == match['FTAG'] else '2'),
                    'Home_Odds': match.get('AvgH', 'N/A'),
                    'Draw_Odds': match.get('AvgD', 'N/A'), 
                    'Away_Odds': match.get('AvgA', 'N/A'),
                    'Over25_Odds': match.get('Avg>2.5', 'N/A'),
                    'Under25_Odds': match.get('Avg<2.5', 'N/A'),
                    'Total_Recommendations': len(recommendations),
                    'Correct_Recommendations': match_correct,
                    'Accuracy_Percentage': round((match_correct / len(recommendations)) * 100, 1) if recommendations else 0,
                    'Recommendations': ' | '.join([f"{r['market']}:{r['confidence']:.1f}%:{'✅' if r['correct'] else '❌'}" for r in match_recommendations])
                })
                
            except Exception as e:
                print(f"⚠️ Errore processando partita {idx}: {str(e)}")
                self.failed_matches += 1
                
                # Salva anche le partite fallite nel report
                match = matches_df.iloc[idx]
                self.detailed_results.append({
                    'Match_ID': idx,
                    'Date': match['Date'],
                    'HomeTeam': match['HomeTeam'],
                    'AwayTeam': match['AwayTeam'],
                    'League': match['league_name'],
                    'Country': match['country'],
                    'Season': match['season_year'],
                    'Result_Home': match['FTHG'],
                    'Result_Away': match['FTAG'],
                    'Total_Goals': match['FTHG'] + match['FTAG'],
                    'Match_Result': '1' if match['FTHG'] > match['FTAG'] else ('X' if match['FTHG'] == match['FTAG'] else '2'),
                    'Home_Odds': match.get('AvgH', 'N/A'),
                    'Draw_Odds': match.get('AvgD', 'N/A'), 
                    'Away_Odds': match.get('AvgA', 'N/A'),
                    'Over25_Odds': match.get('Avg>2.5', 'N/A'),
                    'Under25_Odds': match.get('Avg<2.5', 'N/A'),
                    'Total_Recommendations': 0,
                    'Correct_Recommendations': 0,
                    'Accuracy_Percentage': 0,
                    'Recommendations': f'ERROR: {str(e)}'
                })
                continue
        
        # Stampa risultati finali
        self.print_results(total_recommendations, total_correct)
        
        # Genera report Excel
        self.generate_excel_report()
    
    def print_results(self, total_recommendations, total_correct):
        """Stampa risultati dettagliati del backtest"""
        print(f"\\n📊 RISULTATI BACKTEST")
        print("=" * 60)
        
        # Partite processate
        processed = self.num_matches - self.failed_matches
        print(f"✅ Partite processate: {processed}/{self.num_matches}")
        if self.failed_matches > 0:
            print(f"❌ Partite fallite: {self.failed_matches}")
        
        if total_recommendations == 0:
            print("❌ Nessuna raccomandazione generata!")
            return
        
        print(f"📋 Raccomandazioni totali: {total_recommendations}")
        print(f"✅ Raccomandazioni corrette: {total_correct}")
        
        # Performance per categoria
        categories = {
            'Over/Under': [],
            '1X2': [],
            'Doppia Chance': [],
            'Match Goals Casa': [],
            'Match Goals Ospite': [],
            'BTTS': []
        }
        
        # Categorizza mercati
        for market, stats in self.market_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                
                if 'Over' in market or 'Under' in market:
                    categories['Over/Under'].append((market, accuracy, stats))
                elif '1X2' in market:
                    categories['1X2'].append((market, accuracy, stats))
                elif 'Doppia Chance' in market:
                    categories['Doppia Chance'].append((market, accuracy, stats))
                elif 'Multigol Casa' in market:
                    categories['Match Goals Casa'].append((market, accuracy, stats))
                elif 'Multigol Ospite' in market:
                    categories['Match Goals Ospite'].append((market, accuracy, stats))
                elif 'BTTS' in market:
                    categories['BTTS'].append((market, accuracy, stats))
        
        # Stampa per categoria
        print(f"\\n🎯 PERFORMANCE PER CATEGORIA:")
        print("-" * 60)
        
        for category, markets in categories.items():
            if markets:
                print(f"\\n🔥 {category.upper()}:")
                markets.sort(key=lambda x: x[1], reverse=True)  # Ordina per accuracy
                for market, accuracy, stats in markets:
                    correct = stats['correct']
                    total = stats['total']
                    print(f"  {market:<30}: {accuracy:5.1f}% ({correct}/{total})")
        
        # Performance globale
        overall_accuracy = (total_correct / total_recommendations) * 100
        
        print(f"\\n🎯 PERFORMANCE GLOBALE")
        print("=" * 30)
        print(f"Accuratezza complessiva: {overall_accuracy:.1f}%")
        
        # Valutazione finale
        if overall_accuracy >= 70:
            status = "🚀 ECCELLENTE"
            comment = "Modello pronto per produzione!"
        elif overall_accuracy >= 60:
            status = "👍 BUONO"
            comment = "Modello promettente, ottimizzazioni possibili."
        elif overall_accuracy >= 50:
            status = "⚠️ MEDIOCRE"
            comment = "Modello necessita miglioramenti significativi."
        else:
            status = "❌ SCARSO"
            comment = "Modello necessita revisione completa."
        
        print(f"Status: {status}")
        print(f"Commento: {comment}")
        
        print(f"\n✅ BACKTEST COMPLETATO")
        print(f"🎯 {overall_accuracy:.1f}% di accuratezza su {processed} partite reali")
    
    def generate_excel_report(self):
        """Genera report Excel dettagliato con tutte le partite"""
        print(f"\n📊 GENERANDO REPORT EXCEL...")
        
        try:
            # Crea DataFrame da risultati dettagliati
            df_results = pd.DataFrame(self.detailed_results)
            
            # Ordina per ID partita
            df_results = df_results.sort_values('Match_ID')
            
            # Nome file con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"football_backtest_report_{timestamp}.xlsx"
            
            # Crea file Excel con multiple sheets
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                
                # Sheet 1: Risultati dettagliati per partita
                df_results.to_excel(writer, sheet_name='Match_Details', index=False)
                
                # Sheet 2: Statistiche per mercato
                market_stats_list = []
                for market, stats in self.market_stats.items():
                    if stats['total'] > 0:
                        accuracy = (stats['correct'] / stats['total']) * 100
                        market_stats_list.append({
                            'Market': market,
                            'Total_Recommendations': stats['total'],
                            'Correct': stats['correct'],
                            'Incorrect': stats['incorrect'], 
                            'Accuracy_%': round(accuracy, 1)
                        })
                
                df_market_stats = pd.DataFrame(market_stats_list)
                df_market_stats = df_market_stats.sort_values('Accuracy_%', ascending=False)
                df_market_stats.to_excel(writer, sheet_name='Market_Statistics', index=False)
                
                # Sheet 3: Statistiche per lega
                league_stats = df_results.groupby(['Country', 'League']).agg({
                    'Match_ID': 'count',
                    'Total_Recommendations': 'sum',
                    'Correct_Recommendations': 'sum',
                    'Accuracy_Percentage': 'mean'
                }).reset_index()
                
                league_stats.rename(columns={
                    'Match_ID': 'Matches_Processed',
                    'Accuracy_Percentage': 'Avg_Accuracy_%'
                }, inplace=True)
                
                league_stats['Avg_Accuracy_%'] = league_stats['Avg_Accuracy_%'].round(1)
                league_stats = league_stats.sort_values('Avg_Accuracy_%', ascending=False)
                league_stats.to_excel(writer, sheet_name='League_Statistics', index=False)
                
                # Sheet 4: Sommario generale
                total_matches = len(df_results)
                processed_matches = len(df_results[df_results['Total_Recommendations'] > 0])
                failed_matches = len(df_results[df_results['Total_Recommendations'] == 0])
                total_recs = df_results['Total_Recommendations'].sum()
                total_correct_recs = df_results['Correct_Recommendations'].sum()
                overall_acc = (total_correct_recs / total_recs * 100) if total_recs > 0 else 0
                
                summary_data = {
                    'Metric': [
                        'Total Matches Loaded',
                        'Matches Successfully Processed', 
                        'Matches Failed',
                        'Success Rate %',
                        'Total Recommendations Generated',
                        'Total Correct Recommendations',
                        'Overall Accuracy %',
                        'Best Market (by accuracy)',
                        'Worst Market (by accuracy)',
                        'Most Recommended Market',
                        'Average Recommendations per Match'
                    ],
                    'Value': [
                        total_matches,
                        processed_matches,
                        failed_matches,
                        round((processed_matches/total_matches)*100, 1),
                        total_recs,
                        total_correct_recs,
                        round(overall_acc, 1),
                        df_market_stats.iloc[0]['Market'] if not df_market_stats.empty else 'N/A',
                        df_market_stats.iloc[-1]['Market'] if not df_market_stats.empty else 'N/A',
                        df_market_stats.loc[df_market_stats['Total_Recommendations'].idxmax(), 'Market'] if not df_market_stats.empty else 'N/A',
                        round(total_recs / processed_matches, 1) if processed_matches > 0 else 0
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✅ Report Excel generato: {excel_filename}")
            print(f"📋 Sheets inclusi:")
            print(f"   • Match_Details: Dettagli di ogni partita")
            print(f"   • Market_Statistics: Performance per mercato")  
            print(f"   • League_Statistics: Performance per lega")
            print(f"   • Summary: Statistiche generali")
            
        except Exception as e:
            print(f"❌ Errore generando Excel: {str(e)}")
            print("💡 Installare openpyxl: pip install openpyxl")

def main():
    """Entry point del backtest"""
    backtest = FootballBacktest(num_matches=2000)
    backtest.run_backtest()

if __name__ == "__main__":
    main()