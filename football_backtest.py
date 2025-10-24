#!/usr/bin/env python3
"""
🎯 Football Model Backtesting & Performance Analysis
Analizza la performance del modello di predizione su dati storici
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

class ModelBacktester:
    def __init__(self):
        self.results = {}
    
    def load_predictions(self, csv_path):
        """Carica le predizioni del modello"""
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 Caricati {len(df)} predizioni da {csv_path}")
            return df
        except Exception as e:
            print(f"❌ Errore caricamento file: {e}")
            return None
    
    def analyze_over_under_performance(self, df):
        """Analizza performance Over/Under"""
        print("\n🎯 ANALISI OVER/UNDER PERFORMANCE")
        print("=" * 50)
        
        # Calcola gol totali reali
        df['actual_total_goals'] = df['actual_home_goals'] + df['actual_away_goals']
        
        # Analizza Over/Under 0.5
        over_05_predicted = df['O_0_5'] > 0.5
        over_05_actual = df['actual_total_goals'] > 0.5
        over_05_accuracy = (over_05_predicted == over_05_actual).mean()
        
        # Analizza Over/Under 1.5
        over_15_predicted = df['O_1_5'] > 0.5
        over_15_actual = df['actual_total_goals'] > 1.5
        over_15_accuracy = (over_15_predicted == over_15_actual).mean()
        
        # Analizza Over/Under 2.5
        if 'O_2_5' in df.columns:
            over_25_predicted = df['O_2_5'] > 0.5
            over_25_actual = df['actual_total_goals'] > 2.5
            over_25_accuracy = (over_25_predicted == over_25_actual).mean()
        else:
            # Calcola O2.5 dalla distribuzione Poisson se non presente
            over_25_prob = self.calculate_over_25_poisson(df)
            over_25_predicted = over_25_prob > 0.5
            over_25_actual = df['actual_total_goals'] > 2.5
            over_25_accuracy = (over_25_predicted == over_25_actual).mean()
        
        print(f"Over 0.5 Accuracy: {over_05_accuracy:.3f} ({over_05_accuracy*100:.1f}%)")
        print(f"Over 1.5 Accuracy: {over_15_accuracy:.3f} ({over_15_accuracy*100:.1f}%)")
        print(f"Over 2.5 Accuracy: {over_25_accuracy:.3f} ({over_25_accuracy*100:.1f}%)")
        
        # Dettagli distribuzione gol
        print(f"\n📈 Distribuzione gol reali:")
        goals_dist = df['actual_total_goals'].value_counts().sort_index()
        for goals, count in goals_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {goals} gol: {count} partite ({percentage:.1f}%)")
        
        return {
            'over_05_accuracy': over_05_accuracy,
            'over_15_accuracy': over_15_accuracy,
            'over_25_accuracy': over_25_accuracy,
            'goals_distribution': goals_dist.to_dict()
        }
    
    def calculate_over_25_poisson(self, df):
        """Calcola probabilità Over 2.5 dalla distribuzione Poisson"""
        lambda_home = df['lambda_home'].values
        lambda_away = df['lambda_away'].values
        
        # P(X + Y > 2.5) = 1 - P(X + Y <= 2)
        # Dove X ~ Poisson(λ_home) e Y ~ Poisson(λ_away)
        prob_0 = np.exp(-(lambda_home + lambda_away))
        prob_1 = (lambda_home + lambda_away) * np.exp(-(lambda_home + lambda_away))
        prob_2 = ((lambda_home + lambda_away)**2 / 2) * np.exp(-(lambda_home + lambda_away))
        
        prob_under_25 = prob_0 + prob_1 + prob_2
        prob_over_25 = 1 - prob_under_25
        
        return prob_over_25
    
    def analyze_1x2_performance(self, df):
        """Analizza performance 1X2"""
        print("\n⚽ ANALISI 1X2 PERFORMANCE")
        print("=" * 50)
        
        # Determina risultato reale
        df['actual_result'] = np.where(
            df['actual_home_goals'] > df['actual_away_goals'], 'H',
            np.where(df['actual_home_goals'] < df['actual_away_goals'], 'A', 'D')
        )
        
        # Predizione modello (maggiore probabilità)
        df['predicted_result'] = df[['1X2_H', '1X2_D', '1X2_A']].idxmax(axis=1)
        df['predicted_result'] = df['predicted_result'].map({'1X2_H': 'H', '1X2_D': 'D', '1X2_A': 'A'})
        
        # Accuracy complessiva
        accuracy_1x2 = (df['actual_result'] == df['predicted_result']).mean()
        
        # Accuracy per tipo di risultato
        home_wins_actual = (df['actual_result'] == 'H').sum()
        draws_actual = (df['actual_result'] == 'D').sum()
        away_wins_actual = (df['actual_result'] == 'A').sum()
        
        home_predicted_correct = ((df['actual_result'] == 'H') & (df['predicted_result'] == 'H')).sum()
        draws_predicted_correct = ((df['actual_result'] == 'D') & (df['predicted_result'] == 'D')).sum()
        away_predicted_correct = ((df['actual_result'] == 'A') & (df['predicted_result'] == 'A')).sum()
        
        print(f"Accuracy 1X2 complessiva: {accuracy_1x2:.3f} ({accuracy_1x2*100:.1f}%)")
        print(f"\nDettaglio per risultato:")
        print(f"  Home wins: {home_predicted_correct}/{home_wins_actual} ({(home_predicted_correct/home_wins_actual*100) if home_wins_actual > 0 else 0:.1f}%)")
        print(f"  Draws: {draws_predicted_correct}/{draws_actual} ({(draws_predicted_correct/draws_actual*100) if draws_actual > 0 else 0:.1f}%)")
        print(f"  Away wins: {away_predicted_correct}/{away_wins_actual} ({(away_predicted_correct/away_wins_actual*100) if away_wins_actual > 0 else 0:.1f}%)")
        
        return {
            'accuracy_1x2': accuracy_1x2,
            'home_accuracy': home_predicted_correct/home_wins_actual if home_wins_actual > 0 else 0,
            'draw_accuracy': draws_predicted_correct/draws_actual if draws_actual > 0 else 0,
            'away_accuracy': away_predicted_correct/away_wins_actual if away_wins_actual > 0 else 0
        }
    
    def analyze_expected_vs_actual(self, df):
        """Analizza lambda predetti vs gol reali"""
        print("\n📊 ANALISI LAMBDA vs REALTÀ")
        print("=" * 50)
        
        # Errori predizione
        home_error = np.abs(df['lambda_home'] - df['actual_home_goals'])
        away_error = np.abs(df['lambda_away'] - df['actual_away_goals'])
        total_error = np.abs((df['lambda_home'] + df['lambda_away']) - (df['actual_home_goals'] + df['actual_away_goals']))
        
        print(f"Errore medio λ_home: {home_error.mean():.3f}")
        print(f"Errore medio λ_away: {away_error.mean():.3f}")
        print(f"Errore medio totale: {total_error.mean():.3f}")
        
        # Correlazioni
        home_corr = np.corrcoef(df['lambda_home'], df['actual_home_goals'])[0, 1]
        away_corr = np.corrcoef(df['lambda_away'], df['actual_away_goals'])[0, 1]
        
        print(f"\nCorrelazione λ_home vs gol_casa: {home_corr:.3f}")
        print(f"Correlazione λ_away vs gol_ospiti: {away_corr:.3f}")
        
        return {
            'home_mae': home_error.mean(),
            'away_mae': away_error.mean(),
            'total_mae': total_error.mean(),
            'home_correlation': home_corr,
            'away_correlation': away_corr
        }
    
    def analyze_multigol_performance(self, df):
        """Analizza performance Multigol Casa e Ospite"""
        print("\n🎯 ANALISI MULTIGOL PERFORMANCE")
        print("=" * 50)
        
        results = {}
        
        # Multigol Casa
        for mg_range in ['1_3', '1_4', '1_5']:
            col_name = f'MG_Casa_{mg_range}'
            if col_name in df.columns:
                # Determina range gol casa
                min_goals, max_goals = map(int, mg_range.split('_'))
                actual_mg = (df['actual_home_goals'] >= min_goals) & (df['actual_home_goals'] <= max_goals)
                predicted_mg = df[col_name] > 0.5
                
                accuracy = (actual_mg == predicted_mg).mean()
                actual_count = actual_mg.sum()
                predicted_count = predicted_mg.sum()
                
                print(f"MG Casa {min_goals}-{max_goals}:")
                print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"  Actual occurrences: {actual_count}/{len(df)} ({actual_count/len(df)*100:.1f}%)")
                print(f"  Predicted (>50%): {predicted_count}/{len(df)} ({predicted_count/len(df)*100:.1f}%)")
                
                results[f'mg_casa_{mg_range}_accuracy'] = accuracy
        
        # Multigol Ospite  
        for mg_range in ['1_3', '1_4', '1_5']:
            col_name = f'MG_Ospite_{mg_range}'
            if col_name in df.columns:
                # Determina range gol ospite
                min_goals, max_goals = map(int, mg_range.split('_'))
                actual_mg = (df['actual_away_goals'] >= min_goals) & (df['actual_away_goals'] <= max_goals)
                predicted_mg = df[col_name] > 0.5
                
                accuracy = (actual_mg == predicted_mg).mean()
                actual_count = actual_mg.sum()
                predicted_count = predicted_mg.sum()
                
                print(f"MG Ospite {min_goals}-{max_goals}:")
                print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"  Actual occurrences: {actual_count}/{len(df)} ({actual_count/len(df)*100:.1f}%)")
                print(f"  Predicted (>50%): {predicted_count}/{len(df)} ({predicted_count/len(df)*100:.1f}%)")
                
                results[f'mg_ospite_{mg_range}_accuracy'] = accuracy
        
        return results
    
    def betting_simulation(self, df, stake=10):
        """Simula betting con le predizioni del modello"""
        print(f"\n💰 SIMULAZIONE BETTING (Stake: €{stake})")
        print("=" * 50)
        
        total_bets = 0
        total_stake = 0
        total_winnings = 0
        
        # Over 1.5 strategy (bet quando probabilità > 70%)
        over15_bets = df[df['O_1_5'] > 0.7].copy()
        over15_wins = over15_bets[over15_bets['actual_home_goals'] + over15_bets['actual_away_goals'] > 1.5]
        
        if len(over15_bets) > 0:
            over15_stake = len(over15_bets) * stake
            over15_winnings = len(over15_wins) * stake * 1.8  # Assumendo quota media 1.8
            over15_profit = over15_winnings - over15_stake
            over15_roi = (over15_profit / over15_stake) * 100
            
            print(f"Over 1.5 Strategy (prob > 70%):")
            print(f"  Scommesse: {len(over15_bets)}")
            print(f"  Vincite: {len(over15_wins)}")
            print(f"  Win Rate: {(len(over15_wins)/len(over15_bets)*100):.1f}%")
            print(f"  Profitto: €{over15_profit:.2f}")
            print(f"  ROI: {over15_roi:.1f}%")
            
            total_bets += len(over15_bets)
            total_stake += over15_stake
            total_winnings += over15_winnings
        
        # 1X2 strategy (bet quando probabilità > 60%)
        confident_1x2 = df[df[['1X2_H', '1X2_D', '1X2_A']].max(axis=1) > 0.6].copy()
        if len(confident_1x2) > 0:
            # Semplificazione: assumiamo vincita se predizione corretta con quota 2.5
            x1x2_wins = confident_1x2[confident_1x2['actual_result'] == confident_1x2['predicted_result']]
            x1x2_stake = len(confident_1x2) * stake
            x1x2_winnings = len(x1x2_wins) * stake * 2.5
            x1x2_profit = x1x2_winnings - x1x2_stake
            x1x2_roi = (x1x2_profit / x1x2_stake) * 100 if x1x2_stake > 0 else 0
            
            print(f"\n1X2 Strategy (prob > 60%):")
            print(f"  Scommesse: {len(confident_1x2)}")
            print(f"  Vincite: {len(x1x2_wins)}")
            print(f"  Win Rate: {(len(x1x2_wins)/len(confident_1x2)*100):.1f}%")
            print(f"  Profitto: €{x1x2_profit:.2f}")
            print(f"  ROI: {x1x2_roi:.1f}%")
            
            total_bets += len(confident_1x2)
            total_stake += x1x2_stake
            total_winnings += x1x2_winnings
        
        if total_stake > 0:
            total_profit = total_winnings - total_stake
            total_roi = (total_profit / total_stake) * 100
            
            print(f"\n📊 TOTALE SIMULAZIONE:")
            print(f"  Scommesse totali: {total_bets}")
            print(f"  Stake totale: €{total_stake:.2f}")
            print(f"  Vincite totali: €{total_winnings:.2f}")
            print(f"  Profitto totale: €{total_profit:.2f}")
            print(f"  ROI totale: {total_roi:.1f}%")
        
        return {
            'total_bets': total_bets,
            'total_profit': total_profit if total_stake > 0 else 0,
            'total_roi': total_roi if total_stake > 0 else 0
        }
    
    def generate_backtest_report(self, df, output_path):
        """Genera report completo del backtesting"""
        print(f"\n📝 Generando report completo...")
        
        # Analizza tutte le metriche
        ou_results = self.analyze_over_under_performance(df)
        x1x2_results = self.analyze_1x2_performance(df)
        lambda_results = self.analyze_expected_vs_actual(df)
        mg_results = self.analyze_multigol_performance(df)
        betting_results = self.betting_simulation(df)
        
        # Genera report HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Football Model Backtesting Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .good {{ background: #d4edda; }}
                .medium {{ background: #fff3cd; }}
                .poor {{ background: #f8d7da; }}
                h2 {{ color: #333; border-bottom: 2px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>🎯 Football Prediction Model - Backtesting Report</h1>
            
            <h2>📊 Dataset Overview</h2>
            <div class="metric">
                <strong>Total Matches:</strong> {len(df)}<br>
                <strong>Date Range:</strong> {df['date'].min()} to {df['date'].max()}<br>
                <strong>Average Goals per Match:</strong> {(df['actual_home_goals'] + df['actual_away_goals']).mean():.2f}
            </div>
            
            <h2>🎯 Over/Under Performance</h2>
            <div class="metric {'good' if ou_results['over_15_accuracy'] > 0.6 else 'medium' if ou_results['over_15_accuracy'] > 0.5 else 'poor'}">
                <strong>Over 0.5 Accuracy:</strong> {ou_results['over_05_accuracy']:.1%}<br>
                <strong>Over 1.5 Accuracy:</strong> {ou_results['over_15_accuracy']:.1%}<br>
                <strong>Over 2.5 Accuracy:</strong> {ou_results['over_25_accuracy']:.1%}
            </div>
            
            <h2>⚽ 1X2 Performance</h2>
            <div class="metric {'good' if x1x2_results['accuracy_1x2'] > 0.5 else 'medium' if x1x2_results['accuracy_1x2'] > 0.4 else 'poor'}">
                <strong>Overall 1X2 Accuracy:</strong> {x1x2_results['accuracy_1x2']:.1%}<br>
                <strong>Home Wins Accuracy:</strong> {x1x2_results['home_accuracy']:.1%}<br>
                <strong>Draws Accuracy:</strong> {x1x2_results['draw_accuracy']:.1%}<br>
                <strong>Away Wins Accuracy:</strong> {x1x2_results['away_accuracy']:.1%}
            </div>
            
            <h2>🎯 Multigol Performance</h2>
            <div class="metric">
                <strong>MG Casa 1-3:</strong> {mg_results.get('mg_casa_1_3_accuracy', 0):.1%}<br>
                <strong>MG Casa 1-4:</strong> {mg_results.get('mg_casa_1_4_accuracy', 0):.1%}<br>
                <strong>MG Casa 1-5:</strong> {mg_results.get('mg_casa_1_5_accuracy', 0):.1%}<br>
                <strong>MG Ospite 1-3:</strong> {mg_results.get('mg_ospite_1_3_accuracy', 0):.1%}<br>
                <strong>MG Ospite 1-4:</strong> {mg_results.get('mg_ospite_1_4_accuracy', 0):.1%}<br>
                <strong>MG Ospite 1-5:</strong> {mg_results.get('mg_ospite_1_5_accuracy', 0):.1%}
            </div>
            
            <h2>📈 Lambda vs Reality</h2>
            <div class="metric">
                <strong>Home Goals MAE:</strong> {lambda_results['home_mae']:.3f}<br>
                <strong>Away Goals MAE:</strong> {lambda_results['away_mae']:.3f}<br>
                <strong>Home Correlation:</strong> {lambda_results['home_correlation']:.3f}<br>
                <strong>Away Correlation:</strong> {lambda_results['away_correlation']:.3f}
            </div>
            
            <h2>💰 Betting Simulation</h2>
            <div class="metric {'good' if betting_results['total_roi'] > 5 else 'medium' if betting_results['total_roi'] > -5 else 'poor'}">
                <strong>Total Bets:</strong> {betting_results['total_bets']}<br>
                <strong>Total Profit:</strong> €{betting_results['total_profit']:.2f}<br>
                <strong>ROI:</strong> {betting_results['total_roi']:.1f}%
            </div>
            
            <h2>🏆 Model Assessment</h2>
            <div class="metric">
                {'<strong>✅ Good Model:</strong> Over 1.5 accuracy > 60%' if ou_results['over_15_accuracy'] > 0.6 else 
                 '<strong>⚠️ Average Model:</strong> Over 1.5 accuracy 50-60%' if ou_results['over_15_accuracy'] > 0.5 else
                 '<strong>❌ Poor Model:</strong> Over 1.5 accuracy < 50%'}
            </div>
            
            <p><em>Report generato il {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report salvato in: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Football Model Backtesting')
    parser.add_argument('--predictions', required=True, help='CSV file with predictions')
    parser.add_argument('--report', default='backtest_report.html', help='Output HTML report')
    parser.add_argument('--stake', type=float, default=10, help='Betting stake for simulation')
    
    args = parser.parse_args()
    
    # Verifica file exists
    if not Path(args.predictions).exists():
        print(f"❌ File non trovato: {args.predictions}")
        sys.exit(1)
    
    print("🎯 FOOTBALL MODEL BACKTESTING")
    print("=" * 50)
    
    # Inizializza backtester
    backtester = ModelBacktester()
    
    # Carica predizioni
    df = backtester.load_predictions(args.predictions)
    if df is None:
        sys.exit(1)
    
    # Esegui analisi
    backtester.analyze_over_under_performance(df)
    backtester.analyze_1x2_performance(df)
    backtester.analyze_expected_vs_actual(df)
    backtester.analyze_multigol_performance(df)
    backtester.betting_simulation(df, args.stake)
    
    # Genera report
    backtester.generate_backtest_report(df, args.report)
    
    print(f"\n✅ Backtesting completo! Report: {args.report}")

if __name__ == "__main__":
    main()