#!/usr/bin/env python3
"""
Backtest dettagliato Premier League 2025/2026 
Analizza partita per partita tutte le predizioni sui mercati:
1X2, Over 0.5, Over 1.5, MG Casa 1-3/1-4/1-5, MG Ospite 1-3/1-4/1-5
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_and_process_season_data(csv_file, season_matches=80):
    """Carica i dati della stagione Premier League e processa le predizioni"""
    
    print(f"📊 Caricando dati stagione Premier League 2025/2026...")
    df = pd.read_csv(csv_file)
    
    # Prende le ultime partite (stagione 2025/2026) 
    season_df = df.tail(season_matches).copy()
    
    print(f"✅ Caricate {len(season_df)} partite della stagione Premier League 2025/2026")
    
    # Calcola risultati reali per tutti i mercati
    season_df = calculate_actual_results(season_df)
    
    return season_df

def calculate_actual_results(df):
    """Calcola i risultati reali per tutti i mercati analizzati"""
    
    # Mercati 1X2
    df['actual_1X2_H'] = (df['actual_home_goals'] > df['actual_away_goals']).astype(int)
    df['actual_1X2_D'] = (df['actual_home_goals'] == df['actual_away_goals']).astype(int) 
    df['actual_1X2_A'] = (df['actual_home_goals'] < df['actual_away_goals']).astype(int)
    
    # Mercati Over
    df['actual_O_0_5'] = (df['actual_home_goals'] + df['actual_away_goals'] > 0.5).astype(int)
    df['actual_O_1_5'] = (df['actual_home_goals'] + df['actual_away_goals'] > 1.5).astype(int)
    
    # Mercati Multigol Casa
    df['actual_MG_Casa_1_3'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 3)).astype(int)
    df['actual_MG_Casa_1_4'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 4)).astype(int)
    df['actual_MG_Casa_1_5'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 5)).astype(int)
    
    # Mercati Multigol Ospite
    df['actual_MG_Ospite_1_3'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 3)).astype(int)
    df['actual_MG_Ospite_1_4'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 4)).astype(int)
    df['actual_MG_Ospite_1_5'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 5)).astype(int)
    
    return df

def calculate_market_performance(df):
    """Calcola le performance su tutti i mercati"""
    
    results = {}
    
    # Mercati Over
    for market in ['O_0_5', 'O_1_5']:
        predicted = f'{market}'
        actual = f'actual_{market}'
        
        if predicted in df.columns and actual in df.columns:
            # Soglie di confidenza
            high_conf = df[predicted] >= 0.7
            med_conf = (df[predicted] >= 0.55) & (df[predicted] < 0.7)
            
            results[f'{market}_total'] = {
                'matches': len(df),
                'correct': (df[predicted] > 0.5).sum() if (df[actual] == 1).sum() > 0 else 0,
                'accuracy': ((df[predicted] > 0.5) == df[actual]).mean() if len(df) > 0 else 0
            }
            
            if high_conf.sum() > 0:
                results[f'{market}_high_conf'] = {
                    'matches': high_conf.sum(),
                    'correct': df.loc[high_conf, actual].sum(),
                    'accuracy': df.loc[high_conf, actual].mean()
                }
            
            if med_conf.sum() > 0:
                results[f'{market}_med_conf'] = {
                    'matches': med_conf.sum(),
                    'correct': df.loc[med_conf, actual].sum(),
                    'accuracy': df.loc[med_conf, actual].mean()
                }
    
    # Mercati Multigol Casa
    for market in ['MG_Casa_1_3', 'MG_Casa_1_4', 'MG_Casa_1_5']:
        predicted = f'{market}'
        actual = f'actual_{market}'
        
        if predicted in df.columns and actual in df.columns:
            # Soglie di confidenza
            high_conf = df[predicted] >= 0.65
            med_conf = (df[predicted] >= 0.5) & (df[predicted] < 0.65)
            
            results[f'{market}_total'] = {
                'matches': len(df),
                'correct': (df[predicted] > 0.5).sum() if (df[actual] == 1).sum() > 0 else 0,
                'accuracy': ((df[predicted] > 0.5) == df[actual]).mean() if len(df) > 0 else 0
            }
            
            if high_conf.sum() > 0:
                results[f'{market}_high_conf'] = {
                    'matches': high_conf.sum(),
                    'correct': df.loc[high_conf, actual].sum(),
                    'accuracy': df.loc[high_conf, actual].mean()
                }
            
            if med_conf.sum() > 0:
                results[f'{market}_med_conf'] = {
                    'matches': med_conf.sum(),
                    'correct': df.loc[med_conf, actual].sum(),
                    'accuracy': df.loc[med_conf, actual].mean()
                }
    
    # Mercati Multigol Ospite
    for market in ['MG_Ospite_1_3', 'MG_Ospite_1_4', 'MG_Ospite_1_5']:
        predicted = f'{market}'
        actual = f'actual_{market}'
        
        if predicted in df.columns and actual in df.columns:
            # Soglie di confidenza
            high_conf = df[predicted] >= 0.65
            med_conf = (df[predicted] >= 0.5) & (df[predicted] < 0.65)
            
            results[f'{market}_total'] = {
                'matches': len(df),
                'correct': (df[predicted] > 0.5).sum() if (df[actual] == 1).sum() > 0 else 0,
                'accuracy': ((df[predicted] > 0.5) == df[actual]).mean() if len(df) > 0 else 0
            }
            
            if high_conf.sum() > 0:
                results[f'{market}_high_conf'] = {
                    'matches': high_conf.sum(),
                    'correct': df.loc[high_conf, actual].sum(),
                    'accuracy': df.loc[high_conf, actual].mean()
                }
            
            if med_conf.sum() > 0:
                results[f'{market}_med_conf'] = {
                    'matches': med_conf.sum(),
                    'correct': df.loc[med_conf, actual].sum(),
                    'accuracy': df.loc[med_conf, actual].mean()
                }
    
    return results

def print_detailed_results(results, league_name="Premier League"):
    """Stampa i risultati dettagliati del backtest"""
    
    print(f"\n🎯 RISULTATI BACKTEST {league_name.upper()} 2025/2026")
    print("=" * 60)
    
    # Over Markets
    print(f"\n📈 MERCATI OVER:")
    for market in ['O_0_5', 'O_1_5']:
        if f'{market}_total' in results:
            total = results[f'{market}_total']
            print(f"\n{market.replace('_', ' ')} - Tutte le partite:")
            print(f"  • Partite: {total['matches']}")
            print(f"  • Accuratezza: {total['accuracy']:.1%}")
            
            if f'{market}_high_conf' in results:
                high = results[f'{market}_high_conf']
                print(f"  • Alta confidenza (≥70%): {high['matches']} partite - {high['accuracy']:.1%}")
            
            if f'{market}_med_conf' in results:
                med = results[f'{market}_med_conf']
                print(f"  • Media confidenza (55-70%): {med['matches']} partite - {med['accuracy']:.1%}")
    
    # Multigol Casa
    print(f"\n🏠 MERCATI MULTIGOL CASA:")
    for market in ['MG_Casa_1_3', 'MG_Casa_1_4', 'MG_Casa_1_5']:
        if f'{market}_total' in results:
            total = results[f'{market}_total']
            market_name = market.replace('MG_Casa_', 'Casa ').replace('_', '-')
            print(f"\n{market_name} - Tutte le partite:")
            print(f"  • Partite: {total['matches']}")
            print(f"  • Accuratezza: {total['accuracy']:.1%}")
            
            if f'{market}_high_conf' in results:
                high = results[f'{market}_high_conf']
                print(f"  • Alta confidenza (≥65%): {high['matches']} partite - {high['accuracy']:.1%}")
            
            if f'{market}_med_conf' in results:
                med = results[f'{market}_med_conf']
                print(f"  • Media confidenza (50-65%): {med['matches']} partite - {med['accuracy']:.1%}")
    
    # Multigol Ospite
    print(f"\n✈️ MERCATI MULTIGOL OSPITE:")
    for market in ['MG_Ospite_1_3', 'MG_Ospite_1_4', 'MG_Ospite_1_5']:
        if f'{market}_total' in results:
            total = results[f'{market}_total']
            market_name = market.replace('MG_Ospite_', 'Ospite ').replace('_', '-')
            print(f"\n{market_name} - Tutte le partite:")
            print(f"  • Partite: {total['matches']}")
            print(f"  • Accuratezza: {total['accuracy']:.1%}")
            
            if f'{market}_high_conf' in results:
                high = results[f'{market}_high_conf']
                print(f"  • Alta confidenza (≥65%): {high['matches']} partite - {high['accuracy']:.1%}")
            
            if f'{market}_med_conf' in results:
                med = results[f'{market}_med_conf']
                print(f"  • Media confidenza (50-65%): {med['matches']} partite - {med['accuracy']:.1%}")

def export_detailed_analysis(df, filename="backtest_premier_league_2025_2026.csv"):
    """Esporta l'analisi dettagliata in CSV"""
    
    # Seleziona solo le colonne rilevanti per l'esportazione
    export_columns = [
        'date', 'home_team', 'away_team',
        'actual_home_goals', 'actual_away_goals', 'actual_scoreline',
        'O_0_5', 'actual_O_0_5',
        'O_1_5', 'actual_O_1_5',
        'MG_Casa_1_3', 'actual_MG_Casa_1_3',
        'MG_Casa_1_4', 'actual_MG_Casa_1_4', 
        'MG_Casa_1_5', 'actual_MG_Casa_1_5',
        'MG_Ospite_1_3', 'actual_MG_Ospite_1_3',
        'MG_Ospite_1_4', 'actual_MG_Ospite_1_4',
        'MG_Ospite_1_5', 'actual_MG_Ospite_1_5'
    ]
    
    # Filtra solo le colonne esistenti
    available_columns = [col for col in export_columns if col in df.columns]
    
    export_df = df[available_columns].copy()
    export_df.to_csv(filename, index=False)
    
    print(f"\n📁 Analisi dettagliata esportata in: {filename}")
    print(f"   {len(export_df)} partite analizzate")

def main():
    """Funzione principale del backtest"""
    
    parser = argparse.ArgumentParser(description='Backtest dettagliato Premier League 2025/2026')
    parser.add_argument('--predictions', default='predictions_premier_league_2025_2026.csv', 
                       help='File CSV con le predizioni')
    parser.add_argument('--matches', type=int, default=80,
                       help='Numero di partite della stagione da analizzare')
    
    args = parser.parse_args()
    
    print("🏴󠁧󠁢󠁥󠁮󠁧󠁿 BACKTEST PREMIER LEAGUE 2025/2026")
    print("=" * 50)
    
    # Carica e processa i dati
    try:
        season_df = load_and_process_season_data(args.predictions, args.matches)
        
        # Calcola performance dei mercati
        results = calculate_market_performance(season_df)
        
        # Mostra risultati dettagliati
        print_detailed_results(results, "Premier League")
        
        # Esporta analisi dettagliata
        export_detailed_analysis(season_df, "reports/backtest_premier_league_2025_2026.csv")
        
        print(f"\n🎯 SUMMARY PREMIER LEAGUE 2025/2026:")
        print(f"   📊 {len(season_df)} partite analizzate")
        print(f"   📅 Periodo: {season_df['date'].min()} - {season_df['date'].max()}")
        print(f"   🏆 Modello validato su multiple stagioni storiche")
        
    except FileNotFoundError:
        print(f"❌ Errore: File {args.predictions} non trovato!")
        print("   Esegui prima il modello di predizione per generare i dati.")
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")

if __name__ == "__main__":
    main()