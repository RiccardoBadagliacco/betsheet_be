#!/usr/bin/env python3
"""
Backtest dettagliato 2025/2026 - Serie A
Analizza partita per partita tutte le predizioni sui mercati:
1X2, Over 0.5, Over 1.5, MG Casa 1-3/1-4/1-5, MG Ospite 1-3/1-4/1-5
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_and_process_season_data(csv_file, season_matches=30):
    """Carica i dati della stagione e processa le predizioni"""
    
    print(f"📊 Caricando dati stagione 2025/2026...")
    df = pd.read_csv(csv_file)
    
    # Prende le ultime partite (stagione 2025/2026) 
    season_df = df.tail(season_matches).copy()
    
    print(f"✅ Caricate {len(season_df)} partite della stagione 2025/2026")
    
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

def format_prediction_confidence(predicted, actual, format_percent=True):
    """Formatta la predizione con indicatore di successo/fallimento"""
    
    if format_percent:
        pred_str = f"{predicted:.1%}"
    else:
        pred_str = f"{predicted:.3f}"
        
    if actual == 1:
        return f"✅ {pred_str}"
    else:
        return f"❌ {pred_str}"

def analyze_match_predictions(df):
    """Analizza le predizioni partita per partita"""
    
    print(f"\n🏟️  BACKTEST DETTAGLIATO SERIE A 2025/2026")
    print("=" * 80)
    print(f"📅 Periodo: {df['date'].min()} → {df['date'].max()}")
    print(f"🎯 Partite analizzate: {len(df)}")
    
    print(f"\n📋 LEGENDA:")
    print(f"✅ = Predizione corretta | ❌ = Predizione errata")
    print(f"Quote: 1=Casa | X=Pareggio | 2=Ospite")
    print(f"MG = Multigol | O = Over")
    
    # Analisi dettagliata partita per partita
    for idx, match in df.iterrows():
        print(f"\n" + "="*70)
        print(f"🏠 {match['home_team']} vs {match['away_team']} 🏃")
        print(f"📅 Data: {match['date']} | Risultato: {match['actual_scoreline']}")
        
        # Quote se disponibili
        if pd.notna(match.get('odds_1')):
            print(f"💰 Quote: 1:{match['odds_1']:.2f} X:{match['odds_X']:.2f} 2:{match['odds_2']:.2f}")
        
        print(f"\n🎯 PREDIZIONI vs RISULTATI:")
        
        # Mercato 1X2
        print(f"\n📊 MERCATO 1X2:")
        h_pred = format_prediction_confidence(match['1X2_H'], match['actual_1X2_H'])
        d_pred = format_prediction_confidence(match['1X2_D'], match['actual_1X2_D']) 
        a_pred = format_prediction_confidence(match['1X2_A'], match['actual_1X2_A'])
        print(f"  Casa (1): {h_pred} | Pareggio (X): {d_pred} | Ospite (2): {a_pred}")
        
        # Mercato Over
        print(f"\n📈 MERCATO OVER:")
        o05_pred = format_prediction_confidence(match['O_0_5'], match['actual_O_0_5'])
        o15_pred = format_prediction_confidence(match['O_1_5'], match['actual_O_1_5'])
        print(f"  Over 0.5: {o05_pred} | Over 1.5: {o15_pred}")
        
        # Mercato Multigol Casa
        print(f"\n🏠 MULTIGOL CASA:")
        mgc13_pred = format_prediction_confidence(match['MG_Casa_1_3'], match['actual_MG_Casa_1_3'])
        mgc14_pred = format_prediction_confidence(match['MG_Casa_1_4'], match['actual_MG_Casa_1_4'])
        mgc15_pred = format_prediction_confidence(match['MG_Casa_1_5'], match['actual_MG_Casa_1_5'])
        print(f"  MG Casa 1-3: {mgc13_pred} | MG Casa 1-4: {mgc14_pred} | MG Casa 1-5: {mgc15_pred}")
        
        # Mercato Multigol Ospite  
        print(f"\n🏃 MULTIGOL OSPITE:")
        mgo13_pred = format_prediction_confidence(match['MG_Ospite_1_3'], match['actual_MG_Ospite_1_3'])
        mgo14_pred = format_prediction_confidence(match['MG_Ospite_1_4'], match['actual_MG_Ospite_1_4'])
        mgo15_pred = format_prediction_confidence(match['MG_Ospite_1_5'], match['actual_MG_Ospite_1_5'])
        print(f"  MG Ospite 1-3: {mgo13_pred} | MG Ospite 1-4: {mgo14_pred} | MG Ospite 1-5: {mgo15_pred}")

def generate_market_summary(df):
    """Genera un riassunto delle performance per mercato"""
    
    print(f"\n\n🏆 RIASSUNTO PERFORMANCE PER MERCATO")
    print("=" * 60)
    
    markets = {
        '1X2_H': 'Casa (1)',
        '1X2_D': 'Pareggio (X)', 
        '1X2_A': 'Ospite (2)',
        'O_0_5': 'Over 0.5',
        'O_1_5': 'Over 1.5', 
        'MG_Casa_1_3': 'MG Casa 1-3',
        'MG_Casa_1_4': 'MG Casa 1-4',
        'MG_Casa_1_5': 'MG Casa 1-5',
        'MG_Ospite_1_3': 'MG Ospite 1-3',
        'MG_Ospite_1_4': 'MG Ospite 1-4', 
        'MG_Ospite_1_5': 'MG Ospite 1-5'
    }
    
    for pred_col, market_name in markets.items():
        actual_col = f'actual_{pred_col}'
        
        if actual_col not in df.columns:
            continue
            
        # Statistiche base
        total_matches = len(df)
        correct_predictions = (df[pred_col].round() == df[actual_col]).sum()
        accuracy = correct_predictions / total_matches * 100
        
        # Media confidence delle predizioni
        avg_confidence = df[pred_col].mean() * 100
        
        # Predizioni vincenti con alta confidence (>70%)
        high_conf_matches = df[df[pred_col] >= 0.70]
        if len(high_conf_matches) > 0:
            high_conf_accuracy = (high_conf_matches[pred_col].round() == high_conf_matches[actual_col]).sum()
            high_conf_rate = high_conf_accuracy / len(high_conf_matches) * 100
            high_conf_info = f" | Alta Conf (≥70%): {high_conf_accuracy}/{len(high_conf_matches)} ({high_conf_rate:.1f}%)"
        else:
            high_conf_info = " | Alta Conf: N/A"
        
        print(f"📊 {market_name:15s}: {correct_predictions:2d}/{total_matches} ({accuracy:5.1f}%) | Conf Media: {avg_confidence:5.1f}%{high_conf_info}")

def generate_top_predictions_summary(df):
    """Genera riassunto delle migliori predizioni per mercato"""
    
    print(f"\n\n🎯 TOP PREDIZIONI PER MERCATO (Confidence ≥ 70%)")
    print("=" * 70)
    
    markets = [
        ('MG_Casa_1_4', 'MG Casa 1-4', 'actual_MG_Casa_1_4'),
        ('MG_Casa_1_5', 'MG Casa 1-5', 'actual_MG_Casa_1_5'),
        ('MG_Ospite_1_4', 'MG Ospite 1-4', 'actual_MG_Ospite_1_4'),
        ('O_1_5', 'Over 1.5', 'actual_O_1_5'),
        ('1X2_H', 'Vittoria Casa', 'actual_1X2_H')
    ]
    
    for pred_col, market_name, actual_col in markets:
        print(f"\n🏆 {market_name.upper()}:")
        
        # Filtra predizioni con alta confidence
        high_conf = df[df[pred_col] >= 0.70].copy()
        
        if len(high_conf) == 0:
            print(f"  ⚠️  Nessuna predizione con confidence ≥ 70%")
            continue
            
        # Ordina per confidence
        high_conf = high_conf.sort_values(pred_col, ascending=False)
        
        correct = 0
        total = len(high_conf)
        
        for _, match in high_conf.head(10).iterrows():
            is_correct = match[actual_col] == 1
            if is_correct:
                correct += 1
                
            icon = "✅" if is_correct else "❌" 
            confidence = match[pred_col] * 100
            
            print(f"  {icon} {match['home_team']} vs {match['away_team']} ({confidence:.1f}%) → {match['actual_scoreline']}")
        
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"  📈 Performance: {correct}/{total} ({accuracy:.1f}%)")

def export_detailed_results(df, output_file="backtest_2025_2026_detailed.csv"):
    """Esporta risultati dettagliati in CSV per ulteriori analisi"""
    
    # Seleziona colonne rilevanti per l'export
    export_cols = [
        'date', 'home_team', 'away_team', 'actual_scoreline',
        'actual_home_goals', 'actual_away_goals'
    ]
    
    # Aggiungi quote se disponibili
    if 'odds_1' in df.columns:
        export_cols.extend(['odds_1', 'odds_X', 'odds_2'])
    
    # Predizioni e risultati per tutti i mercati
    pred_actual_cols = [
        # 1X2
        '1X2_H', 'actual_1X2_H', '1X2_D', 'actual_1X2_D', '1X2_A', 'actual_1X2_A',
        # Over
        'O_0_5', 'actual_O_0_5', 'O_1_5', 'actual_O_1_5',
        # MG Casa
        'MG_Casa_1_3', 'actual_MG_Casa_1_3', 'MG_Casa_1_4', 'actual_MG_Casa_1_4', 'MG_Casa_1_5', 'actual_MG_Casa_1_5',
        # MG Ospite
        'MG_Ospite_1_3', 'actual_MG_Ospite_1_3', 'MG_Ospite_1_4', 'actual_MG_Ospite_1_4', 'MG_Ospite_1_5', 'actual_MG_Ospite_1_5'
    ]
    
    # Aggiungi solo colonne esistenti
    for col in pred_actual_cols:
        if col in df.columns:
            export_cols.append(col)
    
    # Export
    export_df = df[export_cols].copy()
    export_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Risultati dettagliati esportati in: {output_file}")
    print(f"📊 Colonne esportate: {len(export_cols)}")
    print(f"📋 Righe esportate: {len(export_df)}")

def main():
    parser = argparse.ArgumentParser(description='Backtest dettagliato Serie A 2025/2026')
    parser.add_argument('--predictions', default='predictions_final_odds.csv', help='File CSV predizioni')
    parser.add_argument('--matches', type=int, default=30, help='Numero partite stagione (default: 30)')
    parser.add_argument('--export', help='File CSV per export risultati dettagliati')
    
    args = parser.parse_args()
    
    # Carica e processa i dati
    df = load_and_process_season_data(args.predictions, args.matches)
    
    # Analisi dettagliata partita per partita
    analyze_match_predictions(df)
    
    # Riassunti performance 
    generate_market_summary(df)
    generate_top_predictions_summary(df)
    
    # Export se richiesto
    if args.export:
        export_detailed_results(df, args.export)
    
    print(f"\n🏁 Backtest completato! Analizzate {len(df)} partite della stagione 2025/2026")

if __name__ == "__main__":
    main()