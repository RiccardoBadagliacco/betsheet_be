#!/usr/bin/env python3
"""
🚀 Football Betting Assistant - Quick Analysis Tool
Analisi rapida per le partite del giorno con suggerimenti betting
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

def analyze_todays_matches(predictions_file, min_confidence=0.7):
    """Analizza le partite con alta confidenza per suggerimenti betting"""
    
    print("🚀 FOOTBALL BETTING ASSISTANT")
    print("=" * 50)
    
    try:
        df = pd.read_csv(predictions_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"📊 Caricati {len(df)} predizioni")
        
        # Filtra per alta confidenza
        mg_cols = [col for col in df.columns if col.startswith('MG_Casa_') or col.startswith('MG_Ospite_')]
        mg_condition = df[mg_cols].max(axis=1) >= min_confidence if mg_cols else False
        
        high_confidence = df[
            (df['O_1_5'] >= min_confidence) | 
            (df[['1X2_H', '1X2_D', '1X2_A']].max(axis=1) >= min_confidence) |
            mg_condition
        ].copy()
        
        if len(high_confidence) == 0:
            print(f"❌ Nessuna partita trovata con confidenza >= {min_confidence}")
            return
        
        print(f"🎯 Trovate {len(high_confidence)} partite ad alta confidenza")
        print("\n" + "="*80)
        
        for idx, match in high_confidence.iterrows():
            print(f"\n📅 {match['date'].strftime('%Y-%m-%d')} | {match['home_team']} vs {match['away_team']}")
            print("-" * 60)
            
            # Over/Under suggerimenti
            if match['O_1_5'] >= min_confidence:
                profit_potential = (match['O_1_5'] - 0.5) * 200  # Stima profitto
                print(f"🎯 OVER 1.5 RACCOMANDATO")
                print(f"   Probabilità: {match['O_1_5']:.1%}")
                print(f"   Confidenza: {'⭐' * int((match['O_1_5'] - 0.5) * 10)}")
                print(f"   Profit Potential: {profit_potential:.0f}%")
            
            # 1X2 suggerimenti
            max_1x2_prob = match[['1X2_H', '1X2_D', '1X2_A']].max()
            if max_1x2_prob >= min_confidence:
                result_map = {'1X2_H': 'HOME WIN', '1X2_D': 'DRAW', '1X2_A': 'AWAY WIN'}
                predicted_result = match[['1X2_H', '1X2_D', '1X2_A']].idxmax()
                result_name = result_map[predicted_result]
                
                print(f"🏆 {result_name} RACCOMANDATO")
                print(f"   Probabilità: {max_1x2_prob:.1%}")
                print(f"   Confidenza: {'⭐' * int((max_1x2_prob - 0.5) * 10)}")
            
            # Multigol suggerimenti
            mg_casa_cols = [col for col in match.index if col.startswith('MG_Casa_')]
            mg_ospite_cols = [col for col in match.index if col.startswith('MG_Ospite_')]
            
            for col in mg_casa_cols:
                if match[col] >= min_confidence:
                    range_name = col.replace('MG_Casa_', '').replace('_', '-')
                    print(f"🎯 MULTIGOL CASA {range_name} RACCOMANDATO")
                    print(f"   Probabilità: {match[col]:.1%}")
                    print(f"   Confidenza: {'⭐' * int((match[col] - 0.5) * 10)}")
            
            for col in mg_ospite_cols:
                if match[col] >= min_confidence:
                    range_name = col.replace('MG_Ospite_', '').replace('_', '-')
                    print(f"🎯 MULTIGOL OSPITE {range_name} RACCOMANDATO")
                    print(f"   Probabilità: {match[col]:.1%}")
                    print(f"   Confidenza: {'⭐' * int((match[col] - 0.5) * 10)}")
            
            # Dettagli tecnici
            print(f"\n📊 Dettagli tecnici:")
            print(f"   λ_home: {match['lambda_home']:.2f} | λ_away: {match['lambda_away']:.2f}")
            print(f"   O0.5: {match['O_0_5']:.1%} | O1.5: {match['O_1_5']:.1%}")
            
            # Risultato reale se disponibile
            if pd.notna(match.get('actual_home_goals')):
                actual_score = f"{int(match['actual_home_goals'])}-{int(match['actual_away_goals'])}"
                actual_total = int(match['actual_home_goals'] + match['actual_away_goals'])
                print(f"   ✅ Risultato reale: {actual_score} (Tot: {actual_total})")
            
            print("   " + "="*50)
        
        # Statistiche finali
        over15_recommendations = len(high_confidence[high_confidence['O_1_5'] >= min_confidence])
        x1x2_recommendations = len(high_confidence[high_confidence[['1X2_H', '1X2_D', '1X2_A']].max(axis=1) >= min_confidence])
        
        # Conta raccomandazioni multigol
        mg_recommendations = 0
        for _, match in high_confidence.iterrows():
            mg_cols = [col for col in match.index if col.startswith('MG_Casa_') or col.startswith('MG_Ospite_')]
            if any(match[col] >= min_confidence for col in mg_cols):
                mg_recommendations += 1
        
        print(f"\n📈 RIEPILOGO RACCOMANDAZIONI")
        print(f"Over 1.5 consigliate: {over15_recommendations}")
        print(f"1X2 consigliate: {x1x2_recommendations}")
        print(f"Multigol consigliate: {mg_recommendations}")
        print(f"Confidenza minima: {min_confidence:.0%}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

def main():
    parser = argparse.ArgumentParser(description='Football Betting Assistant')
    parser.add_argument('--predictions', required=True, help='CSV file with predictions')
    parser.add_argument('--confidence', type=float, default=0.7, help='Minimum confidence threshold (0.7 = 70%)')
    
    args = parser.parse_args()
    
    if not Path(args.predictions).exists():
        print(f"❌ File non trovato: {args.predictions}")
        print("\nGenerare prima le predizioni con:")
        print("python simple_football_model.py --data leagues_csv_unified/CAMPIONATO.csv --out predictions.csv")
        sys.exit(1)
    
    analyze_todays_matches(args.predictions, args.confidence)

if __name__ == "__main__":
    main()