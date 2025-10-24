#!/usr/bin/env python3
"""
Simulazione Betting Multigol Casa 1-4 e 1-5
Analizza le performance betting con diverse soglie di confidence
"""

import pandas as pd
import numpy as np
import argparse

def load_and_process_data(csv_file):
    """Carica e processa i dati delle predizioni"""
    df = pd.read_csv(csv_file)
    
    # Calcola risultati reali per ogni mercato multigol
    df['mg_casa_1_3_real'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 3)).astype(int)
    df['mg_casa_1_4_real'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 4)).astype(int)
    df['mg_casa_1_5_real'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 5)).astype(int)
    
    df['mg_ospite_1_3_real'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 3)).astype(int)
    df['mg_ospite_1_4_real'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 4)).astype(int)
    df['mg_ospite_1_5_real'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 5)).astype(int)
    
    return df

def analyze_single_multigol_market(df, market_name, prediction_col, actual_col, stake=10, thresholds=[0.6, 0.65, 0.7, 0.75, 0.8]):
    """Analizza un singolo mercato multigol"""
    
    print(f"\n� {market_name.upper()} BETTING ANALYSIS")
    print("-" * 40)
    
    results = {}
    
    for threshold in thresholds:
        filtered_df = df[df[prediction_col] >= threshold]
        
        if len(filtered_df) == 0:
            continue
            
        wins = filtered_df[actual_col].sum()
        total_bets = len(filtered_df)
        win_rate = wins / total_bets if total_bets > 0 else 0
        # Profitto corretto: (vincite * quota * stake) - (total_bets * stake)
        profit = (wins * 2.0 * stake) - (total_bets * stake)  # Quota multigol ~2.0
        roi = (profit / (total_bets * stake)) * 100 if total_bets > 0 else 0
        
        results[threshold] = {
            'bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'profit': profit,
            'roi': roi
        }
        
        print(f"Soglia {threshold:.1%}:")
        print(f"  Scommesse: {total_bets}")
        print(f"  Vincite: {wins}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Profitto: €{profit:.2f}")
        print(f"  ROI: {roi:.1f}%")
        print()
    
    return results

def analyze_multigol_with_smart_filters(df, stake=10):
    """Analizza tutti i mercati multigol con filtri intelligenti basati sulle quote 1X2"""
    
    print(f"\n🎯 FILTRI INTELLIGENTI PER MERCATI MULTIGOL")
    print("-" * 50)
    
    # Filtri per MG Casa (casa deve avere vantaggi)
    home_strong_favorite = df['odds_1'] <= 1.60    # Casa molto favorita
    home_moderate_favorite = df['odds_1'] <= 1.90  # Casa moderatamente favorita  
    away_not_dominant = df['odds_2'] >= 2.20       # Ospite non dominante
    balanced_match = (df['odds_1'] <= 2.50) & (df['odds_2'] <= 2.50)  # Partite equilibrate
    
    # Filtri per MG Ospite (ospite deve avere vantaggi)
    away_strong_favorite = df['odds_2'] <= 1.60    # Ospite molto favorito
    away_moderate_favorite = df['odds_2'] <= 1.90  # Ospite moderatamente favorito
    home_not_dominant = df['odds_1'] >= 2.20       # Casa non dominante
    
    # Contesti favorevoli per MG Casa
    mg_casa_context_strong = home_strong_favorite & away_not_dominant
    mg_casa_context_moderate = (home_moderate_favorite | balanced_match) & away_not_dominant
    
    # Contesti favorevoli per MG Ospite  
    mg_ospite_context_strong = away_strong_favorite & home_not_dominant
    mg_ospite_context_moderate = (away_moderate_favorite | balanced_match) & home_not_dominant
    
    print(f"📊 Analisi contesti di gioco:")
    print(f"  - Casa molto favorita (≤1.60): {home_strong_favorite.sum()} partite")
    print(f"  - Casa moderatamente favorita (≤1.90): {home_moderate_favorite.sum()} partite")
    print(f"  - Ospite molto favorito (≤1.60): {away_strong_favorite.sum()} partite") 
    print(f"  - Ospite moderatamente favorito (≤1.90): {away_moderate_favorite.sum()} partite")
    print(f"  - Partite equilibrate: {balanced_match.sum()} partite")
    
    # Analisi MG Casa con filtri
    analyze_mg_market_with_filter(df, "MG Casa 1-3", 'MG_Casa_1_3', 'mg_casa_1_3_real', 
                                 mg_casa_context_strong, mg_casa_context_moderate, stake, 0.65, 0.70)
    
    analyze_mg_market_with_filter(df, "MG Casa 1-4", 'MG_Casa_1_4', 'mg_casa_1_4_real', 
                                 mg_casa_context_strong, mg_casa_context_moderate, stake, 0.70, 0.75)
                                 
    analyze_mg_market_with_filter(df, "MG Casa 1-5", 'MG_Casa_1_5', 'mg_casa_1_5_real', 
                                 mg_casa_context_strong, mg_casa_context_moderate, stake, 0.75, 0.80)
    
    # Analisi MG Ospite con filtri
    analyze_mg_market_with_filter(df, "MG Ospite 1-3", 'MG_Ospite_1_3', 'mg_ospite_1_3_real', 
                                 mg_ospite_context_strong, mg_ospite_context_moderate, stake, 0.65, 0.70)
                                 
    analyze_mg_market_with_filter(df, "MG Ospite 1-4", 'MG_Ospite_1_4', 'mg_ospite_1_4_real', 
                                 mg_ospite_context_strong, mg_ospite_context_moderate, stake, 0.70, 0.75)
                                 
    analyze_mg_market_with_filter(df, "MG Ospite 1-5", 'MG_Ospite_1_5', 'mg_ospite_1_5_real', 
                                 mg_ospite_context_strong, mg_ospite_context_moderate, stake, 0.75, 0.80)

def analyze_mg_market_with_filter(df, market_name, pred_col, actual_col, strong_context, moderate_context, stake, threshold1, threshold2):
    """Analizza un mercato MG con filtri di contesto"""
    
    print(f"\n🎯 {market_name.upper()} CON FILTRI INTELLIGENTI")
    print("-" * 45)
    
    # Filtro conservativo (contesto forte + soglia alta)
    conservative_filter = strong_context & (df[pred_col] >= threshold2)
    
    # Filtro bilanciato (contesto moderato + soglia media)  
    balanced_filter = moderate_context & (df[pred_col] >= threshold1)
    
    # Analizza filtro conservativo
    if conservative_filter.sum() > 0:
        analyze_filter_performance(df, conservative_filter, actual_col, stake, f"{market_name} - CONSERVATIVO")
        
        # Mostra esempi
        examples = df[conservative_filter].head(3)
        for _, match in examples.iterrows():
            result_icon = "✅" if match[actual_col] == 1 else "❌"
            print(f"  {result_icon} {match['home_team']} vs {match['away_team']} - Quote: 1:{match['odds_1']:.2f} 2:{match['odds_2']:.2f}")
    
    # Analizza filtro bilanciato
    if balanced_filter.sum() > 0:
        analyze_filter_performance(df, balanced_filter, actual_col, stake, f"{market_name} - BILANCIATO")

def analyze_filter_performance(df, filter_mask, actual_col, stake, strategy_name):
    """Analizza le performance di un filtro specifico"""
    
    filtered_df = df[filter_mask]
    if len(filtered_df) == 0:
        return
        
    wins = filtered_df[actual_col].sum()
    total_bets = len(filtered_df)
    win_rate = wins / total_bets
    
    # Profitto corretto: vincite - perdite (assumendo quota ~2.0 per multigol)
    profit = (wins * stake * 2.0) - (total_bets * stake)
    roi = (profit / (total_bets * stake)) * 100
    
    print(f"📈 {strategy_name}:")
    print(f"  Scommesse: {total_bets}")
    print(f"  Vincite: {wins}")  
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Profitto: €{profit:.2f}")
    print(f"  ROI: {roi:.1f}%")

def analyze_multigol_betting(df, stake=10):
    """Analizza le performance del betting multigol completo con diverse soglie"""
    
    print("🏠� SIMULAZIONE BETTING MULTIGOL COMPLETA + ANALISI QUOTE")
    print("=" * 70)
    print(f"📊 Analizzando {len(df)} predizioni...")
    
    # Analisi tutti i mercati MG Casa
    mg_casa_results = {}
    mg_casa_results['1-3'] = analyze_single_multigol_market(df, "MG Casa 1-3", 'MG_Casa_1_3', 'mg_casa_1_3_real', stake)
    mg_casa_results['1-4'] = analyze_single_multigol_market(df, "MG Casa 1-4", 'MG_Casa_1_4', 'mg_casa_1_4_real', stake)
    mg_casa_results['1-5'] = analyze_single_multigol_market(df, "MG Casa 1-5", 'MG_Casa_1_5', 'mg_casa_1_5_real', stake)
    
    # Analisi tutti i mercati MG Ospite
    mg_ospite_results = {}
    mg_ospite_results['1-3'] = analyze_single_multigol_market(df, "MG Ospite 1-3", 'MG_Ospite_1_3', 'mg_ospite_1_3_real', stake)
    mg_ospite_results['1-4'] = analyze_single_multigol_market(df, "MG Ospite 1-4", 'MG_Ospite_1_4', 'mg_ospite_1_4_real', stake)
    mg_ospite_results['1-5'] = analyze_single_multigol_market(df, "MG Ospite 1-5", 'MG_Ospite_1_5', 'mg_ospite_1_5_real', stake)
    
    # Verifica se abbiamo le quote 1X2
    has_odds = 'odds_1' in df.columns and 'odds_X' in df.columns and 'odds_2' in df.columns
    
    if has_odds:
        # Analisi completa con filtri intelligenti per tutti i mercati multigol
        analyze_multigol_with_smart_filters(df, stake)
    else:
        print("⚠️ Quote non disponibili nel dataset - usando solo confidence multigol")
    
    # Analisi combinata: migliori opportunità
    print("\n🚀 STRATEGIA COMBINATA OTTIMALE")
    print("-" * 40)
    
    # MG Casa 1-4 con soglia 70% + MG Casa 1-5 con soglia 75%
    best_1_4 = df[df['MG_Casa_1_4'] >= 0.70]
    best_1_5 = df[df['MG_Casa_1_5'] >= 0.75]
    
    if len(best_1_4) > 0:
        wins_best_1_4 = best_1_4['mg_casa_1_4_real'].sum()
        total_stake_1_4 = len(best_1_4) * stake
        profit_1_4 = (wins_best_1_4 * stake * 2.0) - total_stake_1_4
        
        print(f"MG Casa 1-4 (soglia 70%):")
        print(f"  Scommesse: {len(best_1_4)}")
        print(f"  Win Rate: {wins_best_1_4/len(best_1_4):.1%}")
        print(f"  Profitto: €{profit_1_4:.2f}")
    
    if len(best_1_5) > 0:
        wins_best_1_5 = best_1_5['mg_casa_1_5_real'].sum()
        total_stake_1_5 = len(best_1_5) * stake
        profit_1_5 = (wins_best_1_5 * stake * 1.8) - total_stake_1_5
        
        print(f"MG Casa 1-5 (soglia 75%):")
        print(f"  Scommesse: {len(best_1_5)}")
        print(f"  Win Rate: {wins_best_1_5/len(best_1_5):.1%}")
        print(f"  Profitto: €{profit_1_5:.2f}")
    
    # Totale combinato
    total_combined_profit = profit_1_4 + profit_1_5
    total_combined_stake = total_stake_1_4 + total_stake_1_5
    combined_roi = (total_combined_profit / total_combined_stake * 100) if total_combined_stake > 0 else 0
    
    print(f"\n💰 TOTALE STRATEGIA COMBINATA:")
    print(f"  Profitto totale: €{total_combined_profit:.2f}")
    print(f"  Stake totale: €{total_combined_stake:.2f}")
    print(f"  ROI combinato: {combined_roi:.1f}%")
    
    # Dettagli partite migliori CON FILTRO QUOTE
    print("\n📋 TOP OPPORTUNITIES (MG Casa 1-4 > 70% + Filtro Quote)")
    print("-" * 50)
    
    # Applica stesso filtro quote usato sopra
    if has_odds:
        home_favorite = df['odds_1'] <= 1.70
        away_not_dominant = df['odds_2'] >= 2.20
        balanced_match = (df['odds_1'] <= 2.50) & (df['odds_2'] <= 2.50)
        good_context = (home_favorite | balanced_match) & away_not_dominant
        
        top_opportunities = df[(df['MG_Casa_1_4'] >= 0.70) & good_context].sort_values('MG_Casa_1_4', ascending=False)
        print(f"📊 Partite con filtro quote applicato: {len(top_opportunities)}")
    else:
        top_opportunities = df[df['MG_Casa_1_4'] >= 0.70].sort_values('MG_Casa_1_4', ascending=False)
        print("⚠️  Quote non disponibili - mostrando tutte le partite > 70%")
    
    for _, match in top_opportunities.iterrows():
        result_icon = "✅" if match['mg_casa_1_4_real'] == 1 else "❌"
        
        if has_odds and pd.notna(match.get('odds_1')):
            quote_info = f"Quote: 1:{match['odds_1']:.2f} X:{match['odds_X']:.2f} 2:{match['odds_2']:.2f}"
        else:
            quote_info = "Quote: N/A"
            
        print(f"{result_icon} {match['home_team']} vs {match['away_team']}")
        print(f"   {quote_info}")
        print(f"   MG Casa 1-4: {match['MG_Casa_1_4']:.1%} | Risultato: {match['actual_scoreline']}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulazione Betting Multigol Completa')
    parser.add_argument('--predictions', required=True, help='File CSV predizioni')
    parser.add_argument('--stake', type=float, default=10.0, help='Stake per scommessa (default: 10)')
    
    args = parser.parse_args()
    
    # Carica e processa il dataset
    df = load_and_process_data(args.predictions)
    
    # Esegue l'analisi completa
    analyze_multigol_betting(df, args.stake)