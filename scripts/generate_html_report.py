#!/usr/bin/env python3
"""
Generatore Report HTML - Backtest Stagione 2025/2026
Crea un report HTML interattivo con predizioni dettagliate per ogni partita
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json

def load_prediction_data(csv_file):
    """Carica i dati delle predizioni e calcola risultati reali"""
    
    df = pd.read_csv(csv_file)
    
    # Calcola risultati reali per tutti i mercati
    df['actual_1X2_H'] = (df['actual_home_goals'] > df['actual_away_goals']).astype(int)
    df['actual_1X2_D'] = (df['actual_home_goals'] == df['actual_away_goals']).astype(int) 
    df['actual_1X2_A'] = (df['actual_home_goals'] < df['actual_away_goals']).astype(int)
    
    df['actual_O_0_5'] = (df['actual_home_goals'] + df['actual_away_goals'] > 0.5).astype(int)
    df['actual_O_1_5'] = (df['actual_home_goals'] + df['actual_away_goals'] > 1.5).astype(int)
    
    df['actual_MG_Casa_1_3'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 3)).astype(int)
    df['actual_MG_Casa_1_4'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 4)).astype(int)
    df['actual_MG_Casa_1_5'] = ((df['actual_home_goals'] >= 1) & (df['actual_home_goals'] <= 5)).astype(int)
    
    df['actual_MG_Ospite_1_3'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 3)).astype(int)
    df['actual_MG_Ospite_1_4'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 4)).astype(int)
    df['actual_MG_Ospite_1_5'] = ((df['actual_away_goals'] >= 1) & (df['actual_away_goals'] <= 5)).astype(int)
    
    return df

def get_prediction_accuracy(predicted, actual):
    """Calcola accuratezza e icona per una predizione"""
    is_correct = (predicted >= 0.5 and actual == 1) or (predicted < 0.5 and actual == 0)
    icon = "✅" if is_correct else "❌"
    confidence_class = "high-confidence" if predicted >= 0.7 else "medium-confidence" if predicted >= 0.5 else "low-confidence"
    return is_correct, icon, confidence_class

def get_recommended_bets(match):
    """Determina le scommesse raccomandate basate sulle soglie del modello"""
    
    recommended_bets = []
    
    # Soglie stabilite durante l'analisi
    thresholds = {
        'O_0_5': 0.85,  # Over 0.5 - soglia alta per alta accuracy
        'O_1_5': 0.70,  # Over 1.5 - soglia media-alta
        'MG_Casa_1_3': 0.70,  # MG Casa 1-3 - soglia alta
        'MG_Casa_1_4': 0.75,  # MG Casa 1-4 - soglia alta per ROI ottimale
        'MG_Casa_1_5': 0.75,  # MG Casa 1-5 - soglia alta
        'MG_Ospite_1_3': 0.70,  # MG Ospite 1-3 - soglia alta
        'MG_Ospite_1_4': 0.70,  # MG Ospite 1-4 - soglia alta
        'MG_Ospite_1_5': 0.75,  # MG Ospite 1-5 - soglia molto alta
        '1X2_H': 0.60,  # Vittoria Casa - soglia moderata
        '1X2_A': 0.60,  # Vittoria Ospite - soglia moderata
    }
    
    bet_names = {
        'O_0_5': 'Over 0.5 Gol',
        'O_1_5': 'Over 1.5 Gol', 
        'MG_Casa_1_3': 'Multigol Casa 1-3',
        'MG_Casa_1_4': 'Multigol Casa 1-4',
        'MG_Casa_1_5': 'Multigol Casa 1-5',
        'MG_Ospite_1_3': 'Multigol Ospite 1-3',
        'MG_Ospite_1_4': 'Multigol Ospite 1-4',
        'MG_Ospite_1_5': 'Multigol Ospite 1-5',
        '1X2_H': f"Vittoria {match['home_team']}",
        '1X2_A': f"Vittoria {match['away_team']}"
    }
    
    # Verifica filtri quote per MG Casa (se disponibili)
    has_good_context_casa = True
    has_good_context_ospite = True
    
    if pd.notna(match.get('odds_1')):
        # Filtro MG Casa: casa favorita OR equilibrata AND ospite non dominante
        home_favorite = match['odds_1'] <= 1.70
        away_not_dominant = match['odds_2'] >= 2.20
        balanced_match = match['odds_1'] <= 2.50 and match['odds_2'] <= 2.50
        has_good_context_casa = (home_favorite or balanced_match) and away_not_dominant
        
        # Filtro MG Ospite: ospite favorito OR equilibrata AND casa non dominante  
        away_favorite = match['odds_2'] <= 1.70
        home_not_dominant = match['odds_1'] >= 2.20
        has_good_context_ospite = (away_favorite or balanced_match) and home_not_dominant
    
    # Controlla ogni mercato
    for market, threshold in thresholds.items():
        if market in match:
            confidence = match[market]
            
            if confidence >= threshold:
                # Applica filtri quote per mercati MG
                include_bet = True
                bet_type = ""
                
                if market.startswith('MG_Casa') and not has_good_context_casa:
                    include_bet = False
                    bet_type = " (❌ Filtro Quote)"
                elif market.startswith('MG_Ospite') and not has_good_context_ospite:
                    include_bet = False  
                    bet_type = " (❌ Filtro Quote)"
                elif market.startswith('MG_Casa') and has_good_context_casa:
                    bet_type = " (✅ Filtro Quote)"
                elif market.startswith('MG_Ospite') and has_good_context_ospite:
                    bet_type = " (✅ Filtro Quote)"
                
                if include_bet:
                    # Determina risultato atteso per colore
                    actual_col = f"actual_{market}"
                    result_icon = ""
                    result_class = ""
                    
                    if actual_col in match:
                        if match[actual_col] == 1:
                            result_icon = "✅"
                            result_class = "winning-bet"
                        else:
                            result_icon = "❌"  
                            result_class = "losing-bet"
                    
                    recommended_bets.append({
                        'market': market,
                        'name': bet_names[market] + bet_type,
                        'confidence': confidence,
                        'result_icon': result_icon,
                        'result_class': result_class,
                        'stake_suggested': 10  # Stake base
                    })
    
    return recommended_bets

def generate_match_card_html(match):
    """Genera HTML per una singola partita"""
    
    # Informazioni base partita
    home_team = match['home_team']
    away_team = match['away_team'] 
    date = match['date']
    result = match['actual_scoreline']
    
    # Quote se disponibili
    odds_html = ""
    if pd.notna(match.get('odds_1')):
        odds_html = f"""
        <div class="odds-section">
            <h4>💰 Quote Bookmaker</h4>
            <div class="odds-grid">
                <span class="odds-item">1: {match['odds_1']:.2f}</span>
                <span class="odds-item">X: {match['odds_X']:.2f}</span>
                <span class="odds-item">2: {match['odds_2']:.2f}</span>
            </div>
        </div>
        """
    
    # Mercati da analizzare
    markets = [
        ('1X2_H', 'actual_1X2_H', '1 (Casa)', 'home-market'),
        ('1X2_D', 'actual_1X2_D', 'X (Pareggio)', 'draw-market'),  
        ('1X2_A', 'actual_1X2_A', '2 (Ospite)', 'away-market'),
        ('O_0_5', 'actual_O_0_5', 'Over 0.5', 'over-market'),
        ('O_1_5', 'actual_O_1_5', 'Over 1.5', 'over-market'),
        ('MG_Casa_1_3', 'actual_MG_Casa_1_3', 'MG Casa 1-3', 'mg-casa-market'),
        ('MG_Casa_1_4', 'actual_MG_Casa_1_4', 'MG Casa 1-4', 'mg-casa-market'),
        ('MG_Casa_1_5', 'actual_MG_Casa_1_5', 'MG Casa 1-5', 'mg-casa-market'),
        ('MG_Ospite_1_3', 'actual_MG_Ospite_1_3', 'MG Ospite 1-3', 'mg-ospite-market'),
        ('MG_Ospite_1_4', 'actual_MG_Ospite_1_4', 'MG Ospite 1-4', 'mg-ospite-market'),
        ('MG_Ospite_1_5', 'actual_MG_Ospite_1_5', 'MG Ospite 1-5', 'mg-ospite-market')
    ]
    
    # Genera HTML per ogni mercato
    markets_html = []
    correct_predictions = 0
    total_predictions = 0
    
    for pred_col, actual_col, market_name, market_class in markets:
        if pred_col in match and actual_col in match:
            predicted = match[pred_col]
            actual = match[actual_col]
            
            is_correct, icon, confidence_class = get_prediction_accuracy(predicted, actual)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            markets_html.append(f"""
            <div class="market-prediction {market_class} {confidence_class}">
                <span class="market-name">{market_name}</span>
                <span class="prediction-value">{predicted:.1%}</span>
                <span class="prediction-icon">{icon}</span>
            </div>
            """)
    
    # Calcola accuracy della partita
    match_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    accuracy_class = "excellent" if match_accuracy >= 75 else "good" if match_accuracy >= 60 else "poor"
    
    # Ottieni scommesse raccomandate
    recommended_bets = get_recommended_bets(match)
    
    # Genera HTML per scommesse raccomandate
    bets_html = ""
    if recommended_bets:
        bets_rows = []
        total_stake = 0
        winning_bets = 0
        
        for bet in recommended_bets:
            total_stake += bet['stake_suggested']
            if bet['result_class'] == 'winning-bet':
                winning_bets += 1
                
            bets_rows.append(f"""
            <tr class="bet-row {bet['result_class']}">
                <td class="bet-name">{bet['name']}</td>
                <td class="bet-confidence">{bet['confidence']:.1%}</td>
                <td class="bet-stake">€{bet['stake_suggested']}</td>
                <td class="bet-result">{bet['result_icon']}</td>
            </tr>
            """)
        
        bet_success_rate = (winning_bets / len(recommended_bets) * 100) if recommended_bets else 0
        bet_success_class = "excellent" if bet_success_rate >= 75 else "good" if bet_success_rate >= 50 else "poor"
        
        bets_html = f"""
        <div class="recommended-bets">
            <h4>🎯 Scommesse Raccomandate dal Modello</h4>
            <div class="bets-summary">
                <span class="bets-count">Scommesse: {len(recommended_bets)}</span>
                <span class="bets-stake">Stake Totale: €{total_stake}</span>
                <span class="bets-success {bet_success_class}">Successo: {winning_bets}/{len(recommended_bets)} ({bet_success_rate:.1f}%)</span>
            </div>
            <table class="bets-table">
                <thead>
                    <tr>
                        <th>Scommessa</th>
                        <th>Confidence</th>
                        <th>Stake</th>
                        <th>Risultato</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(bets_rows)}
                </tbody>
            </table>
        </div>
        """
    else:
        bets_html = """
        <div class="recommended-bets">
            <h4>🎯 Scommesse Raccomandate dal Modello</h4>
            <p class="no-bets">❌ Nessuna scommessa supera le soglie di sicurezza del modello</p>
        </div>
        """
    
    # Genera HTML completo della partita
    match_html = f"""
    <div class="match-card {accuracy_class}">
        <div class="match-header">
            <h3 class="match-teams">🏠 {home_team} vs {away_team} 🏃</h3>
            <div class="match-info">
                <span class="match-date">📅 {date}</span>
                <span class="match-result">⚽ Risultato: {result}</span>
                <span class="match-accuracy accuracy-{accuracy_class}">
                    🎯 Accuracy: {match_accuracy:.1f}% ({correct_predictions}/{total_predictions})
                </span>
            </div>
        </div>
        
        {odds_html}
        
        {bets_html}
        
        <div class="predictions-section">
            <h4>🎯 Predizioni vs Risultati</h4>
            
            <div class="market-group">
                <h5>📊 Mercato 1X2</h5>
                <div class="market-grid">
                    {markets_html[0] + markets_html[1] + markets_html[2]}
                </div>
            </div>
            
            <div class="market-group">  
                <h5>📈 Mercato Over</h5>
                <div class="market-grid">
                    {markets_html[3] + markets_html[4]}
                </div>
            </div>
            
            <div class="market-group">
                <h5>🏠 Multigol Casa</h5>
                <div class="market-grid">
                    {markets_html[5] + markets_html[6] + markets_html[7]}
                </div>
            </div>
            
            <div class="market-group">
                <h5>🏃 Multigol Ospite</h5>
                <div class="market-grid">
                    {markets_html[8] + markets_html[9] + markets_html[10]}
                </div>
            </div>
        </div>
    </div>
    """
    
    return match_html, match_accuracy, correct_predictions, total_predictions

def calculate_global_stats(df):
    """Calcola statistiche globali per il riassunto"""
    
    markets = {
        '1X2_H': ('actual_1X2_H', 'Casa (1)'),
        '1X2_D': ('actual_1X2_D', 'Pareggio (X)'), 
        '1X2_A': ('actual_1X2_A', 'Ospite (2)'),
        'O_0_5': ('actual_O_0_5', 'Over 0.5'),
        'O_1_5': ('actual_O_1_5', 'Over 1.5'),
        'MG_Casa_1_3': ('actual_MG_Casa_1_3', 'MG Casa 1-3'),
        'MG_Casa_1_4': ('actual_MG_Casa_1_4', 'MG Casa 1-4'),
        'MG_Casa_1_5': ('actual_MG_Casa_1_5', 'MG Casa 1-5'),
        'MG_Ospite_1_3': ('actual_MG_Ospite_1_3', 'MG Ospite 1-3'),
        'MG_Ospite_1_4': ('actual_MG_Ospite_1_4', 'MG Ospite 1-4'),
        'MG_Ospite_1_5': ('actual_MG_Ospite_1_5', 'MG Ospite 1-5')
    }
    
    stats = []
    
    for pred_col, (actual_col, market_name) in markets.items():
        if pred_col in df.columns and actual_col in df.columns:
            # Accuratezza generale
            correct_general = ((df[pred_col] >= 0.5) == (df[actual_col] == 1)).sum()
            total_general = len(df)
            accuracy_general = correct_general / total_general * 100
            
            # Accuratezza alta confidence (≥70%)
            high_conf_mask = df[pred_col] >= 0.70
            high_conf_df = df[high_conf_mask]
            
            if len(high_conf_df) > 0:
                correct_high_conf = ((high_conf_df[pred_col] >= 0.5) == (high_conf_df[actual_col] == 1)).sum()
                accuracy_high_conf = correct_high_conf / len(high_conf_df) * 100
                high_conf_info = f"{correct_high_conf}/{len(high_conf_df)} ({accuracy_high_conf:.1f}%)"
            else:
                high_conf_info = "N/A"
            
            # Confidence media
            avg_confidence = df[pred_col].mean() * 100
            
            stats.append({
                'market': market_name,
                'accuracy_general': accuracy_general,
                'correct_general': correct_general,
                'total_general': total_general,
                'avg_confidence': avg_confidence,
                'high_conf_info': high_conf_info
            })
    
    return stats

def generate_html_report(df, output_file="report_stagione_2025_2026.html"):
    """Genera il report HTML completo"""
    
    print(f"🏗️  Generando report HTML per {len(df)} partite...")
    
    # Genera HTML per ogni partita
    matches_html = []
    total_accuracy = 0
    total_correct = 0
    total_predictions = 0
    
    # Statistiche scommesse raccomandate
    total_recommended_bets = 0
    total_winning_bets = 0
    total_stake_recommended = 0
    
    for idx, match in df.iterrows():
        match_html, accuracy, correct, predictions = generate_match_card_html(match)
        matches_html.append(match_html)
        total_accuracy += accuracy
        total_correct += correct
        total_predictions += predictions
        
        # Conta scommesse raccomandate per statistiche generali
        recommended_bets = get_recommended_bets(match)
        total_recommended_bets += len(recommended_bets)
        total_stake_recommended += len(recommended_bets) * 10  # stake base €10
        
        for bet in recommended_bets:
            if bet['result_class'] == 'winning-bet':
                total_winning_bets += 1
    
    # Calcola statistiche globali
    global_stats = calculate_global_stats(df)
    
    # Genera tabella statistiche
    stats_rows = []
    for stat in global_stats:
        accuracy_class = "excellent" if stat['accuracy_general'] >= 75 else "good" if stat['accuracy_general'] >= 60 else "poor"
        stats_rows.append(f"""
        <tr class="{accuracy_class}">
            <td>{stat['market']}</td>
            <td>{stat['correct_general']}/{stat['total_general']}</td>
            <td>{stat['accuracy_general']:.1f}%</td>
            <td>{stat['avg_confidence']:.1f}%</td>
            <td>{stat['high_conf_info']}</td>
        </tr>
        """)
    
    # HTML completo
    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report Predizioni Serie A 2025/2026</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            
            .header .subtitle {{
                font-size: 1.2em;
                opacity: 0.9;
                margin-bottom: 20px;
            }}
            
            .global-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            
            .stat-card {{
                background: rgba(255,255,255,0.9);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .stat-number {{
                font-size: 2.5em;
                font-weight: bold;
                color: #2c3e50;
                display: block;
            }}
            
            .stat-label {{
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 5px;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .summary-section {{
                margin-bottom: 40px;
            }}
            
            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .summary-table th {{
                background: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            
            .summary-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ecf0f1;
            }}
            
            .summary-table tr.excellent {{ background-color: #d4edda; }}
            .summary-table tr.good {{ background-color: #fff3cd; }}
            .summary-table tr.poor {{ background-color: #f8d7da; }}
            
            .matches-section {{
                margin-top: 40px;
            }}
            
            .section-title {{
                font-size: 2em;
                color: #2c3e50;
                margin-bottom: 30px;
                text-align: center;
                position: relative;
            }}
            
            .section-title::after {{
                content: '';
                position: absolute;
                bottom: -10px;
                left: 50%;
                transform: translateX(-50%);
                width: 100px;
                height: 3px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 2px;
            }}
            
            .match-card {{
                background: white;
                margin-bottom: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border-left: 5px solid #bdc3c7;
            }}
            
            .match-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            }}
            
            .match-card.excellent {{ border-left-color: #27ae60; }}
            .match-card.good {{ border-left-color: #f39c12; }}
            .match-card.poor {{ border-left-color: #e74c3c; }}
            
            .match-header {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 25px;
                border-bottom: 1px solid #dee2e6;
            }}
            
            .match-teams {{
                font-size: 1.5em;
                color: #2c3e50;
                margin-bottom: 15px;
                text-align: center;
            }}
            
            .match-info {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }}
            
            .match-info span {{
                padding: 8px 15px;
                border-radius: 20px;
                background: rgba(255,255,255,0.8);
                font-size: 0.9em;
                font-weight: 500;
            }}
            
            .accuracy-excellent {{ background-color: #d4edda !important; color: #155724; }}
            .accuracy-good {{ background-color: #fff3cd !important; color: #856404; }}
            .accuracy-poor {{ background-color: #f8d7da !important; color: #721c24; }}
            
            .odds-section {{
                padding: 20px 25px;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            
            .odds-section h4 {{
                color: #495057;
                margin-bottom: 15px;
                font-size: 1.1em;
            }}
            
            .odds-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
            }}
            
            .odds-item {{
                background: white;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                color: #495057;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            
            .predictions-section {{
                padding: 25px;
            }}
            
            .predictions-section h4 {{
                color: #495057;
                margin-bottom: 20px;
                font-size: 1.2em;
            }}
            
            .market-group {{
                margin-bottom: 25px;
            }}
            
            .market-group h5 {{
                color: #6c757d;
                margin-bottom: 15px;
                font-size: 1em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .market-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }}
            
            .market-prediction {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 15px;
                border-radius: 8px;
                background: #f8f9fa;
                transition: all 0.3s ease;
            }}
            
            .market-prediction:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .market-prediction.high-confidence {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-left: 4px solid #28a745;
            }}
            
            .market-prediction.medium-confidence {{
                background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
                border-left: 4px solid #ffc107;
            }}
            
            .market-prediction.low-confidence {{
                background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
                border-left: 4px solid #dc3545;
            }}
            
            .market-name {{
                font-weight: 600;
                color: #495057;
                font-size: 0.9em;
            }}
            
            .prediction-value {{
                font-weight: bold;
                color: #2c3e50;
            }}
            
            .prediction-icon {{
                font-size: 1.2em;
            }}
            
            .recommended-bets {{
                padding: 20px 25px;
                background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%);
                border-bottom: 1px solid #dee2e6;
                border-left: 4px solid #28a745;
            }}
            
            .recommended-bets h4 {{
                color: #155724;
                margin-bottom: 15px;
                font-size: 1.1em;
            }}
            
            .bets-summary {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                flex-wrap: wrap;
                gap: 10px;
            }}
            
            .bets-summary span {{
                padding: 6px 12px;
                border-radius: 15px;
                font-size: 0.85em;
                font-weight: 600;
                background: rgba(255,255,255,0.8);
            }}
            
            .bets-success.excellent {{
                background-color: #d4edda !important;
                color: #155724;
            }}
            
            .bets-success.good {{
                background-color: #fff3cd !important;
                color: #856404;
            }}
            
            .bets-success.poor {{
                background-color: #f8d7da !important;
                color: #721c24;
            }}
            
            .bets-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .bets-table th {{
                background: #28a745;
                color: white;
                padding: 10px 12px;
                text-align: left;
                font-size: 0.9em;
                font-weight: 600;
            }}
            
            .bets-table td {{
                padding: 8px 12px;
                border-bottom: 1px solid #f1f3f4;
                font-size: 0.9em;
            }}
            
            .bet-row.winning-bet {{
                background-color: rgba(40, 167, 69, 0.1);
            }}
            
            .bet-row.losing-bet {{
                background-color: rgba(220, 53, 69, 0.1);
            }}
            
            .bet-name {{
                font-weight: 600;
                color: #2c3e50;
            }}
            
            .bet-confidence {{
                font-weight: bold;
                color: #28a745;
            }}
            
            .bet-stake {{
                font-weight: bold;
                color: #007bff;
            }}
            
            .bet-result {{
                font-size: 1.1em;
                text-align: center;
            }}
            
            .no-bets {{
                text-align: center;
                color: #6c757d;
                font-style: italic;
                margin: 15px 0;
                padding: 15px;
                background: rgba(108, 117, 125, 0.1);
                border-radius: 8px;
            }}
            
            .filter-section {{
                margin-bottom: 30px;
                text-align: center;
            }}
            
            .filter-btn {{
                display: inline-block;
                padding: 10px 20px;
                margin: 5px;
                border: none;
                border-radius: 25px;
                background: #6c757d;
                color: white;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 0.9em;
            }}
            
            .filter-btn:hover, .filter-btn.active {{
                background: #007bff;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,123,255,0.3);
            }}
            
            @media (max-width: 768px) {{
                .header h1 {{ font-size: 2em; }}
                .match-info {{ flex-direction: column; text-align: center; }}
                .odds-grid {{ grid-template-columns: 1fr; }}
                .market-grid {{ grid-template-columns: 1fr; }}
                .global-stats {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏆 Report Predizioni Serie A</h1>
                <div class="subtitle">Stagione 2025/2026 - Analisi Dettagliata</div>
                
                <div class="global-stats">
                    <div class="stat-card">
                        <span class="stat-number">{len(df)}</span>
                        <span class="stat-label">Partite Analizzate</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{(total_accuracy/len(df)):.1f}%</span>
                        <span class="stat-label">Accuracy Media</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{total_recommended_bets}</span>
                        <span class="stat-label">Scommesse Raccomandate</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{(total_winning_bets/total_recommended_bets*100) if total_recommended_bets > 0 else 0:.1f}%</span>
                        <span class="stat-label">Win Rate Scommesse</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">€{total_stake_recommended}</span>
                        <span class="stat-label">Stake Totale</span>
                    </div>
                </div>
            </div>
            
            <div class="content">
                <div class="summary-section">
                    <h2 class="section-title">📊 Riassunto Performance per Mercato</h2>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>Mercato</th>
                                <th>Corrette/Totali</th>
                                <th>Accuracy</th>
                                <th>Confidence Media</th>
                                <th>Alta Confidence (≥70%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(stats_rows)}
                        </tbody>
                    </table>
                </div>
                
                <div class="matches-section">
                    <h2 class="section-title">🏟️ Dettaglio Partite</h2>
                    
                    <div class="filter-section">
                        <button class="filter-btn active" onclick="filterMatches('all')">Tutte le Partite</button>
                        <button class="filter-btn" onclick="filterMatches('excellent')">Eccellenti (≥75%)</button>
                        <button class="filter-btn" onclick="filterMatches('good')">Buone (60-74%)</button>
                        <button class="filter-btn" onclick="filterMatches('poor')">Da Migliorare (&lt;60%)</button>
                    </div>
                    
                    <div class="matches-container">
                        {''.join(matches_html)}
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function filterMatches(category) {{
                const matches = document.querySelectorAll('.match-card');
                const buttons = document.querySelectorAll('.filter-btn');
                
                // Reset button states
                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                
                // Filter matches
                matches.forEach(match => {{
                    if (category === 'all') {{
                        match.style.display = 'block';
                    }} else {{
                        if (match.classList.contains(category)) {{
                            match.style.display = 'block';
                        }} else {{
                            match.style.display = 'none';
                        }}
                    }}
                }});
            }}
            
            // Smooth scrolling for better UX
            document.querySelectorAll('.match-card').forEach(card => {{
                card.addEventListener('click', function() {{
                    this.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Salva il file HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Report HTML generato: {output_file}")
    print(f"📊 Statistiche incluse:")
    print(f"  - {len(df)} partite analizzate")
    print(f"  - {total_correct}/{total_predictions} predizioni corrette")
    print(f"  - {(total_accuracy/len(df)):.1f}% accuracy media")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generatore Report HTML Serie A 2025/2026')
    parser.add_argument('--predictions', default='predictions_2025_2026_complete.csv', help='File CSV predizioni')
    parser.add_argument('--output', default='report_stagione_2025_2026.html', help='File HTML output')
    
    args = parser.parse_args()
    
    # Carica e processa i dati
    df = load_prediction_data(args.predictions)
    
    # Genera report HTML
    output_file = generate_html_report(df, args.output)
    
    print(f"\n🎉 Report completato! Apri {output_file} nel browser per visualizzarlo.")

if __name__ == "__main__":
    main()