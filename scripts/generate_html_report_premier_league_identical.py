#!/usr/bin/env python3
"""
Generatore Report HTML Premier League 2025/2026 - Stile identico a Serie A
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_and_process_data(csv_file):
    """Carica e processa i dati delle predizioni Premier League"""
    print(f"📊 Caricando predizioni Premier League 2025/2026...")
    df = pd.read_csv(csv_file)
    
    # Prende le ultime 80 partite (stagione 2025/2026)
    season_df = df.tail(80).copy()
    
    # Converti date
    season_df['date'] = pd.to_datetime(season_df['date'])
    
    print(f"✅ Caricate {len(season_df)} partite Premier League 2025/2026")
    
    return season_df

def calculate_market_accuracy(df):
    """Calcola l'accuratezza per ogni mercato"""
    results = {}
    
    markets = {
        'Casa (1)': ('1X2_H', 'actual_home_goals', 'actual_away_goals', lambda h, a: h > a),
        'Pareggio (X)': ('1X2_D', 'actual_home_goals', 'actual_away_goals', lambda h, a: h == a),  
        'Ospite (2)': ('1X2_A', 'actual_home_goals', 'actual_away_goals', lambda h, a: h < a),
        'Over 0.5': ('O_0_5', 'actual_home_goals', 'actual_away_goals', lambda h, a: (h + a) > 0.5),
        'Over 1.5': ('O_1_5', 'actual_home_goals', 'actual_away_goals', lambda h, a: (h + a) > 1.5),
        'MG Casa 1-3': ('MG_Casa_1_3', 'actual_home_goals', None, lambda h, _: 1 <= h <= 3),
        'MG Casa 1-4': ('MG_Casa_1_4', 'actual_home_goals', None, lambda h, _: 1 <= h <= 4),
        'MG Casa 1-5': ('MG_Casa_1_5', 'actual_home_goals', None, lambda h, _: 1 <= h <= 5),
        'MG Ospite 1-3': ('MG_Ospite_1_3', 'actual_away_goals', None, lambda a, _: 1 <= a <= 3),
        'MG Ospite 1-4': ('MG_Ospite_1_4', 'actual_away_goals', None, lambda a, _: 1 <= a <= 4),
        'MG Ospite 1-5': ('MG_Ospite_1_5', 'actual_away_goals', None, lambda a, _: 1 <= a <= 5),
    }
    
    completed_matches = df.dropna(subset=['actual_home_goals', 'actual_away_goals'])
    
    for market_name, (pred_col, goal_col1, goal_col2, condition) in markets.items():
        if pred_col in df.columns:
            if goal_col2:
                actual_results = completed_matches.apply(
                    lambda row: condition(row[goal_col1], row[goal_col2]), axis=1
                )
            else:
                actual_results = completed_matches.apply(
                    lambda row: condition(row[goal_col1], None), axis=1
                )
            
            predictions = completed_matches[pred_col] > 0.5
            accuracy = (predictions == actual_results).mean() if len(actual_results) > 0 else 0
            
            # Calcola accuratezza alta confidenza
            high_conf_mask = completed_matches[pred_col] >= 0.7
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (
                    completed_matches.loc[high_conf_mask, pred_col] > 0.5
                ) == actual_results[high_conf_mask]
                high_conf_acc = high_conf_accuracy.mean()
                high_conf_count = high_conf_mask.sum()
                high_conf_correct = high_conf_accuracy.sum()
            else:
                high_conf_acc = None
                high_conf_count = 0
                high_conf_correct = 0
            
            results[market_name] = {
                'correct': (predictions == actual_results).sum(),
                'total': len(actual_results),
                'accuracy': accuracy,
                'avg_confidence': completed_matches[pred_col].mean(),
                'high_conf_count': high_conf_count,
                'high_conf_correct': high_conf_correct,
                'high_conf_acc': high_conf_acc
            }
    
    return results

def get_match_predictions_and_results(match):
    """Ottieni predizioni e risultati per una singola partita"""
    predictions = {}
    
    # 1X2
    predictions['1X2_H'] = match.get('1X2_H', 0)
    predictions['1X2_D'] = match.get('1X2_D', 0) 
    predictions['1X2_A'] = match.get('1X2_A', 0)
    
    # Over
    predictions['O_0_5'] = match.get('O_0_5', 0)
    predictions['O_1_5'] = match.get('O_1_5', 0)
    
    # Multigol
    for mg_type in ['Casa', 'Ospite']:
        for mg_range in ['1_3', '1_4', '1_5']:
            key = f'MG_{mg_type}_{mg_range}'
            predictions[key] = match.get(key, 0)
    
    # Risultati reali se disponibili
    if pd.notna(match.get('actual_home_goals')):
        home_goals = int(match['actual_home_goals'])
        away_goals = int(match['actual_away_goals'])
        
        predictions['actual_1X2_H'] = home_goals > away_goals
        predictions['actual_1X2_D'] = home_goals == away_goals
        predictions['actual_1X2_A'] = home_goals < away_goals
        predictions['actual_O_0_5'] = (home_goals + away_goals) > 0.5
        predictions['actual_O_1_5'] = (home_goals + away_goals) > 1.5
        predictions['actual_MG_Casa_1_3'] = 1 <= home_goals <= 3
        predictions['actual_MG_Casa_1_4'] = 1 <= home_goals <= 4 
        predictions['actual_MG_Casa_1_5'] = 1 <= home_goals <= 5
        predictions['actual_MG_Ospite_1_3'] = 1 <= away_goals <= 3
        predictions['actual_MG_Ospite_1_4'] = 1 <= away_goals <= 4
        predictions['actual_MG_Ospite_1_5'] = 1 <= away_goals <= 5
    
    return predictions

def get_recommended_bets(match_data):
    """Genera raccomandazioni scommesse per una partita con soglie specifiche PL"""
    recommendations = []
    
    # Over 0.5 - Premier League è molto offensiva
    if match_data.get('O_0_5', 0) >= 0.85:
        confidence = match_data['O_0_5'] * 100
        stake = 20 if confidence >= 90 else 15
        recommendations.append({
            'market': 'Over 0.5 Goals',
            'confidence': f"{confidence:.1f}%",
            'stake': f"€{stake}",
            'actual': match_data.get('actual_O_0_5', None)
        })
    
    # Over 1.5 
    if match_data.get('O_1_5', 0) >= 0.72:
        confidence = match_data['O_1_5'] * 100
        stake = 20 if confidence >= 75 else 15
        recommendations.append({
            'market': 'Over 1.5 Goals',
            'confidence': f"{confidence:.1f}%", 
            'stake': f"€{stake}",
            'actual': match_data.get('actual_O_1_5', None)
        })
    
    # Multigol Casa
    for mg_range, threshold in [('1_3', 0.70), ('1_4', 0.75), ('1_5', 0.78)]:
        mg_prob = match_data.get(f'MG_Casa_{mg_range}', 0)
        if mg_prob >= threshold:
            confidence = mg_prob * 100
            stake = 15 if confidence >= 75 else 10
            range_display = mg_range.replace('_', '-')
            recommendations.append({
                'market': f'Multigol Casa {range_display}',
                'confidence': f"{confidence:.1f}%",
                'stake': f"€{stake}",
                'actual': match_data.get(f'actual_MG_Casa_{mg_range}', None)
            })
    
    # Multigol Ospite
    for mg_range, threshold in [('1_3', 0.68), ('1_4', 0.72), ('1_5', 0.75)]:
        mg_prob = match_data.get(f'MG_Ospite_{mg_range}', 0)
        if mg_prob >= threshold:
            confidence = mg_prob * 100
            stake = 15 if confidence >= 75 else 10
            range_display = mg_range.replace('_', '-')
            recommendations.append({
                'market': f'Multigol Ospite {range_display}',
                'confidence': f"{confidence:.1f}%",
                'stake': f"€{stake}",
                'actual': match_data.get(f'actual_MG_Ospite_{mg_range}', None)
            })
    
    return recommendations

def generate_html_report(df, market_stats, output_file):
    """Genera report HTML identico a quello Serie A"""
    
    # Statistiche globali
    total_matches = len(df)
    completed_matches = df.dropna(subset=['actual_home_goals', 'actual_away_goals'])
    total_completed = len(completed_matches)
    
    # Calcola accuracy media
    accuracies = [stats['accuracy'] for stats in market_stats.values() if stats['accuracy'] > 0]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    # Conta scommesse raccomandate e win rate
    total_recommendations = 0
    winning_bets = 0
    total_stake = 0
    
    for _, match in df.iterrows():
        match_data = get_match_predictions_and_results(match)
        recommendations = get_recommended_bets(match_data)
        total_recommendations += len(recommendations)
        
        for bet in recommendations:
            stake_amount = int(bet['stake'].replace('€', ''))
            total_stake += stake_amount
            if bet['actual'] is True:
                winning_bets += 1
    
    win_rate = (winning_bets / total_recommendations * 100) if total_recommendations > 0 else 0
    
    # Inizia HTML - copia ESATTO dalla Serie A
    html_content = f"""
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Report Predizioni Premier League 2025/2026</title>
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
                <h1>🏴󠁧󠁢󠁥󠁮󠁧󠁿 Report Predizioni Premier League</h1>
                <div class="subtitle">Stagione 2025/2026 - Analisi Dettagliata</div>
                
                <div class="global-stats">
                    <div class="stat-card">
                        <span class="stat-number">{total_matches}</span>
                        <span class="stat-label">Partite Analizzate</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{avg_accuracy:.1%}</span>
                        <span class="stat-label">Accuracy Media</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{total_recommendations}</span>
                        <span class="stat-label">Scommesse Raccomandate</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{win_rate:.1f}%</span>
                        <span class="stat-label">Win Rate Scommesse</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">€{total_stake}</span>
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
    """
    
    # Aggiungi righe tabella riassunto
    for market_name, stats in market_stats.items():
        if stats['accuracy'] >= 0.75:
            row_class = 'excellent'
        elif stats['accuracy'] >= 0.6:
            row_class = 'good'
        else:
            row_class = 'poor'
        
        high_conf_display = 'N/A'
        if stats['high_conf_count'] > 0:
            high_conf_display = f"{stats['high_conf_correct']}/{stats['high_conf_count']} ({stats['high_conf_acc']:.1%})"
        
        html_content += f"""
        <tr class="{row_class}">
            <td>{market_name}</td>
            <td>{stats['correct']}/{stats['total']}</td>
            <td>{stats['accuracy']:.1%}</td>
            <td>{stats['avg_confidence']:.1%}</td>
            <td>{high_conf_display}</td>
        </tr>
        """
    
    html_content += """
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
    """
    
    # Aggiungi le partite
    for _, match in df.iterrows():
        match_data = get_match_predictions_and_results(match)
        recommendations = get_recommended_bets(match_data)
        
        # Calcola accuracy per la partita
        correct_predictions = 0
        total_predictions = 0
        
        for market in ['1X2_H', '1X2_D', '1X2_A', 'O_0_5', 'O_1_5', 'MG_Casa_1_3', 'MG_Casa_1_4', 'MG_Casa_1_5', 'MG_Ospite_1_3', 'MG_Ospite_1_4', 'MG_Ospite_1_5']:
            if f'actual_{market}' in match_data:
                prediction = match_data[market] > 0.5
                actual = match_data[f'actual_{market}']
                if prediction == actual:
                    correct_predictions += 1
                total_predictions += 1
        
        match_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Determina classe qualità
        if match_accuracy >= 0.75:
            card_class = 'excellent'
            accuracy_class = 'accuracy-excellent'
        elif match_accuracy >= 0.6:
            card_class = 'good' 
            accuracy_class = 'accuracy-good'
        else:
            card_class = 'poor'
            accuracy_class = 'accuracy-poor'
        
        # Calcola successo scommesse
        if recommendations:
            winning_bets_match = sum(1 for bet in recommendations if bet['actual'] is True)
            bet_success_rate = winning_bets_match / len(recommendations)
            
            if bet_success_rate >= 0.75:
                bet_success_class = 'excellent'
            elif bet_success_rate >= 0.5:
                bet_success_class = 'good'
            else:
                bet_success_class = 'poor'
        else:
            bet_success_class = 'good'
            bet_success_rate = 0
        
        # Dati partita
        date_str = match['date'].strftime('%Y-%m-%d')
        home_team = match['home_team']
        away_team = match['away_team']
        
        if pd.notna(match.get('actual_home_goals')):
            result = f"{int(match['actual_home_goals'])}-{int(match['actual_away_goals'])}"
        else:
            result = "N/A"
        
        html_content += f"""
    <div class="match-card {card_class}">
        <div class="match-header">
            <h3 class="match-teams">🏠 {home_team} vs {away_team} 🏃</h3>
            <div class="match-info">
                <span class="match-date">📅 {date_str}</span>
                <span class="match-result">⚽ Risultato: {result}</span>
                <span class="match-accuracy {accuracy_class}">
                    🎯 Accuracy: {match_accuracy:.1%} ({correct_predictions}/{total_predictions})
                </span>
            </div>
        </div>
        """
        
        # Quote se disponibili
        if pd.notna(match.get('odds_1')):
            html_content += f"""
        <div class="odds-section">
            <h4>💰 Quote Bookmaker</h4>
            <div class="odds-grid">
                <span class="odds-item">1: {match['odds_1']:.2f}</span>
                <span class="odds-item">X: {match['odds_X']:.2f}</span>
                <span class="odds-item">2: {match['odds_2']:.2f}</span>
            </div>
        </div>
        """
        
        # Sezione predizioni
        html_content += """
        <div class="predictions-section">
            <h4>🎯 Predizioni del Modello</h4>
        """
        
        # Predizioni 1X2
        html_content += """
            <div class="market-group">
                <h5>⚽ Risultato Finale (1X2)</h5>
                <div class="market-grid">
        """
        
        for market, label in [('1X2_H', '1 (Casa)'), ('1X2_D', 'X (Pareggio)'), ('1X2_A', '2 (Ospite)')]:
            prob = match_data.get(market, 0)
            actual = match_data.get(f'actual_{market}')
            
            if prob >= 0.7:
                conf_class = 'high-confidence'
            elif prob >= 0.5:
                conf_class = 'medium-confidence'  
            else:
                conf_class = 'low-confidence'
            
            icon = '✅' if actual is True else ('❌' if actual is False else '❓')
            
            html_content += f"""
                    <div class="market-prediction {conf_class}">
                        <span class="market-name">{label}</span>
                        <span class="prediction-value">{prob:.1%} {icon}</span>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
        
        # Predizioni Over
        html_content += """
            <div class="market-group">
                <h5>📊 Mercati Over Goals</h5>
                <div class="market-grid">
        """
        
        for market, label in [('O_0_5', 'Over 0.5'), ('O_1_5', 'Over 1.5')]:
            prob = match_data.get(market, 0)
            actual = match_data.get(f'actual_{market}')
            
            if prob >= 0.7:
                conf_class = 'high-confidence'
            elif prob >= 0.5:
                conf_class = 'medium-confidence'
            else:
                conf_class = 'low-confidence'
            
            icon = '✅' if actual is True else ('❌' if actual is False else '❓')
            
            html_content += f"""
                    <div class="market-prediction {conf_class}">
                        <span class="market-name">{label}</span>
                        <span class="prediction-value">{prob:.1%} {icon}</span>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
        
        # Predizioni Multigol
        for team, team_label in [('Casa', '🏠 Multigol Casa'), ('Ospite', '✈️ Multigol Ospite')]:
            html_content += f"""
            <div class="market-group">
                <h5>{team_label}</h5>
                <div class="market-grid">
            """
            
            for mg_range in ['1_3', '1_4', '1_5']:
                market = f'MG_{team}_{mg_range}'
                label = f'{team} {mg_range.replace("_", "-")}'
                prob = match_data.get(market, 0)
                actual = match_data.get(f'actual_{market}')
                
                if prob >= 0.7:
                    conf_class = 'high-confidence'
                elif prob >= 0.5:
                    conf_class = 'medium-confidence'
                else:
                    conf_class = 'low-confidence'
                
                icon = '✅' if actual is True else ('❌' if actual is False else '❓')
                
                html_content += f"""
                        <div class="market-prediction {conf_class}">
                            <span class="market-name">{label}</span>
                            <span class="prediction-value">{prob:.1%} {icon}</span>
                        </div>
                """
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </div>
        """
        
        # Scommesse raccomandate
        if recommendations:
            html_content += f"""
        <div class="recommended-bets">
            <h4>💰 Scommesse Raccomandate</h4>
            <div class="bets-summary">
                <span>🎯 {len(recommendations)} scommesse</span>
                <span class="bets-success {bet_success_class}">✅ {winning_bets_match}/{len(recommendations)} vincenti ({bet_success_rate:.1%})</span>
                <span>💰 Stake totale: €{sum(int(bet['stake'].replace('€', '')) for bet in recommendations)}</span>
            </div>
            
            <table class="bets-table">
                <thead>
                    <tr>
                        <th>Mercato</th>
                        <th>Confidence</th>
                        <th>Stake</th>
                        <th>Risultato</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for bet in recommendations:
                bet_class = 'winning-bet' if bet['actual'] is True else ('losing-bet' if bet['actual'] is False else '')
                result_icon = '✅' if bet['actual'] is True else ('❌' if bet['actual'] is False else '❓')
                
                html_content += f"""
                    <tr class="bet-row {bet_class}">
                        <td class="bet-name">{bet['market']}</td>
                        <td class="bet-confidence">{bet['confidence']}</td>
                        <td class="bet-stake">{bet['stake']}</td>
                        <td class="bet-result">{result_icon}</td>
                    </tr>
                """
            
            html_content += """
                </tbody>
            </table>
        </div>
            """
        else:
            html_content += """
        <div class="no-bets">
            Nessuna scommessa raccomandata per questa partita
        </div>
            """
        
        html_content += """
    </div>
        """
    
    # Chiusura HTML
    html_content += """
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function filterMatches(type) {
                const cards = document.querySelectorAll('.match-card');
                const buttons = document.querySelectorAll('.filter-btn');
                
                buttons.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');
                
                cards.forEach(card => {
                    if (type === 'all') {
                        card.style.display = 'block';
                    } else {
                        if (card.classList.contains(type)) {
                            card.style.display = 'block';
                        } else {
                            card.style.display = 'none';
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    
    # Salva file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Report HTML generato: {output_file}")
    return html_content

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Generatore Report HTML Premier League identico a Serie A')
    parser.add_argument('--predictions', default='reports/predictions_premier_league_2025_2026.csv',
                       help='File CSV con le predizioni Premier League')
    parser.add_argument('--output', default='reports/report_premier_league_2025_2026_con_scommesse.html',
                       help='File HTML di output')
    
    args = parser.parse_args()
    
    print("🏴󠁧󠁢󠁥󠁮󠁧󠁿 GENERATORE REPORT PREMIER LEAGUE - STILE SERIE A")
    print("=" * 60)
    
    try:
        # Carica dati
        df = load_and_process_data(args.predictions)
        
        # Calcola statistiche mercati
        print("📊 Calcolando statistiche mercati...")
        market_stats = calculate_market_accuracy(df)
        
        # Genera report HTML
        print("📄 Generando report HTML identico a Serie A...")
        generate_html_report(df, market_stats, args.output)
        
        print(f"\n✅ Report completato con successo!")
        print(f"   📄 File: {args.output}")
        print(f"   🎯 Stile identico al report Serie A")
        
    except FileNotFoundError:
        print(f"❌ Errore: File {args.predictions} non trovato!")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()