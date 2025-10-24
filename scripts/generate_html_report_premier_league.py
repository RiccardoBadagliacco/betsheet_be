#!/usr/bin/env python3
"""
Generatore Report HTML Premier League 2025/2026
Crea un report dettagliato con raccomandazioni di scommessa
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

def load_predictions_data(csv_file):
    """Carica i dati delle predizioni dalla Premier League"""
    
    print(f"📊 Caricando predizioni Premier League 2025/2026...")
    df = pd.read_csv(csv_file)
    
    # Prende le ultime 80 partite (stagione 2025/2026)
    season_df = df.tail(80).copy()
    
    # Converti date
    season_df['date'] = pd.to_datetime(season_df['date'])
    
    print(f"✅ Caricate {len(season_df)} partite Premier League 2025/2026")
    
    return season_df

def get_recommended_bets(df):
    """Identifica le scommesse raccomandate dal modello per Premier League"""
    
    recommendations = []
    
    for idx, match in df.iterrows():
        match_recs = {
            'date': match['date'].strftime('%d/%m/%Y'),
            'match': f"{match['home_team']} vs {match['away_team']}",
            'actual_score': f"{int(match['actual_home_goals'])}-{int(match['actual_away_goals'])}" if pd.notna(match.get('actual_home_goals')) else 'N/A',
            'recommendations': []
        }
        
        # Over 0.5 - Soglia alta per Premier League (molto offensiva)
        if match.get('O_0_5', 0) >= 0.85:
            confidence = match['O_0_5'] * 100
            match_recs['recommendations'].append({
                'market': 'Over 0.5 Gol',
                'probability': f"{confidence:.1f}%",
                'confidence': 'ALTA',
                'reason': f'Probabilità {confidence:.1f}% - Entrambe squadre offensive'
            })
        
        # Over 1.5 - Soglia per Premier League
        if match.get('O_1_5', 0) >= 0.72:
            confidence = match['O_1_5'] * 100
            match_recs['recommendations'].append({
                'market': 'Over 1.5 Gol',
                'probability': f"{confidence:.1f}%",
                'confidence': 'ALTA' if confidence >= 75 else 'MEDIA',
                'reason': f'Probabilità {confidence:.1f}% - Partita con gol'
            })
        
        # Multigol Casa - Soglie per Premier League
        for mg_range, threshold in [('1_3', 0.70), ('1_4', 0.75), ('1_5', 0.78)]:
            mg_prob = match.get(f'MG_Casa_{mg_range}', 0)
            if mg_prob >= threshold:
                confidence = mg_prob * 100
                range_display = mg_range.replace('_', '-')
                match_recs['recommendations'].append({
                    'market': f'Multigol Casa {range_display}',
                    'probability': f"{confidence:.1f}%",
                    'confidence': 'ALTA' if confidence >= 75 else 'MEDIA',
                    'reason': f'Casa forte in attacco - Probabilità {confidence:.1f}%'
                })
        
        # Multigol Ospite - Soglie per Premier League  
        for mg_range, threshold in [('1_3', 0.68), ('1_4', 0.72), ('1_5', 0.75)]:
            mg_prob = match.get(f'MG_Ospite_{mg_range}', 0)
            if mg_prob >= threshold:
                confidence = mg_prob * 100
                range_display = mg_range.replace('_', '-')
                match_recs['recommendations'].append({
                    'market': f'Multigol Ospite {range_display}',
                    'probability': f"{confidence:.1f}%",
                    'confidence': 'ALTA' if confidence >= 75 else 'MEDIA',
                    'reason': f'Ospite in forma - Probabilità {confidence:.1f}%'
                })
        
        # Solo se ci sono raccomandazioni
        if match_recs['recommendations']:
            recommendations.append(match_recs)
    
    return recommendations

def generate_html_report(df, recommendations, output_file="report_premier_league_2025_2026_con_scommesse.html"):
    """Genera il report HTML completo per Premier League"""
    
    # Statistiche generali
    total_matches = len(df)
    matches_with_results = df.dropna(subset=['actual_home_goals', 'actual_away_goals'])
    total_completed = len(matches_with_results)
    
    # Performance sui mercati principali
    over_05_acc = ((df['O_0_5'] > 0.5) == (df['actual_home_goals'] + df['actual_away_goals'] > 0.5)).mean() if total_completed > 0 else 0
    over_15_acc = ((df['O_1_5'] > 0.5) == (df['actual_home_goals'] + df['actual_away_goals'] > 1.5)).mean() if total_completed > 0 else 0
    
    # Conteggio raccomandazioni
    total_recommendations = sum(len(r['recommendations']) for r in recommendations)
    high_conf_recs = sum(len([rec for rec in r['recommendations'] if rec['confidence'] == 'ALTA']) for r in recommendations)
    
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
            
            .odd-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.3s ease;
            }}
            
            .odd-item:hover {{
                transform: scale(1.02);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .odd-item.recommended {{
                border-color: #27ae60;
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            }}
            
            .odd-label {{
                font-weight: 600;
                color: #495057;
                margin-bottom: 8px;
                font-size: 0.9em;
            }}
            
            .odd-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            
            .odd-confidence {{
                font-size: 0.8em;
                color: #27ae60;
                margin-top: 5px;
                font-weight: 500;
            }}
            
            .predictions-section {{
                padding: 20px 25px;
                background: white;
            }}
            
            .predictions-section h4 {{
                color: #495057;
                margin-bottom: 20px;
                font-size: 1.2em;
                text-align: center;
            }}
            
            .predictions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }}
            
            .prediction-group {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid #34495e;
            }}
            
            .prediction-group h5 {{
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.1em;
                text-align: center;
            }}
            
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 8px 0;
                padding: 8px 12px;
                background: white;
                border-radius: 6px;
                font-size: 0.9em;
            }}
            
            .prediction-label {{
                font-weight: 500;
                color: #495057;
            }}
            
            .prediction-value {{
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
            }}
            
            .probability-high {{ background: #d4edda; color: #155724; }}
            .probability-medium {{ background: #fff3cd; color: #856404; }}
            .probability-low {{ background: #f8d7da; color: #721c24; }}
            
            .recommendations-section {{
                padding: 20px 25px;
                background: #e8f4fd;
                border-top: 3px solid #3498db;
            }}
            
            .recommendations-section h4 {{
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.2em;
                text-align: center;
            }}
            
            .recommendation {{
                background: white;
                padding: 15px 20px;
                margin: 10px 0;
                border-radius: 10px;
                border-left: 5px solid #3498db;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .recommendation.high-confidence {{
                border-left-color: #27ae60;
                background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
            }}
            
            .recommendation.medium-confidence {{
                border-left-color: #f39c12;
                background: linear-gradient(135deg, #ffffff 0%, #fffcf8 100%);
            }}
            
            .recommendation-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }}
            
            .recommendation-market {{
                font-weight: bold;
                font-size: 1.1em;
                color: #2c3e50;
            }}
            
            .recommendation-probability {{
                background: #3498db;
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 0.9em;
                font-weight: 600;
            }}
            
            .recommendation.high-confidence .recommendation-probability {{
                background: #27ae60;
            }}
            
            .recommendation.medium-confidence .recommendation-probability {{
                background: #f39c12;
            }}
            
            .recommendation-reason {{
                color: #7f8c8d;
                font-size: 0.9em;
                font-style: italic;
            }}
            
            .filters {{
                background: #ecf0f1;
                padding: 25px;
                border-bottom: 1px solid #bdc3c7;
            }}
            
            .filters h3 {{
                color: #2c3e50;
                margin-bottom: 20px;
                text-align: center;
            }}
            
            .filter-row {{
                display: flex;
                justify-content: center;
                gap: 30px;
                flex-wrap: wrap;
            }}
            
            .filter-group {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .filter-group label {{
                font-weight: 600;
                color: #34495e;
                min-width: 80px;
            }}
            
            .filter-group select {{
                padding: 8px 15px;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                background: white;
                font-size: 0.9em;
                min-width: 150px;
            }}
            
            .filter-group select:focus {{
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }}
            
            .no-matches {{
                text-align: center;
                padding: 60px 20px;
                color: #7f8c8d;
            }}
            
            .no-matches h3 {{
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    margin: 10px;
                    border-radius: 10px;
                }}
                
                .header {{
                    padding: 20px;
                }}
                
                .header h1 {{
                    font-size: 2em;
                }}
                
                .content {{
                    padding: 20px;
                }}
                
                .global-stats {{
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }}
                
                .stat-number {{
                    font-size: 2em;
                }}
                
                .predictions-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .odds-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
                
                .filter-row {{
                    flex-direction: column;
                    gap: 15px;
                }}
                
                .match-info {{
                    flex-direction: column;
                    align-items: flex-start;
                }}
            }}
            
            @media (max-width: 480px) {{
                body {{
                    padding: 10px;
                }}
                
                .global-stats {{
                    grid-template-columns: 1fr;
                }}
                
                .odds-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .stat-number {{
                    font-size: 1.8em;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏴󠁧󠁢󠁥󠁮󠁧󠁿 Report Predizioni Premier League 2025/2026</h1>
                <div class="subtitle">
                    Analisi dettagliata delle performance del modello • Generato il {datetime.now().strftime('%d/%m/%Y alle %H:%M')}
                </div>
                <div class="global-stats">
                    <div class="stat-card">
                        <span class="stat-number">{total_matches}</span>
                        <div class="stat-label">Partite Totali</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{total_completed}</span>
                        <div class="stat-label">Partite Giocate</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{over_05_acc:.1%}</span>
                        <div class="stat-label">Accuratezza Over 0.5</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{over_15_acc:.1%}</span>
                        <div class="stat-label">Accuratezza Over 1.5</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{total_recommendations}</span>
                        <div class="stat-label">Raccomandazioni</div>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{high_conf_recs}</span>
                        <div class="stat-label">Alta Confidenza</div>
                    </div>
                </div>
            </div>
            
            <div class="content">
                <div class="matches-section">
                    <div class="section-title">🎯 Analisi Dettagliata Partite con Raccomandazioni</div>
                    
                    <div class="filters">
                        <h3>🔍 Filtra Risultati</h3>
                        <div class="filter-row">
                            <div class="filter-group">
                                <label for="confidenceFilter">Confidenza:</label>
                                <select id="confidenceFilter" onchange="filterMatches()">
                                    <option value="all">Tutte le confidenze</option>
                                    <option value="high-confidence">Solo Alta Confidenza</option>
                                    <option value="medium-confidence">Solo Media Confidenza</option>
                                </select>
                            </div>
                            <div class="filter-group">
                                <label for="marketFilter">Mercato:</label>
                                <select id="marketFilter" onchange="filterMatches()">
                                    <option value="all">Tutti i mercati</option>
                                    <option value="over">Over Goals</option>
                                    <option value="multigol-casa">Multigol Casa</option>
                                    <option value="multigol-ospite">Multigol Ospite</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div id="matches-container">
    """
    
    # Aggiungi le partite con raccomandazioni
    if recommendations:
        for rec in recommendations:
            # Determina la classe di qualità
            high_count = sum(1 for r in rec['recommendations'] if r['confidence'] == 'ALTA')
            if high_count >= 2:
                card_class = 'excellent'
            elif high_count >= 1:
                card_class = 'good'
            else:
                card_class = 'poor'
            
            html_content += f"""
                    <div class="match-card {card_class}" data-match-id="match-{rec['match'].lower().replace(' ', '-').replace('vs', '')}">
                        <div class="match-header">
                            <div class="match-teams">{rec['match']}</div>
                            <div class="match-info">
                                <span>📅 {rec['date']}</span>
                                <span>⚽ Risultato: {rec['actual_score']}</span>
                                <span>🎯 {len(rec['recommendations'])} raccomandazioni</span>
                                <span>⭐ {high_count} alta confidenza</span>
                            </div>
                        </div>
                        
                        <div class="recommendations-section">
                            <h4>🎲 Raccomandazioni Scommesse</h4>
            """
            
            for recommendation in rec['recommendations']:
                conf_class = 'high-confidence' if recommendation['confidence'] == 'ALTA' else 'medium-confidence'
                market_type = 'over' if 'Over' in recommendation['market'] else ('multigol-casa' if 'Casa' in recommendation['market'] else 'multigol-ospite')
                
                html_content += f"""
                            <div class="recommendation {conf_class}" data-confidence="{conf_class}" data-market="{market_type}">
                                <div class="recommendation-header">
                                    <div class="recommendation-market">{recommendation['market']}</div>
                                    <div class="recommendation-probability">{recommendation['probability']}</div>
                                </div>
                                <div class="recommendation-reason">{recommendation['reason']}</div>
                            </div>
                """
            
            html_content += """
                        </div>
                    </div>
            """
    else:
        html_content += """
                    <div class="no-matches">
                        <h3>Nessuna partita trovata</h3>
                        <p>Non ci sono partite che corrispondono ai criteri selezionati.</p>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
        </div>
        
        <script>
            function filterMatches() {{
                const confidenceFilter = document.getElementById('confidenceFilter').value;
                const marketFilter = document.getElementById('marketFilter').value;
                
                const matchCards = document.querySelectorAll('.match-card');
                let visibleCount = 0;
                
                matchCards.forEach(card => {{
                    const recommendations = card.querySelectorAll('.recommendation');
                    let hasVisibleRecs = false;
                    
                    recommendations.forEach(rec => {{
                        const confidence = rec.getAttribute('data-confidence');
                        const market = rec.getAttribute('data-market');
                        
                        let showConfidence = confidenceFilter === 'all' || confidence === confidenceFilter;
                        let showMarket = marketFilter === 'all' || market === marketFilter;
                        
                        if (showConfidence && showMarket) {{
                            rec.style.display = 'block';
                            hasVisibleRecs = true;
                        }} else {{
                            rec.style.display = 'none';
                        }}
                    }});
                    
                    if (hasVisibleRecs) {{
                        card.style.display = 'block';
                        visibleCount++;
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});
                
                // Mostra/nascondi messaggio "nessuna partita"
                const noMatches = document.querySelector('.no-matches');
                const container = document.getElementById('matches-container');
                
                if (visibleCount === 0 && !noMatches) {{
                    container.innerHTML = `
                        <div class="no-matches">
                            <h3>Nessuna partita trovata</h3>
                            <p>Non ci sono partite che corrispondono ai criteri selezionati.</p>
                        </div>
                    `;
                }} else if (visibleCount > 0 && noMatches) {{
                    // Ripristina le partite originali se erano state nascoste
                    location.reload();
                }}
            }}
            
            // Inizializza i filtri al caricamento della pagina
            document.addEventListener('DOMContentLoaded', function() {{
                filterMatches();
            }});
            
            // Aggiungi animazioni smooth al scroll
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{
                            behavior: 'smooth',
                            block: 'start'
                        }});
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Salva il file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Report HTML generato: {output_file}")
    print(f"   📊 {total_matches} partite analizzate")
    print(f"   🎯 {len(recommendations)} partite con raccomandazioni")
    print(f"   ⭐ {total_recommendations} raccomandazioni totali")

def main():
    """Funzione principale"""
    
    parser = argparse.ArgumentParser(description='Generatore Report HTML Premier League 2025/2026')
    parser.add_argument('--predictions', default='predictions_premier_league_2025_2026.csv',
                       help='File CSV con le predizioni Premier League')
    parser.add_argument('--output', default='reports/report_premier_league_2025_2026_con_scommesse.html',
                       help='File HTML di output')
    
    args = parser.parse_args()
    
    print("🏴󠁧󠁢󠁥󠁮󠁧󠁿 GENERATORE REPORT PREMIER LEAGUE 2025/2026")
    print("=" * 50)
    
    try:
        # Carica dati
        df = load_predictions_data(args.predictions)
        
        # Genera raccomandazioni
        print("🎯 Generando raccomandazioni scommesse...")
        recommendations = get_recommended_bets(df)
        
        # Genera report HTML
        print("📄 Creando report HTML...")
        generate_html_report(df, recommendations, args.output)
        
        print(f"\n✅ Report completato con successo!")
        print(f"   📄 File: {args.output}")
        
    except FileNotFoundError:
        print(f"❌ Errore: File {args.predictions} non trovato!")
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()