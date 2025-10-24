#!/usr/bin/env python3
"""
Script per generare CSV unificati per ogni lega con tutte le stagioni
ordinate cronologicamente dalla più vecchia alla più nuova.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.database_football import get_football_db_session
from app.db.models_football import League, Season, Match, Team
from app.constants.leagues import LEAGUES

def get_match_data_for_csv(match):
    """
    Converte un match in formato dizionario per CSV.
    Mantiene la struttura originale dei CSV di football-data.co.uk
    """
    return {
        'Date': match.match_date.strftime('%d/%m/%Y') if match.match_date else '',
        'Time': match.match_time or '',
        'HomeTeam': match.home_team.name if match.home_team else '',
        'AwayTeam': match.away_team.name if match.away_team else '',
        'FTHG': match.home_goals_ft if match.home_goals_ft is not None else '',
        'FTAG': match.away_goals_ft if match.away_goals_ft is not None else '',
        'HTHG': match.home_goals_ht if match.home_goals_ht is not None else '',
        'HTAG': match.away_goals_ht if match.away_goals_ht is not None else '',
        'HS': match.home_shots if match.home_shots is not None else '',
        'AS': match.away_shots if match.away_shots is not None else '',
        'HST': match.home_shots_target if match.home_shots_target is not None else '',
        'AST': match.away_shots_target if match.away_shots_target is not None else '',
        'AvgH': match.avg_home_odds if match.avg_home_odds is not None else '',
        'AvgD': match.avg_draw_odds if match.avg_draw_odds is not None else '',
        'AvgA': match.avg_away_odds if match.avg_away_odds is not None else '',
        'Avg>2.5': match.avg_over_25_odds if match.avg_over_25_odds is not None else '',
        'Avg<2.5': match.avg_under_25_odds if match.avg_under_25_odds is not None else '',
        # Aggiungi metadati utili
        'Season': match.season.name if match.season else '',
        'SeasonCode': match.season.code if match.season else '',
    }

def generate_league_csv(db: Session, league_code: str, output_dir: str):
    """
    Genera un CSV unificato per una lega specifica.
    
    Args:
        db: Sessione database
        league_code: Codice della lega (es. I1, E0)
        output_dir: Directory di output
    """
    print(f"🔄 Processando lega {league_code}...")
    
    # Ottieni informazioni lega
    league_info = LEAGUES.get(league_code, {
        "name": f"League {league_code}",
        "country": "Unknown"
    })
    
    # Ottieni la lega dal database
    league = db.query(League).filter(League.code == league_code).first()
    if not league:
        print(f"❌ Lega {league_code} non trovata nel database")
        return
    
    # Ottieni tutte le stagioni della lega ordinate per data di inizio (dalla più vecchia)
    seasons = db.query(Season).filter(
        Season.league_id == league.id
    ).order_by(Season.start_date.asc().nullslast(), Season.code.asc()).all()
    
    if not seasons:
        print(f"❌ Nessuna stagione trovata per {league_code}")
        return
    
    print(f"   📊 Trovate {len(seasons)} stagioni per {league_info['name']}")
    
    # Ottieni tutte le partite di tutte le stagioni ordinate per data
    all_matches = []
    total_matches = 0
    
    for season in seasons:
        matches = db.query(Match).filter(
            Match.season_id == season.id
        ).order_by(Match.match_date.asc()).all()
        
        season_matches = len(matches)
        total_matches += season_matches
        all_matches.extend(matches)
        
        print(f"   📅 {season.name}: {season_matches} partite")
    
    if not all_matches:
        print(f"❌ Nessuna partita trovata per {league_code}")
        return
    
    # Ordina tutte le partite per data (dalla più vecchia alla più nuova)
    all_matches.sort(key=lambda x: x.match_date if x.match_date else datetime.min.date())
    
    # Crea nome file
    country_code = league_info['country'].replace(' ', '_')
    league_name_clean = league_info['name'].replace(' ', '_').replace('/', '_')
    filename = f"{country_code}_{league_code}_{league_name_clean}_ALL_SEASONS.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Definisci header CSV
    fieldnames = [
        'Date', 'Time', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'HTHG', 'HTAG',
        'HS', 'AS', 'HST', 'AST',
        'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5',
        'Season', 'SeasonCode'
    ]
    
    # Scrivi CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for match in all_matches:
            row_data = get_match_data_for_csv(match)
            writer.writerow(row_data)
    
    # Statistiche finali
    file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
    date_range = f"{all_matches[0].match_date} → {all_matches[-1].match_date}"
    
    print(f"✅ Generato: {filename}")
    print(f"   📈 {total_matches} partite totali")
    print(f"   📅 Periodo: {date_range}")
    print(f"   💾 Dimensione: {file_size:.2f} MB")
    print()

def main():
    """Funzione principale per generare tutti i CSV delle leghe."""
    
    print("🚀 Avvio generazione CSV unificati per lega")
    print("=" * 60)
    
    # Crea directory di output
    output_dir = "leagues_csv_unified"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"📁 Directory output: {output_dir}")
    print()
    
    # Ottieni sessione database
    db = next(get_football_db_session())
    
    try:
        # Ottieni tutte le leghe disponibili nel database
        available_leagues = db.query(League.code).distinct().all()
        league_codes = [league[0] for league in available_leagues]
        
        print(f"🏆 Trovate {len(league_codes)} leghe nel database:")
        for code in sorted(league_codes):
            league_info = LEAGUES.get(code, {"name": f"League {code}", "country": "Unknown"})
            print(f"   {code}: {league_info['name']} ({league_info['country']})")
        print()
        
        # Genera CSV per ogni lega
        success_count = 0
        for league_code in sorted(league_codes):
            try:
                generate_league_csv(db, league_code, output_dir)
                success_count += 1
            except Exception as e:
                print(f"❌ Errore processando {league_code}: {str(e)}")
                print()
        
        print("=" * 60)
        print(f"🎉 Completato! {success_count}/{len(league_codes)} leghe processate con successo")
        print(f"📁 File salvati in: {os.path.abspath(output_dir)}")
        
        # Mostra riepilogo finale
        csv_files = list(Path(output_dir).glob("*.csv"))
        total_size = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
        print(f"📊 {len(csv_files)} file CSV generati")
        print(f"💾 Dimensione totale: {total_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Errore generale: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main()