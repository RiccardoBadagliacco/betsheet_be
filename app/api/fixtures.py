"""
API semplice per scaricare il palinsesto e salvarlo nel database
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import delete
import pandas as pd
import requests
from datetime import datetime, date
from app.db.database_football import get_football_db, create_football_tables
from app.db.models_football import Fixture, Team, Season, League
import logging
import json
import os
from app.api.recommendations import generate_recommendations
logger = logging.getLogger(__name__)
router = APIRouter()

# URL fisso del CSV fixtures
FIXTURES_CSV_URL = "https://www.football-data.co.uk/fixtures.csv"

def get_league_name_from_code(league_code: str) -> str:
    """
    Converte il codice lega nel nome completo
    
    Args:
        league_code: Codice della lega (es. "E0", "I1")
        
    Returns:
        Nome completo della lega
    """
    league_names = {
        'E0': 'Premier League',
        'E1': 'Championship', 
        'E2': 'League One',
        'E3': 'League Two',
        'SC0': 'Scottish Premiership',
        'SC1': 'Scottish Championship',
        'D1': 'Bundesliga',
        'D2': '2. Bundesliga',
        'I1': 'Serie A',
        'I2': 'Serie B',
        'SP1': 'La Liga',
        'SP2': 'Segunda División',
        'F1': 'Ligue 1',
        'F2': 'Ligue 2',
        'N1': 'Eredivisie',
        'B1': 'Jupiler Pro League',
        'P1': 'Primeira Liga',
        'T1': 'Süper Lig',
        'G1': 'Super League Greece'
    }
    return league_names.get(league_code, f'League {league_code}')


async def generate_all_predictions_and_save(save: bool = True):
    """
    ✅ Usa la nuova logica OverModelV1 con patch7 e soglie
    """
    try:
        db = next(get_football_db())
        try:
            logger.info("🔄 Generazione previsioni Over...")

            # ✅ chiamiamo direttamente la nuova API
            result = await generate_recommendations(save=True)

            result["generated_at"] = datetime.utcnow().isoformat()
            result["generated_by"] = "cron_fixtures_over"

            os.makedirs("data", exist_ok=True)
            json_file = "data/all_predictions.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"✅ Predizioni Over salvate in {json_file}")

            return {
                "success": True,
                "fixtures": result.get("fixtures_count"),
                "json_file": json_file
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"❌ Errore generazione predizioni: {e}")
        return {"success": False, "error": str(e)}

def get_country_from_league_code(league_code: str) -> str:
    """
    Mappa il codice lega al paese corrispondente
    """
    league_country_mapping = {
        # Inghilterra
        'E0': 'England',
        'E1': 'England', 
        'E2': 'England',
        'E3': 'England',
        
        # Scozia
        'SC0': 'Scotland',
        'SC1': 'Scotland',
        
        # Germania
        'D1': 'Germany',
        'D2': 'Germany',
        
        # Italia
        'I1': 'Italy',
        'I2': 'Italy',
        
        # Spagna
        'SP1': 'Spain',
        'SP2': 'Spain',
        
        # Francia
        'F1': 'France',
        'F2': 'France',
        
        # Olanda
        'N1': 'Netherlands',
        
        # Belgio
        'B1': 'Belgium',
        
        # Portogallo
        'P1': 'Portugal',
        
        # Turchia
        'T1': 'Turkey',
        
        # Grecia
        'G1': 'Greece'
    }
    
    return league_country_mapping.get(league_code, 'Unknown')


def organize_predictions_by_hierarchy(fixtures: list) -> dict:
    """
    Organizza le fixture gerarchicamente: data → paese → lega → fixture
    
    Args:
        fixtures: Lista delle fixture con predizioni
        
    Returns:
        Dizionario organizzato gerarchicamente
    """
    organized = {}
    
    for fixture in fixtures:
        # Estrai informazioni
        match_date = fixture.get('match_date', 'Unknown')
        league_code = fixture.get('league_code', 'Unknown')
        country = get_country_from_league_code(league_code)
        league_name = get_league_name_from_code(league_code)
        
        # Crea struttura gerarchica se non esiste
        if match_date not in organized:
            organized[match_date] = {}
        
        if country not in organized[match_date]:
            organized[match_date][country] = {}
        
        if league_code not in organized[match_date][country]:
            organized[match_date][country][league_code] = {
                "league_name": league_name,
                "country": country,
                "fixtures": []
            }
        
        # Aggiungi la fixture alla struttura
        organized[match_date][country][league_code]["fixtures"].append(fixture)
    
    # Ordina le date
    sorted_organized = {}
    for date_key in sorted(organized.keys()):
        sorted_organized[date_key] = {}
        
        # Ordina i paesi alfabeticamente
        for country_key in sorted(organized[date_key].keys()):
            sorted_organized[date_key][country_key] = {}
            
            # Ordina le leghe alfabeticamente
            for league_key in sorted(organized[date_key][country_key].keys()):
                league_data = organized[date_key][country_key][league_key]
                
                # Ordina le fixture per orario se disponibile
                fixtures_sorted = sorted(
                    league_data["fixtures"], 
                    key=lambda x: (x.get('match_time') or '00:00', x.get('home_team', ''))
                )
                
                sorted_organized[date_key][country_key][league_key] = {
                    "league_name": league_data["league_name"],
                    "country": league_data["country"],
                    "fixture_count": len(fixtures_sorted),
                    "fixtures": fixtures_sorted
                }
    
    return sorted_organized


def normalize_team_name(team_name: str) -> str:
    """
    Normalizza il nome della squadra per il matching
    
    Args:
        team_name: Nome originale della squadra
        
    Returns:
        Nome normalizzato
    """
    if not team_name or pd.isna(team_name):
        return ""
    
    # Converti in stringa e rimuovi spazi extra
    name = str(team_name).strip()
    
    # Converti in minuscolo per matching case-insensitive
    name = name.lower()
    
    # Rimuovi caratteri speciali comuni
    replacements = {
        'fc': '',
        'f.c.': '',
        'ac': '',
        'a.c.': '',
        'sc': '',
        's.c.': '',
        'cf': '',
        'c.f.': '',
        'united': 'utd',
        'city': '',
        'town': '',
        'rovers': '',
        'county': '',
        'albion': '',
        '.': '',
        '-': ' ',
        '_': ' ',
    }
    
    # Applica sostituzioni
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Rimuovi spazi multipli e strip
    name = ' '.join(name.split())
    
    return name


def find_or_create_team(db: Session, team_name: str, country_id: str = None) -> Team:
    """
    Trova una squadra esistente o ne crea una nuova
    
    Args:
        db: Sessione database
        team_name: Nome della squadra
        country_id: ID del paese (opzionale)
        
    Returns:
        Oggetto Team
    """
    if not team_name or pd.isna(team_name):
        return None
    
    # Normalizza il nome per il matching
    normalized_name = normalize_team_name(team_name)
    
    # Cerca squadra esistente con nome normalizzato
    existing_team = db.query(Team).filter(
        Team.normalized_name == normalized_name
    ).first()
    
    if existing_team:
        logger.debug(f"Team trovato: {existing_team.name} (ID: {existing_team.id})")
        return existing_team
    
    # Crea nuova squadra
    new_team = Team(
        name=str(team_name).strip(),
        normalized_name=normalized_name,
        country_id=country_id
    )
    
    db.add(new_team)
    db.flush()  # Per ottenere l'ID senza commit
    
    logger.info(f"Team creato: {new_team.name} (ID: {new_team.id})")
    return new_team


def find_current_season(db: Session, league_code: str) -> Season:
    """
    Trova la stagione corrente per una determinata lega
    
    Args:
        db: Sessione database
        league_code: Codice della lega (es. "E0", "I1")
        
    Returns:
        Oggetto Season o None se non trovato
    """
    # Cerca la lega per codice
    league = db.query(League).filter(League.code == league_code).first()
    
    if not league:
        logger.warning(f"Lega non trovata per codice: {league_code}")
        return None
    
    # Cerca la stagione corrente (non completata)
    current_season = db.query(Season).filter(
        Season.league_id == league.id,
        Season.is_completed == False
    ).order_by(Season.start_date.desc()).first()
    
    if current_season:
        logger.debug(f"Stagione corrente trovata: {current_season.code} per lega {league_code}")
        return current_season
    
    logger.warning(f"Nessuna stagione corrente trovata per lega: {league_code}")
    return None


def clean_fixtures_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce i dati del CSV nel formato del progetto
    
    Args:
        df: DataFrame grezzo dal CSV
        
    Returns:
        DataFrame pulito nel formato del progetto
    """
    logger.info(f"Pulizia dati: {len(df)} righe originali")
    logger.info(f"Colonne disponibili: {list(df.columns)}")
    
    # Crea copia per evitare modifiche originale
    cleaned_df = df.copy()
    
    # Mappa le colonne al nostro formato standard
    column_mapping = {
        'Date': 'match_date',
        'Time': 'match_time', 
        'Div': 'league_code',
        'ï»¿Div': 'league_code',  # Gestisce BOM (Byte Order Mark)
        'HomeTeam': 'home_team_name',
        'AwayTeam': 'away_team_name',
        'FTHG': 'home_goals_ft',
        'FTAG': 'away_goals_ft',
        'HTHG': 'home_goals_ht',
        'HTAG': 'away_goals_ht',
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_target',
        'AST': 'away_shots_target',
        'AvgH': 'avg_home_odds',
        'AvgD': 'avg_draw_odds',
        'AvgA': 'avg_away_odds',
        'Avg>2.5': 'avg_over_25_odds',
        'Avg<2.5': 'avg_under_25_odds'
    }
    
    # Rinomina le colonne se esistono
    mapped_columns = []
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={old_col: new_col})
            mapped_columns.append(f"{old_col} -> {new_col}")
            logger.info(f"Mappata colonna: {old_col} -> {new_col}")
    
    logger.info(f"Colonne mappate: {mapped_columns}")
    
    logger.info(f"Colonne dopo mapping: {list(cleaned_df.columns)}")
    
    # Verifica speciale per league_code
    if 'league_code' in cleaned_df.columns:
        unique_leagues = cleaned_df['league_code'].unique()
        logger.info(f"Leghe trovate nel CSV: {unique_leagues}")
    else:
        logger.warning("⚠️ Colonna league_code non trovata dopo il mapping!")    # Rimuovi righe senza data valida
    if 'match_date' in cleaned_df.columns:
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['match_date'])
        logger.info(f"Rimosse {initial_count - len(cleaned_df)} righe senza data valida")
    
    # Rimuovi righe senza squadre
    if 'home_team_name' in cleaned_df.columns and 'away_team_name' in cleaned_df.columns:
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['home_team_name', 'away_team_name'])
        logger.info(f"Rimosse {initial_count - len(cleaned_df)} righe senza squadre")
    
    # Aggiungi il nome della lega basandosi sul codice
    if 'league_code' in cleaned_df.columns:
        cleaned_df['league_name'] = cleaned_df['league_code'].apply(get_league_name_from_code)
        logger.info(f"Aggiunti nomi leghe per {len(cleaned_df['league_code'].unique())} leghe diverse")
        logger.info(f"Leghe trovate: {cleaned_df['league_code'].unique().tolist()}")
    
    # Filtra solo partite future (senza risultato o con data futura)
    if 'home_goals_ft' in cleaned_df.columns and 'away_goals_ft' in cleaned_df.columns:
        # Mantieni solo partite senza risultato
        no_result = (
            cleaned_df['home_goals_ft'].isna() | 
            cleaned_df['away_goals_ft'].isna() |
            (cleaned_df['home_goals_ft'] == '') |
            (cleaned_df['away_goals_ft'] == '')
        )
        cleaned_df = cleaned_df[no_result]
        logger.info(f"Filtrate {len(cleaned_df)} partite senza risultato (fixtures)")
    
    # Aggiungi numero riga per tracciamento
    cleaned_df['csv_row_number'] = range(1, len(cleaned_df) + 1)
    
    logger.info(f"Dati puliti: {len(cleaned_df)} fixtures finali")
    
    return cleaned_df

@router.post("/fixtures/download")
async def download_fixtures():
    """
    Scarica il CSV fixtures dall'URL fisso, lo pulisce e lo salva nel database
    """
    try:
        logger.info(f"Scaricando fixtures da: {FIXTURES_CSV_URL}")
        
        # 1. Scarica il CSV
        response = requests.get(FIXTURES_CSV_URL, timeout=30)
        response.raise_for_status()
        
        # 2. Carica in DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"CSV scaricato: {len(df)} righe")
        
        # 3. Pulisce i dati
        cleaned_df = clean_fixtures_data(df)
        
        if len(cleaned_df) == 0:
            return {
                "success": False,
                "message": "Nessuna fixture valida trovata nel CSV",
                "total_downloaded": len(df),
                "fixtures_saved": 0
            }
        
        # 4. Salva nel database
        fixtures_saved = 0
        fixtures_skipped = 0
        
        # Ottieni sessione database
        db: Session = next(get_football_db())
        
        try:
            # Cancella fixtures esistenti (opzionale - rimuovi se vuoi mantenere storico)
            db.execute(delete(Fixture))
            db.commit()
            logger.info("Fixtures esistenti cancellate")
            
            # Inserisci nuove fixtures
            for _, row in cleaned_df.iterrows():
                # Trova o crea le squadre
                home_team = None
                away_team = None
                season = None
                
                # Recupera i dati base
                home_team_name = row.get('home_team_name')
                away_team_name = row.get('away_team_name')
                league_code = row.get('league_code')
                
                # Trova la stagione corrente per la lega prima di tutto
                season = None
                if league_code and pd.notna(league_code):
                    season = find_current_season(db, str(league_code))
                    if fixtures_saved < 5:  # Log solo per le prime 5 fixtures
                        logger.info(f"League code: '{league_code}' -> Season: {season.id if season else 'None'}")
                
                # Salta le fixtures senza stagione valida (non creare nemmeno i team)
                if not season:
                    fixtures_skipped += 1
                    if fixtures_skipped <= 10:  # Log prime 10 fixtures saltate
                        logger.info(f"Saltata fixture senza stagione valida: {home_team_name} vs {away_team_name} (League: {league_code})")
                    continue
                
                # Ora crea i team (solo per fixture valide)
                home_team = None
                away_team = None
                
                if home_team_name and pd.notna(home_team_name):
                    home_team = find_or_create_team(db, str(home_team_name))
                
                if away_team_name and pd.notna(away_team_name):
                    away_team = find_or_create_team(db, str(away_team_name))
                
                # Gestisci la data correttamente
                match_date_value = row.get('match_date')
                parsed_date = None
                if pd.notna(match_date_value):
                    if hasattr(match_date_value, 'date'):
                        # È già un datetime
                        parsed_date = match_date_value.date()
                    else:
                        # È una stringa, prova a parsarla
                        try:
                            parsed_date = pd.to_datetime(str(match_date_value)).date()
                        except:
                            logger.warning(f"Impossibile parsare data: {match_date_value}")
                            parsed_date = None
                
                # Crea la fixture con i campi relazionali popolati
                fixture = Fixture(
                    # Campi relazionali
                    season_id=season.id if season else None,
                    home_team_id=home_team.id if home_team else None,
                    away_team_id=away_team.id if away_team else None,
                    
                    # Dati temporali
                    match_date=parsed_date,
                    match_time=row.get('match_time') if pd.notna(row.get('match_time')) else None,
                    
                    # Dati grezzi per identificazione
                    league_code=str(league_code) if pd.notna(league_code) else None,
                    league_name=row.get('league_name') if pd.notna(row.get('league_name')) else None,
                    
                    # Risultati (sempre NULL per fixtures)
                    home_goals_ft=None,
                    away_goals_ft=None,
                    home_goals_ht=None,
                    away_goals_ht=None,
                    home_shots=None,
                    away_shots=None,
                    home_shots_target=None,
                    away_shots_target=None,
                    
                    # Quote
                    avg_home_odds=float(row.get('avg_home_odds')) if pd.notna(row.get('avg_home_odds')) else None,
                    avg_draw_odds=float(row.get('avg_draw_odds')) if pd.notna(row.get('avg_draw_odds')) else None,
                    avg_away_odds=float(row.get('avg_away_odds')) if pd.notna(row.get('avg_away_odds')) else None,
                    avg_over_25_odds=float(row.get('avg_over_25_odds')) if pd.notna(row.get('avg_over_25_odds')) else None,
                    avg_under_25_odds=float(row.get('avg_under_25_odds')) if pd.notna(row.get('avg_under_25_odds')) else None,
                    
                    # Metadati
                    csv_row_number=int(row.get('csv_row_number')) if pd.notna(row.get('csv_row_number')) else None,
                    downloaded_at=datetime.utcnow()
                )
                
                db.add(fixture)
                fixtures_saved += 1
                
                # Log per debug
                if fixtures_saved % 50 == 0:
                    logger.info(f"Processate {fixtures_saved} fixtures...")
                    
                # Log dettagli per le prime fixture
                if fixtures_saved <= 3:
                    home_name = home_team.name if home_team else 'Unknown'
                    away_name = away_team.name if away_team else 'Unknown'
                    logger.info(f"Fixture {fixtures_saved}: {home_name} vs {away_name}")
                    logger.info(f"  - Home Team ID: {home_team.id if home_team else 'None'}")
                    logger.info(f"  - Away Team ID: {away_team.id if away_team else 'None'}")
                    logger.info(f"  - Season ID: {season.id if season else 'None'} ({league_code})")
                    logger.info(f"  - Match Date: {fixture.match_date}")
            
            # Commit delle modifiche
            db.commit()
            logger.info(f"Salvate {fixtures_saved} fixtures nel database")
            if fixtures_skipped > 0:
                logger.info(f"Saltate {fixtures_skipped} fixtures senza stagione valida")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Errore nel salvataggio database: {e}")
            raise HTTPException(status_code=500, detail=f"Errore nel salvataggio: {str(e)}")
        
        finally:
            db.close()
        
        # 5. Calcola automaticamente le predizioni chiamando la logica esistente
        predictions_result = None
        try:
            logger.info("Iniziando calcolo predizioni per tutte le fixture...")
            predictions_result = await generate_all_predictions_and_save()
            logger.info(f"Predizioni calcolate e salvate: {predictions_result.get('total_recommendations', 0)} raccomandazioni")
        except Exception as e:
            logger.error(f"Errore nel calcolo predizioni (non bloccante): {e}")
            predictions_result = {
                "success": False,
                "error": str(e),
                "total_recommendations": 0
            }
        
        return {
            "success": True,
            "message": f"Fixtures scaricate e salvate con successo",
            "total_downloaded": len(df),
            "fixtures_saved": fixtures_saved,
            "fixtures_skipped": fixtures_skipped,
            "url": FIXTURES_CSV_URL,
            "predictions": predictions_result
        }
        
    except requests.RequestException as e:
        logger.error(f"Errore nel download CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Errore nel download: {str(e)}")
    
    except Exception as e:
        logger.error(f"Errore generico: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")


@router.get("/predictions")
async def get_all_predictions():
    """
    Ritorna tutte le predizioni organizzate gerarchicamente per data → paese → lega → fixture
    """
    try:
        json_file = "data/all_predictions.json"
        
        # Verifica se il file esiste
        if not os.path.exists(json_file):
            return {
                "success": False,
                "message": "Nessuna predizione trovata. Esegui prima il download delle fixture.",
                "predictions": None,
                "file_exists": False
            }
        
        # Leggi il file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)
        
        # Organizza i dati gerarchicamente: data → paese → lega → fixture
        organized_data = organize_predictions_by_hierarchy(predictions_data.get('fixtures', []))
        
        # Aggiungi informazioni sul file e metadata
        file_stats = os.stat(json_file)
        
        return {
            "success": True,
            "predictions_by_date": organized_data,
            "metadata": {
                "total_fixtures": predictions_data.get('total_fixtures', 0),
                "successful_predictions": predictions_data.get('successful_predictions', 0),
                "total_recommendations": predictions_data.get('total_recommendations', 0),
                "generated_at": predictions_data.get('generated_at'),
                "generated_by": predictions_data.get('generated_by'),
                "model_info": predictions_data.get('model_info', {}),
                "file_info": {
                    "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    "file_size_kb": round(file_stats.st_size / 1024, 2)
                }
            }
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel parsing del JSON: {e}")
        raise HTTPException(status_code=500, detail="File delle predizioni corrotto")
    except Exception as e:
        logger.error(f"Errore nel recupero predizioni: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")


@router.post("/predictions/regenerate")
async def regenerate_predictions():
    """
    Rigenera le predizioni per tutte le fixture esistenti nel database
    """
    try:
        logger.info("Rigenerazione predizioni richiesta...")
        
        result = await generate_all_predictions_and_save()
        
        return {
            "success": result.get("success", False),
            "message": "Predizioni rigenerate con successo" if result.get("success") else "Errore nella rigenerazione",
            "total_fixtures": result.get("total_fixtures", 0),
            "successful_predictions": result.get("successful_predictions", 0),
            "total_recommendations": result.get("total_recommendations", 0),
            "json_file": result.get("json_file"),
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error(f"Errore nella rigenerazione predizioni: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.get("/fixtures")
async def get_fixtures():
    """
    Recupera tutte le fixtures dal database
    """
    try:
        db: Session = next(get_football_db())
        
        try:
            fixtures = db.query(Fixture).order_by(Fixture.match_date, Fixture.match_time).all()
            
            fixtures_list = []
            for fixture in fixtures:
                fixtures_list.append({
                    "id": str(fixture.id),
                    "match_date": fixture.match_date.strftime("%Y-%m-%d") if fixture.match_date else None,
                    "match_time": fixture.match_time,
                    "home_team_id": str(fixture.home_team_id) if fixture.home_team_id else None,
                    "away_team_id": str(fixture.away_team_id) if fixture.away_team_id else None,
                    "home_team_name": fixture.home_team.name if fixture.home_team else None,
                    "away_team_name": fixture.away_team.name if fixture.away_team else None,
                    "season_id": str(fixture.season_id) if fixture.season_id else None,
                    "season_name": fixture.season.name if fixture.season else None,
                    "league_code": fixture.league_code,
                    "league_name": fixture.league_name,
                    "avg_home_odds": fixture.avg_home_odds,
                    "avg_draw_odds": fixture.avg_draw_odds,
                    "avg_away_odds": fixture.avg_away_odds,
                    "avg_over_25_odds": fixture.avg_over_25_odds,
                    "avg_under_25_odds": fixture.avg_under_25_odds,
                    "downloaded_at": fixture.downloaded_at.strftime("%Y-%m-%d %H:%M:%S") if fixture.downloaded_at else None
                })
            
            return {
                "success": True,
                "fixtures": fixtures_list,
                "count": len(fixtures_list)
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Errore nel recupero fixtures: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.delete("/fixtures")
async def clear_fixtures():
    """
    Cancella tutte le fixtures dal database
    """
    try:
        db: Session = next(get_football_db())
        
        try:
            result = db.execute(delete(Fixture))
            db.commit()
            
            return {
                "success": True,
                "message": f"Cancellate {result.rowcount} fixtures",
                "deleted_count": result.rowcount
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Errore nella cancellazione fixtures: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

@router.post("/fixtures/create-tables")
async def create_fixtures_tables():
    """
    Crea le tabelle del database football (inclusa fixtures)
    """
    try:
        create_football_tables()
        
        return {
            "success": True,
            "message": "Tabelle del database football create con successo"
        }
        
    except Exception as e:
        logger.error(f"Errore nella creazione tabelle: {e}")
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")