"""
Servizi per la gestione dei dati calcistici strutturati
"""

from sqlalchemy.orm import Session
from app.db.models_football import Country, League, Season, Team, Match
from app.db.database_football import get_football_db
from app.constants.leagues import get_all_leagues
from datetime import datetime, date
import pandas as pd
import logging
from typing import Optional, List, Dict, Any
import re
from sqlalchemy import and_, or_

logger = logging.getLogger(__name__)

class FootballDataService:
    """Servizio per gestire i dati calcistici nel database"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_country(self, country_name: str, country_code: str = None) -> Country:
        """Ottieni o crea un paese"""
        
        # Genera codice paese se non fornito
        if not country_code:
            country_code = self._generate_country_code(country_name)
        
        # Cerca paese esistente
        country = self.db.query(Country).filter(
            or_(Country.name == country_name, Country.code == country_code)
        ).first()
        
        if not country:
            country = Country(
                name=country_name,
                code=country_code,
                flag_url=self._get_flag_url(country_code)
            )
            self.db.add(country)
            self.db.commit()
            logger.info(f"✅ Created country: {country_name} ({country_code})")
        
        return country
    
    def get_or_create_league(self, league_code: str) -> League:
        """Ottieni o crea una lega"""
        
        # Cerca lega esistente
        league = self.db.query(League).filter(League.code == league_code).first()
        
        if not league:
            # Ottieni info dalla configurazione
            league_info = get_all_leagues('all').get(league_code, {})
            league_name = league_info.get("name", f"Unknown League {league_code}")
            country_name = league_info.get("country", "Unknown")
            
            # Crea o ottieni paese
            country = self.get_or_create_country(country_name)
            
            # Determina tier dalla configurazione
            tier = self._determine_league_tier(league_code, league_name)
            
            league = League(
                code=league_code,
                name=league_name,
                country_id=country.id,
                tier=tier,
                logo_url=self._get_league_logo_url(league_code)
            )
            self.db.add(league)
            self.db.commit()
            logger.info(f"✅ Created league: {league_name} ({league_code})")
        
        return league
    
    def get_or_create_season(self, league_code: str, season_code: str, csv_file_path: str = None) -> Season:
        """Ottieni o crea una stagione"""
        
        league = self.get_or_create_league(league_code)
        
        # Cerca stagione esistente
        season = self.db.query(Season).filter(
            and_(Season.league_id == league.id, Season.code == season_code)
        ).first()
        
        if not season:
            season_name = self._format_season_name(season_code)
            
            season = Season(
                league_id=league.id,
                name=season_name,
                code=season_code,
                csv_file_path=csv_file_path
            )
            self.db.add(season)
            self.db.commit()
            logger.info(f"✅ Created season: {league.name} {season_name}")
        
        return season
    
    def get_or_create_team(self, team_name: str, country_name: str = None) -> Team:
        """Ottieni o crea una squadra"""
        
        normalized_name = self._normalize_team_name(team_name)
        
        # Cerca squadra esistente per nome normalizzato
        team = self.db.query(Team).filter(Team.normalized_name == normalized_name).first()
        
        if not team:
            country = None
            if country_name:
                country = self.get_or_create_country(country_name)
            
            team = Team(
                name=team_name,
                normalized_name=normalized_name,
                country_id=country.id if country else None,
                logo_url=self._get_team_logo_url(team_name)
            )
            self.db.add(team)
            self.db.commit()
            logger.info(f"✅ Created team: {team_name}")
        
        return team
    
    def process_csv_to_database(self, csv_file_path: str, league_code: str, season_code: str) -> Dict[str, Any]:
        """Processa un file CSV e popola il database"""
        
        try:
            # Leggi CSV
            df = pd.read_csv(csv_file_path)
            
            # Ottieni o crea stagione
            season = self.get_or_create_season(league_code, season_code, csv_file_path)
            country_name = season.league.country.name
            
            matches_created = 0
            matches_updated = 0
            errors = []
            
            for index, row in df.iterrows():
                try:
                    # Estrai dati partita
                    match_data = self._extract_match_data(row, index, country_name)
                    
                    if match_data:
                        # Crea o aggiorna partita
                        match = self._create_or_update_match(season, match_data)
                        if match:
                            matches_created += 1
                    
                except Exception as e:
                        error_msg = f"Row {index}: {str(e)}"
                        errors.append(error_msg)
                        logger.warning(f"⚠️  {error_msg}")
            
            # Aggiorna statistiche stagione
            season.processed_matches = matches_created
            season.total_matches = len(df)
            self.db.flush()
            self.db.refresh(season)
            # Calcola date inizio/fine stagione
            self._update_season_dates(season)
            
            self.db.commit()
            
            result = {
                "success": True,
                "season_id": str(season.id),
                "league": f"{season.league.name} ({league_code})",
                "season": season.name,
                "matches_processed": matches_created,
                "total_rows": len(df),
                "errors_count": len(errors),
                "errors": errors[:10] if errors else []  # Prime 10 errori
            }
            
            logger.info(f"✅ Processed {csv_file_path}: {matches_created}/{len(df)} matches")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process {csv_file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "league": league_code,
                "season": season_code
            }
    
    def _extract_match_data(self, row: pd.Series, row_index: int, country_name: str) -> Optional[Dict]:
        """Estrae i dati di una partita da una riga CSV"""
        
        try:
            # Controlla campi obbligatori
            if pd.isna(row.get('Date')) or pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
                return None
            
            # Parsa data
            match_date = pd.to_datetime(row['Date'], dayfirst=True).date()
            
            # Ottieni squadre
            home_team = self.get_or_create_team(str(row['HomeTeam']).strip(), country_name)
            away_team = self.get_or_create_team(str(row['AwayTeam']).strip(), country_name)
            
            # Estrai statistiche (con gestione valori mancanti)
            match_data = {
                'match_date': match_date,
                'match_time': str(row.get('Time', '')).strip() if not pd.isna(row.get('Time')) else None,
                'home_team_id': home_team.id,
                'away_team_id': away_team.id,
                'home_goals_ft': self._safe_int(row.get('FTHG')),
                'away_goals_ft': self._safe_int(row.get('FTAG')),
                'home_goals_ht': self._safe_int(row.get('HTHG')),
                'away_goals_ht': self._safe_int(row.get('HTAG')),
                'home_shots': self._safe_int(row.get('HS')),
                'away_shots': self._safe_int(row.get('AS')),
                'home_shots_target': self._safe_int(row.get('HST')),
                'away_shots_target': self._safe_int(row.get('AST')),
                'avg_home_odds': self._safe_float(row.get('AvgH')),
                'avg_draw_odds': self._safe_float(row.get('AvgD')),
                'avg_away_odds': self._safe_float(row.get('AvgA')),
                'avg_over_25_odds': self._safe_float(row.get('Avg>2.5')),
                'avg_under_25_odds': self._safe_float(row.get('Avg<2.5')),
                'csv_row_number': row_index
            }
            
            return match_data
            
        except Exception as e:
            logger.warning(f"⚠️  Error extracting match data from row {row_index}: {str(e)}")
            return None
    
    def _create_or_update_match(self, season: Season, match_data: Dict) -> Optional[Match]:
        """Crea o aggiorna una partita"""
        
        try:
            # Cerca partita esistente
            existing_match = self.db.query(Match).filter(
                and_(
                    Match.season_id == season.id,
                    Match.match_date == match_data['match_date'],
                    Match.home_team_id == match_data['home_team_id'],
                    Match.away_team_id == match_data['away_team_id']
                )
            ).first()
            
            if existing_match:
                # Aggiorna partita esistente
                for key, value in match_data.items():
                    if key != 'match_date':  # Non modificare la data
                        setattr(existing_match, key, value)
                return existing_match
            else:
                # Crea nuova partita
                match = Match(season_id=season.id, **match_data)
                self.db.add(match)
                return match
                
        except Exception as e:
            logger.error(f"❌ Error creating/updating match: {str(e)}")
            return None
    
    def _update_season_dates(self, season: Season):
        matches = (
            self.db.query(Match)
            .filter(Match.season_id == season.id, Match.match_date.isnot(None))
            .all()
        )

        if not matches:
            print(f"No matches found for season {season.code}")
            return

        dates = [m.match_date for m in matches]

        season.start_date = min(dates)
        latest_date = max(dates)
        today = date.today()
        days_since_last = (today - latest_date).days

        season.is_completed = self._is_season_completed(season, latest_date, days_since_last)
        print(f"Season {season.code} completed: {season.is_completed}")

        if season.is_completed:
            season.end_date = latest_date

        self.db.add(season)
    
    def _is_season_completed(self, season: Season, latest_date: date, days_since_last: int) -> bool:

        # ✅ Se c'è già una end_date, la stagione è completata
        if season.end_date is not None:
            return True

        start_code = season.code.split("/")[0]
        if len(start_code) == 2:
            season_year = 2000 + int(start_code)
        else:
            season_year = int(start_code)

        # Stagione più vecchia di 2 anni -> completata
        if season_year < date.today().year - 2:
            return True

        today = date.today()
        current_year = today.year
        current_month = today.month

        # calcolo stagione calcistica corrente
        if current_month >= 8:
            current_football_year = current_year
        else:
            current_football_year = current_year - 1

        # se la stagione è nel futuro → non completata
        if season_year > current_football_year:
            return False

        # se la stagione è più vecchia di una prima dell'attuale → completata
        if season_year < current_football_year - 1:
            return True

        # stagione corrente (in corso)
        if season_year == current_football_year:
            return False

        # nessuna partita da 120 giorni → probabilmente finita
        if days_since_last > 120:
            return True

        # siamo nel periodo estivo e l'ultima partita è a giugno o prima → finita
        if 6 <= current_month <= 8 and latest_date.month <= 6:
            return True

        return False


    # Metodi di utilità
    def _generate_country_code(self, country_name: str) -> str:
        """Genera codice paese a 3 lettere"""
        country_codes = {
            "Italy": "ITA", "England": "ENG", "Germany": "GER", "Spain": "ESP",
            "France": "FRA", "Netherlands": "NED", "Belgium": "BEL",
            "Portugal": "POR", "Scotland": "SCO", "Turkey": "TUR","Brazil": "BRA", "Argentina": "ARG",
            "China": "CHN", "Denmark": "DNK", "Ireland": "IRL"
        }
        return country_codes.get(country_name, country_name[:3].upper())
    
    def _get_flag_url(self, country_code: str) -> str:
        """Ottieni URL bandiera paese"""
        return f"https://flagcdn.com/w40/{country_code.lower()}.png"
    
    def _determine_league_tier(self, league_code: str, league_name: str) -> int:
        """Determina il tier della lega (1=prima divisione, 2=seconda, etc.)"""
        first_tier = ["E0", "I1", "D1", "SP1", "F1", "N1", "P1", "SC0", "T1", "B1", "BRA", "ARG", "CHN", "DNK", "IRL"]
        if league_code in first_tier:
            return 1
        elif "2" in league_code or "Second" in league_name or "Championship" in league_name:
            return 2
        else:
            return 1  # Default
    
    def _get_league_logo_url(self, league_code: str) -> str:
        """Ottieni URL logo lega"""
        # Placeholder - potresti implementare una logica per ottenere loghi reali
        return f"https://example.com/logos/leagues/{league_code.lower()}.png"
    
    def _format_season_name(self, season_code: str) -> str:
        """Formatta nome stagione (es. '2324' -> '2023/2024')"""
        if len(season_code) == 4:
            year1 = int(season_code[:2])
            year2 = int(season_code[2:])
            
            if year1 >= 50:
                full_year1 = 1900 + year1
            else:
                full_year1 = 2000 + year1
                
            full_year2 = full_year1 + 1
            return f"{full_year1}/{full_year2}"
        
        return season_code
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalizza nome squadra per matching"""
        # Rimuovi spazi extra, converti minuscolo, rimuovi caratteri speciali
        normalized = re.sub(r'[^\w\s]', '', team_name.lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _get_team_logo_url(self, team_name: str) -> str:
        """Ottieni URL logo squadra"""
        # Placeholder - potresti implementare una logica per ottenere loghi reali
        normalized = self._normalize_team_name(team_name).replace(' ', '_')
        return f"https://example.com/logos/teams/{normalized}.png"
    
    def _safe_int(self, value) -> Optional[int]:
        """Conversione sicura a int"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Conversione sicura a float"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None