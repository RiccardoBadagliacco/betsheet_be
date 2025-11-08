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
BATCH_SIZE = 500

class FootballDataService:
    """Servizio per gestire i dati calcistici nel database"""
    
    def __init__(self, db: Session):
        self.db = db
    
    
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
    
    def is_season_completed(self, league_code: str, season_code: str) -> bool:
        league = self.db.query(League).filter(League.code == league_code.upper()).first()
        if not league:
            return False
        season = self.db.query(Season).filter(and_(Season.league_id==league.id, Season.code==season_code)).first()
        if not season:
            return False
        if season.end_date is not None:
            return True
        # euristica corrente
        start_code = season.code.split("/")[0]
        try:
            year = int(start_code) if len(start_code) == 4 else 2000 + int(start_code)
        except Exception:
            return False
        today = date.today()
        if year < today.year - 2:
            return True
        if season.processed_matches and season.total_matches and season.processed_matches >= season.total_matches:
            return True
        return bool(season.is_completed)
    
    def get_or_create_country(self, country_name: str, country_code: str = None) -> Country:
        if not country_code:
            country_code = self._cc(country_name)
        c = self.db.query(Country).filter(or_(Country.name==country_name, Country.code==country_code)).first()
        if not c:
            c = Country(name=country_name, code=country_code, flag_url=f"https://flagcdn.com/w40/{country_code.lower()}.png")
            self.db.add(c); self.db.commit()
        return c
    
    def get_or_create_league(self, league_code: str) -> League:
        l = self.db.query(League).filter(League.code==league_code).first()
        if not l:
            info = get_all_leagues('all').get(league_code, {})
            name = info.get('name', f"Unknown League {league_code}")
            country = self.get_or_create_country(info.get('country', 'Unknown'))
            tier = 1 if league_code in {"E0","I1","D1","SP1","F1","N1","P1","SC0","T1","B1","BRA","ARG","CHN","DNK","IRL"} else (2 if ("2" in league_code or "Championship" in name or "Second" in name) else 1)
            l = League(code=league_code, name=name, country_id=country.id, tier=tier, logo_url=f"https://example.com/logos/leagues/{league_code.lower()}.png")
            self.db.add(l); self.db.commit()
        return l
    
    def get_or_create_season(self, league_code: str, season_code: str, csv_file_path: Optional[str]=None) -> Season:
        league = self.get_or_create_league(league_code)
        s = self.db.query(Season).filter(and_(Season.league_id==league.id, Season.code==season_code)).first()
        if not s:
            name = self._format_season(season_code)
            s = Season(league_id=league.id, name=name, code=season_code, csv_file_path=csv_file_path)
            self.db.add(s); self.db.commit()
        elif csv_file_path and s.csv_file_path != csv_file_path:
            s.csv_file_path = csv_file_path; self.db.commit()
        return s
    
    def get_or_create_team(self, team_name: str, country_name: Optional[str]) -> Team:
        norm = self._norm(team_name)
        t = self.db.query(Team).filter(Team.normalized_name==norm).first()
        if not t:
            country = self.get_or_create_country(country_name) if country_name else None
            t = Team(name=team_name, normalized_name=norm, country_id=(country.id if country else None), logo_url=f"https://example.com/logos/teams/{norm.replace(' ', '_')}.png")
            self.db.add(t); self.db.commit()
        return t
    
    
    
    
    def process_csv_to_database(self, csv_file_path: str, league_code: str, season_code: str) -> Dict[str, object]:
        try:
            df = pd.read_csv(csv_file_path)
            season = self.get_or_create_season(league_code, season_code, csv_file_path)
            country_name = season.league.country.name


            rows = []
            errors: List[str] = []


            for i, row in df.iterrows():
                try:
                    rec = self._row_to_match(row, i, country_name)
                    if rec:
                        rows.append(rec)
                except Exception as e:
                    errors.append(f"Row {i}: {e}")

            created = 0
            # batch upsert (semplice: cerca esistenti + aggiorna al volo; poi bulk per nuovi)
            new_objs: List[Match] = []
            for r in rows:
                m = self._find_existing_match(season.id, r)
                if m:
                    for k, v in r.items():
                        if k != 'match_date':
                            setattr(m, k, v)
                else:
                    new_objs.append(Match(season_id=season.id, **r))
                    
                if len(new_objs) >= BATCH_SIZE:
                    self.db.bulk_save_objects(new_objs)
                    created += len(new_objs)
                    new_objs.clear()
                    
            if new_objs:
                self.db.bulk_save_objects(new_objs)
                created += len(new_objs)


            # update season stats & dates
            season.processed_matches = created
            season.total_matches = len(df)
            self._update_season_dates(season)
            self.db.commit()


            return {
                "success": True,
                "season_id": str(season.id),
                "league": f"{season.league.name} ({league_code})",
                "season": season.name,
                "matches_processed": created,
                "total_rows": len(df),
                "errors_count": len(errors),
                "errors": errors[:10],
            }
        except Exception as e:
            logger.exception("process_csv_to_database failed")
            return {"success": False, "error": str(e), "league": league_code, "season": season_code}
        
        
    def _row_to_match(self, row: pd.Series, idx: int, country_name: str):
        if pd.isna(row.get('Date')) or pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
            return None
        d = pd.to_datetime(row['Date'], dayfirst=True).date()
        home = self.get_or_create_team(str(row['HomeTeam']).strip(), country_name)
        away = self.get_or_create_team(str(row['AwayTeam']).strip(), country_name)
        def s_int(v):
            if pd.isna(v) or v == '' or v is None: return None
            try: return int(float(v))
            except: return None
        def s_float(v):
            if pd.isna(v) or v == '' or v is None: return None
            try: return float(v)
            except: return None
        return {
            'match_date': d,
            'match_time': (str(row.get('Time', '')).strip() if not pd.isna(row.get('Time')) else None),
            'home_team_id': home.id,
            'away_team_id': away.id,
            'home_goals_ft': s_int(row.get('FTHG')),
            'away_goals_ft': s_int(row.get('FTAG')),
            'home_goals_ht': s_int(row.get('HTHG')),
            'away_goals_ht': s_int(row.get('HTAG')),
            'home_shots': s_int(row.get('HS')),
            'away_shots': s_int(row.get('AS')),
            'home_shots_target': s_int(row.get('HST')),
            'away_shots_target': s_int(row.get('AST')),
            'avg_home_odds': s_float(row.get('AvgH')),
            'avg_draw_odds': s_float(row.get('AvgD')),
            'avg_away_odds': s_float(row.get('AvgA')),
            'avg_over_25_odds': s_float(row.get('Avg>2.5')),
            'avg_under_25_odds': s_float(row.get('Avg<2.5')),
            'csv_row_number': idx,
        }
    
    def _find_existing_match(self, season_id: str, r: dict) -> Optional[Match]:
        return self.db.query(Match).filter(and_(
            Match.season_id==season_id,
            Match.match_date==r['match_date'],
            Match.home_team_id==r['home_team_id'],
            Match.away_team_id==r['away_team_id']
            )).first()
        
    def _update_season_dates(self, season: Season):
        matches = self.db.query(Match).filter(Match.season_id==season.id, Match.match_date.isnot(None)).all()
        if not matches:
            return
        dates = [m.match_date for m in matches]
        season.start_date = min(dates)
        latest = max(dates)
        season.is_completed = self._is_completed_heuristic(season, latest)
        if season.is_completed:
            season.end_date = latest
        self.db.add(season)
        
    def _is_completed_heuristic(self, season: Season, latest_date: date) -> bool:
        today = date.today()
        days = (today - latest_date).days
        # euristiche semplici
        if season.end_date:
            return True
        if days > 120:
            return True
        # estate e ultima partita <= giugno
        if 6 <= today.month <= 8 and latest_date.month <= 6:
            return True
        return False
    
    # utils
    def _cc(self, country_name: str) -> str:
        mapping = {"Italy":"ITA","England":"ENG","Germany":"GER","Spain":"ESP","France":"FRA","Netherlands":"NED","Belgium":"BEL","Portugal":"POR","Scotland":"SCO","Turkey":"TUR","Brazil":"BRA","Argentina":"ARG","China":"CHN","Denmark":"DNK","Ireland":"IRL"}
        return mapping.get(country_name, country_name[:3].upper())
    def _format_season(self, code: str) -> str:
        if len(code)==4:
            y1 = 1900+int(code[:2]) if int(code[:2])>=50 else 2000+int(code[:2])
            return f"{y1}/{y1+1}"
        return code
    def _norm(self, s: str) -> str:
        s = re.sub(r'[^\w\s]', '', s.lower().strip())
        return re.sub(r'\s+', ' ', s)


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