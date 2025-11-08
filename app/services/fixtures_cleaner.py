
from __future__ import annotations
import logging
import pandas as pd
from typing import Optional, Dict
from app.constants.leagues import get_league_name_from_code

logger = logging.getLogger(__name__)

# -------- Normalization helpers --------
def normalize_team_name(team_name: str) -> str:
    if not team_name or pd.isna(team_name):
        return ""
    name = str(team_name).strip().lower()
    replacements = {
        'f.c.': '', 'fc': '', 'a.c.': '', 'ac': '', 's.c.': '', 'sc': '',
        'c.f.': '', 'cf': '', 'united': 'utd', 'city': '', 'town': '',
        'rovers': '', 'county': '', 'albion': '', '.': '', '-': ' ', '_': ' ',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = ' '.join(name.split())
    return name

# -------- CSV cleaning --------
COL_MAP: Dict[str, str] = {
    'Date': 'match_date',
    'Time': 'match_time', 
    'Div': 'league_code',
    'ï»¿Div': 'league_code',  # BOM
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
    'Avg<2.5': 'avg_under_25_odds',
    'Home': 'home_team_name',
    'Away': 'away_team_name',
}

def clean_fixtures_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Cleaning CSV base (rows=%d, cols=%d)", len(df), len(df.columns))
    cleaned = df.copy()
    for src, dst in COL_MAP.items():
        if src in cleaned.columns:
            cleaned = cleaned.rename(columns={src: dst})
    if 'league_code' in cleaned.columns:
        unique = cleaned['league_code'].dropna().unique().tolist()
        logger.info("Leghe nel CSV principale: %s", unique)
    # drop invalid dates
    if 'match_date' in cleaned.columns:
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=['match_date'])
        logger.info("Drop righe senza data: %d → %d", before, len(cleaned))
    # drop rows without teams
    if {'home_team_name','away_team_name'}.issubset(cleaned.columns):
        before = len(cleaned)
        cleaned = cleaned.dropna(subset=['home_team_name','away_team_name'])
        logger.info("Drop righe senza squadre: %d → %d", before, len(cleaned))
    # add league_name
    if 'league_code' in cleaned.columns:
        cleaned['league_name'] = cleaned['league_code'].apply(get_league_name_from_code)
    # keep fixtures only (no FT result)
    if {'home_goals_ft','away_goals_ft'}.issubset(cleaned.columns):
        mask = (
            cleaned['home_goals_ft'].isna() | cleaned['away_goals_ft'].isna() |
            (cleaned['home_goals_ft'] == '') | (cleaned['away_goals_ft'] == '')
        )
        cleaned = cleaned[mask]
        logger.info("Tenute %d righe senza risultato (fixtures)", len(cleaned))
    cleaned['csv_row_number'] = range(1, len(cleaned) + 1)
    logger.info("CSV pulito: %d righe", len(cleaned))
    return cleaned

# -------- Second CSV processing (other leagues) --------
def map_league_code_from_db(db, country_name: str, league_name: str) -> Optional[str]:
    from app.db.models_football import Country
    country = db.query(Country).filter(Country.name.ilike(country_name)).first()
    if not country:
        logger.warning("Paese non trovato: %s", country_name)
        return None
    league = next((l for l in country.leagues if l.name.lower() == league_name.lower()), None)
    if not league:
        logger.warning("Lega non trovata: %s (%s)", league_name, country_name)
        return None
    return country.code

def process_additional_csv_with_db(df_other: pd.DataFrame, db) -> pd.DataFrame:
    logger.info("Pulizia CSV aggiuntivo (rows=%d)", len(df_other))
    cleaned = df_other.copy()
    def map_div(row):
        country = row.get("Country") or row.get("ï»¿Country")
        league = row.get("League")
        return map_league_code_from_db(db, country, league) if (country and league) else None
    cleaned["Div"] = cleaned.apply(map_div, axis=1)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["Div"])
    logger.info("Drop righe senza Div valido: %d → %d", before, len(cleaned))
    cleaned = clean_fixtures_data(cleaned)
    return cleaned


