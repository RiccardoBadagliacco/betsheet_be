
from __future__ import annotations
import logging
import pandas as pd
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
from app.db.models_football import Fixture, Team, Season, League

from app.services.fixtures_cleaner import normalize_team_name

logger = logging.getLogger(__name__)
BATCH_SIZE = 500

def find_current_season(db: Session, league_code: str) -> Optional[Season]:
    league = db.query(League).filter(League.code == league_code).first()
    if not league:
        return None
    season = db.query(Season).filter(
        Season.league_id == league.id,
        Season.is_completed == False
    ).order_by(Season.start_date.desc()).first()
    return season

def import_fixtures_dataframe(df: pd.DataFrame, db: Session) -> Dict[str, int]:
    if df is None or df.empty:
        return {"saved": 0, "skipped": 0}

    team_cache: Dict[str, Team] = {t.normalized_name: t for t in db.query(Team).all()}
    def get_team_cached(name: str) -> Team:
        norm = normalize_team_name(name)
        t = team_cache.get(norm)
        if t:
            return t
        # Crea automaticamente la squadra se non esiste
        t = Team(name=name.strip(), normalized_name=norm)
        db.add(t)
        team_cache[norm] = t
        logger.info(f"🆕 Squadra creata automaticamente: {name.strip()}")
        return t

    saved, skipped = 0, 0
    batch: List[Fixture] = []

    for _, row in df.iterrows():
        league_code = str(row.get('league_code')) if pd.notna(row.get('league_code')) else None
        if not league_code:
            skipped += 1
            continue
        season = find_current_season(db, league_code)
        if not season:
            skipped += 1
            continue
        ht = row.get('home_team_name'); at = row.get('away_team_name')
        if not (pd.notna(ht) and pd.notna(at)):
            skipped += 1; continue
        home = get_team_cached(str(ht)); away = get_team_cached(str(at))

        md = row.get('match_date') or row.get('Date')
        parsed_date = None
        if pd.notna(md):
            try:
                parsed_date = pd.to_datetime(str(md), dayfirst=True, errors='coerce').date()
            except Exception:
                parsed_date = None
        
        fx = Fixture(
            season_id=season.id,
            home_team_id=home.id if home else None,
            away_team_id=away.id if away else None,
            match_time=row.get('match_time') if pd.notna(row.get('match_time')) else None,
            league_code=league_code,
            league_name=row.get('league_name') if pd.notna(row.get('league_name')) else None,
            home_goals_ft=None, away_goals_ft=None, home_goals_ht=None, away_goals_ht=None,
            home_shots=None, away_shots=None, home_shots_target=None, away_shots_target=None,
            avg_home_odds=float(row.get('avg_home_odds')) if pd.notna(row.get('avg_home_odds')) else None,
            avg_draw_odds=float(row.get('avg_draw_odds')) if pd.notna(row.get('avg_draw_odds')) else None,
            avg_away_odds=float(row.get('avg_away_odds')) if pd.notna(row.get('avg_away_odds')) else None,
            avg_over_25_odds=float(row.get('avg_over_25_odds')) if pd.notna(row.get('avg_over_25_odds')) else None,
            avg_under_25_odds=float(row.get('avg_under_25_odds')) if pd.notna(row.get('avg_under_25_odds')) else None,
            csv_row_number=int(row.get('csv_row_number')) if pd.notna(row.get('csv_row_number')) else None,
            downloaded_at=datetime.utcnow(),
            match_date=parsed_date,

        )
        batch.append(fx); saved += 1

        if len(batch) >= BATCH_SIZE:
            db.bulk_save_objects(batch); db.commit(); batch.clear()

    if batch:
        db.bulk_save_objects(batch); db.commit(); batch.clear()

    logger.info("Fixtures importate: saved=%d, skipped=%d", saved, skipped)
    return {"saved": saved, "skipped": skipped}


def parse_match_date(value) -> Optional[str]:
    """
    Converte un valore generico in formato data standard 'YYYY-MM-DD'.
    Gestisce formati 12/05/24, 2024-05-12, ecc.
    """
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        dt = pd.to_datetime(str(value), errors='coerce', dayfirst=True)
        if pd.isna(dt):
            dt = pd.to_datetime(str(value), errors='coerce', dayfirst=False)
        if not pd.isna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None
