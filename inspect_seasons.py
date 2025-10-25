#!/usr/bin/env python3
"""
Script per ispezionare le stagioni nel database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.db.database_football import FootballSessionLocal
from app.db.models_football import Season, League, Match
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_database():
    """Ispeziona il contenuto del database"""
    
    db = FootballSessionLocal()
    
    try:
        # Controlla tutte le stagioni
        seasons = db.query(Season).join(League).order_by(Season.code.desc()).all()
        
        logger.info(f"📊 REPORT STAGIONI DATABASE")
        logger.info(f"=" * 60)
        logger.info(f"Totale stagioni: {len(seasons)}")
        logger.info(f"Anno corrente: {datetime.now().year}")
        
        # Raggruppa per stato
        completed_seasons = [s for s in seasons if s.is_completed]
        active_seasons = [s for s in seasons if not s.is_completed]
        
        logger.info(f"\n📈 STATISTICHE:")
        logger.info(f"   Stagioni completate: {len(completed_seasons)}")
        logger.info(f"   Stagioni attive: {len(active_seasons)}")
        
        # Mostra le stagioni più recenti
        logger.info(f"\n🆕 STAGIONI PIÙ RECENTI:")
        for season in seasons[:10]:
            match_count = db.query(Match).filter(Match.season_id == season.id).count()
            status = "✅ Completata" if season.is_completed else "🟡 Attiva"
            logger.info(f"   {season.league.code} {season.name} ({season.code}) - {match_count} partite - {status}")
        
        # Mostra le stagioni attive
        if active_seasons:
            logger.info(f"\n🟡 STAGIONI ANCORA ATTIVE:")
            for season in active_seasons:
                match_count = db.query(Match).filter(Match.season_id == season.id).count()
                
                # Calcola l'anno
                season_start_year = int(season.code[:2])
                if season_start_year > 50:
                    season_start_year += 1900
                else:
                    season_start_year += 2000
                
                years_old = datetime.now().year - season_start_year
                logger.info(f"   {season.league.code} {season.name} - {match_count} partite - {years_old} anni fa")
        
        # Cerca stagioni specifiche che hai menzionato
        target_codes = ['2122', '1819']  # 2021/2022, 2018/2019
        
        logger.info(f"\n🔍 CONTROLLO STAGIONI SPECIFICHE:")
        for code in target_codes:
            specific_seasons = db.query(Season).filter(Season.code == code).all()
            
            if specific_seasons:
                logger.info(f"   Stagioni con codice {code}:")
                for season in specific_seasons:
                    match_count = db.query(Match).filter(Match.season_id == season.id).count()
                    status = "✅ Completata" if season.is_completed else "🟡 Attiva"
                    logger.info(f"     {season.league.code} {season.name} - {match_count} partite - {status}")
            else:
                logger.info(f"   ❌ Nessuna stagione trovata con codice {code}")
        
        # Controlla se ci sono problemi con gli ID che hai menzionato
        target_ids = ['3122d5a287f8456e81e7fb24d89de94e', '310881c64e654ecab9de5b1d285f1f4e']
        
        logger.info(f"\n🔍 CONTROLLO ID SPECIFICI:")
        for season_id in target_ids:
            season = db.query(Season).filter(Season.id == season_id).first()
            
            if season:
                match_count = db.query(Match).filter(Match.season_id == season.id).count()
                status = "✅ Completata" if season.is_completed else "🟡 Attiva"
                logger.info(f"   ID {season_id}:")
                logger.info(f"     {season.league.code} {season.name} ({season.code}) - {match_count} partite - {status}")
            else:
                logger.info(f"   ❌ Nessuna stagione trovata con ID {season_id}")
        
    finally:
        db.close()

if __name__ == "__main__":
    inspect_database()