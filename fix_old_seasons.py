#!/usr/bin/env python3
"""
Script per correggere le stagioni vecchie nel database
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

def analyze_seasons():
    """Analizza le stagioni nel database"""
    
    db = FootballSessionLocal()
    
    try:
        current_year = datetime.now().year
        logger.info(f"Analizzando stagioni - Anno corrente: {current_year}")
        
        seasons = db.query(Season).join(League).all()
        logger.info(f"Trovate {len(seasons)} stagioni totali")
        
        old_active_seasons = []
        
        for season in seasons:
            # Determina l'anno di inizio della stagione
            season_start_year = int(season.code[:2])
            if season_start_year > 50:  # 1900+
                season_start_year += 1900
            else:  # 2000+
                season_start_year += 2000
            
            # Conta le partite
            match_count = db.query(Match).filter(Match.season_id == season.id).count()
            
            # Se la stagione non ha end_date, deve rimanere attiva (is_completed = False)
            # Se ha end_date ma is_completed è False, è un errore da correggere
            has_end_date = season.end_date is not None
            should_be_completed = has_end_date
            
            if should_be_completed and not season.is_completed:
                old_active_seasons.append({
                    "id": str(season.id),
                    "league_code": season.league.code,
                    "league_name": season.league.name,
                    "season_name": season.name,
                    "season_code": season.code,
                    "season_start_year": season_start_year,
                    "match_count": match_count,
                    "years_old": current_year - season_start_year
                })
        
        logger.info(f"🔍 RISULTATI ANALISI:")
        logger.info(f"   Stagioni totali: {len(seasons)}")
        logger.info(f"   Stagioni vecchie ancora attive: {len(old_active_seasons)}")
        
        if old_active_seasons:
            logger.info(f"\n📋 STAGIONI DA CORREGGERE:")
            for season in old_active_seasons[:10]:  # Mostra prime 10
                logger.info(f"   {season['league_code']} {season['season_name']} - {season['match_count']} partite - {season['years_old']} anni fa")
            
            if len(old_active_seasons) > 10:
                logger.info(f"   ... e altre {len(old_active_seasons) - 10} stagioni")
        
        return old_active_seasons
        
    finally:
        db.close()

def fix_old_seasons(dry_run=True):
    """Corregge le stagioni vecchie"""
    
    db = FootballSessionLocal()
    
    try:
        current_year = datetime.now().year
        seasons_to_fix = []
        
        seasons = db.query(Season).join(League).filter(
            Season.is_completed == False
        ).all()
        
        for season in seasons:
            # Se la stagione ha end_date ma non è marcata come completata, deve essere corretta
            if season.end_date is not None and not season.is_completed:
                seasons_to_fix.append(season)
        
        logger.info(f"🔧 {'DRY RUN - ' if dry_run else ''}CORREZIONE STAGIONI:")
        logger.info(f"   Stagioni con end_date da correggere: {len(seasons_to_fix)}")
        
        if not dry_run and seasons_to_fix:
            updated_count = 0
            
            for season in seasons_to_fix:
                season.is_completed = True
                season.updated_at = datetime.utcnow()
                updated_count += 1
                
                logger.info(f"   ✅ {season.league.code} {season.name} → completata (aveva end_date)")
            
            db.commit()
            logger.info(f"✅ Aggiornate {updated_count} stagioni nel database")
        
        elif dry_run and seasons_to_fix:
            logger.info(f"   DRY RUN - Le seguenti stagioni verrebbero corrette:")
            for season in seasons_to_fix[:5]:
                logger.info(f"     {season.league.code} {season.name}")
            if len(seasons_to_fix) > 5:
                logger.info(f"     ... e altre {len(seasons_to_fix) - 5}")
        
        return len(seasons_to_fix)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Errore nella correzione: {e}")
        raise
    finally:
        db.close()

def main():
    """Funzione principale"""
    
    logger.info("🚀 CORREZIONE STAGIONI VECCHIE")
    logger.info("=" * 50)
    
    # 1. Analizza le stagioni
    old_seasons = analyze_seasons()
    
    if not old_seasons:
        logger.info("✅ Nessuna stagione da correggere!")
        return
    
    # 2. Dry run
    logger.info(f"\n🧪 DRY RUN:")
    seasons_count = fix_old_seasons(dry_run=True)
    
    # 3. Chiedi conferma
    print(f"\n❓ Vuoi correggere {seasons_count} stagioni? (y/N): ", end="")
    response = input().lower().strip()
    
    if response in ['y', 'yes', 'si', 's']:
        logger.info(f"\n🔧 APPLICANDO CORREZIONI...")
        fix_old_seasons(dry_run=False)
        logger.info(f"✅ Correzioni completate!")
    else:
        logger.info(f"❌ Operazione annullata")

if __name__ == "__main__":
    main()