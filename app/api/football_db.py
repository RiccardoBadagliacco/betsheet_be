"""
API endpoints per gestire il database calcistico
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from app.db.database import get_db
from app.db.database_football import get_football_db
from typing import Dict, Any
import os

router = APIRouter(prefix="/football-db", tags=["Football Database"])

@router.get("/status")
async def get_database_status(
    main_db: Session = Depends(get_db),
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Restituisce lo stato dei database (principale e calcistico)
    """
    try:
        # Controlla database principale
        main_inspector = inspect(main_db.bind)
        main_tables = main_inspector.get_table_names()
        
        # Controlla database calcistico
        football_inspector = inspect(football_db.bind)
        football_tables = football_inspector.get_table_names()
        
        # Conta record nelle tabelle principali del database calcistico
        football_counts = {}
        for table in football_tables:
            result = football_db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            football_counts[table] = count
        
        # Informazioni sui file database
        main_db_path = "./bets.db"
        football_db_path = "./football_dataset.db"
        
        main_db_size = os.path.getsize(main_db_path) if os.path.exists(main_db_path) else 0
        football_db_size = os.path.getsize(football_db_path) if os.path.exists(football_db_path) else 0
        
        return {
            "status": "success",
            "main_database": {
                "path": main_db_path,
                "size_bytes": main_db_size,
                "size_mb": round(main_db_size / (1024 * 1024), 2),
                "tables": main_tables
            },
            "football_database": {
                "path": football_db_path,
                "size_bytes": football_db_size,
                "size_mb": round(football_db_size / (1024 * 1024), 2),
                "tables": football_tables,
                "record_counts": football_counts
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel controllo database: {str(e)}")

@router.get("/tables/{table_name}")
async def get_table_info(
    table_name: str,
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Restituisce informazioni dettagliate su una tabella specifica
    """
    try:
        inspector = inspect(football_db.bind)
        tables = inspector.get_table_names()
        
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Tabella {table_name} non trovata")
        
        # Ottieni schema della tabella
        columns = inspector.get_columns(table_name)
        indexes = inspector.get_indexes(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        
        # Conta record
        result = football_db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        record_count = result.scalar()
        
        return {
            "table_name": table_name,
            "record_count": record_count,
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel recupero informazioni tabella: {str(e)}")

@router.delete("/clear/{table_name}")
async def clear_table(
    table_name: str,
    football_db: Session = Depends(get_football_db)
) -> Dict[str, Any]:
    """
    Svuota una tabella specifica (ATTENZIONE: operazione irreversibile)
    """
    try:
        inspector = inspect(football_db.bind)
        tables = inspector.get_table_names()
        
        if table_name not in tables:
            raise HTTPException(status_code=404, detail=f"Tabella {table_name} non trovata")
        
        # Conta record prima della cancellazione
        result = football_db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        records_before = result.scalar()
        
        # Svuota la tabella
        football_db.execute(text(f"DELETE FROM {table_name}"))
        football_db.commit()
        
        return {
            "status": "success",
            "table_name": table_name,
            "records_deleted": records_before,
            "message": f"Tabella {table_name} svuotata con successo"
        }
    except HTTPException:
        raise
    except Exception as e:
        football_db.rollback()
        raise HTTPException(status_code=500, detail=f"Errore nel svuotare la tabella: {str(e)}")