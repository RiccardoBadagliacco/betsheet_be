from fastapi import APIRouter, Depends
from datetime import datetime
from app.db.database import get_db
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/health", tags=["Health"])
def health(db: Session = Depends(get_db)):
    """Simple health check: returns server time and whether DB is reachable."""

    return {"status": "ok" , "timestamp": datetime.utcnow().isoformat() + "Z"}
