from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.models_football import Country
from sqlalchemy.exc import IntegrityError
from app.db.database_football import get_football_db

router = APIRouter()

@router.delete("/countries/{country_id}", response_model=dict)
def delete_country(country_id: str, db: Session = Depends(get_football_db)):
    """
    Elimina un Country dal database dato il suo ID.
    """
    try:
        country_uuid = UUID(country_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    country = db.query(Country).filter(Country.id == country_uuid).first()
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")

    try:
        db.delete(country)
        db.commit()
        return {"message": "Country deleted successfully"}
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Cannot delete country due to related data")