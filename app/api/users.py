from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Any
from app.schemas.user import UserCreate, UserResponse, User as UserSchema, ErrorResponse
from app.db.database import get_db
from app.db.models import User
from app.core.security import get_password_hash, is_password_strong
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registra nuovo utente",
    description="Registra un nuovo utente nel database",
    responses={
        201: {"description": "Utente creato"},
        400: {"description": "Dati non validi", "model": ErrorResponse}
    },
    tags=["Users"]
)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)) -> Any:
    # Normalize inputs
    username = user_in.username.strip().lower()
    email = user_in.email.strip().lower()

    # Pre-check
    existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if existing:
        if existing.username == username:
            raise HTTPException(status_code=400, detail="Username già in uso")
        if existing.email == email:
            raise HTTPException(status_code=400, detail="Email già in uso")

    ok, msg = is_password_strong(user_in.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    hashed = get_password_hash(user_in.password)
    now = datetime.utcnow()
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed,
        created_at=now,
        updated_at=now,
        is_active=True,
    )
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
    except IntegrityError as e:
        db.rollback()
        logger.exception("IntegrityError while creating user")
        raise HTTPException(status_code=409, detail="Username o email già in uso")

    user_out = UserSchema(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at,
        updated_at=db_user.updated_at,
    )

    return UserResponse(success=True, message="Utente registrato", data=user_out)


@router.get("/{user_id}", response_model=UserResponse, tags=["Users"])
def get_user(user_id: int, db: Session = Depends(get_db)) -> Any:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")

    user_out = UserSchema(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
    return UserResponse(success=True, message="Utente recuperato", data=user_out)
