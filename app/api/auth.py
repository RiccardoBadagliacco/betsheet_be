from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.schemas.auth import Token, RefreshPayload
from app.db.database import get_db
from app.db.models import User, RefreshToken
from app.core.security import verify_password, get_password_hash
from app.core.auth import get_current_active_user
from app.core.jwt import create_access_token, create_refresh_token
from datetime import timedelta, datetime
from app.core.settings import settings

router = APIRouter()


@router.post("/login", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # OAuth2PasswordRequestForm provides username and password fields which
    # enables the Swagger UI "Authorize" button for the password flow.
    # allow login with either username or email (case-insensitive)
    identifier = form_data.username.strip().lower()
    user = (
        db.query(User)
        .filter((User.username == identifier) | (User.email == identifier))
        .first()
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(minutes=int(settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    token = create_access_token({"sub": user.username}, expires_delta=access_token_expires)

    # create refresh token and store its hash
    refresh_token = create_refresh_token()
    rt_hash = get_password_hash(refresh_token)
    expires_at = datetime.utcnow() + timedelta(days=30)
    db_rt = RefreshToken(token_hash=rt_hash, user_id=user.id, expires_at=expires_at)
    db.add(db_rt)
    db.commit()

    return Token(access_token=token, refresh_token=refresh_token)


@router.get("/me", tags=["Auth"])
def me(current_user: User = Depends(get_current_active_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
    }


@router.post("/refresh", tags=["Auth"])
def refresh(payload: RefreshPayload, db: Session = Depends(get_db)):
    # find matching refresh token by verifying hash
    candidates = db.query(RefreshToken).filter(RefreshToken.revoked == False).all()
    matched = None
    for c in candidates:
        try:
            if verify_password(payload.refresh_token, c.token_hash):
                matched = c
                break
        except Exception:
            continue
    if not matched:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = db.query(User).filter(User.id == matched.user_id).first()
    access_token_expires = timedelta(minutes=int(settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    token = create_access_token({"sub": user.username}, expires_delta=access_token_expires)
    return Token(access_token=token)


@router.post("/logout", tags=["Auth"])
def logout(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    db.query(RefreshToken).filter(RefreshToken.user_id == current_user.id).update({"revoked": True})
    db.commit()
    return {"ok": True}
