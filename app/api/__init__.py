from fastapi import APIRouter
from . import users, auth, backrolls, health, bets

# Istanza di router principale dell'API
api_router = APIRouter()

# Includi i singoli router modulari
api_router.include_router(users.router, prefix="/user", tags=["User"])
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(backrolls.router, prefix="/backrolls", tags=["Backrolls"])
api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(bets.router, prefix="/bets", tags=["Bets"])
