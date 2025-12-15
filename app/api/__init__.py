from fastapi import APIRouter
from . import matches, users, auth, backrolls, health, bets, csv_download, leagues_seasons, fixtures, seasons,countries,engine

# Istanza di router principale dell'API
api_router = APIRouter()

# Includi i singoli router modulari
api_router.include_router(users.router, prefix="/user", tags=["User"])
api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(backrolls.router, prefix="/backrolls", tags=["Backrolls"])
api_router.include_router(health.router, prefix="", tags=["Health"])
api_router.include_router(bets.router, prefix="/bets", tags=["Bets"])
api_router.include_router(csv_download.router, prefix="/csv", tags=["CSV Download"])
api_router.include_router(leagues_seasons.router, prefix="/api/v1", tags=["Leagues & Seasons"])
api_router.include_router(fixtures.router, prefix="/api/v1", tags=["Fixtures"])
api_router.include_router(seasons.router, prefix="/api/v1", tags=["Seasons Management"])
api_router.include_router(countries.router, prefix="/api/v1", tags=["Countries"])
api_router.include_router(matches.router, prefix="/api/v1", tags=["Model Predict"])
api_router.include_router(engine.router, prefix="/engine", tags=["Profeta"])
