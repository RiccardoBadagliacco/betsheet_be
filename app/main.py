from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api import api_router
import uvicorn

from app.db.database import engine
from app.db.models import Base
from app.core.settings import settings
from app.core.logging import setup_logging
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from sqlalchemy.exc import IntegrityError
from app.schemas.user import ErrorResponse


# --- CORS ---
origins = [str(o).rstrip('/') for o in settings.CORS_ORIGINS]
print(origins)
app = FastAPI()

# Setup logging
setup_logging()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # origini consentite
    allow_credentials=True,      # consente cookie, token
    allow_methods=["*"],         # consente tutti i metodi HTTP (GET, POST, etc.)
    allow_headers=["*"],         # consente tutti gli headers
)

# --- Monta il router delle API ---
app.include_router(api_router)


@app.on_event("startup")
def on_startup():
    # In development it's convenient creare le tabelle se mancano.
    # In produzione preferire Alembic per le migrazioni.
    Base.metadata.create_all(bind=engine)
    
    # Auto-load saved models at startup
    try:
        from app.storage.model_storage import model_storage
        from app.api.model_management import loaded_models
        
        saved_models = model_storage.list_saved_models()
        loaded_count = 0
        
        for league_code in saved_models.keys():
            try:
                if model_storage.is_model_fresh(league_code, max_age_hours=48):  # Load if < 48h old
                    predictor = model_storage.load_model(league_code)
                    if predictor:
                        loaded_models[league_code] = predictor
                        loaded_count += 1
            except Exception as e:
                print(f"⚠️  Failed to auto-load model {league_code}: {e}")
        
        if loaded_count > 0:
            print(f"🚀 Auto-loaded {loaded_count} fresh models at startup")
        else:
            print("📋 No fresh models found. Generate them using /api/v1/models/generate-all-models")
            
    except Exception as e:
        print(f"⚠️  Error during model auto-loading: {e}")


# Global exception handler to return ErrorResponse shape
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Handle common DB integrity error specially
    if isinstance(exc, IntegrityError):
        return JSONResponse(status_code=409, content=ErrorResponse(success=False, message="Conflict", error=str(exc)).dict())
    # Default
    return JSONResponse(status_code=500, content=ErrorResponse(success=False, message="Internal Server Error", error=str(exc)).dict())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
