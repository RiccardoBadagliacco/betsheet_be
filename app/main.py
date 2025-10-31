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
