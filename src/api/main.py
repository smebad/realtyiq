import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import assistant, listings, predict, search
from src.db.database import create_all_tables
from src.db.crud import count_listings
from src.db.database import SessionLocal
from src.api.schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App definition
app = FastAPI(
    title       = "RealtyIQ API",
    description = "AI-powered real estate intelligence platform",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — allows Streamlit UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Routers
app.include_router(listings.router)
app.include_router(predict.router)
app.include_router(search.router)
app.include_router(assistant.router)


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting RealtyIQ API ...")
    create_all_tables()
    logger.info("Database tables verified.")


# Health check - used by Docker, deployment platforms, and monitoring tools
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    # Check DB
    db_ok    = False
    total    = 0
    try:
        db       = SessionLocal()
        total    = count_listings(db)
        db_ok    = True
        db.close()
    except Exception:
        pass

    # Check model
    model_ok = False
    try:
        from src.ml.predict import get_model_metadata
        get_model_metadata()
        model_ok = True
    except Exception:
        pass

    return HealthResponse(
        status         = "ok" if (db_ok and model_ok) else "degraded",
        model_loaded   = model_ok,
        db_connected   = db_ok,
        total_listings = total,
    )

# Root endpoint - simple message and links to docs/health
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "RealtyIQ API is running",
        "docs":    "/docs",
        "health":  "/health",
    }