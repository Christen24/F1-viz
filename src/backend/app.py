"""
F1 Visualization System — FastAPI Application

Entry point for the backend server.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from src.backend.config import settings
from src.backend.routers.session import router as session_router
from src.backend.routers.video import router as video_router
from src.backend.routers.track import router as track_router
from src.backend.routers.chat import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class _FastF1DtypeWarningFilter(logging.Filter):
    """Suppress noisy, non-fatal FastF1 telemetry dtype merge warnings."""

    _needle = "Failed to preserve data type for column"

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return self._needle not in msg


_fastf1_core_logger = logging.getLogger("fastf1.fastf1.core")
_fastf1_core_logger.addFilter(_FastF1DtypeWarningFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("F1 Viz Backend starting up")
    logger.info("Data directory: %s", settings.data_dir)
    logger.info("FastF1 cache: %s", settings.fastf1_cache_dir)
    logger.info(
        "LLM config â€” gemini_key=%s db_url=%s",
        "set" if (settings.gemini_api_key or settings.gemini_api_key_unprefixed) else "missing",
        "set" if settings.database_url else "missing",
    )
    yield
    logger.info("F1 Viz Backend shutting down")


app = FastAPI(
    title="F1 Race Intelligence Platform",
    description="Race highlights with telemetry analysis and ML-driven insights",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip middleware for automatic compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(session_router)
app.include_router(video_router)
app.include_router(track_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.backend.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
