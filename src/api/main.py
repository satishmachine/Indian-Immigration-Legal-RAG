"""
api.main
========
FastAPI Root Application Entrypoint for Statutory Legal AI Backend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.endpoints.health import router as health_router
from api.v1.router import api_v1_router
from core.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifespan context manager."""
    logger.info("Initializing Statutory Legal AI FastAPI Server...")
    yield
    logger.info("Shutting down Statutory Legal AI FastAPI Server...")


def create_app() -> FastAPI:
    """FastAPI Application Factory."""
    settings = get_settings()

    app = FastAPI(
        title="Statutory Legal AI Platform API",
        description="RESTful API for Indian Statutory Legal RAG Search, Ingestion, and Citation Engine.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS
    origins = [str(origin) for origin in settings.api.cors_origins] if hasattr(settings, "api") else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    app.include_router(health_router)
    app.include_router(api_v1_router, prefix="/api")

    # Mount static files directory
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = get_settings().api if hasattr(get_settings(), "api") else None
    host = getattr(cfg, "host", "0.0.0.0")
    port = getattr(cfg, "port", 8000)
    uvicorn.run("api.main:app", host=host, port=port, reload=True)
