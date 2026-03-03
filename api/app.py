"""
FastAPI application factory — assembles middleware, routes, and lifecycle.
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from swaadstack.config import api_config
from swaadstack.inference.engine import InferenceEngine
from swaadstack.api.middleware import TimingMiddleware
from swaadstack.api.routes import recommend, menu, health
from swaadstack.utils.logging import logger

# Module-level engine (accessed by route modules)
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — load model on startup."""
    global engine
    logger.info("api_starting", host=api_config.host, port=api_config.port)

    engine = InferenceEngine()
    try:
        engine.load()
        logger.info("engine_ready", num_items=len(engine.menu_data))
    except Exception as e:
        logger.error("engine_load_failed", error=str(e))

    yield

    logger.info("api_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=api_config.title,
        description=api_config.description,
        version=api_config.version,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router)
    app.include_router(recommend.router)
    app.include_router(menu.router)

    return app


# Default app instance
app = create_app()
