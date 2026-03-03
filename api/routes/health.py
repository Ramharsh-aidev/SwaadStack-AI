"""Health and root routes — / and /health endpoints."""

import time
from datetime import datetime, timezone

from fastapi import APIRouter

from swaadstack.config import api_config

router = APIRouter(tags=["health"])
_start_time = time.time()


@router.get("/")
async def root():
    """API root — returns service info."""
    return {
        "name": api_config.title,
        "version": api_config.version,
        "description": api_config.description,
        "endpoints": {
            "POST /recommend": "Get meal completion recommendations",
            "POST /predict": "Alias for /recommend",
            "GET /menu": "Browse menu items",
            "GET /menu/{item_id}": "Get specific menu item",
            "GET /health": "System health check",
        },
    }


@router.get("/health")
async def health_check():
    """System health check."""
    from swaadstack.api.app import engine

    health = {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": api_config.version,
    }
    if engine:
        health["engine"] = engine.health_check()
    return health
