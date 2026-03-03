"""Pydantic request/response schemas for API validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """POST /recommend request body."""
    cart_items: List[str] = Field(..., description="List of item IDs in the cart")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    geohash: Optional[str] = Field(None, description="Location geohash")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="Day of week (0=Mon)")
    top_n: Optional[int] = Field(None, ge=1, le=20, description="Number of recommendations")
    diversity: Optional[float] = Field(None, ge=0.0, le=1.0, description="MMR lambda (0=diverse, 1=relevant)")


class RecommendationItem(BaseModel):
    item_id: str
    name: str
    category: str
    price: float
    mmr_score: float
    relevance_score: float


class CartSummary(BaseModel):
    items: List[str]
    item_count: int
    total_value: float
    categories: Optional[List[str]] = None
    diversity_score: Optional[float] = None
    missing_categories: Optional[List[str]] = None


class RecommendResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    cart_summary: Dict[str, Any]
    context: Dict[str, Any]
    latency: Dict[str, float]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    engine: Optional[Dict[str, Any]] = None
    timestamp: str
    version: str


class MenuItemResponse(BaseModel):
    item_id: str
    name: str
    category: str
    price: float
    cuisine: Optional[str] = None
    dietary: Optional[List[str]] = None
    description: Optional[str] = None
