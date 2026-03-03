"""Recommendation routes — /recommend and /predict endpoints."""

from fastapi import APIRouter, HTTPException

from swaadstack.api.schemas import RecommendRequest

router = APIRouter(tags=["recommendations"])


@router.post("/recommend")
async def recommend(request: RecommendRequest):
    """Generate meal completion recommendations based on current cart."""
    from swaadstack.api.app import engine

    if not engine or not engine._is_loaded:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    valid_items = [iid for iid in request.cart_items if iid in engine.menu_data]
    if not valid_items and request.cart_items:
        raise HTTPException(status_code=404, detail="No valid items found in cart")

    result = engine.recommend(
        cart_items=request.cart_items,
        user_id=request.user_id,
        geohash=request.geohash,
        hour=request.hour,
        day_of_week=request.day_of_week,
        top_n=request.top_n or 5,
        lambda_mmr=request.diversity,
    )
    return result


@router.post("/predict")
async def predict(request: RecommendRequest):
    """Alias for /recommend."""
    return await recommend(request)
