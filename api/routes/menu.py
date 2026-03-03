"""Menu routes — /menu endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["menu"])


@router.get("/menu")
async def list_menu(category: Optional[str] = Query(None)):
    """List all menu items, optionally filtered by category."""
    from swaadstack.api.app import engine

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    items = list(engine.menu_data.values())
    if category:
        items = [i for i in items if i.get("category") == category]

    return [
        {
            "item_id": i["item_id"], "name": i["name"],
            "category": i["category"], "price": i.get("price", 0),
            "cuisine": i.get("cuisine", ""), "dietary": i.get("dietary", []),
            "description": i.get("description", ""),
        }
        for i in items
    ]


@router.get("/menu/{item_id}")
async def get_menu_item(item_id: str):
    """Get a specific menu item by ID."""
    from swaadstack.api.app import engine

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not ready")

    item = engine.menu_data.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found")

    return {
        "item_id": item["item_id"], "name": item["name"],
        "category": item["category"], "price": item.get("price", 0),
        "cuisine": item.get("cuisine", ""), "dietary": item.get("dietary", []),
        "description": item.get("description", ""),
    }
