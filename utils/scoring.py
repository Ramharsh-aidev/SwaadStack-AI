"""Scoring utilities — RFM scores, cart diversity metrics."""

import time
from typing import Any, Dict, List


def category_diversity_score(cart_categories: List[str]) -> float:
    """
    Compute diversity score for categories in cart.
    Higher = more diverse (more different categories).
    """
    if not cart_categories:
        return 0.0
    return len(set(cart_categories)) / len(cart_categories)


def compute_rfm_scores(
    user_orders: List[Dict[str, Any]],
    current_timestamp: float,
) -> Dict[str, float]:
    """
    Compute RFM (Recency, Frequency, Monetary) scores for a user.

    Args:
        user_orders: List of dicts with 'timestamp' and 'total_value'
        current_timestamp: Current time for recency calculation

    Returns:
        Dict with recency, frequency, monetary scores (0–1 normalized)
    """
    if not user_orders:
        return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}

    timestamps = [o["timestamp"] for o in user_orders]
    values = [o["total_value"] for o in user_orders]

    days_since_last = (current_timestamp - max(timestamps)) / 86400
    recency = max(0.0, 1.0 - days_since_last / 30.0)

    recent_orders = sum(1 for t in timestamps if (current_timestamp - t) < 30 * 86400)
    frequency = min(1.0, recent_orders / 20.0)

    avg_value = sum(values) / len(values)
    monetary = min(1.0, avg_value / 1000.0)

    return {
        "recency": round(recency, 4),
        "frequency": round(frequency, 4),
        "monetary": round(monetary, 4),
    }
