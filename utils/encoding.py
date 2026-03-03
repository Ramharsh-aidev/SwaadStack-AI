"""
Temporal & geographic feature encoding utilities.

- Cyclical sin/cos encoding for time features (prevents hour 23/0 distance problem)
- Geohash string to bucket index mapping
- Mealtime label classification
"""

import math
import hashlib

import numpy as np


def encode_temporal_features(hour: int, day_of_week: int) -> np.ndarray:
    """
    Encode temporal features using cyclical sine/cosine transforms.

    Prevents the model from seeing hour=23 and hour=0 as maximally
    distant when they are actually adjacent.

    Args:
        hour: Hour of day (0-23)
        day_of_week: Day of week (0-6, Monday=0)

    Returns:
        numpy array of shape (4,) → [hour_sin, hour_cos, dow_sin, dow_cos]
    """
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)
    return np.array([hour_sin, hour_cos, dow_sin, dow_cos], dtype=np.float32)


def get_mealtime_label(hour: int) -> str:
    """Map hour to meal period for contextual reasoning."""
    if 6 <= hour < 11:
        return "breakfast"
    elif 11 <= hour < 15:
        return "lunch"
    elif 15 <= hour < 18:
        return "snacks"
    elif 18 <= hour < 23:
        return "dinner"
    else:
        return "late_night"


def geohash_to_bucket(geohash: str, num_buckets: int = 100) -> int:
    """Convert geohash string to a bucket index for embedding lookup."""
    hash_val = int(hashlib.md5(geohash.encode()).hexdigest(), 16)
    return hash_val % num_buckets
