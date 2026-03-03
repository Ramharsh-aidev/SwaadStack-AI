"""Shared test fixtures used across all test modules."""

import time

import numpy as np
import pytest


@pytest.fixture
def sample_query():
    """Generate a random query vector."""
    np.random.seed(42)
    vec = np.random.randn(128).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.copy()


@pytest.fixture
def sample_embeddings():
    """Generate random candidate embeddings."""
    np.random.seed(123)
    vecs = np.random.randn(10, 128).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).copy()


@pytest.fixture
def sample_candidate_info():
    """Generate sample candidate metadata."""
    return [
        {"item_id": f"ITEM_{i:03d}", "name": f"Item {i}",
         "category": ["Main", "Side", "Beverage", "Dessert"][i % 4],
         "price": 100 + i * 20}
        for i in range(10)
    ]
