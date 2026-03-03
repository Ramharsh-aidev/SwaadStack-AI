"""Unit tests for embedding normalization and similarity."""

import numpy as np
import pytest

from swaadstack.utils.embeddings import normalize_embedding, batch_normalize_embeddings


class TestEmbeddingUtilities:
    def test_normalize_returns_unit_vector(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normed = normalize_embedding(vec)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-5

    def test_normalize_zero_vector(self):
        vec = np.zeros(10, dtype=np.float32)
        normed = normalize_embedding(vec)
        assert np.all(normed == 0)
        assert not np.any(np.isnan(normed))

    def test_batch_normalize(self):
        vecs = np.array([[3, 4], [0, 0], [1, 0]], dtype=np.float32)
        normed = batch_normalize_embeddings(vecs)
        assert abs(np.linalg.norm(normed[0]) - 1.0) < 1e-5
        assert abs(np.linalg.norm(normed[1])) < 1e-5
        assert abs(np.linalg.norm(normed[2]) - 1.0) < 1e-5
