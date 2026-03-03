"""Performance and latency validation tests."""

import time

import numpy as np
import torch
import pytest

from swaadstack.utils.mmr import mmr_rerank
from swaadstack.utils.encoding import encode_temporal_features


class TestPerformance:
    def test_mmr_performance(self, sample_query, sample_embeddings, sample_candidate_info):
        start = time.perf_counter()
        for _ in range(100):
            mmr_rerank(query_vec=sample_query, candidate_vecs=sample_embeddings,
                       candidate_info=sample_candidate_info, top_n=5)
        elapsed = (time.perf_counter() - start) / 100 * 1000
        assert elapsed < 50, f"MMR too slow: {elapsed:.2f}ms"

    def test_temporal_encoding_performance(self):
        start = time.perf_counter()
        for _ in range(10000):
            encode_temporal_features(12, 3)
        elapsed = (time.perf_counter() - start) / 10000 * 1000
        assert elapsed < 1, f"Temporal encoding too slow: {elapsed:.4f}ms"

    def test_model_forward_pass_performance(self):
        from swaadstack.models import SwaadStackModel
        model = SwaadStackModel()
        model.eval()
        B = 1
        cart = torch.randn(B, 5, 384)
        target = torch.randn(B, 384)
        temporal = torch.randn(B, 4)
        geohash = torch.randint(0, 100, (B,))
        with torch.no_grad():
            model(cart_embeddings=cart, target_embeddings=target,
                  temporal_features=temporal, geohash_buckets=geohash)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                model(cart_embeddings=cart, target_embeddings=target,
                      temporal_features=temporal, geohash_buckets=geohash)
        elapsed = (time.perf_counter() - start) / 50 * 1000
        assert elapsed < 200, f"Forward pass too slow: {elapsed:.2f}ms"
