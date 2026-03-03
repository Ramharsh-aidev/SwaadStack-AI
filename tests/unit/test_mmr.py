"""Unit tests for MMR re-ranking logic."""

import numpy as np
import pytest

from swaadstack.utils.mmr import mmr_rerank


class TestMMRReranking:
    def test_mmr_returns_correct_count(self, sample_query, sample_embeddings, sample_candidate_info):
        results = mmr_rerank(query_vec=sample_query, candidate_vecs=sample_embeddings,
                             candidate_info=sample_candidate_info, lambda_param=0.7, top_n=5)
        assert len(results) == 5

    def test_mmr_returns_fewer_when_not_enough_candidates(self, sample_query):
        small_embeddings = np.random.randn(3, 128).astype(np.float32)
        small_info = [
            {"item_id": "A", "name": "A", "category": "Main", "price": 100},
            {"item_id": "B", "name": "B", "category": "Side", "price": 80},
            {"item_id": "C", "name": "C", "category": "Beverage", "price": 50},
        ]
        results = mmr_rerank(query_vec=sample_query, candidate_vecs=small_embeddings,
                             candidate_info=small_info, top_n=10)
        assert len(results) == 3

    def test_mmr_excludes_cart_items(self, sample_query, sample_embeddings, sample_candidate_info):
        exclude = {"ITEM_000", "ITEM_001"}
        results = mmr_rerank(query_vec=sample_query, candidate_vecs=sample_embeddings,
                             candidate_info=sample_candidate_info, top_n=5, exclude_ids=exclude)
        result_ids = {r["item_id"] for r in results}
        assert not result_ids.intersection(exclude)

    def test_mmr_diversity_vs_relevance(self, sample_query):
        dim = 128
        np.random.seed(123)
        base = sample_query.copy()
        similar_a = base + np.random.randn(dim).astype(np.float32) * 0.01
        similar_b = base + np.random.randn(dim).astype(np.float32) * 0.01
        different = np.random.randn(dim).astype(np.float32)
        candidates = np.array([similar_a, similar_b, different])
        info = [
            {"item_id": "S1", "name": "Similar 1"},
            {"item_id": "S2", "name": "Similar 2"},
            {"item_id": "D1", "name": "Different 1"},
        ]
        relevant = mmr_rerank(sample_query, candidates, info, lambda_param=1.0, top_n=3)
        diverse = mmr_rerank(sample_query, candidates, info, lambda_param=0.0, top_n=3)
        assert len(relevant) == 3
        assert len(diverse) == 3

    def test_mmr_scores_are_present(self, sample_query, sample_embeddings, sample_candidate_info):
        results = mmr_rerank(query_vec=sample_query, candidate_vecs=sample_embeddings,
                             candidate_info=sample_candidate_info, top_n=3)
        for r in results:
            assert "mmr_score" in r
            assert "relevance_score" in r
            assert isinstance(r["mmr_score"], float)

    def test_mmr_empty_candidates(self, sample_query):
        results = mmr_rerank(query_vec=sample_query,
                             candidate_vecs=np.array([]).reshape(0, 128).astype(np.float32),
                             candidate_info=[], top_n=5)
        assert results == []
