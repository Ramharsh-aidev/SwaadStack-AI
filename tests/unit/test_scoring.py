"""Unit tests for scoring utilities."""

import time
import pytest

from swaadstack.utils.scoring import category_diversity_score, compute_rfm_scores


class TestDiversityScore:
    def test_empty_cart(self):
        assert category_diversity_score([]) == 0.0

    def test_single_item(self):
        assert category_diversity_score(["Main"]) == 1.0

    def test_all_same_category(self):
        assert category_diversity_score(["Main", "Main", "Main"]) == pytest.approx(1/3)

    def test_all_different_categories(self):
        assert category_diversity_score(["Main", "Side", "Beverage", "Dessert"]) == 1.0

    def test_partial_diversity(self):
        score = category_diversity_score(["Main", "Main", "Side", "Beverage"])
        assert 0.5 < score < 1.0


class TestRFMScores:
    def test_empty_orders(self):
        scores = compute_rfm_scores([], current_timestamp=time.time())
        assert scores["recency"] == 0.0
        assert scores["frequency"] == 0.0
        assert scores["monetary"] == 0.0

    def test_recent_order(self):
        now = time.time()
        orders = [{"timestamp": now - 3600, "total_value": 500}]
        scores = compute_rfm_scores(orders, current_timestamp=now)
        assert scores["recency"] > 0.9

    def test_old_orders(self):
        now = time.time()
        orders = [{"timestamp": now - 60 * 86400, "total_value": 500}]
        scores = compute_rfm_scores(orders, current_timestamp=now)
        assert scores["recency"] == 0.0
