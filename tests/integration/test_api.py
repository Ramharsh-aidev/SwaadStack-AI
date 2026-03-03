"""Integration tests for FastAPI API endpoints."""

import pytest
from unittest.mock import MagicMock


class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        import swaadstack.api.app as app_module

        mock_engine = MagicMock()
        mock_engine._is_loaded = True
        mock_engine.menu_data = {
            "MAIN_001": {"item_id": "MAIN_001", "name": "Butter Chicken", "category": "Main",
                         "price": 320, "cuisine": "North Indian", "dietary": ["Non-Vegetarian", "Spicy"],
                         "description": "Rich tomato-based chicken curry"},
            "SIDE_001": {"item_id": "SIDE_001", "name": "Garlic Naan", "category": "Side",
                         "price": 60, "cuisine": "North Indian", "dietary": ["Vegetarian"],
                         "description": "Tandoor-baked bread with garlic"},
            "BEV_001": {"item_id": "BEV_001", "name": "Coca-Cola", "category": "Beverage",
                        "price": 50, "cuisine": "Fast Food", "dietary": ["Vegetarian"], "description": "Classic cola"},
        }
        mock_engine.recommend.return_value = {
            "recommendations": [
                {"item_id": "SIDE_001", "name": "Garlic Naan", "category": "Side", "price": 60, "mmr_score": 0.85, "relevance_score": 0.92},
                {"item_id": "BEV_001", "name": "Coca-Cola", "category": "Beverage", "price": 50, "mmr_score": 0.78, "relevance_score": 0.88},
            ],
            "cart_summary": {"items": ["MAIN_001"], "item_count": 1, "total_value": 320,
                             "categories": ["Main"], "diversity_score": 1.0, "missing_categories": ["Side", "Beverage", "Dessert"]},
            "context": {"mealtime": "lunch", "geohash": "tdr1y", "personalized": False},
            "latency": {"feature_fetch_ms": 2.5, "encoding_ms": 15.3, "retrieval_ms": 3.2, "ranking_ms": 1.8, "total_ms": 22.8},
            "metadata": {"candidates_retrieved": 20, "candidates_after_filter": 18, "mmr_lambda": 0.7, "model_version": "swaadstack-v1.0"},
        }
        mock_engine.health_check.return_value = {
            "loaded": True, "model_loaded": True, "faiss_loaded": True, "num_menu_items": 3, "num_indexed_items": 3,
        }
        app_module.engine = mock_engine
        self.client = TestClient(app_module.app)
        self.mock_engine = mock_engine

    def test_root_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "SwaadStack AI"

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_recommend_golden_path(self):
        response = self.client.post("/recommend", json={"cart_items": ["MAIN_001"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) > 0
        assert data["cart_summary"]["item_count"] == 1

    def test_recommend_with_context(self):
        response = self.client.post("/recommend", json={
            "cart_items": ["MAIN_001"], "user_id": "user_0001", "geohash": "tdr1y",
            "hour": 13, "day_of_week": 3, "top_n": 3, "diversity": 0.7,
        })
        assert response.status_code == 200

    def test_recommend_empty_cart(self):
        response = self.client.post("/recommend", json={"cart_items": []})
        assert response.status_code in [200, 404]

    def test_recommend_invalid_items(self):
        self.mock_engine.menu_data = {"MAIN_001": {"item_id": "MAIN_001", "name": "Test"}}
        response = self.client.post("/recommend", json={"cart_items": ["INVALID_001", "INVALID_002"]})
        assert response.status_code == 404

    def test_predict_alias(self):
        response = self.client.post("/predict", json={"cart_items": ["MAIN_001"]})
        assert response.status_code == 200

    def test_menu_listing(self):
        response = self.client.get("/menu")
        assert response.status_code == 200
        assert len(response.json()) > 0

    def test_menu_item_by_id(self):
        response = self.client.get("/menu/MAIN_001")
        assert response.status_code == 200
        assert response.json()["item_id"] == "MAIN_001"

    def test_menu_item_not_found(self):
        assert self.client.get("/menu/NONEXISTENT").status_code == 404

    def test_menu_filter_by_category(self):
        response = self.client.get("/menu?category=Main")
        assert response.status_code == 200
        for item in response.json():
            assert item["category"] == "Main"

    def test_response_has_timing_header(self):
        assert "x-response-time" in self.client.get("/health").headers

    def test_latency_in_response(self):
        response = self.client.post("/recommend", json={"cart_items": ["MAIN_001"]})
        assert "total_ms" in response.json()["latency"]
