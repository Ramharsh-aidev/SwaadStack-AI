"""Unit tests for data generation — menu items, embeddings, sessions, dataset."""

import numpy as np
import pandas as pd
import torch
import pytest


class TestMenuItems:
    def test_menu_items_exist(self):
        from swaadstack.data.menu_items import MENU_ITEMS
        assert len(MENU_ITEMS) == 50

    def test_menu_item_schema(self):
        from swaadstack.data.menu_items import MENU_ITEMS
        required = {"item_id", "name", "category", "price", "description", "cuisine", "dietary"}
        for item in MENU_ITEMS:
            for field in required:
                assert field in item, f"Missing '{field}' in {item.get('item_id')}"

    def test_category_distribution(self):
        from swaadstack.data.menu_items import MENU_ITEMS
        categories = {item["category"] for item in MENU_ITEMS}
        assert categories == {"Main", "Side", "Beverage", "Dessert"}


class TestEmbeddings:
    def test_random_embeddings(self):
        from swaadstack.data.embeddings import generate_random_embeddings
        from swaadstack.data.menu_items import MENU_ITEMS
        embeddings = generate_random_embeddings(MENU_ITEMS[:5], dim=384)
        assert len(embeddings) == 5
        for emb in embeddings.values():
            assert emb.shape == (384,)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_random_embeddings_deterministic(self):
        from swaadstack.data.embeddings import generate_random_embeddings
        from swaadstack.data.menu_items import MENU_ITEMS
        emb1 = generate_random_embeddings(MENU_ITEMS[:5])
        emb2 = generate_random_embeddings(MENU_ITEMS[:5])
        for item_id in emb1:
            assert np.allclose(emb1[item_id], emb2[item_id])


class TestSessionSimulation:
    def test_session_simulation(self):
        from swaadstack.data.generator import simulate_sessions
        from swaadstack.data.menu_items import MENU_ITEMS
        df = simulate_sessions(MENU_ITEMS, num_sessions=50, num_users=10)
        assert len(df) > 0
        assert "user_id" in df.columns
        assert "sequence_item_ids" in df.columns
        assert "target_item_id" in df.columns
        assert "timestamp" in df.columns
        assert "mealtime" in df.columns


class TestDataset:
    def test_sequence_padding(self):
        from swaadstack.training.dataset import CartCompletionDataset
        from swaadstack.data.embeddings import generate_random_embeddings
        from swaadstack.data.menu_items import MENU_ITEMS
        embeddings = generate_random_embeddings(MENU_ITEMS[:10])
        sessions_df = pd.DataFrame([{
            "user_id": "user_0001", "session_id": "sess_000001",
            "sequence_item_ids": "MAIN_001|SIDE_001", "target_item_id": "BEV_001",
            "target_category": "Beverage", "cart_size": 2,
            "timestamp": "2025-01-15T12:30:00", "hour": 12,
            "day_of_week": 2, "mealtime": "lunch", "geohash": "tdr1y",
        }])
        dataset = CartCompletionDataset(sessions_df, MENU_ITEMS[:10], embeddings, max_seq_length=5, num_negatives=1)
        sample = dataset[0]
        assert sample["cart_embeddings"].shape == (5, 384)
        assert sample["padding_mask"].shape == (5,)
        assert sample["temporal_features"].shape == (4,)

    def test_padding_mask_correctness(self):
        from swaadstack.training.dataset import CartCompletionDataset
        from swaadstack.data.embeddings import generate_random_embeddings
        from swaadstack.data.menu_items import MENU_ITEMS
        embeddings = generate_random_embeddings(MENU_ITEMS[:10])
        sessions_df = pd.DataFrame([{
            "user_id": "user_0001", "session_id": "sess_000001",
            "sequence_item_ids": "MAIN_001", "target_item_id": "SIDE_001",
            "target_category": "Side", "cart_size": 1,
            "timestamp": "2025-01-15T12:30:00", "hour": 12,
            "day_of_week": 2, "mealtime": "lunch", "geohash": "tdr1y",
        }])
        dataset = CartCompletionDataset(sessions_df, MENU_ITEMS[:10], embeddings, max_seq_length=5, num_negatives=0)
        mask = dataset[0]["padding_mask"]
        assert mask[0] == False
        assert all(mask[i] == True for i in range(1, 5))
