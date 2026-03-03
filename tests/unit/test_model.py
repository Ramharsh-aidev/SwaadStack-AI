"""Unit tests for model architecture — shapes, normalization, forward pass."""

import torch
import pytest


class TestModelArchitecture:
    def test_item_tower_shapes(self):
        from swaadstack.models.towers import ItemTower
        tower = ItemTower(input_dim=384, hidden_dim=256, output_dim=128)
        x = torch.randn(4, 384)
        assert tower(x).shape == (4, 128)
        x_seq = torch.randn(4, 3, 384)
        assert tower(x_seq).shape == (4, 3, 128)

    def test_item_tower_normalization(self):
        from swaadstack.models.towers import ItemTower
        tower = ItemTower(input_dim=384, hidden_dim=256, output_dim=128)
        out = tower(torch.randn(4, 384))
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_user_tower_shapes(self):
        from swaadstack.models.towers import SequentialUserTower
        tower = SequentialUserTower(input_dim=384, hidden_dim=256, output_dim=128,
                                    num_heads=4, num_layers=2, context_dim=32)
        B, S, D = 4, 3, 384
        x = torch.randn(B, S, D)
        mask = torch.tensor([[False, False, False], [False, False, True],
                              [False, True, True], [False, False, False]])
        context = torch.randn(B, 32)
        assert tower(x, padding_mask=mask, context_vector=context).shape == (B, 128)

    def test_user_tower_normalization(self):
        from swaadstack.models.towers import SequentialUserTower
        tower = SequentialUserTower(input_dim=384, hidden_dim=256, output_dim=128,
                                    num_heads=4, num_layers=2, context_dim=32)
        out = tower(torch.randn(4, 3, 384), context_vector=torch.randn(4, 32))
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_full_model_forward_pass(self):
        from swaadstack.models import SwaadStackModel
        model = SwaadStackModel()
        B = 4
        output = model(
            cart_embeddings=torch.randn(B, 3, 384), target_embeddings=torch.randn(B, 384),
            padding_mask=torch.zeros(B, 3, dtype=torch.bool), temporal_features=torch.randn(B, 4),
            geohash_buckets=torch.randint(0, 100, (B,)), labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        assert output["logits"].shape == (B,)
        assert output["user_embedding"].shape == (B, 128)
        assert output["item_embedding"].shape == (B, 128)
        assert output["loss"].dim() == 0

    def test_model_no_labels(self):
        from swaadstack.models import SwaadStackModel
        model = SwaadStackModel()
        output = model(cart_embeddings=torch.randn(1, 2, 384), target_embeddings=torch.randn(1, 384))
        assert "logits" in output
        assert "loss" not in output

    def test_model_score_computation(self):
        from swaadstack.models import SwaadStackModel
        model = SwaadStackModel()
        user_emb = torch.nn.functional.normalize(torch.randn(1, 128), p=2, dim=-1)
        item_embs = torch.nn.functional.normalize(torch.randn(10, 128), p=2, dim=-1)
        assert model.compute_scores(user_emb, item_embs).shape == (1, 10)
