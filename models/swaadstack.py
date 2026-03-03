"""SwaadStackModel (training) and SwaadStackInference (production) wrappers."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from swaadstack.config import model_config, ModelConfig
from swaadstack.models.towers import ItemTower, ContextEncoder, SequentialUserTower


class SwaadStackModel(nn.Module):
    """Complete Two-Tower model for training with BCEWithLogitsLoss."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = model_config
        self.config = config

        self.item_tower = ItemTower(
            input_dim=config.input_embedding_dim, hidden_dim=config.hidden_dim,
            output_dim=config.output_embedding_dim, dropout=config.transformer_dropout,
        )
        self.context_encoder = ContextEncoder(
            temporal_dim=config.temporal_feature_dim, num_geohash_buckets=config.num_geohash_buckets,
            geohash_embedding_dim=config.geohash_embedding_dim,
            output_dim=config.context_dim + config.geohash_embedding_dim,
        )
        self.user_tower = SequentialUserTower(
            input_dim=config.input_embedding_dim, hidden_dim=config.hidden_dim,
            output_dim=config.output_embedding_dim, num_heads=config.num_transformer_heads,
            num_layers=config.num_transformer_layers, dropout=config.transformer_dropout,
            max_seq_length=config.max_seq_length, feedforward_dim=config.transformer_feedforward_dim,
            context_dim=config.context_dim + config.geohash_embedding_dim,
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, cart_embeddings: torch.Tensor, target_embeddings: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None, temporal_features: Optional[torch.Tensor] = None,
                geohash_buckets: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        context = None
        if temporal_features is not None and geohash_buckets is not None:
            context = self.context_encoder(temporal_features, geohash_buckets)

        user_embedding = self.user_tower(cart_embeddings, padding_mask=padding_mask, context_vector=context)
        item_embedding = self.item_tower(target_embeddings)
        logits = torch.sum(user_embedding * item_embedding, dim=-1) / self.temperature.abs()

        output = {"logits": logits, "user_embedding": user_embedding, "item_embedding": item_embedding}
        if labels is not None:
            output["loss"] = self.loss_fn(logits, labels.float())
        return output

    def get_user_embedding(self, cart_embeddings, padding_mask=None, temporal_features=None, geohash_buckets=None):
        context = None
        if temporal_features is not None and geohash_buckets is not None:
            context = self.context_encoder(temporal_features, geohash_buckets)
        return self.user_tower(cart_embeddings, padding_mask=padding_mask, context_vector=context)

    def get_item_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_embeddings)

    def compute_scores(self, user_embedding: torch.Tensor, candidate_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.matmul(user_embedding, candidate_embeddings.T) / self.temperature.abs()


class SwaadStackInference(nn.Module):
    """Inference-optimized wrapper — no gradients, numpy output for FAISS."""

    def __init__(self, model: SwaadStackModel):
        super().__init__()
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def encode_cart(self, cart_embeddings, padding_mask=None, temporal_features=None, geohash_buckets=None) -> np.ndarray:
        user_emb = self.model.get_user_embedding(cart_embeddings, padding_mask, temporal_features, geohash_buckets)
        return user_emb.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_items(self, item_embeddings: torch.Tensor) -> np.ndarray:
        projected = self.model.get_item_embeddings(item_embeddings)
        return projected.cpu().numpy().astype(np.float32)
