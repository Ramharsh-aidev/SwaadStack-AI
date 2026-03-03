"""
Neural network building blocks — towers and encoders.

- PositionalEncoding: Sinusoidal position injection for sequence order
- ItemTower: MLP projector for item embeddings (384 → 128)
- ContextEncoder: Temporal + geographic feature encoder
- SequentialUserTower: Transformer-based cart sequence encoder
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ItemTower(nn.Module):
    """
    Item Tower — projects text embeddings into shared latent space.

    Architecture: Input(384) → Linear → LN → GELU → Linear → LN → GELU → Linear → L2-norm → Output(128)
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return F.normalize(projected, p=2, dim=-1)


class ContextEncoder(nn.Module):
    """Encodes temporal (cyclical sin/cos) + geographic (geohash embedding) context."""

    def __init__(self, temporal_dim: int = 4, num_geohash_buckets: int = 100,
                 geohash_embedding_dim: int = 16, output_dim: int = 32):
        super().__init__()
        self.geohash_embedding = nn.Embedding(num_geohash_buckets, geohash_embedding_dim)
        context_input_dim = temporal_dim + geohash_embedding_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, temporal_features: torch.Tensor, geohash_buckets: torch.Tensor) -> torch.Tensor:
        geo_emb = self.geohash_embedding(geohash_buckets)
        context = torch.cat([temporal_features, geo_emb], dim=-1)
        return self.context_mlp(context)


class SequentialUserTower(nn.Module):
    """
    User/Sequence Tower — TransformerEncoder + [CLS] token + context fusion.

    Understands that 'Butter Chicken → Garlic Naan → ???' logically leads to a Beverage.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, output_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 max_seq_length: int = 5, feedforward_dim: int = 512, context_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, max_len=max_seq_length + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, item_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = item_embeddings.size(0)

        x = self.input_projection(item_embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.positional_encoding(x)

        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            attention_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            attention_mask = None

        encoded = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        sequence_repr = encoded[:, 0, :]

        if context_vector is not None:
            combined = torch.cat([sequence_repr, context_vector], dim=-1)
        else:
            zero_ctx = torch.zeros(
                batch_size, self.final_projection[0].in_features - self.hidden_dim,
                device=sequence_repr.device,
            )
            combined = torch.cat([sequence_repr, zero_ctx], dim=-1)

        output = self.final_projection(combined)
        return F.normalize(output, p=2, dim=-1)
