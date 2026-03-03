"""Model factory — create and load SwaadStack models."""

from typing import Optional

import torch

from swaadstack.config import model_config, ModelConfig
from swaadstack.models.swaadstack import SwaadStackModel


def create_model(config: Optional[ModelConfig] = None) -> SwaadStackModel:
    """Create a new SwaadStack model."""
    if config is None:
        config = model_config

    model = SwaadStackModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 SwaadStack Model Created:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Input dim:  {config.input_embedding_dim}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Output dim: {config.output_embedding_dim}")
    print(f"   Transformer: {config.num_transformer_layers} layers, {config.num_transformer_heads} heads")

    return model


def load_model(path: str, config: Optional[ModelConfig] = None) -> SwaadStackModel:
    """Load a trained model from disk."""
    model = create_model(config)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
