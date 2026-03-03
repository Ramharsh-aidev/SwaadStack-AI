"""
SwaadStack AI - Centralized Configuration
==========================================
All hyperparameters, paths, model settings, and deployment constants.
Single source of truth for the entire system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # swaad-stack-ai/
SWAADSTACK_DIR = Path(__file__).resolve().parent.parent  # swaad-stack-ai/swaadstack/
ARTIFACTS_DIR = SWAADSTACK_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODEL_DIR = ARTIFACTS_DIR / "models"
LOG_DIR = SWAADSTACK_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Data Generation
# ==============================================================================
@dataclass
class DataConfig:
    """Configuration for synthetic data generation pipeline."""
    num_menu_items: int = 50
    num_sessions: int = 5000
    num_users: int = 500
    max_sequence_length: int = 5
    embedding_dim: int = 384
    embedding_model_name: str = "all-MiniLM-L6-v2"

    menu_file: Path = DATA_DIR / "menu.json"
    sessions_file: Path = DATA_DIR / "sessions.csv"
    embeddings_file: Path = DATA_DIR / "item_embeddings.npy"
    id_mapping_file: Path = DATA_DIR / "id_mapping.json"

    meal_flow_probs: dict = field(default_factory=lambda: {
        "Main": {"Side": 0.45, "Beverage": 0.35, "Dessert": 0.15, "Main": 0.05},
        "Side": {"Beverage": 0.50, "Dessert": 0.30, "Main": 0.10, "Side": 0.10},
        "Beverage": {"Dessert": 0.55, "Side": 0.20, "Main": 0.15, "Beverage": 0.10},
        "Dessert": {"Beverage": 0.40, "Side": 0.25, "Main": 0.25, "Dessert": 0.10},
    })

    category_distribution: dict = field(default_factory=lambda: {
        "Main": 18, "Side": 12, "Beverage": 12, "Dessert": 8,
    })

    geohashes: list = field(default_factory=lambda: [
        "tdr1y", "tdr1x", "tdnu8", "tdnub", "tsj2u",
        "tsj2v", "tepz0", "tk3s6", "tw3eq", "tgr76",
    ])


# ==============================================================================
# Model Architecture
# ==============================================================================
@dataclass
class ModelConfig:
    """Configuration for the Hybrid Two-Tower + Sequential Transformer."""
    input_embedding_dim: int = 384
    hidden_dim: int = 256
    output_embedding_dim: int = 128

    num_transformer_heads: int = 4
    num_transformer_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_feedforward_dim: int = 512

    max_seq_length: int = 5
    padding_idx: int = 0

    num_geohash_buckets: int = 100
    geohash_embedding_dim: int = 16
    temporal_feature_dim: int = 4
    context_dim: int = 20

    model_weights_path: Path = MODEL_DIR / "swaadstack_model.pth"
    item_tower_path: Path = MODEL_DIR / "item_tower.pth"
    user_tower_path: Path = MODEL_DIR / "user_tower.pth"
    faiss_index_path: Path = MODEL_DIR / "faiss_index.bin"


# ==============================================================================
# Training
# ==============================================================================
@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    num_epochs: int = 15
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 100

    num_negative_samples: int = 5

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    patience: int = 5
    min_delta: float = 1e-4

    log_interval: int = 50
    eval_interval: int = 1
    max_grad_norm: float = 1.0

    device: str = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"


# ==============================================================================
# Inference
# ==============================================================================
@dataclass
class InferenceConfig:
    """Configuration for the real-time inference engine."""
    faiss_top_k: int = 20
    faiss_nprobe: int = 10

    mmr_lambda: float = 0.7
    mmr_top_n: int = 5

    max_latency_ms: int = 300
    feature_fetch_budget_ms: int = 15
    retrieval_budget_ms: int = 25
    ranking_budget_ms: int = 80
    business_logic_budget_ms: int = 20

    popularity_fallback_top_k: int = 10
    min_interactions_for_personalization: int = 3


# ==============================================================================
# Feature Store
# ==============================================================================
@dataclass
class FeatureStoreConfig:
    """Configuration for the Redis-based online feature store."""
    redis_host: str = os.environ.get("REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("REDIS_PORT", "6379"))
    redis_db: int = int(os.environ.get("REDIS_DB", "0"))
    redis_password: Optional[str] = os.environ.get("REDIS_PASSWORD", None)

    user_vector_prefix: str = "user:vector:"
    restaurant_vector_prefix: str = "restaurant:vector:"
    cart_context_prefix: str = "cart:context:"
    popularity_prefix: str = "popularity:"

    user_vector_ttl: int = 86400
    restaurant_vector_ttl: int = 3600
    cart_context_ttl: int = 1800
    popularity_ttl: int = 3600


# ==============================================================================
# API
# ==============================================================================
@dataclass
class APIConfig:
    """Configuration for the FastAPI serving layer."""
    host: str = os.environ.get("API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("API_PORT", "8000"))
    workers: int = int(os.environ.get("API_WORKERS", "4"))
    title: str = "SwaadStack AI"
    description: str = "Real-Time Meal Completion Recommendation Engine"
    version: str = "1.0.0"
    debug: bool = os.environ.get("DEBUG", "false").lower() == "true"


# ==============================================================================
# Constants
# ==============================================================================
FOOD_CATEGORIES = ["Main", "Side", "Beverage", "Dessert"]

CUISINE_TYPES = [
    "North Indian", "South Indian", "Chinese", "Italian",
    "Continental", "Street Food", "Fast Food", "Healthy",
]

DIETARY_TAGS = [
    "Vegetarian", "Non-Vegetarian", "Vegan", "Egg",
    "Jain", "Keto", "Gluten-Free", "Spicy",
]

# ==============================================================================
# Singleton Instances
# ==============================================================================
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
feature_store_config = FeatureStoreConfig()
api_config = APIConfig()
