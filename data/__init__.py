"""Data package — synthetic data generation pipeline."""

from swaadstack.data.menu_items import MENU_ITEMS
from swaadstack.data.embeddings import generate_random_embeddings, generate_sbert_embeddings
from swaadstack.data.generator import simulate_sessions
from swaadstack.data.pipeline import run_data_generation_pipeline

__all__ = [
    "MENU_ITEMS",
    "generate_random_embeddings", "generate_sbert_embeddings",
    "simulate_sessions",
    "run_data_generation_pipeline",
]
