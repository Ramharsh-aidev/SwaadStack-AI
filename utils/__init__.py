"""Utilities package - shared helpers across the system."""

from swaadstack.utils.logging import setup_logging, logger
from swaadstack.utils.encoding import encode_temporal_features, get_mealtime_label, geohash_to_bucket
from swaadstack.utils.embeddings import normalize_embedding, batch_normalize_embeddings, cosine_similarity
from swaadstack.utils.mmr import mmr_rerank
from swaadstack.utils.scoring import category_diversity_score, compute_rfm_scores
from swaadstack.utils.helpers import save_json, load_json, print_banner, print_metrics, create_progress_bar, timeit

__all__ = [
    "setup_logging", "logger",
    "encode_temporal_features", "get_mealtime_label", "geohash_to_bucket",
    "normalize_embedding", "batch_normalize_embeddings", "cosine_similarity",
    "mmr_rerank",
    "category_diversity_score", "compute_rfm_scores",
    "save_json", "load_json", "print_banner", "print_metrics",
    "create_progress_bar", "timeit",
]
