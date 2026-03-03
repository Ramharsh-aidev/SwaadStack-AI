"""Embedding generation — Sentence-BERT and random fallback."""

import hashlib
from typing import Any, Dict, List

import numpy as np

from swaadstack.utils.helpers import console


def generate_sbert_embeddings(
    items: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, np.ndarray]:
    """Generate Sentence-BERT embeddings for menu items (zero-shot cold-start)."""
    try:
        from sentence_transformers import SentenceTransformer

        console.print(f"[bold cyan]Loading Sentence-BERT model: {model_name}[/bold cyan]")
        model = SentenceTransformer(model_name)

        descriptions = []
        item_ids = []
        for item in items:
            text = (
                f"{item['name']}. {item['description']}. "
                f"Category: {item['category']}. Cuisine: {item['cuisine']}. "
                f"Dietary: {', '.join(item['dietary'])}."
            )
            descriptions.append(text)
            item_ids.append(item["item_id"])

        console.print(f"[cyan]Generating embeddings for {len(descriptions)} items...[/cyan]")
        embeddings = model.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)

        embedding_dict = {}
        for item_id, embedding in zip(item_ids, embeddings):
            embedding_dict[item_id] = embedding.astype(np.float32)

        console.print(f"[green]✓ Generated {len(embedding_dict)} embeddings of dim {embeddings.shape[1]}[/green]")
        return embedding_dict

    except ImportError:
        console.print("[yellow]⚠ sentence-transformers not available. Using random embeddings.[/yellow]")
        return generate_random_embeddings(items)


def generate_random_embeddings(
    items: List[Dict[str, Any]],
    dim: int = 384,
) -> Dict[str, np.ndarray]:
    """Deterministic pseudo-random embeddings as fallback — category-biased for clustering."""
    embedding_dict = {}
    for item in items:
        seed_str = f"{item['item_id']}_{item['name']}_{item['category']}_{item['cuisine']}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)

        vec = rng.randn(dim).astype(np.float32)

        category_bias = {
            "Main": np.array([1, 0, 0, 0] * (dim // 4)),
            "Side": np.array([0, 1, 0, 0] * (dim // 4)),
            "Beverage": np.array([0, 0, 1, 0] * (dim // 4)),
            "Dessert": np.array([0, 0, 0, 1] * (dim // 4)),
        }
        bias = category_bias.get(item["category"], np.zeros(dim))
        vec = vec + 0.5 * bias.astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        embedding_dict[item["item_id"]] = vec

    return embedding_dict
