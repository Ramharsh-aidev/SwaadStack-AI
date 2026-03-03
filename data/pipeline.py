"""Data generation pipeline — orchestrates menu, embeddings, and session generation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from swaadstack.config import data_config, DataConfig
from swaadstack.data.menu_items import MENU_ITEMS
from swaadstack.data.embeddings import generate_sbert_embeddings, generate_random_embeddings
from swaadstack.data.generator import simulate_sessions
from swaadstack.utils.helpers import save_json, print_banner, console


def run_data_generation_pipeline(
    config: Optional[DataConfig] = None,
    use_sentence_bert: bool = True,
) -> Tuple[List[Dict], pd.DataFrame, Dict[str, np.ndarray]]:
    """Execute the complete Synthetic Data Generation pipeline."""
    if config is None:
        config = data_config

    print_banner("SwaadStack AI - Synthetic Data Generation Pipeline")

    # Step 1: Menu Items
    console.print("[bold]Step 1/4: Preparing menu items...[/bold]")
    menu_items = MENU_ITEMS[:config.num_menu_items]
    console.print(f"  📋 {len(menu_items)} menu items across {len(set(i['category'] for i in menu_items))} categories")

    # Step 2: Generate Embeddings
    console.print("\n[bold]Step 2/4: Generating item embeddings...[/bold]")
    if use_sentence_bert:
        embeddings = generate_sbert_embeddings(menu_items, config.embedding_model_name)
    else:
        embeddings = generate_random_embeddings(menu_items, config.embedding_dim)

    # Step 3: Simulate Sessions
    console.print("\n[bold]Step 3/4: Simulating user sessions...[/bold]")
    sessions_df = simulate_sessions(
        menu_items,
        num_sessions=config.num_sessions,
        num_users=config.num_users,
        geohashes=config.geohashes,
    )

    # Step 4: Save Artifacts
    console.print("\n[bold]Step 4/4: Saving artifacts...[/bold]")

    menu_with_embeddings = []
    for item in menu_items:
        item_copy = item.copy()
        item_copy["embedding"] = embeddings[item["item_id"]].tolist()
        menu_with_embeddings.append(item_copy)

    save_json(menu_with_embeddings, str(config.menu_file))
    console.print(f"  💾 Menu saved to: {config.menu_file}")

    sessions_df.to_csv(str(config.sessions_file), index=False)
    console.print(f"  💾 Sessions saved to: {config.sessions_file}")

    item_ids = sorted(embeddings.keys())
    embedding_matrix = np.array([embeddings[iid] for iid in item_ids])
    np.save(str(config.embeddings_file), embedding_matrix)
    console.print(f"  💾 Embeddings saved to: {config.embeddings_file}")

    id_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
    save_json(id_mapping, str(config.id_mapping_file))
    console.print(f"  💾 ID mapping saved to: {config.id_mapping_file}")

    # Summary
    console.print("\n[bold green]✅ Data Generation Complete![/bold green]")
    console.print(f"  📊 Menu items: {len(menu_items)}")
    console.print(f"  📊 Training samples: {len(sessions_df)}")
    console.print(f"  📊 Unique users: {sessions_df['user_id'].nunique()}")
    console.print(f"  📊 Embedding dimension: {embedding_matrix.shape[1]}")
    console.print(f"  📊 Date range: {sessions_df['timestamp'].min()} to {sessions_df['timestamp'].max()}")

    console.print("\n  📊 Target category distribution:")
    for cat, count in sessions_df["target_category"].value_counts().items():
        console.print(f"     {cat}: {count} ({count/len(sessions_df)*100:.1f}%)")

    return menu_items, sessions_df, embeddings
