"""Training pipeline — data loading, splitting, training, and embedding export."""

import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from swaadstack.config import data_config, model_config, training_config, DataConfig, MODEL_DIR
from swaadstack.models import SwaadStackModel, create_model
from swaadstack.training.dataset import CartCompletionDataset
from swaadstack.training.trainer import Trainer
from swaadstack.utils.helpers import print_banner, console


def load_data(config: DataConfig = None):
    config = config or data_config
    console.print("[cyan]Loading data artifacts...[/cyan]")
    sessions_df = pd.read_csv(str(config.sessions_file))
    console.print(f"  📋 Sessions: {len(sessions_df)} samples")

    with open(str(config.menu_file), "r", encoding="utf-8") as f:
        menu_data = json.load(f)
    console.print(f"  📋 Menu items: {len(menu_data)}")

    embeddings = {}
    for item in menu_data:
        embeddings[item["item_id"]] = np.array(item["embedding"], dtype=np.float32)
    return sessions_df, menu_data, embeddings


def temporal_split(sessions_df, train_ratio=0.7, val_ratio=0.15):
    df = sessions_df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    console.print(f"  Temporal Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df


def save_item_embeddings(model, embeddings, id_mapping_path, output_path):
    model.eval()
    with open(id_mapping_path, "r") as f:
        id_mapping = json.load(f)
    sorted_ids = sorted(id_mapping.keys(), key=lambda x: id_mapping[x])
    emb_matrix = np.array([embeddings[iid] for iid in sorted_ids])
    emb_tensor = torch.tensor(emb_matrix, dtype=torch.float32)
    with torch.no_grad():
        projected = model.get_item_embeddings(emb_tensor)
        projected_np = projected.cpu().numpy().astype(np.float32)
    np.save(output_path, projected_np)
    console.print(f"  💾 Projected item embeddings: {output_path} ({projected_np.shape})")
    return projected_np


def run_training_pipeline():
    print_banner("SwaadStack AI - Training Pipeline")

    console.print("\n[bold]Step 1: Loading Data[/bold]")
    sessions_df, menu_data, embeddings = load_data()

    console.print("\n[bold]Step 2: Temporal Split[/bold]")
    train_df, val_df, test_df = temporal_split(sessions_df, training_config.train_ratio, training_config.val_ratio)

    console.print("\n[bold]Step 3: Creating Datasets[/bold]")
    train_dataset = CartCompletionDataset(train_df, menu_data, embeddings,
                                          max_seq_length=model_config.max_seq_length,
                                          num_negatives=training_config.num_negative_samples)
    val_dataset = CartCompletionDataset(val_df, menu_data, embeddings,
                                        max_seq_length=model_config.max_seq_length,
                                        num_negatives=training_config.num_negative_samples)

    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    console.print(f"  Train: {len(train_dataset)} samples → {len(train_loader)} batches")
    console.print(f"  Val:   {len(val_dataset)} samples → {len(val_loader)} batches")

    console.print("\n[bold]Step 4: Training[/bold]")
    model = create_model(model_config)
    trainer = Trainer(model, train_loader, val_loader, training_config)
    results = trainer.train()

    console.print("\n[bold]Step 5: Saving FAISS Embeddings[/bold]")
    projected_path = str(MODEL_DIR / "projected_item_embeddings.npy")
    save_item_embeddings(model, embeddings, str(data_config.id_mapping_file), projected_path)

    console.print("\n[bold green]🎉 Training Pipeline Complete![/bold green]")
    return model, results
