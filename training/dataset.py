"""CartCompletionDataset — PyTorch dataset with negative sampling."""

import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from swaadstack.utils.encoding import encode_temporal_features, geohash_to_bucket


class CartCompletionDataset(Dataset):
    """
    Each sample: cart embeddings + target (pos/neg) + context + label.
    Negative sampling: 50% hard (same category), 50% easy (random).
    """

    def __init__(self, sessions_df, menu_data: List[Dict[str, Any]],
                 embeddings: Dict[str, np.ndarray], max_seq_length: int = 5, num_negatives: int = 5):
        self.sessions = sessions_df.reset_index(drop=True)
        self.menu_data = {item["item_id"]: item for item in menu_data}
        self.embeddings = embeddings
        self.max_seq_length = max_seq_length
        self.num_negatives = num_negatives

        self.category_items: Dict[str, List[str]] = {}
        for item in menu_data:
            cat = item["category"]
            if cat not in self.category_items:
                self.category_items[cat] = []
            self.category_items[cat].append(item["item_id"])

        self.all_item_ids = list(self.embeddings.keys())
        self.embedding_dim = next(iter(self.embeddings.values())).shape[0]

    def __len__(self) -> int:
        return len(self.sessions) * (1 + self.num_negatives)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = idx // (1 + self.num_negatives)
        is_positive = (idx % (1 + self.num_negatives)) == 0
        row = self.sessions.iloc[sample_idx]

        cart_ids = row["sequence_item_ids"].split("|")

        cart_embeddings = []
        padding_mask = []
        for i in range(self.max_seq_length):
            if i < len(cart_ids) and cart_ids[i] in self.embeddings:
                cart_embeddings.append(self.embeddings[cart_ids[i]])
                padding_mask.append(False)
            else:
                cart_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                padding_mask.append(True)

        cart_tensor = torch.tensor(np.array(cart_embeddings), dtype=torch.float32)
        mask_tensor = torch.tensor(padding_mask, dtype=torch.bool)

        target_id = row["target_item_id"]
        if is_positive:
            target_emb = self.embeddings.get(target_id, np.zeros(self.embedding_dim, dtype=np.float32))
            label = 1.0
        else:
            neg_id = self._sample_negative(target_id, cart_ids)
            target_emb = self.embeddings.get(neg_id, np.zeros(self.embedding_dim, dtype=np.float32))
            label = 0.0

        target_tensor = torch.tensor(target_emb, dtype=torch.float32)

        hour = row.get("hour", 12)
        dow = row.get("day_of_week", 0)
        temporal = torch.tensor(encode_temporal_features(int(hour), int(dow)), dtype=torch.float32)

        geohash = row.get("geohash", "tdr1y")
        geohash_bucket = torch.tensor(geohash_to_bucket(str(geohash)), dtype=torch.long)

        return {
            "cart_embeddings": cart_tensor,
            "target_embedding": target_tensor,
            "padding_mask": mask_tensor,
            "temporal_features": temporal,
            "geohash_bucket": geohash_bucket,
            "label": torch.tensor(label, dtype=torch.float32),
        }

    def _sample_negative(self, positive_id: str, cart_ids: List[str]) -> str:
        exclude = set(cart_ids + [positive_id])
        if random.random() < 0.5 and positive_id in self.menu_data:
            target_cat = self.menu_data[positive_id]["category"]
            candidates = [iid for iid in self.category_items.get(target_cat, []) if iid not in exclude]
            if candidates:
                return random.choice(candidates)
        candidates = [iid for iid in self.all_item_ids if iid not in exclude]
        if candidates:
            return random.choice(candidates)
        return random.choice(self.all_item_ids)
