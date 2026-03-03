"""
Real-time inference engine — FAISS ANN search + MMR diversity re-ranking.

Pipeline (within 300ms budget):
  Feature Fetch (<15ms) → Cart Encoding (<80ms) → FAISS Search (<25ms) → MMR Re-rank (<20ms)
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from swaadstack.config import model_config, inference_config, data_config, InferenceConfig, MODEL_DIR
from swaadstack.models import SwaadStackInference, load_model, create_model
from swaadstack.inference.feature_store import FeatureStore
from swaadstack.utils import (
    mmr_rerank, encode_temporal_features, geohash_to_bucket,
    get_mealtime_label, category_diversity_score, logger, timeit,
)


class InferenceEngine:
    """Production inference engine for SwaadStack recommendations."""

    def __init__(self, model_path=None, menu_path=None, embeddings_path=None,
                 id_mapping_path=None, config=None):
        self.config = config or inference_config
        self.model: Optional[SwaadStackInference] = None
        self.faiss_index = None
        self.menu_data: Dict[str, Dict] = {}
        self.raw_embeddings: Dict[str, np.ndarray] = {}
        self.projected_embeddings: Optional[np.ndarray] = None
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.feature_store: Optional[FeatureStore] = None
        self._is_loaded = False

        self._model_path = model_path or str(model_config.model_weights_path)
        self._menu_path = menu_path or str(data_config.menu_file)
        self._embeddings_path = embeddings_path or str(MODEL_DIR / "projected_item_embeddings.npy")
        self._id_mapping_path = id_mapping_path or str(data_config.id_mapping_file)

    def load(self):
        start = time.perf_counter()
        logger.info("inference_engine_loading")
        self._load_menu()
        self._load_model()
        self._build_faiss_index()
        try:
            self.feature_store = FeatureStore()
        except Exception as e:
            logger.warning("feature_store_init_failed", error=str(e))
            self.feature_store = FeatureStore()
        elapsed = (time.perf_counter() - start) * 1000
        self._is_loaded = True
        logger.info("inference_engine_loaded", elapsed_ms=round(elapsed, 2), num_items=len(self.menu_data))

    def _load_menu(self):
        with open(self._menu_path, "r", encoding="utf-8") as f:
            menu_list = json.load(f)
        for item in menu_list:
            self.menu_data[item["item_id"]] = item
            self.raw_embeddings[item["item_id"]] = np.array(item["embedding"], dtype=np.float32)
        with open(self._id_mapping_path, "r") as f:
            self.id_to_idx = json.load(f)
            self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}
        logger.info("menu_loaded", num_items=len(self.menu_data))

    def _load_model(self):
        try:
            model = load_model(self._model_path)
            self.model = SwaadStackInference(model)
            logger.info("model_loaded", path=self._model_path)
        except Exception as e:
            logger.warning("model_load_failed", error=str(e))
            model = create_model()
            self.model = SwaadStackInference(model)
            logger.info("model_created_fresh")

    def _build_faiss_index(self):
        try:
            import faiss
            try:
                self.projected_embeddings = np.load(self._embeddings_path)
            except FileNotFoundError:
                logger.warning("projected_embeddings_not_found", msg="Generating on-the-fly")
                sorted_ids = sorted(self.id_to_idx.keys(), key=lambda x: self.id_to_idx[x])
                raw_matrix = np.array([self.raw_embeddings[iid] for iid in sorted_ids])
                raw_tensor = torch.tensor(raw_matrix, dtype=torch.float32)
                self.projected_embeddings = self.model.encode_items(raw_tensor)

            dim = self.projected_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.projected_embeddings)
            self.faiss_index.add(self.projected_embeddings)
            logger.info("faiss_index_built", num_vectors=self.faiss_index.ntotal, dim=dim)
        except ImportError:
            logger.warning("faiss_not_available", msg="Using numpy fallback")
            try:
                self.projected_embeddings = np.load(self._embeddings_path)
            except FileNotFoundError:
                sorted_ids = sorted(self.id_to_idx.keys(), key=lambda x: self.id_to_idx[x])
                self.projected_embeddings = np.array([self.raw_embeddings[iid] for iid in sorted_ids])
            norms = np.linalg.norm(self.projected_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.projected_embeddings = self.projected_embeddings / norms

    @timeit
    def recommend(self, cart_items, user_id=None, geohash=None, hour=None,
                  day_of_week=None, top_n=None, lambda_mmr=None):
        if not self._is_loaded:
            raise RuntimeError("Inference engine not loaded. Call .load() first.")
        top_n = top_n or self.config.mmr_top_n
        lambda_mmr = lambda_mmr if lambda_mmr is not None else self.config.mmr_lambda
        latency = {}
        start_total = time.perf_counter()

        valid_cart = [iid for iid in cart_items if iid in self.menu_data]
        if not valid_cart:
            return self._cold_start_fallback(geohash, hour, top_n)

        # Feature Fetch
        t0 = time.perf_counter()
        cart_embeddings = [self.raw_embeddings[iid] for iid in valid_cart[:model_config.max_seq_length]]
        padding_mask = []
        while len(cart_embeddings) < model_config.max_seq_length:
            cart_embeddings.append(np.zeros(model_config.input_embedding_dim, dtype=np.float32))
            padding_mask.append(True)
        padding_mask = [False] * len(valid_cart[:model_config.max_seq_length]) + padding_mask[:]
        padding_mask = padding_mask[:model_config.max_seq_length]

        if hour is not None and day_of_week is not None:
            temporal = encode_temporal_features(hour, day_of_week)
        else:
            from datetime import datetime
            now = datetime.now()
            temporal = encode_temporal_features(now.hour, now.weekday())
            hour = now.hour
        geo_bucket = geohash_to_bucket(geohash or "tdr1y")
        latency["feature_fetch_ms"] = (time.perf_counter() - t0) * 1000

        # Cart Encoding
        t1 = time.perf_counter()
        cart_tensor = torch.tensor(np.array(cart_embeddings), dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor([padding_mask], dtype=torch.bool)
        temporal_tensor = torch.tensor(temporal, dtype=torch.float32).unsqueeze(0)
        geo_tensor = torch.tensor([geo_bucket], dtype=torch.long)
        query_vector = self.model.encode_cart(cart_tensor, padding_mask=mask_tensor,
                                              temporal_features=temporal_tensor, geohash_buckets=geo_tensor)
        latency["encoding_ms"] = (time.perf_counter() - t1) * 1000

        # FAISS Search
        t2 = time.perf_counter()
        candidates = self._search_candidates(query_vector, top_k=self.config.faiss_top_k)
        latency["retrieval_ms"] = (time.perf_counter() - t2) * 1000

        # MMR Re-ranking
        t3 = time.perf_counter()
        cart_set = set(valid_cart)
        candidate_vecs, candidate_info = [], []
        for idx, score in candidates:
            item_id = self.idx_to_id.get(idx)
            if item_id and item_id not in cart_set:
                item_meta = self.menu_data.get(item_id, {})
                candidate_vecs.append(self.projected_embeddings[idx])
                candidate_info.append({
                    "item_id": item_id, "name": item_meta.get("name", "Unknown"),
                    "category": item_meta.get("category", "Unknown"),
                    "price": item_meta.get("price", 0), "cuisine": item_meta.get("cuisine", "Unknown"),
                    "dietary": item_meta.get("dietary", []), "retrieval_score": float(score),
                })
        if candidate_vecs:
            reranked = mmr_rerank(query_vec=query_vector.flatten(),
                                  candidate_vecs=np.array(candidate_vecs, dtype=np.float32),
                                  candidate_info=candidate_info, lambda_param=lambda_mmr,
                                  top_n=top_n, exclude_ids=cart_set)
        else:
            reranked = []
        latency["ranking_ms"] = (time.perf_counter() - t3) * 1000
        latency["total_ms"] = (time.perf_counter() - start_total) * 1000

        cart_categories = [self.menu_data[iid]["category"] for iid in valid_cart]
        cart_total = sum(self.menu_data[iid].get("price", 0) for iid in valid_cart)
        return {
            "recommendations": reranked,
            "cart_summary": {
                "items": valid_cart, "item_count": len(valid_cart), "total_value": cart_total,
                "categories": cart_categories, "diversity_score": category_diversity_score(cart_categories),
                "missing_categories": self._find_missing_categories(cart_categories),
            },
            "context": {"mealtime": get_mealtime_label(hour or 12), "geohash": geohash, "personalized": user_id is not None},
            "latency": {k: round(v, 2) for k, v in latency.items()},
            "metadata": {"candidates_retrieved": len(candidates), "candidates_after_filter": len(candidate_info),
                         "mmr_lambda": lambda_mmr, "model_version": "swaadstack-v1.0"},
        }

    def _search_candidates(self, query_vector, top_k=20):
        query = query_vector.reshape(1, -1).astype(np.float32)
        if self.faiss_index is not None:
            import faiss
            faiss.normalize_L2(query)
            scores, indices = self.faiss_index.search(query, top_k)
            return list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            scores = (self.projected_embeddings @ query_norm.T).flatten()
            top_indices = np.argsort(-scores)[:top_k]
            return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _cold_start_fallback(self, geohash, hour, top_n):
        mealtime = get_mealtime_label(hour or 12)
        if self.feature_store and geohash:
            popular = self.feature_store.get_popularity(geohash, mealtime)
            if popular:
                return {
                    "recommendations": popular[:top_n],
                    "cart_summary": {"items": [], "item_count": 0, "total_value": 0},
                    "context": {"mealtime": mealtime, "geohash": geohash, "personalized": False, "fallback": "contextual_popularity"},
                    "latency": {"total_ms": 0},
                    "metadata": {"note": "Cold-start: contextual popularity fallback"},
                }
        popular_items = []
        for cat in ["Main", "Beverage", "Side", "Dessert"]:
            cat_items = [item for item in self.menu_data.values() if item.get("category") == cat]
            if cat_items:
                cat_items.sort(key=lambda x: abs(x.get("price", 200) - 200))
                popular_items.append({
                    "item_id": cat_items[0]["item_id"], "name": cat_items[0]["name"],
                    "category": cat_items[0]["category"], "price": cat_items[0].get("price", 0),
                    "cuisine": cat_items[0].get("cuisine", ""), "mmr_score": 0.5, "relevance_score": 0.5,
                })
        return {
            "recommendations": popular_items[:top_n],
            "cart_summary": {"items": [], "item_count": 0, "total_value": 0},
            "context": {"mealtime": mealtime, "geohash": geohash, "personalized": False, "fallback": "default_popularity"},
            "latency": {"total_ms": 0},
            "metadata": {"note": "Cold-start: default popularity fallback"},
        }

    def _find_missing_categories(self, cart_categories):
        return sorted({"Main", "Side", "Beverage", "Dessert"} - set(cart_categories))

    def health_check(self):
        return {
            "loaded": self._is_loaded, "model_loaded": self.model is not None,
            "faiss_loaded": self.faiss_index is not None, "num_menu_items": len(self.menu_data),
            "num_indexed_items": self.faiss_index.ntotal if self.faiss_index else 0,
            "feature_store": self.feature_store.health_check() if self.feature_store else None,
        }
