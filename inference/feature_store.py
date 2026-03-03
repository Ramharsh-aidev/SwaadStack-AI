"""
Redis-backed online feature store with graceful in-memory fallback.

Handles: user vectors (24h TTL), cart context (session-based),
popularity rankings (cold-start), and batched pipeline reads (<15ms).
"""

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from swaadstack.config import feature_store_config, FeatureStoreConfig
from swaadstack.utils.logging import logger


class FeatureStore:
    """Redis-backed feature store with in-memory fallback for dev/testing."""

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or feature_store_config
        self.redis_client = None
        self._fallback_store: Dict[str, Any] = {}
        self._connect()

    def _connect(self):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.config.redis_host, port=self.config.redis_port,
                db=self.config.redis_db, password=self.config.redis_password,
                decode_responses=False, socket_timeout=2, socket_connect_timeout=2,
                retry_on_timeout=True,
            )
            self.redis_client.ping()
            logger.info("feature_store_connected", backend="redis",
                        host=self.config.redis_host, port=self.config.redis_port)
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e), fallback="in_memory")
            self.redis_client = None

    @property
    def is_redis_available(self) -> bool:
        if self.redis_client is None:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False

    # ── User Vector Operations ──

    def store_user_vector(self, user_id: str, vector: np.ndarray, metadata: Optional[Dict] = None):
        key = f"{self.config.user_vector_prefix}{user_id}"
        data = {"vector": vector.tobytes(), "dim": str(vector.shape[0]), "updated_at": str(time.time())}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        if self.is_redis_available:
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.config.user_vector_ttl)
            pipe.execute()
        else:
            self._fallback_store[key] = data

    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        key = f"{self.config.user_vector_prefix}{user_id}"
        if self.is_redis_available:
            result = self.redis_client.hgetall(key)
            if result and b"vector" in result:
                dim = int(result[b"dim"])
                return np.frombuffer(result[b"vector"], dtype=np.float32).reshape(dim)
        else:
            if key in self._fallback_store:
                data = self._fallback_store[key]
                dim = int(data["dim"])
                return np.frombuffer(data["vector"], dtype=np.float32).copy().reshape(dim)
        return None

    # ── Cart Context Operations ──

    def store_cart_context(self, session_id: str, cart_embedding: np.ndarray,
                           cart_items: List[str], cart_total: float, diversity_score: float):
        key = f"{self.config.cart_context_prefix}{session_id}"
        data = {
            "embedding": cart_embedding.tobytes(), "dim": str(cart_embedding.shape[0]),
            "items": json.dumps(cart_items), "total": str(cart_total),
            "diversity": str(diversity_score), "item_count": str(len(cart_items)),
            "updated_at": str(time.time()),
        }
        if self.is_redis_available:
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.config.cart_context_ttl)
            pipe.execute()
        else:
            self._fallback_store[key] = data

    def get_cart_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = f"{self.config.cart_context_prefix}{session_id}"
        if self.is_redis_available:
            result = self.redis_client.hgetall(key)
            if result and b"embedding" in result:
                dim = int(result[b"dim"])
                return {
                    "embedding": np.frombuffer(result[b"embedding"], dtype=np.float32).reshape(dim),
                    "items": json.loads(result[b"items"]),
                    "total": float(result[b"total"]),
                    "diversity": float(result[b"diversity"]),
                    "item_count": int(result[b"item_count"]),
                }
        else:
            if key in self._fallback_store:
                data = self._fallback_store[key]
                dim = int(data["dim"])
                return {
                    "embedding": np.frombuffer(data["embedding"], dtype=np.float32).copy().reshape(dim),
                    "items": json.loads(data["items"]),
                    "total": float(data["total"]),
                    "diversity": float(data["diversity"]),
                    "item_count": int(data["item_count"]),
                }
        return None

    # ── Popularity (Cold-Start) ──

    def store_popularity(self, geohash: str, mealtime: str, rankings: List[Dict[str, Any]]):
        key = f"{self.config.popularity_prefix}{geohash}:{mealtime}"
        data = {"rankings": json.dumps(rankings), "updated_at": str(time.time())}
        if self.is_redis_available:
            pipe = self.redis_client.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, self.config.popularity_ttl)
            pipe.execute()
        else:
            self._fallback_store[key] = data

    def get_popularity(self, geohash: str, mealtime: str) -> Optional[List[Dict]]:
        key = f"{self.config.popularity_prefix}{geohash}:{mealtime}"
        if self.is_redis_available:
            result = self.redis_client.hgetall(key)
            if result and b"rankings" in result:
                return json.loads(result[b"rankings"])
        else:
            if key in self._fallback_store:
                return json.loads(self._fallback_store[key]["rankings"])
        return None

    # ── Batch Operations ──

    def get_features_batch(self, user_id: str, session_id: str) -> Dict[str, Any]:
        features = {"user_vector": None, "cart_context": None}
        if self.is_redis_available:
            pipe = self.redis_client.pipeline()
            pipe.hgetall(f"{self.config.user_vector_prefix}{user_id}")
            pipe.hgetall(f"{self.config.cart_context_prefix}{session_id}")
            results = pipe.execute()
            if results[0] and b"vector" in results[0]:
                dim = int(results[0][b"dim"])
                features["user_vector"] = np.frombuffer(results[0][b"vector"], dtype=np.float32).reshape(dim)
            if results[1] and b"embedding" in results[1]:
                dim = int(results[1][b"dim"])
                features["cart_context"] = {
                    "embedding": np.frombuffer(results[1][b"embedding"], dtype=np.float32).reshape(dim),
                    "items": json.loads(results[1][b"items"]),
                }
        else:
            features["user_vector"] = self.get_user_vector(user_id)
            cart_ctx = self.get_cart_context(session_id)
            if cart_ctx:
                features["cart_context"] = cart_ctx
        return features

    def health_check(self) -> Dict[str, Any]:
        return {
            "backend": "redis" if self.is_redis_available else "in_memory",
            "connected": self.is_redis_available,
            "host": self.config.redis_host,
            "port": self.config.redis_port,
            "fallback_entries": len(self._fallback_store),
        }
