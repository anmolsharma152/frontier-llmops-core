"""Semantic response cache.

Stores LLM responses keyed by embedding vectors. Supports
exact match and fuzzy (semantic similarity) lookups with a
configurable threshold. Falls back to in-memory dict when
Redis is unavailable.
"""

import json
import time
import hashlib
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(self, config: dict):
        self.backend = config.get("backend", "memory")
        self.ttl = config.get("ttl", 3600)
        self.threshold = config.get("similarity_threshold", 0.92)
        self.max_size = config.get("max_size", 10000)

        model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
        self.embedder = SentenceTransformer(model_name)

        self._redis = None
        if self.backend == "redis":
            try:
                import redis
                self._redis = redis.Redis.from_url(config.get("redis_url", "redis://localhost:6379/0"))
                self._redis.ping()
            except Exception as e:
                print(f"[WARN] Redis unavailable ({e}), falling back to in-memory cache")
                self.backend = "memory"

        if self.backend == "memory":
            self._store: dict[str, dict] = {}
            self._embeddings: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, prompt: str) -> tuple[bool, Any]:
        prompt_emb = self._embed(prompt)

        if self.backend == "redis":
            return self._redis_get(prompt_emb)

        best_key, best_sim = None, 0.0
        for key, emb in self._embeddings.items():
            sim = float(np.dot(prompt_emb, emb))
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key and best_sim >= self.threshold:
            entry = self._store.get(best_key)
            if entry and (entry["expires"] > time.time()):
                return True, entry["response"]
        return False, None

    def _redis_get(self, prompt_emb: np.ndarray) -> tuple[bool, Any]:
        keys = self._redis.keys("cache:*")
        best_sim, best_data = 0.0, None
        for key in keys:
            stored_emb = np.frombuffer(self._redis.get(key), dtype=np.float32)
            sim = float(np.dot(prompt_emb, stored_emb))
            if sim > best_sim:
                best_sim = sim
                best_data = self._redis.get(f"{key.decode()}:data")

        if best_sim >= self.threshold and best_data:
            return True, json.loads(best_data)
        return False, None

    def set(self, prompt: str, response: Any) -> None:
        emb = self._embed(prompt)
        key = self._key(prompt)
        expires = time.time() + self.ttl

        if self.backend == "redis":
            self._redis.setex(f"cache:{key}", self.ttl, emb.tobytes())
            self._redis.setex(f"cache:{key}:data", self.ttl, json.dumps(response))
        else:
            if len(self._store) >= self.max_size:
                oldest = min(self._store.keys(), key=lambda k: self._store[k]["created"])
                del self._store[oldest]
                del self._embeddings[oldest]
            self._store[key] = {"response": response, "expires": expires, "created": time.time()}
            self._embeddings[key] = emb

    def stats(self) -> dict:
        if self.backend == "redis":
            keys = self._redis.keys("cache:*data")
            return {"backend": "redis", "entries": len(keys) // 2}
        return {
            "backend": "memory",
            "entries": len(self._store),
            "max_size": self.max_size,
        }
