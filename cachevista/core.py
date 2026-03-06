"""
CacheVista Core Module
Multimodal Memory Caching for Vision-Language Models
"""

import hashlib
import numpy as np
from abc import ABC, abstractmethod


class BaseCacheStrategy(ABC):
    """All caching strategies must implement this interface."""

    @abstractmethod
    def retrieve(self, image_embedding: np.ndarray) -> np.ndarray | None:
        pass

    @abstractmethod
    def store(self, image_embedding: np.ndarray) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class NoCacheStrategy(BaseCacheStrategy):
    """Baseline: never caches anything."""

    def retrieve(self, image_embedding):
        return None

    def store(self, image_embedding):
        pass

    def clear(self):
        pass


class StaticCacheStrategy(BaseCacheStrategy):
    """Stores everything, never invalidates."""

    def __init__(self):
        self._store = {}

    def retrieve(self, image_embedding):
        return self._store.get(self._hash(image_embedding), None)

    def store(self, image_embedding):
        self._store[self._hash(image_embedding)] = image_embedding

    def clear(self):
        self._store.clear()

    def _hash(self, embedding):
        return hashlib.md5(embedding.tobytes()).hexdigest()


class CacheVista(BaseCacheStrategy):
    """
    Hybrid 3-layer cache:
      Layer 1: Exact hash match
      Layer 2: FAISS semantic similarity  (Day 6)
      Layer 3: MLP drift predictor        (Day 7)
    """

    def __init__(self, similarity_threshold=0.90):
        self.similarity_threshold = similarity_threshold
        self._store = {}
        self.hits = 0
        self.misses = 0

    def retrieve(self, image_embedding):
        key = self._hash(image_embedding)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def store(self, image_embedding):
        self._store[self._hash(image_embedding)] = image_embedding

    def clear(self):
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def cache_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def _hash(self, embedding):
        return hashlib.md5(embedding.tobytes()).hexdigest()


def generate_with_strategy(strategy: BaseCacheStrategy, image_embedding: np.ndarray):
    """Retrieve from cache or store and return the embedding."""
    cached = strategy.retrieve(image_embedding)
    if cached is not None:
        return cached, "HIT"
    strategy.store(image_embedding)
    return image_embedding, "MISS"

