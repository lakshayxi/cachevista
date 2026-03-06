import hashlib
from abc import ABC, abstractmethod

import faiss
import numpy as np


class BaseCacheStrategy(ABC):
    @abstractmethod
    def retrieve(self, embedding: np.ndarray) -> np.ndarray | None:
        pass

    @abstractmethod
    def store(self, embedding: np.ndarray) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class NoCacheStrategy(BaseCacheStrategy):
    def retrieve(self, embedding): return None
    def store(self, embedding): pass
    def clear(self): pass


class StaticCacheStrategy(BaseCacheStrategy):
    def __init__(self):
        self._store = {}

    def retrieve(self, embedding):
        return self._store.get(_md5(embedding))

    def store(self, embedding):
        self._store[_md5(embedding)] = embedding

    def clear(self):
        self._store.clear()


class CacheVista(BaseCacheStrategy):
    def __init__(self, threshold=0.90, dim=768):
        self.threshold = threshold
        self.dim = dim
        self._hash_store = {}
        self._index = faiss.IndexFlatIP(dim)
        self._vectors = []
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0

    def retrieve(self, embedding: np.ndarray) -> np.ndarray | None:
        key = _md5(embedding)
        if key in self._hash_store:
            self.hits_l1 += 1
            return self._hash_store[key]

        if self._index.ntotal > 0:
            q = embedding.reshape(1, -1).astype(np.float32)
            sims, idxs = self._index.search(q, k=1)
            if float(sims[0][0]) >= self.threshold:
                self.hits_l2 += 1
                return self._vectors[int(idxs[0][0])]

        self.misses += 1
        return None

    def store(self, embedding: np.ndarray) -> None:
        self._hash_store[_md5(embedding)] = embedding
        self._index.add(embedding.reshape(1, -1).astype(np.float32))
        self._vectors.append(embedding)

    def clear(self) -> None:
        self._hash_store.clear()
        self._index = faiss.IndexFlatIP(self.dim)
        self._vectors.clear()
        self.hits_l1 = self.hits_l2 = self.misses = 0

    def hit_rate(self) -> float:
        total = self.hits_l1 + self.hits_l2 + self.misses
        return (self.hits_l1 + self.hits_l2) / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits_l1": self.hits_l1,
            "hits_l2": self.hits_l2,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
            "index_size": self._index.ntotal,
        }


def generate_with_strategy(strategy: BaseCacheStrategy, embedding: np.ndarray):
    cached = strategy.retrieve(embedding)
    if cached is not None:
        return cached, "HIT"
    strategy.store(embedding)
    return embedding, "MISS"


def _md5(embedding: np.ndarray) -> str:
    return hashlib.md5(embedding.tobytes()).hexdigest()
