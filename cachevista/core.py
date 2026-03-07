import hashlib
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

import faiss
import numpy as np


class BaseCacheStrategy(ABC):
    @abstractmethod
    def retrieve(self, query_hash: str, embedding: np.ndarray) -> np.ndarray | None:
        pass

    @abstractmethod
    def store(self, query_hash: str, embedding: np.ndarray) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def stats(self) -> dict:
        pass


class NoCacheStrategy(BaseCacheStrategy):
    def retrieve(self, query_hash, embedding): return None
    def store(self, query_hash, embedding): pass
    def clear(self): pass

    def stats(self):
        return {"hit_rate": 0.0, "hits_l1": 0, "hits_l2": 0, "misses": 0,
                "drift_rejections": 0, "index_size": 0}


class StaticCacheStrategy(BaseCacheStrategy):
    def __init__(self, max_size=1000):
        self._store = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def retrieve(self, query_hash, embedding):
        with self._lock:
            if query_hash in self._store:
                self._store.move_to_end(query_hash)
                self.hits += 1
                return self._store[query_hash]
            self.misses += 1
            return None

    def store(self, query_hash, embedding):
        with self._lock:
            if query_hash in self._store:
                self._store.move_to_end(query_hash)
                return
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[query_hash] = embedding

    def clear(self):
        with self._lock:
            self._store.clear()
            self.hits = self.misses = 0

    def stats(self):
        total = self.hits + self.misses
        return {
            "hit_rate": self.hits / total if total > 0 else 0.0,
            "hits_l1": self.hits, "hits_l2": 0,
            "misses": self.misses, "drift_rejections": 0,
            "index_size": len(self._store),
        }


class CacheVista(BaseCacheStrategy):
    def __init__(self, threshold=0.90, dim=None, mlp=None, max_size=1000):
        self.threshold = threshold
        self.mlp = mlp
        self.max_size = max_size

        self._dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim)) if dim else None

        self._lock = threading.RLock()
        self._hash_store = OrderedDict()   # query_hash -> normalized embedding
        self._id_to_emb = {}               # faiss id -> embedding
        self._hash_to_id = {}              # query_hash -> faiss id
        self._next_id = 0

        self.hits_l1 = 0
        self.hits_l2 = 0
        self.misses = 0
        self.drift_rejections = 0

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _init_index(self, dim: int):
        self._dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def retrieve(self, query_hash: str, embedding: np.ndarray) -> np.ndarray | None:
        emb = self._normalize(embedding)

        with self._lock:
            if query_hash in self._hash_store:
                self._hash_store.move_to_end(query_hash)
                self.hits_l1 += 1
                return self._hash_store[query_hash]

            if self._index is not None and self._index.ntotal > 0:
                sims, ids = self._index.search(emb.reshape(1, -1), k=1)
                sim = float(sims[0][0])
                candidate_id = int(ids[0][0])

                if sim >= self.threshold and candidate_id in self._id_to_emb:
                    candidate = self._id_to_emb[candidate_id]

                    if self.mlp is not None:
                        if self.mlp.predict(emb, candidate) > 0.5:
                            self.drift_rejections += 1
                            self.misses += 1
                            return None

                    self.hits_l2 += 1
                    return candidate

            self.misses += 1
            return None

    def store(self, query_hash: str, embedding: np.ndarray) -> None:
        emb = self._normalize(embedding)

        with self._lock:
            if self._index is None:
                self._init_index(emb.shape[0])

            if query_hash in self._hash_store:
                self._hash_store.move_to_end(query_hash)
                return

            if len(self._hash_store) >= self.max_size:
                evicted_hash, _ = self._hash_store.popitem(last=False)
                evicted_id = self._hash_to_id.pop(evicted_hash, None)
                if evicted_id is not None:
                    self._index.remove_ids(np.array([evicted_id], dtype=np.int64))
                    del self._id_to_emb[evicted_id]

            vec_id = self._next_id
            self._next_id += 1

            self._hash_store[query_hash] = emb
            self._hash_to_id[query_hash] = vec_id
            self._index.add_with_ids(emb.reshape(1, -1), np.array([vec_id], dtype=np.int64))
            self._id_to_emb[vec_id] = emb

    def clear(self) -> None:
        with self._lock:
            self._hash_store.clear()
            self._hash_to_id.clear()
            self._id_to_emb.clear()
            self._next_id = 0
            if self._dim:
                self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))
            else:
                self._index = None
            self.hits_l1 = self.hits_l2 = self.misses = self.drift_rejections = 0

    def hit_rate(self) -> float:
        total = self.hits_l1 + self.hits_l2 + self.misses
        return (self.hits_l1 + self.hits_l2) / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits_l1": self.hits_l1,
            "hits_l2": self.hits_l2,
            "misses": self.misses,
            "drift_rejections": self.drift_rejections,
            "hit_rate": self.hit_rate(),
            "index_size": self._index.ntotal if self._index else 0,
        }


def generate_with_strategy(
    strategy: BaseCacheStrategy,
    query_hash: str,
    embedding: np.ndarray,
):
    cached = strategy.retrieve(query_hash, embedding)
    if cached is not None:
        return cached, "HIT"
    strategy.store(query_hash, embedding)
    return embedding, "MISS"


def hash_query(image_bytes: bytes, question: str) -> str:
    """Joint cache key — matches only if both image AND question are identical."""
    h = hashlib.md5(image_bytes)
    h.update(question.encode("utf-8"))
    return h.hexdigest()


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()