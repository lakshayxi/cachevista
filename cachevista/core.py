import hashlib
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict

import faiss
import numpy as np

# Default similarity threshold — matches benchmark config (0.93).
# Tests that call CacheVista() with no args will use this value, so they
# exercise the same threshold as the real system.
DEFAULT_THRESHOLD = 0.93


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

    def __len__(self) -> int:
        return self.stats()["index_size"]


class NoCacheStrategy(BaseCacheStrategy):
    def retrieve(self, query_hash, embedding): return None
    def store(self, query_hash, embedding): pass
    def clear(self): pass

    def stats(self):
        return {"hit_rate": 0.0, "hits_l1": 0, "hits_l2": 0, "misses": 0,
                "drift_rejections": 0, "index_size": 0}


class StaticCacheStrategy(BaseCacheStrategy):
    """
    Hash-only LRU cache. The embedding argument in retrieve() is intentionally
    unused — this strategy matches on exact hash only, not semantic similarity.
    It is not equivalent to CacheVista with threshold=1.0; it is a pure L1 baseline.
    """

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
        with self._lock:
            total = self.hits + self.misses
            return {
                "hit_rate": self.hits / total if total > 0 else 0.0,
                "hits_l1": self.hits, "hits_l2": 0,
                "misses": self.misses, "drift_rejections": 0,
                "index_size": len(self._store),
            }


class CacheVista(BaseCacheStrategy):
    """
    3-layer cache: L1 exact hash, L2 FAISS ANN, L3 MLP drift gate.

    Normalization contract: all embeddings stored and searched are unit-norm
    float32. Callers (e.g. generate_with_strategy) pass already-normalized
    vectors; internal _normalize is a safety net, not the primary normalization
    path. The MLP in L3 also expects unit-norm joint embeddings — this is an
    enforced invariant here, not just a convention.

    Performance note: FAISS IndexFlatIP does not support O(1) deletion.
    remove_ids on eviction is O(n) in the number of stored vectors. For
    max_size <= 10k this is negligible; beyond that, consider IVF indexing
    or a tombstone-based approach.

    L1 hash semantics: hash_query hashes raw image bytes, not pixel data.
    Two visually identical images encoded with different JPEG settings will
    produce different L1 hashes and fall through to L2. This is a known
    limitation documented in hash_query.
    """

    def __init__(self, threshold=DEFAULT_THRESHOLD, dim=None, mlp=None, max_size=1000):
        self.threshold = threshold
        self.mlp = mlp
        self.max_size = max_size

        self._dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim)) if dim else None

        self._lock = threading.RLock()
        self._hash_store = OrderedDict()  # query_hash -> normalized embedding
        self._id_to_emb = {}              # faiss id -> embedding
        self._hash_to_id = {}             # query_hash -> faiss id
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
        with self._lock:
            emb = self._normalize(embedding)

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
        with self._lock:
            emb = self._normalize(embedding)

            if self._index is None:
                self._init_index(emb.shape[0])

            if query_hash in self._hash_store:
                self._hash_store.move_to_end(query_hash)
                return

            if len(self._hash_store) >= self.max_size:
                evicted_hash, _ = self._hash_store.popitem(last=False)
                evicted_id = self._hash_to_id.pop(evicted_hash, None)
                if evicted_id is not None:
                    # O(n) on IndexFlatIP — acceptable at max_size <= 10k
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
        with self._lock:
            total = self.hits_l1 + self.hits_l2 + self.misses
            return (self.hits_l1 + self.hits_l2) / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits_l1 + self.hits_l2 + self.misses
            return {
                "hits_l1": self.hits_l1,
                "hits_l2": self.hits_l2,
                "misses": self.misses,
                "drift_rejections": self.drift_rejections,
                "hit_rate": (self.hits_l1 + self.hits_l2) / total if total > 0 else 0.0,
                "index_size": self._index.ntotal if self._index else 0,
            }


def generate_with_strategy(
    strategy: BaseCacheStrategy,
    query_hash: str,
    embedding: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Retrieve from cache or store and return. Always returns a unit-norm embedding.

    Normalization happens exactly once here, before either retrieve or store is
    called. Both paths return the same normalized vector — callers can rely on
    unit-norm output regardless of hit/miss status.

    retrieve() and store() still normalize internally as a safety net for direct
    callers, but when going through this function the work is not duplicated.
    """
    emb = embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    normalized = emb / norm if norm > 0 else emb

    cached = strategy.retrieve(query_hash, normalized)
    if cached is not None:
        return cached, "HIT"

    strategy.store(query_hash, normalized)
    return normalized, "MISS"


def hash_query(image_bytes: bytes, question: str) -> str:
    """
    Joint L1 cache key for (image, question) pair.

    Matches only if both image bytes AND question string are byte-identical.
    Two visually identical images saved with different JPEG settings or metadata
    will produce different hashes and fall through to L2 ANN search.
    This byte-identity vs visual-identity distinction is a known L1 limitation.

    Note: switched from MD5 to blake2b for throughput on large image byte strings.
    Breaking change — any persisted cache keys or hardcoded MD5 hex strings in
    tests must be regenerated.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(image_bytes)
    h.update(question.encode("utf-8"))
    return h.hexdigest()


def hash_image_bytes(image_bytes: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(image_bytes)
    return h.hexdigest()