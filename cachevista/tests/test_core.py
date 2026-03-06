import numpy as np
from cachevista.core import NoCacheStrategy, StaticCacheStrategy, CacheVista, generate_with_strategy

emb = np.random.rand(768).astype(np.float32)

def test_no_cache_always_misses():
    strategy = NoCacheStrategy()
    _, status = generate_with_strategy(strategy, emb)
    assert status == "MISS"

def test_static_cache_hits_on_second_call():
    strategy = StaticCacheStrategy()
    _, s1 = generate_with_strategy(strategy, emb)
    _, s2 = generate_with_strategy(strategy, emb)
    assert s1 == "MISS" and s2 == "HIT"

def test_cachevista_hit_rate():
    cv = CacheVista()
    generate_with_strategy(cv, emb)
    generate_with_strategy(cv, emb)
    assert cv.hit_rate() == 0.5