import numpy as np
from PIL import Image

from cachevista.core import CacheVista, generate_with_strategy
from cachevista.encoder import CLIPEncoder

CAT = "data/test_cat.jpg"
DOG = "data/test_dog.jpg"


def load(path):
    return Image.open(path).convert("RGB")


def crop(path, pct=0.05):
    img = load(path)
    w, h = img.size
    dx, dy = int(w * pct), int(h * pct)
    return img.crop((dx, dy, w - dx, h - dy))


def test_l1_hit():
    encoder = CLIPEncoder()
    cache = CacheVista()
    emb = encoder.encode(load(CAT))
    _, s1 = generate_with_strategy(cache, emb)
    _, s2 = generate_with_strategy(cache, emb)
    assert s1 == "MISS" and s2 == "HIT"
    assert cache.hits_l1 == 1 and cache.hits_l2 == 0
    print(f"\nL1 hit. stats={cache.stats()}")


def test_l2_hit_on_cropped_image():
    encoder = CLIPEncoder()
    cache = CacheVista(threshold=0.90)
    emb_orig = encoder.encode(load(CAT))
    generate_with_strategy(cache, emb_orig)
    emb_crop = encoder.encode(crop(CAT, pct=0.05))
    sim = float(np.dot(emb_orig, emb_crop))
    print(f"\noriginal vs cropped similarity: {sim:.4f}")
    _, status = generate_with_strategy(cache, emb_crop)
    assert sim > 0.90, f"similarity too low: {sim:.4f}"
    assert status == "HIT" and cache.hits_l2 == 1
    print(f"L2 semantic hit. stats={cache.stats()}")


def test_l2_miss_on_different_image():
    encoder = CLIPEncoder()
    cache = CacheVista(threshold=0.90)
    emb_cat = encoder.encode(load(CAT))
    generate_with_strategy(cache, emb_cat)
    emb_dog = encoder.encode(load(DOG))
    sim = float(np.dot(emb_cat, emb_dog))
    print(f"\ncat vs dog similarity: {sim:.4f}")
    _, status = generate_with_strategy(cache, emb_dog)
    assert status == "MISS" and cache.hits_l2 == 0
    print(f"correctly missed. stats={cache.stats()}")


def test_stats():
    encoder = CLIPEncoder()
    cache = CacheVista()
    emb_cat = encoder.encode(load(CAT))
    emb_dog = encoder.encode(load(DOG))
    generate_with_strategy(cache, emb_cat)
    generate_with_strategy(cache, emb_cat)
    generate_with_strategy(cache, emb_dog)
    s = cache.stats()
    print(f"\nstats={s}")
    assert s["misses"] == 2
    assert s["hits_l1"] == 1
    assert s["index_size"] == 2
    assert abs(s["hit_rate"] - 1/3) < 0.01
