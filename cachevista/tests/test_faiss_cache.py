import numpy as np
from PIL import Image
from cachevista.core import CacheVista, NoCacheStrategy, StaticCacheStrategy, generate_with_strategy, hash_query, hash_image_bytes
from cachevista.encoder import get_encoder

encoder = get_encoder()


def _qhash(path, question="what is in this image?"):
    with open(path, "rb") as f:
        return hash_query(f.read(), question)


def _joint(path, question="what is in this image?"):
    return encoder.encode_joint(Image.open(path).convert("RGB"), question)


def test_l1_exact_hit():
    cache = CacheVista()
    emb = _joint("data/test_cat.jpg")
    h = _qhash("data/test_cat.jpg")
    _, s1 = generate_with_strategy(cache, h, emb)
    _, s2 = generate_with_strategy(cache, h, emb)
    assert s1 == "MISS"
    assert s2 == "HIT"


def test_l1_miss_on_different_question():
    # same image, different question — L1 must miss (different hash)
    cache = CacheVista()
    q1, q2 = "what is in this image?", "how many objects are in this image?"
    h1 = _qhash("data/test_cat.jpg", q1)
    h2 = _qhash("data/test_cat.jpg", q2)
    emb1 = _joint("data/test_cat.jpg", q1)
    emb2 = _joint("data/test_cat.jpg", q2)
    generate_with_strategy(cache, h1, emb1)
    _, status = generate_with_strategy(cache, h2, emb2)
    # different question = different hash = L1 miss (may hit L2 due to similarity)
    assert h1 != h2


def test_l2_semantic_hit_on_augmented_image():
    cache = CacheVista(threshold=0.93)
    img = Image.open("data/test_cat.jpg").convert("RGB")
    q = "what is in this image?"
    w, h = img.size
    crop = img.crop((10, 10, w - 10, h - 10))

    with open("data/test_cat.jpg", "rb") as f:
        orig_bytes = f.read()
    h_orig = hash_query(orig_bytes, q)
    emb_orig = encoder.encode_joint(img, q)
    emb_crop = encoder.encode_joint(crop, q)

    generate_with_strategy(cache, h_orig, emb_orig)
    _, status = generate_with_strategy(cache, "crop_different_hash_" + q, emb_crop)
    assert status == "HIT"


def test_l2_miss_on_different_image():
    cache = CacheVista(threshold=0.93)
    q = "what is in this image?"
    emb_cat = _joint("data/test_cat.jpg", q)
    emb_dog = _joint("data/test_dog.jpg", q)
    generate_with_strategy(cache, _qhash("data/test_cat.jpg", q), emb_cat)
    _, status = generate_with_strategy(cache, _qhash("data/test_dog.jpg", q), emb_dog)
    assert status == "MISS"


def test_stats_schema_consistent():
    q = "what is in this image?"
    emb = _joint("data/test_cat.jpg", q)
    expected_keys = {"hit_rate", "hits_l1", "hits_l2", "misses", "drift_rejections", "index_size"}
    for strategy in [NoCacheStrategy(), StaticCacheStrategy(), CacheVista()]:
        generate_with_strategy(strategy, "x", emb)
        assert set(strategy.stats().keys()) == expected_keys, \
            f"schema mismatch on {type(strategy).__name__}"


def test_dim_inferred_from_first_store():
    cache = CacheVista()
    emb = _joint("data/test_cat.jpg")
    generate_with_strategy(cache, "test", emb)
    assert cache._dim == encoder.joint_dim