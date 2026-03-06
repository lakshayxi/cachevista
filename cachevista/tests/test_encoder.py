import numpy as np
from PIL import Image
from cachevista.encoder import CLIPEncoder
from cachevista.core import CacheVista, generate_with_strategy

CAT_IMG = "data/test_cat.jpg"
DOG_IMG = "data/test_dog.jpg"

def get_test_image(path):
    return Image.open(path).convert("RGB")

def test_embedding_shape():
    encoder = CLIPEncoder()
    embedding = encoder.encode(get_test_image(CAT_IMG))
    assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
    print(f"\nEmbedding shape: {embedding.shape}")

def test_embedding_is_normalized():
    encoder = CLIPEncoder()
    embedding = encoder.encode(get_test_image(CAT_IMG))
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5, f"Expected norm=1.0, got {norm}"
    print(f"\nEmbedding norm: {norm:.6f}")

def test_same_image_cache_hit():
    encoder = CLIPEncoder()
    cache = CacheVista()
    img = get_test_image(CAT_IMG)
    emb1 = encoder.encode(img)
    emb2 = encoder.encode(img)
    _, s1 = generate_with_strategy(cache, emb1)
    _, s2 = generate_with_strategy(cache, emb2)
    assert s1 == "MISS" and s2 == "HIT"
    print(f"\nSame image: {s1} -> {s2}")

def test_different_images_cache_miss():
    encoder = CLIPEncoder()
    cache = CacheVista()
    emb_cat = encoder.encode(get_test_image(CAT_IMG))
    emb_dog = encoder.encode(get_test_image(DOG_IMG))
    _, s1 = generate_with_strategy(cache, emb_cat)
    _, s2 = generate_with_strategy(cache, emb_dog)
    assert s1 == "MISS" and s2 == "MISS"
    print(f"\nDifferent images: {s1} -> {s2}")

def test_embedding_similarity():
    encoder = CLIPEncoder()
    img = get_test_image(CAT_IMG)
    emb1 = encoder.encode(img)
    emb2 = encoder.encode(img)
    similarity = np.dot(emb1, emb2)
    assert similarity > 0.9999, f"Expected ~1.0, got {similarity}"
    print(f"\nCosine similarity: {similarity:.6f}")
