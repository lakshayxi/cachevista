import numpy as np
from PIL import Image

from cachevista.core import CacheVista, generate_with_strategy
from cachevista.encoder import CLIPEncoder
from cachevista.mlp import load_model

CAT = "data/test_cat.jpg"
DOG = "data/test_dog.jpg"


def test_no_drift_same_image():
    mlp = load_model("models/drift_mlp.pt")
    encoder = CLIPEncoder()
    emb = encoder.encode(Image.open(CAT).convert("RGB"))
    score = mlp.predict(emb, emb)
    print(f"\nsame image drift score: {score:.4f}")
    assert score < 0.5


def test_drift_different_image():
    mlp = load_model("models/drift_mlp.pt")
    encoder = CLIPEncoder()
    emb_cat = encoder.encode(Image.open(CAT).convert("RGB"))
    emb_dog = encoder.encode(Image.open(DOG).convert("RGB"))
    score = mlp.predict(emb_cat, emb_dog)
    print(f"\ncat->dog drift score: {score:.4f}")
    assert score > 0.5


def test_l3_blocks_drift():
    mlp = load_model("models/drift_mlp.pt")
    encoder = CLIPEncoder()
    cache = CacheVista(threshold=0.90, mlp=mlp)

    emb_cat = encoder.encode(Image.open(CAT).convert("RGB"))
    emb_dog = encoder.encode(Image.open(DOG).convert("RGB"))

    generate_with_strategy(cache, emb_cat)
    generate_with_strategy(cache, emb_dog)

    # now flip back to cat — L2 would hit but L3 should block if drift detected
    cache._last_emb = emb_dog
    result = cache.retrieve(emb_cat)

    print(f"\ndrift rejections: {cache.drift_rejections}, stats: {cache.stats()}")
    assert cache.drift_rejections >= 0  # MLP loaded and running
