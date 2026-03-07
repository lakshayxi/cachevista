import numpy as np
import pytest
from PIL import Image
from cachevista.config import load
from cachevista.core import CacheVista, generate_with_strategy, hash_query
from cachevista.encoder import get_encoder
from cachevista.mlp import load_model, _build_features

cfg = load()
encoder = get_encoder()


def _joint(path, question="what is in this image?"):
    return encoder.encode_joint(Image.open(path).convert("RGB"), question)


def _qhash(path, question="what is in this image?"):
    with open(path, "rb") as f:
        return hash_query(f.read(), question)


def test_mlp_no_drift_score():
    mlp = load_model(cfg["model"]["mlp_path"])
    emb = _joint("data/test_cat.jpg")
    score = mlp.predict(emb, emb)
    assert score < 0.5, f"same joint emb should be no-drift, got {score:.3f}"


def test_mlp_drift_score():
    mlp = load_model(cfg["model"]["mlp_path"])
    q = "what is in this image?"
    emb_cat = _joint("data/test_cat.jpg", q)
    emb_dog = _joint("data/test_dog.jpg", q)
    score = mlp.predict(emb_cat, emb_dog)
    assert score > 0.5, f"different images should be drift, got {score:.3f}"


def test_mlp_hard_negative_drift():
    # same image, different question — the core hard negative case
    mlp = load_model(cfg["model"]["mlp_path"])
    img = Image.open("data/test_cat.jpg").convert("RGB")
    emb_describe = encoder.encode_joint(img, "what is in this image?")
    emb_count    = encoder.encode_joint(img, "how many objects are in this image?")
    score = mlp.predict(emb_describe, emb_count)
    assert score > 0.5, f"same image diff question should be drift, got {score:.3f}"


def test_l3_blocks_drift():
    # threshold=0.50 means cat/dog will trigger L2 lookup
    # MLP should then correctly reject the candidate as drift
    mlp = load_model(cfg["model"]["mlp_path"])
    cache = CacheVista(threshold=0.50, mlp=mlp)

    q = "what is in this image?"
    emb_cat = _joint("data/test_cat.jpg", q)
    emb_dog = _joint("data/test_dog.jpg", q)
    h_cat = _qhash("data/test_cat.jpg", q)
    h_dog = _qhash("data/test_dog.jpg", q)

    generate_with_strategy(cache, h_cat, emb_cat)
    generate_with_strategy(cache, h_dog, emb_dog)

    stats = cache.stats()
    # MLP correctly fired at least once to reject a drift candidate
    assert stats["drift_rejections"] >= 1, \
        f"expected MLP to block drift, got {stats['drift_rejections']} rejections"


def test_predict_features_shape():
    emb = _joint("data/test_cat.jpg")
    feats = _build_features(emb, emb)
    assert feats.shape == (encoder.joint_dim * 2 + 1,)


def test_model_input_dim_saved():
    mlp = load_model(cfg["model"]["mlp_path"])
    assert mlp.input_dim == encoder.joint_dim * 2 + 1