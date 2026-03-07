import numpy as np
from PIL import Image
from cachevista.core import CacheVista, generate_with_strategy, hash_image_bytes
from cachevista.encoder import get_encoder

encoder = get_encoder()


def _hash(path):
    with open(path, "rb") as f:
        return hash_image_bytes(f.read())


def test_embedding_shape():
    emb = encoder.encode(Image.open("data/test_cat.jpg").convert("RGB"))
    assert emb.shape == (encoder.image_dim,)


def test_embedding_normalized():
    emb = encoder.encode(Image.open("data/test_cat.jpg").convert("RGB"))
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_joint_embedding_shape():
    emb = encoder.encode_joint(Image.open("data/test_cat.jpg").convert("RGB"), "what is in this image?")
    assert emb.shape == (encoder.joint_dim,)


def test_joint_embedding_normalized():
    emb = encoder.encode_joint(Image.open("data/test_cat.jpg").convert("RGB"), "what color is it?")
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_same_image_same_question_close():
    img = Image.open("data/test_cat.jpg").convert("RGB")
    q = "what is in this image?"
    e1 = encoder.encode_joint(img, q)
    e2 = encoder.encode_joint(img, q)
    assert np.allclose(e1, e2, atol=1e-5)


def test_same_image_different_question_differs():
    img = Image.open("data/test_cat.jpg").convert("RGB")
    e1 = encoder.encode_joint(img, "what is in this image?")
    e2 = encoder.encode_joint(img, "how many objects are in this image?")
    sim = float(np.dot(e1, e2))
    # should be similar but not identical — question changes the joint embedding
    assert sim < 0.999
    assert sim > 0.5


def test_singleton_returns_same_instance():
    from cachevista.encoder import get_encoder
    enc1 = get_encoder()
    enc2 = get_encoder()
    assert enc1 is enc2


def test_repr():
    assert "CLIPEncoder" in repr(encoder)
    assert str(encoder.image_dim) in repr(encoder)


def test_cache_hit_same_image_same_question():
    cache = CacheVista()
    img = Image.open("data/test_cat.jpg").convert("RGB")
    q = "what is in this image?"
    from cachevista.core import hash_query
    with open("data/test_cat.jpg", "rb") as f:
        img_bytes = f.read()
    qhash = hash_query(img_bytes, q)
    emb = encoder.encode_joint(img, q)
    _, s1 = generate_with_strategy(cache, qhash, emb)
    _, s2 = generate_with_strategy(cache, qhash, emb)
    assert s1 == "MISS"
    assert s2 == "HIT"


def test_encode_batch_shape():
    imgs = [Image.open("data/test_cat.jpg").convert("RGB"),
            Image.open("data/test_dog.jpg").convert("RGB")]
    embs = encoder.encode_batch(imgs)
    assert embs.shape == (2, encoder.image_dim)


def test_encode_batch_normalized():
    imgs = [Image.open("data/test_cat.jpg").convert("RGB")]
    embs = encoder.encode_batch(imgs)
    assert abs(np.linalg.norm(embs[0]) - 1.0) < 1e-5