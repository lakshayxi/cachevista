"""
Generates training data for the drift MLP using (image, question) joint embeddings.

Label definition — grounded in VQA output equivalence:
  drift=0 (reuse is safe):
      Same image + same question type. BLIP-VQA produces identical answer.
      Joint embeddings are near-identical. Cache hit is valid.

  drift=1, easy (do not reuse):
      Different image + any question. Different content, different answer.
      Joint embeddings are distant. Trivially separable — included for balance.

  drift=1, hard (the critical tier):
      Same image + different question type. CLIP image sim = 1.0 (identical image)
      but the question asks about different aspects → different VQA answer.
      Joint embedding diverges due to question embedding difference.
      FAISS on image-only embedding would wrongly serve cached response.
      This is the boundary L3 exists to detect.

The hard negatives have CLIP_image_sim = 1.0 but joint_sim < 1.0 — a case
FAISS operating on image embeddings alone cannot distinguish from a cache hit.

Class imbalance note: by construction each split has n_easy positives and
2*n_easy negatives (n_easy easy + n_hard hard), a 1:2 ratio. The recommended
fix is to pass pos_weight=torch.tensor([2.0]) to BCEWithLogitsLoss in mlp.py.
The imbalance ratio is logged explicitly at the end of generate().

Test split note: this script generates train and val only. Val is used for
early stopping in mlp.py. Reported eval metrics are therefore validation
metrics, not held-out test metrics. A separate test split should be carved
out before final reporting.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from cachevista.config import load
from cachevista.encoder import get_encoder
from cachevista.mlp import _build_features

QUESTION_TYPES = ["describe", "count", "color", "spatial", "yn"]


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    ops = [
        lambda x: x.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))),
        lambda x: ImageEnhance.Brightness(x).enhance(rng.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Contrast(x).enhance(rng.uniform(0.8, 1.2)),
        lambda x: x.rotate(rng.uniform(-5, 5)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0, 1.5))),
    ]
    for op in rng.sample(ops, k=rng.randint(1, 2)):
        img = op(img)
    return img.convert("RGB")


def generate(n_easy=500, n_hard=500, seed=42, val_split=0.2):
    random.seed(seed)
    np.random.seed(seed)
    # torch seeded only if needed — no torch ops happen in this file directly,
    # and the encoder is already initialized before generate() is called

    cfg = load()
    encoder = get_encoder()
    img_dir = Path(cfg["data"]["coco_dir"])
    data_dir = Path(cfg["data"]["drift_data_dir"])
    questions_path = data_dir / "questions.json"

    if not questions_path.exists():
        raise RuntimeError(
            f"questions not found at {questions_path}\n"
            f"run: PYTHONPATH=. python cachevista/generate_questions.py"
        )

    paths = sorted(img_dir.glob("*.jpg"))
    with open(questions_path) as f:
        _raw = json.load(f)

    # support both old format (flat dict per image) and new format (with top-level
    # "question_templates" and "images" keys from updated generate_questions.py)
    if "images" in _raw:
        qdata = _raw["images"]
        question_templates = _raw.get("question_templates", {})
    else:
        # legacy format: qdata[img_name]["questions"][qtype]
        qdata = _raw
        # legacy format has question strings per image under "questions" key
        # validate that at least the first entry has them
        question_templates = {}
        _sample = next(iter(qdata.values()), {})
        if "questions" not in _sample:
            raise RuntimeError(
                "questions.json is in an unrecognised format — missing both top-level "
                "'question_templates' and per-image 'questions' keys. "
                "Re-run generate_questions.py to regenerate."
            )

    missing = [p.name for p in paths if p.name not in qdata]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing questions — re-run generate_questions.py")

    print(f"found {len(paths)} images with questions")

    # image-level split — prevents same image appearing in both train and val
    all_idx = list(range(len(paths)))
    random.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * val_split))
    val_idx = sorted(all_idx[:n_val])
    train_idx = sorted(all_idx[n_val:])
    print(f"train images: {len(train_idx)}  val images: {len(val_idx)}")

    # Scale pair counts for val proportionally to avoid val image reuse being
    # 5x higher than train (which skews val positive distribution).
    # With 500 images and val_split=0.2: train has 400 images, val has 100.
    # Using n_easy=500 for both means val samples each image ~5x vs ~1.25x for
    # train. Scale val counts to keep per-image reuse rate consistent.
    val_scale = len(val_idx) / max(len(train_idx), 1)
    n_easy_val = max(50, int(n_easy * val_scale))
    n_hard_val = max(50, int(n_hard * val_scale))

    # All images are loaded for encoding then immediately freed.
    # Peak RAM during this block is ~450MB for 500 COCO images.
    # A chunked approach (encode_joint_batch per N images) would reduce peak
    # memory but complicates the reshape — acceptable for a one-time data
    # generation script on a machine with >=2GB free RAM.
    print("encoding joint embeddings for all (image, question) pairs...")
    all_questions = []
    all_images_for_encoding = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        for qtype in QUESTION_TYPES:
            _q = question_templates.get(qtype) or qdata[p.name].get("questions", {}).get(qtype)
            if not _q:
                raise RuntimeError(f"question string missing for type '{qtype}' in image '{p.name}'")
            all_questions.append(_q)
            all_images_for_encoding.append(img)

    joint_embs_flat = encoder.encode_joint_batch(all_images_for_encoding, all_questions)
    del all_images_for_encoding  # free after encoding — joint_embs is all we need

    # reshape to (N_images, N_qtypes, joint_dim)
    joint_embs = joint_embs_flat.reshape(len(paths), len(QUESTION_TYPES), -1)
    print(f"encoded {len(paths)} x {len(QUESTION_TYPES)} joint embeddings, dim={joint_embs.shape[2]}")

    # log joint sim distribution to understand boundary
    sample_sims = []
    for _ in range(500):
        i, j = random.sample(range(len(paths)), 2)
        qt = random.randrange(len(QUESTION_TYPES))
        sim = float(np.dot(joint_embs[i, qt], joint_embs[j, qt]))
        sample_sims.append(sim)
    print(f"random pair joint_sim: mean={np.mean(sample_sims):.3f}  "
          f"min={np.min(sample_sims):.3f}  max={np.max(sample_sims):.3f}")

    # log same-image different-question sim (the hard negative signature)
    same_img_diff_q_sims = []
    for i in random.sample(range(len(paths)), min(50, len(paths))):
        qt1, qt2 = random.sample(range(len(QUESTION_TYPES)), 2)
        sim = float(np.dot(joint_embs[i, qt1], joint_embs[i, qt2]))
        same_img_diff_q_sims.append(sim)
    print(f"same-image diff-question joint_sim: mean={np.mean(same_img_diff_q_sims):.3f}  "
          f"min={np.min(same_img_diff_q_sims):.3f}  max={np.max(same_img_diff_q_sims):.3f}")

    def make_pairs(idx_list, n_pos, n_neg_easy, n_neg_hard, split_seed):
        # Re-seed independently per split so val augmentations don't depend on
        # how many random draws the train split consumed.
        rng = random.Random(split_seed)
        np_rng = np.random.default_rng(split_seed)

        features, labels, meta = [], [], []

        # positives: same image + same question type (augmented)
        for _ in range(n_pos):
            gi = rng.choice(idx_list)
            qt_idx = rng.randrange(len(QUESTION_TYPES))
            qtype = QUESTION_TYPES[qt_idx]
            _name = paths[gi].name
            question = question_templates.get(qtype) or qdata[_name].get("questions", {}).get(qtype)
            if not question:
                raise RuntimeError(f"question string missing for type '{qtype}' in image '{_name}'")

            img = Image.open(paths[gi]).convert("RGB")
            aug = augment(img, rng)

            emb_aug = encoder.encode_joint(aug, question)
            emb_orig = joint_embs[gi, qt_idx]

            features.append(_build_features(emb_orig, emb_aug))
            labels.append(0.0)
            meta.append({"i": gi, "j": gi, "qt1": qtype, "qt2": qtype,
                         "type": "aug_positive", "label": 0})

        # easy negatives: different image + any question
        # Reject pairs where joint_sim > 0.7 to keep tier boundaries clean.
        sampled = 0
        attempts = 0
        max_attempts = n_neg_easy * 20
        while sampled < n_neg_easy and attempts < max_attempts:
            attempts += 1
            gi, gj = rng.sample(idx_list, 2)
            qt_i = rng.randrange(len(QUESTION_TYPES))
            qt_j = rng.randrange(len(QUESTION_TYPES))
            sim = float(np.dot(joint_embs[gi, qt_i], joint_embs[gj, qt_j]))
            if sim > 0.7:
                continue  # exclude near-duplicates that blur tier boundary
            features.append(_build_features(joint_embs[gi, qt_i], joint_embs[gj, qt_j]))
            labels.append(1.0)
            meta.append({"i": gi, "j": gj,
                         "qt1": QUESTION_TYPES[qt_i], "qt2": QUESTION_TYPES[qt_j],
                         "type": "easy_negative", "joint_sim": round(sim, 4), "label": 1})
            sampled += 1

        if sampled < n_neg_easy:
            print(f"  warning: only got {sampled}/{n_neg_easy} easy negatives after {attempts} attempts")

        # hard negatives: same image + different question type
        # Both embeddings come from the precomputed canonical grid (no augmentation).
        # Known limitation: inference candidates may be augmented; this is a
        # train/inference distribution mismatch that could be closed by augmenting
        # one side of hard negative pairs in a future iteration.
        for _ in range(n_neg_hard):
            gi = rng.choice(idx_list)
            qt_i, qt_j = rng.sample(range(len(QUESTION_TYPES)), 2)
            sim = float(np.dot(joint_embs[gi, qt_i], joint_embs[gi, qt_j]))
            features.append(_build_features(joint_embs[gi, qt_i], joint_embs[gi, qt_j]))
            labels.append(1.0)
            meta.append({"i": gi, "j": gi,
                         "qt1": QUESTION_TYPES[qt_i], "qt2": QUESTION_TYPES[qt_j],
                         "type": "hard_negative", "joint_sim": round(sim, 4), "label": 1})

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        perm = np_rng.permutation(len(X))
        return X[perm], y[perm], [meta[k] for k in perm]

    print("generating train pairs...")
    X_train, y_train, meta_train = make_pairs(
        train_idx, n_easy, n_easy, n_hard, split_seed=seed
    )
    print("generating val pairs...")
    X_val, y_val, meta_val = make_pairs(
        val_idx, n_easy_val, n_easy_val, n_hard_val, split_seed=seed + 1
    )

    for split, X, y, meta in [
        ("train", X_train, y_train, meta_train),
        ("val", X_val, y_val, meta_val),
    ]:
        n_pos = int((y == 0).sum())
        n_neg = int((y == 1).sum())
        hard = sum(1 for m in meta if m["type"] == "hard_negative")
        ratio = n_neg / n_pos if n_pos > 0 else float("inf")
        print(
            f"{split}: {len(y)} pairs  pos={n_pos}  neg={n_neg}  "
            f"hard_neg={hard}  neg:pos ratio={ratio:.2f}  feature_dim={X.shape[1]}"
        )
        if ratio >= 1.5:
            print(
                f"  WARNING: {split} neg:pos={ratio:.2f} — pass "
                f"pos_weight=torch.tensor([{ratio:.1f}]) to BCEWithLogitsLoss in mlp.py"
            )

    return X_train, y_train, X_val, y_val, meta_train, meta_val


if __name__ == "__main__":
    cfg = load()
    X_train, y_train, X_val, y_val, meta_train, meta_val = generate()

    out_dir = Path(cfg["data"]["drift_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

    with open(out_dir / "meta_train.json", "w") as f:
        json.dump(meta_train, f)
    with open(out_dir / "meta_val.json", "w") as f:
        json.dump(meta_val, f)

    print(f"saved to {out_dir}")
    print(
        "NOTE: reported eval metrics are validation metrics, not held-out test metrics.\n"
        "      Val was also used for early stopping in mlp.py — generate a separate\n"
        "      test split before final reporting."
    )