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
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from cachevista.config import load
from cachevista.encoder import get_encoder
from cachevista.mlp import _build_features

QUESTION_TYPES = ["describe", "count", "color", "spatial", "yn"]


def augment(img: Image.Image) -> Image.Image:
    w, h = img.size
    ops = [
        lambda x: x.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95))),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: x.rotate(random.uniform(-5, 5)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5))),
    ]
    for op in random.sample(ops, k=random.randint(1, 2)):
        img = op(img)
    return img.convert("RGB")


def generate(n_easy=500, n_hard=500, seed=42, val_split=0.2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        qdata = json.load(f)

    missing = [p.name for p in paths if p.name not in qdata]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing questions — re-run generate_questions.py")

    print(f"found {len(paths)} images with questions")

    # image-level split — prevents same image in both train and val
    all_idx = list(range(len(paths)))
    random.shuffle(all_idx)
    n_val = max(1, int(len(all_idx) * val_split))
    val_idx = sorted(all_idx[:n_val])
    train_idx = sorted(all_idx[n_val:])
    print(f"train images: {len(train_idx)}  val images: {len(val_idx)}")

    images = [Image.open(p).convert("RGB") for p in paths]

    # pre-encode all joint embeddings for all (image, question_type) combos
    print("encoding joint embeddings for all (image, question) pairs...")
    all_questions = []
    for p in paths:
        for qtype in QUESTION_TYPES:
            all_questions.append(qdata[p.name]["questions"][qtype])

    all_images_repeated = []
    for img in images:
        for _ in QUESTION_TYPES:
            all_images_repeated.append(img)

    joint_embs_flat = encoder.encode_joint_batch(all_images_repeated, all_questions)
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
    for i in random.sample(range(len(paths)), 50):
        qt1, qt2 = random.sample(range(len(QUESTION_TYPES)), 2)
        sim = float(np.dot(joint_embs[i, qt1], joint_embs[i, qt2]))
        same_img_diff_q_sims.append(sim)
    print(f"same-image diff-question joint_sim: mean={np.mean(same_img_diff_q_sims):.3f}  "
          f"min={np.min(same_img_diff_q_sims):.3f}  max={np.max(same_img_diff_q_sims):.3f}")

    def make_pairs(idx_list):
        features, labels, meta = [], [], []

        # positives: same image + same question type (with augmentation)
        for _ in range(n_easy):
            gi = random.choice(idx_list)
            qt_idx = random.randrange(len(QUESTION_TYPES))
            qtype = QUESTION_TYPES[qt_idx]
            question = qdata[paths[gi].name]["questions"][qtype]

            aug = augment(images[gi])
            emb_aug = encoder.encode_joint(aug, question)
            emb_orig = joint_embs[gi, qt_idx]

            features.append(_build_features(emb_orig, emb_aug))
            labels.append(0.0)
            meta.append({"i": gi, "j": gi, "qt1": qtype, "qt2": qtype,
                         "type": "aug_positive", "label": 0})

        # easy negatives: different image + any question
        for _ in range(n_easy):
            gi, gj = random.sample(idx_list, 2)
            qt_i = random.randrange(len(QUESTION_TYPES))
            qt_j = random.randrange(len(QUESTION_TYPES))
            sim = float(np.dot(joint_embs[gi, qt_i], joint_embs[gj, qt_j]))
            features.append(_build_features(joint_embs[gi, qt_i], joint_embs[gj, qt_j]))
            labels.append(1.0)
            meta.append({"i": gi, "j": gj,
                         "qt1": QUESTION_TYPES[qt_i], "qt2": QUESTION_TYPES[qt_j],
                         "type": "easy_negative", "joint_sim": sim, "label": 1})

        # hard negatives: same image + different question type
        # image CLIP sim = 1.0 but joint sim < 1.0 — FAISS can't detect this
        count = 0
        while count < n_hard:
            gi = random.choice(idx_list)
            qt_i, qt_j = random.sample(range(len(QUESTION_TYPES)), 2)
            sim = float(np.dot(joint_embs[gi, qt_i], joint_embs[gi, qt_j]))
            features.append(_build_features(joint_embs[gi, qt_i], joint_embs[gi, qt_j]))
            labels.append(1.0)
            meta.append({"i": gi, "j": gi,
                         "qt1": QUESTION_TYPES[qt_i], "qt2": QUESTION_TYPES[qt_j],
                         "type": "hard_negative", "joint_sim": sim, "label": 1})
            count += 1

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        perm = np.random.permutation(len(X))
        return X[perm], y[perm], [meta[k] for k in perm]

    print("generating train pairs...")
    X_train, y_train, meta_train = make_pairs(train_idx)
    print("generating val pairs...")
    X_val, y_val, meta_val = make_pairs(val_idx)

    for split, y, meta in [("train", y_train, meta_train), ("val", y_val, meta_val)]:
        n_pos = int((y == 0).sum())
        n_neg = int((y == 1).sum())
        hard = sum(1 for m in meta if m["type"] == "hard_negative")
        print(f"{split}: {len(y)} pairs  pos={n_pos}  neg={n_neg}  "
              f"hard_neg={hard}  balance={y.mean():.2f}  feature_dim={X_train.shape[1]}")

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