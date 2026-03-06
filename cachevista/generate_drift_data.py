import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

from cachevista.config import load
from cachevista.encoder import CLIPEncoder


def load_image(path):
    return Image.open(path).convert("RGB")


def augment(img):
    mode = random.choice(["crop", "brightness", "flip"])
    if mode == "crop":
        w, h = img.size
        dx, dy = int(w * 0.05), int(h * 0.05)
        return img.crop((dx, dy, w - dx, h - dy))
    if mode == "brightness":
        return ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def make_pairs(encoder, image_paths):
    pairs, labels = [], []
    paths = list(image_paths)
    random.shuffle(paths)

    for path in tqdm(paths, desc="generating pairs"):
        img = load_image(path)
        emb = encoder.encode(img)

        # no-drift: same image with random augmentation
        aug_emb = encoder.encode(augment(img))
        pairs.append(np.concatenate([emb, aug_emb]))
        labels.append(0)

        # drift: random image from the dataset
        drift_path = random.choice(paths)
        while drift_path == path:
            drift_path = random.choice(paths)
        drift_emb = encoder.encode(load_image(drift_path))
        pairs.append(np.concatenate([emb, drift_emb]))
        labels.append(1)

    return np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_test_split(X, y, test_size=0.20, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


if __name__ == "__main__":
    cfg = load()
    encoder = CLIPEncoder()

    img_dir = Path(cfg["data"]["coco_dir"])
    image_paths = sorted(img_dir.glob("*.jpg"))
    print(f"found {len(image_paths)} images")

    X, y = make_pairs(encoder, image_paths)
    print(f"generated {len(X)} pairs — no-drift: {int((y==0).sum())}  drift: {int((y==1).sum())}")

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"train: {len(X_train)}  test: {len(X_test)}")

    np.save(cfg["data"]["drift_train_X"], X_train)
    np.save(cfg["data"]["drift_train_y"], y_train)
    np.save(cfg["data"]["drift_test_X"], X_test)
    np.save(cfg["data"]["drift_test_y"], y_test)
    print("saved to data/")
