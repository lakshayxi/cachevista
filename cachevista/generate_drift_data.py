import numpy as np
from PIL import Image, ImageEnhance

from cachevista.encoder import CLIPEncoder


def load(path):
    return Image.open(path).convert("RGB")


def augment(img, mode):
    w, h = img.size
    if mode == "crop":
        dx, dy = int(w * 0.05), int(h * 0.05)
        return img.crop((dx, dy, w - dx, h - dy))
    if mode == "brightness":
        return ImageEnhance.Brightness(img).enhance(1.3)
    if mode == "flip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def make_pairs(encoder, img_a, img_b):
    pairs, labels = [], []

    # no-drift: same image, minor augmentation
    base_a = encoder.encode(img_a)
    base_b = encoder.encode(img_b)

    for mode in ["crop", "brightness", "flip"]:
        aug_a = encoder.encode(augment(img_a, mode))
        pairs.append(np.concatenate([base_a, aug_a]))
        labels.append(0)

        aug_b = encoder.encode(augment(img_b, mode))
        pairs.append(np.concatenate([base_b, aug_b]))
        labels.append(0)

    # drift: switching between two different images
    for _ in range(6):
        pairs.append(np.concatenate([base_a, base_b]))
        labels.append(1)
        pairs.append(np.concatenate([base_b, base_a]))
        labels.append(1)

    return np.array(pairs, dtype=np.float32), np.array(labels, dtype=np.float32)


if __name__ == "__main__":
    encoder = CLIPEncoder()
    cat = load("data/test_cat.jpg")
    dog = load("data/test_dog.jpg")

    X, y = make_pairs(encoder, cat, dog)

    np.save("data/drift_X.npy", X)
    np.save("data/drift_y.npy", y)

    print(f"generated {len(X)} pairs — no-drift: {int((y==0).sum())}  drift: {int((y==1).sum())}")
    print(f"X shape: {X.shape}")
