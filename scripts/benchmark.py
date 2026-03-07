import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from codecarbon import EmissionsTracker
from PIL import Image, ImageEnhance

from cachevista.config import load
from cachevista.core import CacheVista, NoCacheStrategy, StaticCacheStrategy, generate_with_strategy, hash_query
from cachevista.encoder import get_encoder
from cachevista.mlp import load_model


QUESTION_TYPES = ["describe", "count", "color", "spatial", "yn"]


def load_image(path):
    return Image.open(path).convert("RGB")


def augment(img):
    w, h = img.size
    dx, dy = int(w * 0.05), int(h * 0.05)
    return img.crop((dx, dy, w - dx, h - dy))


def build_query_sequence(paths, encoder, questions, n_unique=100,
                         repeat_rate=0.4, augment_rate=0.2, seed=42):
    """
    Each query is an (image, question) pair.
    Repeats reuse the same image + same question → should hit L1.
    Augmented repeats use same question but cropped image → should hit L2.
    New queries use a fresh image + random question → miss.
    """
    random.seed(seed)
    paths = list(paths)
    random.shuffle(paths)
    base_paths = paths[:n_unique]

    base_data = {}
    for p in base_paths:
        img = load_image(p)
        qtype = random.choice(QUESTION_TYPES)
        question = questions[p.name]["questions"][qtype]
        with open(p, "rb") as f:
            img_bytes = f.read()
        qhash = hash_query(img_bytes, question)
        joint_emb = encoder.encode_joint(img, question)
        base_data[p] = {
            "hash": qhash,
            "emb": joint_emb,
            "img": img,
            "question": question,
            "img_bytes": img_bytes,
        }

    sequence = []
    for p in base_paths:
        d = base_data[p]
        sequence.append((d["hash"], d["emb"]))

        r = random.random()
        if r < repeat_rate:
            # exact repeat — same image, same question
            sequence.append((d["hash"], d["emb"]))
        elif r < repeat_rate + augment_rate:
            # augmented repeat — same question, slightly different image
            aug_img = augment(d["img"])
            aug_emb = encoder.encode_joint(aug_img, d["question"])
            aug_hash = hash_query(b"aug_" + d["img_bytes"][:16], d["question"])
            sequence.append((aug_hash, aug_emb))

    random.shuffle(sequence)
    return sequence


def run_strategy(name, strategy, sequence):
    latencies = []
    tracker = EmissionsTracker(project_name=name, logging_logger=None, save_to_file=False)
    tracker.start()

    for qhash, emb in sequence:
        t0 = time.perf_counter()
        generate_with_strategy(strategy, qhash, emb)
        latencies.append(time.perf_counter() - t0)

    emissions = tracker.stop()
    s = strategy.stats()

    return {
        "strategy": name,
        "hit_rate": round(s["hit_rate"], 4),
        "hits_l1": s.get("hits_l1", 0),
        "hits_l2": s.get("hits_l2", 0),
        "drift_rejections": s.get("drift_rejections", 0),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 4),
        "total_latency_ms": round(np.sum(latencies) * 1000, 2),
        "energy_kg_co2": round(emissions or 0.0, 8),
    }


if __name__ == "__main__":
    cfg = load()
    encoder = get_encoder()

    img_dir = Path(cfg["data"]["coco_dir"])
    paths = sorted(img_dir.glob("*.jpg"))
    print(f"found {len(paths)} images")

    questions_path = Path(cfg["data"]["drift_data_dir"]) / "questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    print("building query sequence...")
    sequence = build_query_sequence(
        paths, encoder, questions,
        n_unique=100, repeat_rate=0.4, augment_rate=0.2, seed=42,
    )
    print(f"query sequence: {len(sequence)} queries")

    mlp = load_model(cfg["model"]["mlp_path"])

    # threshold=0.93 calibrated to joint embedding distribution:
    # positives cluster at ~0.987, hard negatives top out at ~0.909
    # 0.93 sits cleanly above the negative ceiling
    strategies = [
        ("no_cache",     NoCacheStrategy()),
        ("static_cache", StaticCacheStrategy()),
        ("cachevista",   CacheVista(threshold=0.93, mlp=mlp)),
    ]

    results = []
    for name, strategy in strategies:
        print(f"  running {name}...")
        result = run_strategy(name, strategy, sequence)
        results.append(result)
        print(f"    hit_rate={result['hit_rate']}  "
              f"l1={result['hits_l1']}  l2={result['hits_l2']}  "
              f"drift_rej={result['drift_rejections']}  "
              f"avg_latency={result['avg_latency_ms']}ms")

    out = Path(cfg["benchmark"]["results_dir"]) / "benchmark.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nsaved to {out}")