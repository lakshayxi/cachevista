import csv
import random
import time
from pathlib import Path

import numpy as np
from codecarbon import EmissionsTracker
from PIL import Image, ImageEnhance

from cachevista.config import load
from cachevista.core import CacheVista, NoCacheStrategy, StaticCacheStrategy, generate_with_strategy
from cachevista.encoder import CLIPEncoder
from cachevista.mlp import load_model


def load_image(path):
    return Image.open(path).convert("RGB")


def augment(img):
    w, h = img.size
    dx, dy = int(w * 0.05), int(h * 0.05)
    return img.crop((dx, dy, w - dx, h - dy))


def build_query_sequence(paths, encoder, n_unique=100, repeat_rate=0.4, augment_rate=0.2):
    random.shuffle(paths)
    base_paths = paths[:n_unique]
    base_embs = {p: encoder.encode(load_image(p)) for p in base_paths}

    sequence = []
    for path in base_paths:
        sequence.append(base_embs[path])

        r = random.random()
        if r < repeat_rate:
            sequence.append(base_embs[path])
        elif r < repeat_rate + augment_rate:
            aug_emb = encoder.encode(augment(load_image(path)))
            sequence.append(aug_emb)

    random.shuffle(sequence)
    return sequence


def run_strategy(name, strategy, embeddings):
    latencies = []
    tracker = EmissionsTracker(project_name=name, logging_logger=None, save_to_file=False)
    tracker.start()

    for emb in embeddings:
        t0 = time.perf_counter()
        generate_with_strategy(strategy, emb)
        latencies.append(time.perf_counter() - t0)

    emissions = tracker.stop()
    hit_rate = strategy.stats()["hit_rate"] if hasattr(strategy, "stats") else 0.0

    return {
        "strategy": name,
        "hit_rate": round(hit_rate, 4),
        "avg_latency_ms": round(np.mean(latencies) * 1000, 4),
        "total_latency_ms": round(np.sum(latencies) * 1000, 2),
        "energy_kg_co2": round(emissions or 0.0, 8),
    }


if __name__ == "__main__":
    cfg = load()
    encoder = CLIPEncoder()

    img_dir = Path(cfg["data"]["coco_dir"])
    paths = sorted(img_dir.glob("*.jpg"))
    print(f"found {len(paths)} images")

    print("building query sequence...")
    sequence = build_query_sequence(paths, encoder, n_unique=100, repeat_rate=0.4, augment_rate=0.2)
    print(f"query sequence: {len(sequence)} queries")

    mlp = load_model(cfg["model"]["mlp_path"])

    strategies = [
        ("no_cache", NoCacheStrategy()),
        ("static_cache", StaticCacheStrategy()),
        ("cachevista", CacheVista(threshold=cfg["cache"]["similarity_threshold"], mlp=mlp)),
    ]

    results = []
    for name, strategy in strategies:
        print(f"  running {name}...")
        result = run_strategy(name, strategy, sequence)
        results.append(result)
        print(f"    hit_rate={result['hit_rate']}  avg_latency={result['avg_latency_ms']}ms  co2={result['energy_kg_co2']}")

    out = Path(cfg["benchmark"]["results_dir"]) / "benchmark.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nsaved to {out}")