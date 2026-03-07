"""
Ablation study for CacheVista.

Compares three configurations to isolate the contribution of each component:

  A) image_only_faiss   — L1 hash + L2 FAISS on 768-dim image embeddings, no MLP
                          Baseline: what you get with standard semantic image caching
                          Cache key: hash(image_bytes) only

  B) joint_faiss        — L1 hash + L2 FAISS on 1152-dim joint embeddings, no MLP
                          Shows the value of question-aware embeddings
                          Cache key: hash(image_bytes + question)

  C) cachevista_full    — L1 + L2 joint FAISS + L3 MLP drift gate (full system)
                          Shows the value of the learned drift detector on top

The delta B-A shows: does adding question context to the embedding improve precision?
The delta C-B shows: does the MLP drift gate add value on top of joint embeddings?

Query sequence design:
  - 100 unique (image, question) pairs
  - 40% exact repeats       → should hit L1 on all configs
  - 20% augmented repeats   → should hit L2 on B and C, may miss on A
  - 10% same image, diff Q  → should MISS on B and C (different hash),
                               A may wrongly HIT (same image hash)
  - remainder = fresh queries → should miss on all
"""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from codecarbon import EmissionsTracker
from PIL import Image, ImageEnhance

from cachevista.config import load
from cachevista.core import (CacheVista, NoCacheStrategy, StaticCacheStrategy,
                              generate_with_strategy, hash_query, hash_image_bytes)
from cachevista.encoder import get_encoder
from cachevista.mlp import load_model


QUESTION_TYPES = ["describe", "count", "color", "spatial", "yn"]


def load_image(path):
    return Image.open(path).convert("RGB")


def augment(img):
    w, h = img.size
    dx, dy = int(w * 0.05), int(h * 0.05)
    return img.crop((dx, dy, w - dx, h - dy))


def build_ablation_sequence(paths, encoder, questions, n_unique=100, seed=42):
    """
    Builds a query sequence with four clearly labelled query types:
      exact_repeat       — same image, same question (should always hit L1)
      augmented_repeat   — cropped image, same question (should hit L2 if sim > threshold)
      same_img_diff_q    — same image, different question (should MISS on joint configs)
      fresh              — new image, new question (always miss)
    Returns: list of (query_hash_image_only, query_hash_joint, image_emb, joint_emb, query_type)
    """
    random.seed(seed)
    paths = list(paths)
    random.shuffle(paths)
    base_paths = paths[:n_unique]

    base_data = {}
    for p in base_paths:
        img = load_image(p)
        qt = random.choice(QUESTION_TYPES)
        question = questions[p.name]["questions"][qt]
        with open(p, "rb") as f:
            img_bytes = f.read()

        base_data[p] = {
            "img": img,
            "img_bytes": img_bytes,
            "question": question,
            "qt": qt,
            "hash_img": hash_image_bytes(img_bytes),
            "hash_joint": hash_query(img_bytes, question),
            "emb_img": encoder.encode(img),
            "emb_joint": encoder.encode_joint(img, question),
        }

    sequence = []

    for p in base_paths:
        d = base_data[p]

        # always add the base query
        sequence.append({
            "type": "fresh",
            "hash_img": d["hash_img"],
            "hash_joint": d["hash_joint"],
            "emb_img": d["emb_img"],
            "emb_joint": d["emb_joint"],
        })

        r = random.random()
        if r < 0.40:
            # exact repeat
            sequence.append({
                "type": "exact_repeat",
                "hash_img": d["hash_img"],
                "hash_joint": d["hash_joint"],
                "emb_img": d["emb_img"],
                "emb_joint": d["emb_joint"],
            })
        elif r < 0.60:
            # augmented repeat — same question, cropped image
            aug = augment(d["img"])
            aug_emb_img = encoder.encode(aug)
            aug_emb_joint = encoder.encode_joint(aug, d["question"])
            aug_hash_img = hash_image_bytes(b"aug_" + d["img_bytes"][:16])
            aug_hash_joint = hash_query(b"aug_" + d["img_bytes"][:16], d["question"])
            sequence.append({
                "type": "augmented_repeat",
                "hash_img": aug_hash_img,
                "hash_joint": aug_hash_joint,
                "emb_img": aug_emb_img,
                "emb_joint": aug_emb_joint,
            })
        elif r < 0.70:
            # same image, different question
            other_qts = [q for q in QUESTION_TYPES if q != d["qt"]]
            qt2 = random.choice(other_qts)
            q2 = questions[p.name]["questions"][qt2]
            emb_joint2 = encoder.encode_joint(d["img"], q2)
            hash_joint2 = hash_query(d["img_bytes"], q2)
            sequence.append({
                "type": "same_img_diff_q",
                "hash_img": d["hash_img"],       # same image hash — A config will hit
                "hash_joint": hash_joint2,        # different joint hash — B/C will miss
                "emb_img": d["emb_img"],
                "emb_joint": emb_joint2,
            })

    random.shuffle(sequence)
    return sequence


def run_config(name, strategy, sequence, use_joint=True):
    latencies = []
    query_type_hits = {"exact_repeat": 0, "augmented_repeat": 0,
                       "same_img_diff_q": 0, "fresh": 0}
    query_type_total = {k: 0 for k in query_type_hits}

    tracker = EmissionsTracker(project_name=name, logging_logger=None, save_to_file=False)
    tracker.start()

    for item in sequence:
        qhash = item["hash_joint"] if use_joint else item["hash_img"]
        emb = item["emb_joint"] if use_joint else item["emb_img"]

        t0 = time.perf_counter()
        _, status = generate_with_strategy(strategy, qhash, emb)
        latencies.append(time.perf_counter() - t0)

        qtype = item["type"]
        query_type_total[qtype] += 1
        if status == "HIT":
            query_type_hits[qtype] += 1

    emissions = tracker.stop()
    s = strategy.stats()

    # precision proxy: among L2 hits, how many were same_img_diff_q (wrong hits)?
    wrong_hits = query_type_hits["same_img_diff_q"]
    total_hits = s["hits_l1"] + s.get("hits_l2", 0)

    return {
        "config": name,
        "hit_rate": round(s["hit_rate"], 4),
        "hits_l1": s.get("hits_l1", s.get("hits", 0)),
        "hits_l2": s.get("hits_l2", 0),
        "drift_rejections": s.get("drift_rejections", 0),
        "wrong_hits_same_img_diff_q": wrong_hits,
        "avg_latency_ms": round(np.mean(latencies) * 1000, 4),
        "energy_kg_co2": round(emissions or 0.0, 8),
        # per-type breakdown
        "exact_repeat_hr": round(query_type_hits["exact_repeat"] / max(query_type_total["exact_repeat"], 1), 3),
        "aug_repeat_hr": round(query_type_hits["augmented_repeat"] / max(query_type_total["augmented_repeat"], 1), 3),
        "same_img_diff_q_hr": round(query_type_hits["same_img_diff_q"] / max(query_type_total["same_img_diff_q"], 1), 3),
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

    print("building ablation query sequence...")
    sequence = build_ablation_sequence(paths, encoder, questions, n_unique=100, seed=42)

    type_counts = {}
    for item in sequence:
        type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
    print(f"sequence: {len(sequence)} queries — {type_counts}")

    mlp = load_model(cfg["model"]["mlp_path"])
    threshold = cfg["cache"].get("ablation_threshold", 0.93)

    configs = [
        # A: image-only FAISS — standard semantic cache, no question context
        ("A_image_only_faiss",
         CacheVista(threshold=0.90, mlp=None),
         False),  # use_joint=False

        # B: joint FAISS, no MLP — question-aware embeddings, threshold only
        ("B_joint_faiss_no_mlp",
         CacheVista(threshold=threshold, mlp=None),
         True),

        # C: full CacheVista — joint FAISS + MLP drift gate
        ("C_cachevista_full",
         CacheVista(threshold=threshold, mlp=mlp),
         True),
    ]

    results = []
    for name, strategy, use_joint in configs:
        print(f"\n  running {name}...")
        result = run_config(name, strategy, sequence, use_joint=use_joint)
        results.append(result)
        print(f"    hit_rate={result['hit_rate']}  "
              f"l1={result['hits_l1']}  l2={result['hits_l2']}  "
              f"drift_rej={result['drift_rejections']}  "
              f"wrong_hits={result['wrong_hits_same_img_diff_q']}  "
              f"avg_latency={result['avg_latency_ms']}ms")
        print(f"    per-type:  exact={result['exact_repeat_hr']}  "
              f"aug={result['aug_repeat_hr']}  "
              f"same_img_diff_q={result['same_img_diff_q_hr']}")

    out = Path(cfg["benchmark"]["results_dir"]) / "ablation.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nsaved to {out}")

    # print paper-ready summary table
    print("\n" + "=" * 70)
    print(f"{'Config':<25} {'HitRate':>8} {'L1':>5} {'L2':>5} {'DriftRej':>9} {'WrongHits':>10} {'Latency(ms)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<25} {r['hit_rate']:>8.4f} {r['hits_l1']:>5} "
              f"{r['hits_l2']:>5} {r['drift_rejections']:>9} "
              f"{r['wrong_hits_same_img_diff_q']:>10} {r['avg_latency_ms']:>12.4f}")
    print("=" * 70)