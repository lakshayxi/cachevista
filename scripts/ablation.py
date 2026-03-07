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

Robust experiments (--robust flag):
  - Seed variance: N seeds, reports mean ± 95% CI for all metrics
  - Tau sweep: [0.85, 0.90, 0.93] to show MLP filtering at lower thresholds
  - Distribution sweep: 4 conversation styles to test ecological validity
"""

import argparse
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

DISTRIBUTIONS = {
    "conservative": dict(exact=0.10, aug=0.05, diff_q=0.05),
    "typical":      dict(exact=0.40, aug=0.20, diff_q=0.10),
    "heavy_reuse":  dict(exact=0.60, aug=0.30, diff_q=0.15),
    "adversarial":  dict(exact=0.05, aug=0.05, diff_q=0.30),
}

TAU_VALUES = [0.85, 0.90, 0.93]


def load_image(path):
    return Image.open(path).convert("RGB")


def augment(img):
    w, h = img.size
    dx, dy = int(w * 0.05), int(h * 0.05)
    return img.crop((dx, dy, w - dx, h - dy))


def build_ablation_sequence(paths, encoder, questions, n_unique=100, seed=42,
                             exact=0.40, aug=0.20, diff_q=0.10):
    """
    Builds a query sequence with four clearly labelled query types:
      exact_repeat       — same image, same question (should always hit L1)
      augmented_repeat   — cropped image, same question (should hit L2 if sim > threshold)
      same_img_diff_q    — same image, different question (should MISS on joint configs)
      fresh              — new image, new question (always miss)
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

        sequence.append({
            "type": "fresh",
            "hash_img": d["hash_img"],
            "hash_joint": d["hash_joint"],
            "emb_img": d["emb_img"],
            "emb_joint": d["emb_joint"],
        })

        r = random.random()
        if r < exact:
            sequence.append({
                "type": "exact_repeat",
                "hash_img": d["hash_img"],
                "hash_joint": d["hash_joint"],
                "emb_img": d["emb_img"],
                "emb_joint": d["emb_joint"],
            })
        elif r < exact + aug:
            aug_img = augment(d["img"])
            aug_emb_img = encoder.encode(aug_img)
            aug_emb_joint = encoder.encode_joint(aug_img, d["question"])
            aug_hash_img = hash_image_bytes(b"aug_" + d["img_bytes"][:16])
            aug_hash_joint = hash_query(b"aug_" + d["img_bytes"][:16], d["question"])
            sequence.append({
                "type": "augmented_repeat",
                "hash_img": aug_hash_img,
                "hash_joint": aug_hash_joint,
                "emb_img": aug_emb_img,
                "emb_joint": aug_emb_joint,
            })
        elif r < exact + aug + diff_q:
            other_qts = [q for q in QUESTION_TYPES if q != d["qt"]]
            qt2 = random.choice(other_qts)
            q2 = questions[p.name]["questions"][qt2]
            emb_joint2 = encoder.encode_joint(d["img"], q2)
            hash_joint2 = hash_query(d["img_bytes"], q2)
            sequence.append({
                "type": "same_img_diff_q",
                "hash_img": d["hash_img"],
                "hash_joint": hash_joint2,
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

    wrong_hits = query_type_hits["same_img_diff_q"]

    return {
        "config": name,
        "hit_rate": round(s["hit_rate"], 4),
        "hits_l1": s.get("hits_l1", s.get("hits", 0)),
        "hits_l2": s.get("hits_l2", 0),
        "drift_rejections": s.get("drift_rejections", 0),
        "wrong_hits_same_img_diff_q": wrong_hits,
        "avg_latency_ms": round(np.mean(latencies) * 1000, 4),
        "energy_kg_co2": round(emissions or 0.0, 8),
        "exact_repeat_hr": round(query_type_hits["exact_repeat"] / max(query_type_total["exact_repeat"], 1), 3),
        "aug_repeat_hr": round(query_type_hits["augmented_repeat"] / max(query_type_total["augmented_repeat"], 1), 3),
        "same_img_diff_q_hr": round(query_type_hits["same_img_diff_q"] / max(query_type_total["same_img_diff_q"], 1), 3),
    }


def _mean_ci(values):
    arr = np.array(values)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0
    se = float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return mean, round(1.96 * se, 4)


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_robust(paths, encoder, questions, mlp, cfg, outdir, n_seeds=30):
    """
    Three experiments:
      1. Seed variance   — all configs over n_seeds, mean ± 95% CI
      2. Tau sweep       — [0.85, 0.90, 0.93], B vs C, shows MLP filtering effect
      3. Distribution sweep — 4 conversation styles, A vs C
    """
    threshold = cfg["cache"].get("ablation_threshold", 0.93)
    seeds = list(range(42, 42 + n_seeds))

    def make_configs(tau):
        return [
            ("A_image_only",   CacheVista(threshold=0.90,  mlp=None), False),
            ("B_joint_no_mlp", CacheVista(threshold=tau,   mlp=None), True),
            ("C_cachevista",   CacheVista(threshold=tau,   mlp=mlp),  True),
        ]

    # experiment 1: seed variance
    print(f"\nExperiment 1: seed variance ({n_seeds} seeds, typical, tau={threshold})")
    seed_rows = []
    for i, seed in enumerate(seeds):
        seq = build_ablation_sequence(paths, encoder, questions,
                                      seed=seed, **DISTRIBUTIONS["typical"])
        for name, strategy, use_joint in make_configs(threshold):
            r = run_config(name, strategy, seq, use_joint)
            r["seed"] = seed
            r["distribution"] = "typical"
            seed_rows.append(r)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n_seeds} seeds done")

    _write_csv(seed_rows, outdir / "ablation_seeds.csv")
    print(f"saved ablation_seeds.csv")

    agg_rows = []
    for cfg_name in ["A_image_only", "B_joint_no_mlp", "C_cachevista"]:
        rows = [r for r in seed_rows if r["config"] == cfg_name]
        agg = {"config": cfg_name, "tau": threshold, "n_seeds": len(rows)}
        for key in ["hit_rate", "hits_l1", "hits_l2", "drift_rejections",
                    "wrong_hits_same_img_diff_q", "avg_latency_ms",
                    "exact_repeat_hr", "aug_repeat_hr", "same_img_diff_q_hr"]:
            m, ci = _mean_ci([r[key] for r in rows])
            agg[f"{key}_mean"] = round(m, 4)
            agg[f"{key}_ci95"] = ci
        agg_rows.append(agg)

    _write_csv(agg_rows, outdir / "ablation_aggregated.csv")
    print(f"saved ablation_aggregated.csv")

    # experiment 2: tau sweep
    print(f"\nExperiment 2: tau sweep {TAU_VALUES}")
    tau_rows = []
    for tau in TAU_VALUES:
        for seed in seeds[:10]:
            seq = build_ablation_sequence(paths, encoder, questions,
                                          seed=seed, **DISTRIBUTIONS["typical"])
            for name, strategy, use_joint in make_configs(tau):
                if name == "A_image_only":
                    continue
                r = run_config(name, strategy, seq, use_joint)
                r["seed"] = seed
                r["tau_override"] = tau
                tau_rows.append(r)
        print(f"  tau={tau} done")

    tau_agg = []
    for tau in TAU_VALUES:
        for cfg_name in ["B_joint_no_mlp", "C_cachevista"]:
            rows = [r for r in tau_rows
                    if r["config"] == cfg_name and r["tau_override"] == tau]
            if not rows:
                continue
            agg = {"config": cfg_name, "tau": tau, "n_seeds": len(rows)}
            for key in ["hit_rate", "drift_rejections",
                        "wrong_hits_same_img_diff_q", "avg_latency_ms"]:
                m, ci = _mean_ci([r[key] for r in rows])
                agg[f"{key}_mean"] = round(m, 4)
                agg[f"{key}_ci95"] = ci
            tau_agg.append(agg)

    _write_csv(tau_agg, outdir / "ablation_tau_sweep.csv")
    print(f"saved ablation_tau_sweep.csv")

    # experiment 3: distribution sweep
    print(f"\nExperiment 3: distribution sweep {list(DISTRIBUTIONS)}")
    distro_rows = []
    for distro_name, distro_params in DISTRIBUTIONS.items():
        for seed in seeds[:10]:
            seq = build_ablation_sequence(paths, encoder, questions,
                                          seed=seed, **distro_params)
            for name, strategy, use_joint in make_configs(threshold):
                if name == "B_joint_no_mlp":
                    continue
                r = run_config(name, strategy, seq, use_joint)
                r["seed"] = seed
                r["distribution"] = distro_name
                r.update(distro_params)
                distro_rows.append(r)
        print(f"  '{distro_name}' done")

    distro_agg = []
    for distro_name in DISTRIBUTIONS:
        for cfg_name in ["A_image_only", "C_cachevista"]:
            rows = [r for r in distro_rows
                    if r["config"] == cfg_name and r["distribution"] == distro_name]
            if not rows:
                continue
            agg = {"config": cfg_name, "distribution": distro_name,
                   "n_seeds": len(rows), **DISTRIBUTIONS[distro_name]}
            for key in ["hit_rate", "wrong_hits_same_img_diff_q", "avg_latency_ms"]:
                m, ci = _mean_ci([r[key] for r in rows])
                agg[f"{key}_mean"] = round(m, 4)
                agg[f"{key}_ci95"] = ci
            distro_agg.append(agg)

    _write_csv(distro_agg, outdir / "ablation_distributions.csv")
    print(f"saved ablation_distributions.csv")

    # summaries
    print(f"\n{'Seed variance (typical, tau=' + str(threshold) + ')':-<60}")
    print(f"{'Config':<22} {'HitRate':>9} {'±CI':>7} {'WrongHits':>10} {'±CI':>7}")
    for row in agg_rows:
        print(f"{row['config']:<22} "
              f"{row['hit_rate_mean']:>9.3f} "
              f"{row['hit_rate_ci95']:>7.3f} "
              f"{row['wrong_hits_same_img_diff_q_mean']:>10.2f} "
              f"{row['wrong_hits_same_img_diff_q_ci95']:>7.2f}")

    print(f"\n{'Tau sweep (B vs C)':-<60}")
    print(f"{'Config':<22} {'tau':>5} {'HitRate':>9} {'DriftRej':>9} {'WrongHits':>10}")
    for row in tau_agg:
        print(f"{row['config']:<22} "
              f"{row['tau']:>5.2f} "
              f"{row['hit_rate_mean']:>9.3f} "
              f"{row['drift_rejections_mean']:>9.2f} "
              f"{row['wrong_hits_same_img_diff_q_mean']:>10.2f}")

    print(f"\n{'Distribution sweep (A vs C)':-<60}")
    print(f"{'Config':<22} {'Distribution':<14} {'HitRate':>9} {'WrongHits':>10}")
    for row in distro_agg:
        print(f"{row['config']:<22} "
              f"{row['distribution']:<14} "
              f"{row['hit_rate_mean']:>9.3f} "
              f"{row['wrong_hits_same_img_diff_q_mean']:>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust", action="store_true",
                        help="run seed variance, tau sweep, and distribution sweep")
    parser.add_argument("--seeds", type=int, default=30,
                        help="number of seeds for robust experiments (default 30)")
    parser.add_argument("--quick", action="store_true",
                        help="5 seeds only, for smoke testing")
    args = parser.parse_args()

    cfg = load()
    encoder = get_encoder()

    img_dir = Path(cfg["data"]["coco_dir"])
    paths = sorted(img_dir.glob("*.jpg"))
    print(f"found {len(paths)} images")

    questions_path = Path(cfg["data"]["drift_data_dir"]) / "questions.json"
    with open(questions_path) as f:
        questions = json.load(f)

    mlp = load_model(cfg["model"]["mlp_path"])
    threshold = cfg["cache"].get("ablation_threshold", 0.93)
    outdir = Path(cfg["benchmark"]["results_dir"])
    outdir.mkdir(exist_ok=True)

    if args.robust:
        n_seeds = 5 if args.quick else args.seeds
        run_robust(paths, encoder, questions, mlp, cfg, outdir, n_seeds=n_seeds)
    else:
        print("building ablation query sequence...")
        sequence = build_ablation_sequence(paths, encoder, questions,
                                           n_unique=100, seed=42)

        type_counts = {}
        for item in sequence:
            type_counts[item["type"]] = type_counts.get(item["type"], 0) + 1
        print(f"sequence: {len(sequence)} queries — {type_counts}")

        configs = [
            ("A_image_only_faiss",
             CacheVista(threshold=0.90, mlp=None),
             False),
            ("B_joint_faiss_no_mlp",
             CacheVista(threshold=threshold, mlp=None),
             True),
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

        out = outdir / "ablation.csv"
        _write_csv(results, out)
        print(f"\nsaved to {out}")

        print("\n" + "=" * 70)
        print(f"{'Config':<25} {'HitRate':>8} {'L1':>5} {'L2':>5} {'DriftRej':>9} {'WrongHits':>10} {'Latency(ms)':>12}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<25} {r['hit_rate']:>8.4f} {r['hits_l1']:>5} "
                  f"{r['hits_l2']:>5} {r['drift_rejections']:>9} "
                  f"{r['wrong_hits_same_img_diff_q']:>10} {r['avg_latency_ms']:>12.4f}")
        print("=" * 70)