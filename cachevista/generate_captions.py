"""
Runs BLIP captioning on all COCO images and saves captions to data/drift/captions.json.

NOTE: captions.json is NOT currently consumed by any downstream script.
generate_drift_data.py constructs MLP labels purely from question type identity
(same question type = no drift, different question type = drift) with no reference
to BLIP output similarity. This script is a remnant of an earlier design where
labels would be grounded in VLM output equivalence — that approach was not
implemented. captions.json is therefore an optional artifact for manual analysis
only, not a required pipeline step.

The required pipeline is:
    download_coco → generate_questions → generate_drift_data → train → evaluate

If caption-based label grounding is implemented in the future, the labelling
logic in generate_drift_data.py must be updated to read captions.json and
compute caption similarity (e.g., BERTScore or sentence similarity) as the
label signal rather than question type identity.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from cachevista.config import load
from cachevista.utils import get_device


def generate_captions(img_dir: Path, out_path: Path, batch_size: int = 16,
                      skip_if_exists: bool = True):
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"no .jpg images found in {img_dir}")

    # load any existing partial results to support resumption after a crash
    captions: dict[str, str] = {}
    if out_path.exists():
        with open(out_path) as f:
            captions = json.load(f)
        already_done = sum(1 for p in paths if p.name in captions)
        # skip only if the file is complete — a partial file should be resumed
        if skip_if_exists and already_done == len(paths):
            print(f"captions already complete at {out_path}, skipping")
            return
        print(f"resuming — {already_done}/{len(paths)} already captioned")

    device = get_device()
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"loading BLIP on {device} ({dtype})...")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    )
    model = model.to(device)
    model.eval()
    print(f"BLIP loaded — captioning {len(paths)} images...")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [p for p in paths if p.name not in captions]

    for batch_start in range(0, len(pending), batch_size):
        batch_paths = pending[batch_start:batch_start + batch_size]

        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            # num_beams=1 (greedy) — one output per input guaranteed
            out = model.generate(**inputs, max_new_tokens=30, num_beams=1)

        batch_captions = processor.batch_decode(out, skip_special_tokens=True)
        if len(batch_captions) != len(batch_paths):
            raise RuntimeError(
                f"batch {batch_start // batch_size}: got {len(batch_captions)} captions "
                f"for {len(batch_paths)} images — count mismatch"
            )

        for p, cap in zip(batch_paths, batch_captions):
            captions[p.name] = cap

        # checkpoint every 10 batches so a crash doesn't lose all progress
        if (batch_start // batch_size + 1) % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(captions, f, indent=2)
            done_count = batch_start + len(batch_paths)
            print(f"  {done_count}/{len(pending)} done (checkpointed)")

    # final save
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"saved {len(captions)} captions to {out_path}")
    for name, cap in list(captions.items())[:3]:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    cfg = load()
    img_dir = Path(cfg["data"]["coco_dir"])
    out_path = Path(cfg["data"]["drift_data_dir"]) / "captions.json"
    generate_captions(img_dir, out_path)