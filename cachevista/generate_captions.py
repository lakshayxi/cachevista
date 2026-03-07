"""
Runs BLIP captioning on all COCO images and saves captions to data/drift/captions.json.
Run this once before generate_drift_data.py.

Captions are used to compute output similarity between image pairs — giving the MLP
ground-truth labels based on whether a VLM would produce equivalent responses,
rather than just embedding similarity.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from cachevista.config import load


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_captions(img_dir: Path, out_path: Path, batch_size: int = 16):
    device = get_device()
    print(f"Loading BLIP on {device}...")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(device)
    model.eval()
    print("BLIP loaded")

    paths = sorted(img_dir.glob("*.jpg"))
    print(f"captioning {len(paths)} images...")

    captions = {}
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)

        batch_captions = processor.batch_decode(out, skip_special_tokens=True)
        for p, cap in zip(batch_paths, batch_captions):
            captions[p.name] = cap

        if (i // batch_size + 1) % 5 == 0:
            print(f"  {i + len(batch_paths)}/{len(paths)} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"saved {len(captions)} captions to {out_path}")
    # show a few samples
    for name, cap in list(captions.items())[:3]:
        print(f"  {name}: {cap}")


if __name__ == "__main__":
    cfg = load()
    img_dir = Path(cfg["data"]["coco_dir"])
    out_path = Path(cfg["data"]["drift_data_dir"]) / "captions.json"
    generate_captions(img_dir, out_path)